"""Backtest loop: turn the agent's signals into money via the broker API.

WHY: this is the measurement apparatus. It walks real candles forward, feeds
each bar's signal to `broker.Broker` - the already-verified execution
simulator - and records the equity path plus the costs. The harness measures
HOW both tracks fail against the $1M target (ceiling.py shows neither can
reach it); a loop that flattered either track would invalidate the study.

Sharpe annualization: the ratio is computed from PER-BAR simple equity
returns and scaled by sqrt(PERIODS_PER_YEAR) with PERIODS_PER_YEAR = 8760,
i.e. hourly bars (24 * 365). Reported alongside every figure so the scaling
is never implicit.

WHAT IT DOES NOT DO: no network, no LLM, no lookahead (signal at bar `i`
sees only bars `0..i`), no randomness beyond the explicit seed. Every
division is guarded: empty/short curves, zero variance, and zero prior
equity all yield a Sharpe of 0.0 instead of NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from agent import Journal, SignalKind, decide
from broker import Broker, RiskViolation
from config import FUNDING_INTERVAL_HOURS, START_CAPITAL_RUB, CostModel, RiskBudget
from data import Candle
from risk import RiskOverlay

# Hourly bars -> 24 * 365 bars per year. The Sharpe scaling assumes this
# cadence; a different bar interval must change this constant, not the code.
PERIODS_PER_YEAR: int = 24 * 365


@dataclass(frozen=True)
class RunResult:
    """Immutable record of one seed on one track."""

    seed: int
    track: str
    bars: int
    final_equity: float
    equity_curve: tuple[float, ...]
    max_drawdown: float
    fees_paid: float
    funding_paid: float
    turnover: float
    n_trades: int
    ruined: bool
    sharpe: float


def is_ruined(equity_curve: Sequence[float], capital: float) -> bool:
    """Whether the path touched the broker's absorbing floor of 0.0."""
    if not equity_curve:
        return False
    if capital <= 0.0:
        raise ValueError(f"capital must be positive, got {capital}")
    return min(equity_curve) <= 0.0


def sharpe_of(equity_curve: Sequence[float]) -> float:
    """Sharpe-like ratio from per-bar simple returns, 0.0 when undefined."""
    if len(equity_curve) < 2:
        return 0.0
    returns: list[float] = []
    for prev, curr in zip(equity_curve, equity_curve[1:]):
        if prev <= 0.0:
            returns.append(0.0)
        else:
            returns.append(curr / prev - 1.0)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if var <= 0.0:
        return 0.0
    return mean / math.sqrt(var) * math.sqrt(PERIODS_PER_YEAR)


def run_backtest(
    bars: Sequence[Candle],
    seed: int,
    budget: RiskBudget,
    cost: CostModel,
    capital: float = START_CAPITAL_RUB,
    symbol: str = "BTCUSDT",
    funding_rate: float = 0.0,
) -> RunResult:
    """Walk `bars` forward on one track. Deterministic in (bars, seed)."""
    if not bars:
        raise ValueError("bars must be non-empty")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if capital <= 0.0:
        raise ValueError(f"capital must be positive, got {capital}")
    overlay = RiskOverlay(budget)
    broker = Broker(capital, cost, budget)
    journal = Journal(capacity=64)
    curve: list[float] = []
    fees = 0.0
    funding_paid = 0.0
    turnover = 0.0
    n_trades = 0
    peak = capital
    max_dd = 0.0

    for i, bar in enumerate(bars):
        price = bar.close
        broker.mark_and_liquidate({symbol: price})
        book = broker.positions()
        prices = {symbol: price} if symbol in book else {}
        drawdown = broker.drawdown(prices if prices else {symbol: price})

        if overlay.should_halt(drawdown):
            if symbol in broker.positions():
                fill = broker.close(symbol, price)
                fees += fill.fee
                turnover += abs(fill.qty) * fill.price
                n_trades += 1
        else:
            signal = decide(bars, i, overlay, seed)
            journal.append(signal.justification)
            held = broker.positions().get(symbol)
            held_dir = 0
            if held is not None:
                held_dir = 1 if held.qty > 0.0 else -1
            want = 0
            if signal.kind is SignalKind.LONG:
                want = 1
            elif signal.kind is SignalKind.SHORT:
                want = -1
            if want == 0:
                if held is not None:
                    fill = broker.close(symbol, price)
                    fees += fill.fee
                    turnover += abs(fill.qty) * fill.price
                    n_trades += 1
            elif want != held_dir:
                if held is not None:
                    fill = broker.close(symbol, price)
                    fees += fill.fee
                    turnover += abs(fill.qty) * fill.price
                    n_trades += 1
                fraction = overlay.clamp_fraction(signal.fraction, drawdown)
                raw_lev = abs(signal.leverage)
                lev = overlay.clamp_leverage(raw_lev, drawdown)
                open_count = len(broker.positions())
                if (
                    fraction > 0.0
                    and lev >= 1.0
                    and overlay.allows_new_position(open_count)
                ):
                    try:
                        fill = broker.open(
                            symbol, price, fraction, lev if want > 0 else -lev
                        )
                    except RiskViolation:
                        pass
                    else:
                        fees += fill.fee
                        turnover += abs(fill.qty) * fill.price

        if (i + 1) % FUNDING_INTERVAL_HOURS == 0 and funding_rate != 0.0:
            held = broker.positions().get(symbol)
            if held is not None:
                amount = abs(held.qty) * held.entry_price * funding_rate
                funding_paid += amount if held.qty > 0.0 else -amount
            broker.apply_funding({symbol: funding_rate})

        book = broker.positions()
        equity = broker.equity({symbol: price} if symbol in book else {})
        curve.append(equity)
        if equity > peak:
            peak = equity
        if peak > 0.0:
            max_dd = max(max_dd, 1.0 - equity / peak)

    curve_t = tuple(curve)
    return RunResult(
        seed=seed,
        track=budget.name,
        bars=len(bars),
        final_equity=curve_t[-1],
        equity_curve=curve_t,
        max_drawdown=max_dd,
        fees_paid=fees,
        funding_paid=funding_paid,
        turnover=turnover,
        n_trades=n_trades,
        ruined=is_ruined(curve_t, capital),
        sharpe=sharpe_of(curve_t),
    )
