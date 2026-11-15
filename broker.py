"""Execution simulator for the two-track agent experiment.

The broker is the component whose bugs would silently invalidate every result: a
sign error in PnL or a missed liquidation still yields a plausible-looking equity
curve. So the invariants it must never break are pinned in test_broker.py, and
the design choices below exist to make those invariants provable rather than
merely intended.

Accounting model
----------------
    notional = available_cash * fraction * |leverage|
    margin   = notional / |leverage|
    qty      = +/- notional / entry_price          (signed: + long, - short)
    pnl(p)   = qty * (p - entry_price)
    equity   = cash + sum(margin + pnl)   over open positions

A position's margin is held in `cash` as encumbered collateral, not spent, so it
is added back when computing equity and released on close. That is why equity at
entry is `capital - fee` rather than `capital - margin - fee`.

Bankruptcy is absorbing, not reflective: a liquidated position loses at most its
margin, cash is clamped at zero, and equity is floored at zero. An account that
blows up is worth 0, never negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config import CostModel, RiskBudget


class RiskViolation(Exception):
    """An order was refused by the track's risk envelope."""


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Position:
    """An open position. `qty` is signed: positive long, negative short."""

    symbol: str
    qty: float
    entry_price: float
    margin: float

    def unrealised(self, mark: float) -> float:
        return self.qty * (mark - self.entry_price)

    def notional_at(self, mark: float) -> float:
        return abs(self.qty) * mark


@dataclass(frozen=True)
class Fill:
    """The record of one executed order, kept for the audit ledger."""

    symbol: str
    side: Side
    qty: float
    price: float
    fee: float


class Broker:
    """Single-account spot/perp simulator with isolated-margin liquidation."""

    def __init__(
        self,
        capital: float,
        cost_model: CostModel,
        risk_budget: RiskBudget,
    ) -> None:
        if capital <= 0.0:
            raise ValueError(f"capital must be positive, got {capital}")
        self.cost = cost_model
        self.budget = risk_budget
        self.cash = capital
        self.peak = capital
        self._positions: dict[str, Position] = {}
        self._last_drawdown: float = 0.0
        self.fills: list[Fill] = []

    # --- introspection ----------------------------------------------------
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for symbol, pos in self._positions.items():
            total += pos.margin + pos.unrealised(prices[symbol])
        return max(0.0, total)

    def drawdown(self, prices: dict[str, float]) -> float:
        self._raise_peak(self.equity(prices))
        if self.peak <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.equity(prices) / self.peak)

    def is_halted(self, prices: dict[str, float] | None = None) -> bool:
        """Whether the track's circuit breaker has tripped.

        With no prices supplied the last observed equity is reused, so a caller
        can query the breaker between bars without re-marking the book.
        """
        if prices is None:
            drawdown = self._last_drawdown
        else:
            drawdown = self.drawdown(prices)
        return self.budget.is_halted(drawdown)

    # --- trading ----------------------------------------------------------
    def open(
        self,
        symbol: str,
        mid: float,
        fraction: float,
        leverage: float,
    ) -> Fill:
        """Open a position sized as `fraction` of cash at `leverage`.

        Negative leverage opens a short. Raises RiskViolation when the track's
        envelope refuses the order, and ValueError when the venue cannot express
        it at all (spot with leverage).
        """
        if mid <= 0.0:
            raise ValueError(f"mid must be positive, got {mid}")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if leverage == 0.0:
            raise ValueError("leverage must be non-zero")
        if not self.cost.supports_leverage and abs(leverage) > 1.0:
            raise ValueError(
                f"{self.cost.venue} is spot and cannot express leverage {leverage}"
            )
        if not self.budget.permits(abs(leverage), fraction):
            raise RiskViolation(
                f"{self.budget.name} refuses leverage={leverage} at fraction={fraction}"
            )

        is_buy = leverage > 0.0
        price = self.cost.fill_price(mid, side_is_buy=is_buy)
        notional = self.cash * fraction * abs(leverage)
        margin = notional / abs(leverage)
        fee = self.cost.fee(notional)
        qty = (notional / price) * (1.0 if is_buy else -1.0)

        if symbol in self._positions:
            raise ValueError(f"position already open on {symbol}")
        self.cash -= margin + fee
        self._positions[symbol] = Position(symbol, qty, price, margin)

        fill = Fill(symbol, Side.BUY if is_buy else Side.SELL, qty, price, fee)
        self.fills.append(fill)
        return fill

    def close(self, symbol: str, mid: float) -> Fill:
        """Close an open position at `mid`, realising PnL and the exit fee."""
        pos = self._positions[symbol]
        price = self.cost.fill_price(mid, side_is_buy=pos.qty < 0.0)
        fee = self.cost.fee(pos.notional_at(price))
        pnl = pos.qty * (price - pos.entry_price)

        self.cash += pos.margin + pnl - fee
        self.cash = max(0.0, self.cash)
        del self._positions[symbol]

        fill = Fill(
            symbol, Side.SELL if pos.qty > 0.0 else Side.BUY, -pos.qty, price, fee
        )
        self.fills.append(fill)
        return fill

    # --- mark-to-market ---------------------------------------------------
    def mark_and_liquidate(self, prices: dict[str, float]) -> list[str]:
        """Liquidate positions whose loss has consumed their margin.

        Returns the symbols liquidated, in a stable order. A position is closed
        at a capped loss of exactly its margin; the account cannot go negative.
        """
        liquidated: list[str] = []
        for symbol in sorted(self._positions):
            pos = self._positions[symbol]
            mark = prices[symbol]
            if pos.unrealised(mark) <= -pos.margin:
                self.cash = max(0.0, self.cash)
                del self._positions[symbol]
                liquidated.append(symbol)
        self._last_drawdown = self.drawdown(prices)
        return liquidated

    def apply_funding(self, rates: dict[str, float]) -> None:
        """Charge or credit perpetual funding for one interval.

        A long pays when the rate is positive; a short receives. Negative rates
        reverse both. Funding is an unconditioned cash flow, so it can push an
        account negative on its own - the equity floor absorbs that.
        """
        for symbol, pos in self._positions.items():
            rate = rates.get(symbol, 0.0)
            amount = pos.notional_at(pos.entry_price) * rate
            self.cash += -amount if pos.qty > 0.0 else amount

    # --- internals --------------------------------------------------------
    def _raise_peak(self, equity: float) -> None:
        if equity > self.peak:
            self.peak = equity
