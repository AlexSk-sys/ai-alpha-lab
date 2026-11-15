"""Invariants for the backtest loop.

The runner turns signals into money via the existing broker API. These tests
pin that flat markets cost nothing, that every metric is guarded (no NaN, no
negatives, no crash on degenerate input), and that ruin is actually reachable
and reported - a harness that cannot report ruin cannot measure failure.
"""

from __future__ import annotations

import dataclasses

import pytest

from config import RISK_BUDGETED, START_CAPITAL_RUB, UNRESTRICTED, CostModel, Venue
from data import Candle
from runner import PERIODS_PER_YEAR, RunResult, is_ruined, run_backtest, sharpe_of


def perp() -> CostModel:
    return CostModel(Venue.PERP_TAKER, 0.00045, 0.0, True)


def trend(up: bool, n: int = 64, drift: float = 0.002) -> list[Candle]:
    bars: list[Candle] = []
    price = 50_000.0
    for i in range(n):
        price = price * (1.0 + drift) if up else price * (1.0 - drift)
        bars.append(Candle(i * 3_600_000, price, price, price, price, 10.0))
    return bars


def flat(n: int = 64) -> list[Candle]:
    return [
        Candle(i * 3_600_000, 50_000.0, 50_000.0, 50_000.0, 50_000.0, 10.0)
        for i in range(n)
    ]


class TestFlatMarket:
    def test_flat_market_costs_nothing(self) -> None:
        result = run_backtest(flat(), 0, RISK_BUDGETED, perp())
        assert result.final_equity == pytest.approx(START_CAPITAL_RUB)
        assert result.fees_paid == pytest.approx(0.0)
        assert result.turnover == pytest.approx(0.0)
        assert result.n_trades == 0
        assert result.ruined is False

    def test_flat_market_has_zero_drawdown_and_zero_sharpe(self) -> None:
        result = run_backtest(flat(), 0, RISK_BUDGETED, perp())
        assert result.max_drawdown == pytest.approx(0.0)
        assert result.sharpe == pytest.approx(0.0)


class TestMetrics:
    def test_curve_covers_every_bar(self) -> None:
        bars = trend(True)
        result = run_backtest(bars, 1, RISK_BUDGETED, perp())
        assert len(result.equity_curve) == len(bars)

    def test_metrics_are_sane_on_a_trend(self) -> None:
        result = run_backtest(trend(True), 1, RISK_BUDGETED, perp())
        assert result.final_equity >= 0.0
        assert result.fees_paid >= 0.0
        assert result.funding_paid == pytest.approx(0.0)
        assert result.turnover >= 0.0
        assert 0.0 <= result.max_drawdown <= 1.0
        assert result.n_trades >= 0

    def test_deterministic_per_seed(self) -> None:
        bars = trend(True)
        first = run_backtest(bars, 5, RISK_BUDGETED, perp())
        second = run_backtest(bars, 5, RISK_BUDGETED, perp())
        assert first.final_equity == second.final_equity
        assert first.equity_curve == second.equity_curve
        assert first.fees_paid == second.fees_paid

    def test_rejects_empty_bars(self) -> None:
        with pytest.raises(ValueError, match="bars"):
            run_backtest([], 0, RISK_BUDGETED, perp())


class TestRuin:
    def test_confiscatory_funding_ruins_the_account(self) -> None:
        result = run_backtest(trend(True), 0, UNRESTRICTED, perp(), funding_rate=1.0)
        assert result.ruined is True

    def test_is_ruined_pins_floor_semantics(self) -> None:
        assert is_ruined((100.0, 50.0, 0.0), START_CAPITAL_RUB) is True
        assert is_ruined((100.0, 50.0, 25.0), START_CAPITAL_RUB) is False

    def test_result_is_a_frozen_record(self) -> None:
        result = run_backtest(flat(n=8), 0, RISK_BUDGETED, perp())
        assert isinstance(result, RunResult)
        # setattr() rather than attribute assignment: a type checker flags the
        # latter on a frozen dataclass, and silencing that flag is exactly the
        # kind of suppression this project bans.
        # FrozenInstanceError specifically, not Exception - a bare Exception
        # would also pass on an AttributeError and hide the real behaviour.
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(result, "final_equity", 0.0)


class TestSharpe:
    def test_annualization_assumes_hourly_bars(self) -> None:
        assert PERIODS_PER_YEAR == 24 * 365

    def test_constant_curve_has_zero_sharpe(self) -> None:
        assert sharpe_of((100.0, 100.0, 100.0)) == pytest.approx(0.0)

    def test_single_point_has_zero_sharpe(self) -> None:
        assert sharpe_of((100.0,)) == pytest.approx(0.0)

    def test_empty_curve_has_zero_sharpe(self) -> None:
        assert sharpe_of(()) == pytest.approx(0.0)

    def test_steady_growth_has_positive_sharpe(self) -> None:
        curve = tuple(100.0 * (1.001**i) for i in range(32))
        assert sharpe_of(curve) > 0.0
