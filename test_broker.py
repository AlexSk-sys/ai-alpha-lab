"""Invariants for the execution simulator.

The broker is the component whose bugs would silently invalidate every result:
a sign error in PnL or a missed liquidation still produces plausible-looking
equity curves. These tests pin the properties that must hold for the study to
be credible, most importantly that equity can never go negative and that a
liquidated position loses exactly its margin and no more.
"""

from __future__ import annotations

import pytest

from broker import Broker, RiskViolation, Side
from config import RISK_BUDGETED, UNRESTRICTED, CostModel, Venue


def perp_cost() -> CostModel:
    return CostModel(Venue.PERP_TAKER, 0.00045, 0.0, True)


def zero_slip_budget():
    return UNRESTRICTED


class TestOpening:
    def test_opening_costs_exactly_the_fee(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        fill = broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        assert fill.fee == pytest.approx(100_000.0 * 0.00045)
        # Equity is untouched by the fee only in the sense that the fee is
        # already deducted from cash; total equity drops by exactly the fee.
        assert broker.equity({"BTCUSDT": 50_000.0}) == pytest.approx(
            100_000.0 - fill.fee
        )

    def test_round_trip_pays_fee_twice(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        open_fill = broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        close_fill = broker.close("BTCUSDT", mid=50_000.0)
        assert close_fill.fee == pytest.approx(open_fill.fee)
        # Flat round trip at an unchanged price loses exactly the two fees.
        assert broker.equity({}) == pytest.approx(
            100_000.0 - open_fill.fee - close_fill.fee
        )

    def test_spot_rejects_leverage(self) -> None:
        spot = CostModel(Venue.SPOT_TAKER, 0.0010, 0.0, False)
        broker = Broker(100_000.0, spot, UNRESTRICTED)
        with pytest.raises(ValueError, match="leverage"):
            broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=3.0)

    def test_risk_budget_rejects_oversized_order(self) -> None:
        broker = Broker(100_000.0, perp_cost(), RISK_BUDGETED)
        with pytest.raises(RiskViolation):
            broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=5.0)

    def test_rejects_unknown_symbol_price(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        with pytest.raises(KeyError):
            broker.equity({"ETHUSDT": 3_000.0})


class TestPnl:
    def test_long_profits_when_price_rises(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        assert broker.equity({"BTCUSDT": 55_000.0}) > 100_000.0

    def test_short_profits_when_price_falls(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=-1.0)
        assert broker.equity({"BTCUSDT": 45_000.0}) > 100_000.0

    def test_long_loses_when_price_falls(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        assert broker.equity({"BTCUSDT": 45_000.0}) < 100_000.0

    def test_equity_is_linear_in_price(self) -> None:
        # Guards against a hidden compounding or nonlinearity bug. The slope
        # must be the position size: a $1 move on N units is $N, so the equity
        # delta is qty * price_delta. Derived from the fill, not hardcoded.
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        fill = broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        a = broker.equity({"BTCUSDT": 51_000.0})
        b = broker.equity({"BTCUSDT": 52_000.0})
        assert b - a == pytest.approx(abs(fill.qty) * 1_000.0)


class TestLiquidation:
    def test_adverse_move_beyond_distance_liquidates(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=10.0)
        # A 20% adverse move is well past 1/10, so the position must die.
        liquidated = broker.mark_and_liquidate({"BTCUSDT": 40_000.0})
        assert liquidated == ["BTCUSDT"]
        assert broker.positions() == {}

    def test_liquidation_costs_exactly_the_margin(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=10.0)
        broker.mark_and_liquidate({"BTCUSDT": 40_000.0})
        # Margin was the whole 100k, so the account is wiped and not negative.
        assert broker.equity({}) == pytest.approx(0.0, abs=1e-6)

    def test_small_move_does_not_liquidate(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=2.0)
        assert broker.mark_and_liquidate({"BTCUSDT": 45_000.0}) == []
        assert "BTCUSDT" in broker.positions()

    def test_favourable_move_never_liquidates(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=20.0)
        assert broker.mark_and_liquidate({"BTCUSDT": 90_000.0}) == []

    def test_equity_never_negative_under_collapse(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=20.0)
        broker.mark_and_liquidate({"BTCUSDT": 1_000.0})
        assert broker.equity({"BTCUSDT": 0.5}) >= 0.0


class TestFunding:
    def test_long_pays_when_rate_positive(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        before = broker.equity({"BTCUSDT": 50_000.0})
        broker.apply_funding({"BTCUSDT": 0.0001})
        after = broker.equity({"BTCUSDT": 50_000.0})
        assert after < before

    def test_short_receives_when_rate_positive(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=-1.0)
        before = broker.equity({"BTCUSDT": 50_000.0})
        broker.apply_funding({"BTCUSDT": 0.0001})
        after = broker.equity({"BTCUSDT": 50_000.0})
        assert after > before

    def test_funding_on_flat_book_is_free(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        before = broker.equity({})
        broker.apply_funding({"BTCUSDT": 0.0001})
        assert broker.equity({}) == pytest.approx(before)

    def test_funding_cannot_drive_equity_negative(self) -> None:
        broker = Broker(100_000.0, perp_cost(), zero_slip_budget())
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=1.0)
        broker.apply_funding({"BTCUSDT": 1.0})
        assert broker.equity({"BTCUSDT": 50_000.0}) >= 0.0


class TestCircuitBreaker:
    def test_halt_engages_at_budget_drawdown(self) -> None:
        # fraction must sit at the budget cap (0.35); a smaller size at this
        # price only produces an 18% drawdown, which is under the 20% trip.
        broker = Broker(100_000.0, perp_cost(), RISK_BUDGETED)
        broker.open("BTCUSDT", mid=50_000.0, fraction=0.35, leverage=1.0)
        broker.mark_and_liquidate({"BTCUSDT": 20_000.0})
        assert broker.is_halted() is True

    def test_unrestricted_track_never_halts(self) -> None:
        broker = Broker(100_000.0, perp_cost(), UNRESTRICTED)
        broker.open("BTCUSDT", mid=50_000.0, fraction=1.0, leverage=10.0)
        broker.mark_and_liquidate({"BTCUSDT": 5_000.0})
        assert broker.is_halted() is False
