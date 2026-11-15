"""Invariants for the canonical study parameters.

These tests exist to stop the parameters drifting. A silently changed fee, a
relaxed risk envelope, or a reworded target would invalidate every result in the
study without breaking any strategy code, so the numbers themselves are pinned.
"""

from __future__ import annotations

import pytest

import config
from config import RUB_PER_USD_ALT, CostModel, RiskBudget, Track, Venue


class TestTarget:
    def test_frozen_rate_demands_1000x(self) -> None:
        assert config.required_multiple() == 1000.0

    def test_alternative_rate_is_reported_too(self) -> None:
        # The headline claim is rate-sensitive; both must stay visible.
        assert config.required_multiple(RUB_PER_USD_ALT) == 250.0

    def test_no_monthly_contributions(self) -> None:
        # Capital is a one-off 100k. Any contribution model would silently
        # change every multiple in the study.
        assert config.START_CAPITAL_RUB == 100_000.0
        assert config.HORIZON_MONTHS == 24

    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_non_positive_rate_rejected(self, rate: float) -> None:
        with pytest.raises(ValueError):
            config.required_multiple(rate)


class TestRiskBudgetValidation:
    def test_rejects_leverage_below_one(self) -> None:
        with pytest.raises(ValueError, match="max_leverage"):
            RiskBudget("bad", 0.5, 0.5, 1)

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_position_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="max_position_fraction"):
            RiskBudget("bad", 1.0, fraction, 1)

    def test_rejects_zero_positions(self) -> None:
        with pytest.raises(ValueError, match="max_positions"):
            RiskBudget("bad", 1.0, 0.5, 0)

    @pytest.mark.parametrize("dd", [0.0, 1.0, -0.2, 2.0])
    def test_rejects_out_of_range_drawdown(self, dd: float) -> None:
        # 0.0 is rejected on purpose: "no breaker" is spelled None, so a bare
        # 0.0 cannot be mistaken for "zero tolerance".
        with pytest.raises(ValueError, match="max_drawdown"):
            RiskBudget("bad", 1.0, 0.5, 1, max_drawdown=dd)

    def test_none_drawdown_is_a_valid_state(self) -> None:
        budget = RiskBudget("no_breaker", 1.0, 1.0, 1, max_drawdown=None)
        assert budget.max_drawdown is None
        assert budget.is_halted(0.99) is False


class TestPermits:
    def test_rejects_oversized_leverage(self) -> None:
        budget = RiskBudget(
            "b", max_leverage=2.0, max_position_fraction=0.5, max_positions=3
        )
        assert budget.permits(leverage=3.0, position_fraction=0.1) is False

    def test_rejects_overconcentrated_position(self) -> None:
        budget = RiskBudget(
            "b", max_leverage=2.0, max_position_fraction=0.5, max_positions=3
        )
        assert budget.permits(leverage=1.0, position_fraction=0.6) is False

    def test_accepts_within_envelope(self) -> None:
        budget = RiskBudget(
            "b", max_leverage=2.0, max_position_fraction=0.5, max_positions=3
        )
        assert budget.permits(leverage=2.0, position_fraction=0.5) is True

    def test_boundary_is_inclusive(self) -> None:
        budget = RiskBudget(
            "b", max_leverage=2.0, max_position_fraction=0.5, max_positions=3
        )
        assert budget.permits(leverage=2.0, position_fraction=0.5) is True


class TestHalt:
    def test_trips_at_threshold(self) -> None:
        budget = RiskBudget("b", 1.0, 1.0, 1, max_drawdown=0.20)
        assert budget.is_halted(0.20) is True

    def test_does_not_trip_below_threshold(self) -> None:
        budget = RiskBudget("b", 1.0, 1.0, 1, max_drawdown=0.20)
        assert budget.is_halted(0.19) is False

    def test_rejects_negative_drawdown(self) -> None:
        budget = RiskBudget("b", 1.0, 1.0, 1, max_drawdown=0.20)
        with pytest.raises(ValueError, match="drawdown"):
            budget.is_halted(-0.1)


class TestTrackSeparation:
    def test_unrestricted_accepts_what_budgeted_rejects(self) -> None:
        # This single asymmetry is the experiment. If it ever collapses, the
        # two tracks are no longer a meaningful comparison.
        assert config.TRACKS[Track.A_UNRESTRICTED].permits(3.0, 0.5) is True
        assert config.TRACKS[Track.B_RISK_BUDGETED].permits(3.0, 0.5) is False

    def test_budgeted_tighter_than_unrestricted(self) -> None:
        a = config.TRACKS[Track.A_UNRESTRICTED]
        b = config.TRACKS[Track.B_RISK_BUDGETED]
        assert b.max_leverage < a.max_leverage
        assert b.max_position_fraction < a.max_position_fraction
        assert b.max_drawdown is not None and a.max_drawdown is None

    def test_control_is_unlevered(self) -> None:
        assert config.TRACKS[Track.C_CONTROL].max_leverage == 1.0


class TestCostModel:
    def test_fee_scales_with_notional(self) -> None:
        model = CostModel(Venue.PERP_TAKER, 0.00045, 4.0, True)
        assert model.fee(100_000.0) == pytest.approx(45.0)

    def test_rejects_negative_notional(self) -> None:
        model = CostModel(Venue.PERP_TAKER, 0.00045, 4.0, True)
        with pytest.raises(ValueError, match="notional"):
            model.fee(-1.0)

    def test_buy_slips_up_and_sell_slips_down(self) -> None:
        model = CostModel(Venue.PERP_TAKER, 0.00045, 4.0, True)
        assert model.fill_price(100.0, side_is_buy=True) > 100.0
        assert model.fill_price(100.0, side_is_buy=False) < 100.0

    def test_zero_slippage_is_neutral(self) -> None:
        model = CostModel(Venue.SPOT_MAKER, 0.0010, 0.0, False)
        assert model.fill_price(100.0, side_is_buy=True) == 100.0
        assert model.fill_price(100.0, side_is_buy=False) == 100.0

    def test_rejects_non_positive_mid(self) -> None:
        model = CostModel(Venue.SPOT_MAKER, 0.0010, 2.0, False)
        with pytest.raises(ValueError, match="mid"):
            model.fill_price(0.0, side_is_buy=True)

    def test_spot_venues_have_no_leverage(self) -> None:
        assert not config.COST_MODELS[Venue.SPOT_TAKER].supports_leverage
        assert not config.COST_MODELS[Venue.SPOT_MAKER].supports_leverage
        assert config.COST_MODELS[Venue.PERP_TAKER].supports_leverage

    def test_perp_cheaper_than_spot_taker(self) -> None:
        assert (
            config.COST_MODELS[Venue.PERP_MAKER].round_trip_fee
            < config.COST_MODELS[Venue.SPOT_TAKER].round_trip_fee
        )
