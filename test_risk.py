"""Invariants for the risk overlay.

The overlay is the ONLY semantic difference between tracks A and B, so these
tests pin that the unrestricted arm is permissive-but-bounded (never disabled)
and that the budgeted arm actually binds on all three knobs.
"""

from __future__ import annotations

import pytest

from config import RISK_BUDGETED, UNRESTRICTED
from risk import RiskOverlay


def unrestricted() -> RiskOverlay:
    return RiskOverlay(UNRESTRICTED)


def budgeted() -> RiskOverlay:
    return RiskOverlay(RISK_BUDGETED)


class TestSemanticDifference:
    def test_unrestricted_passes_full_conviction_through(self) -> None:
        overlay = unrestricted()
        assert overlay.clamp_fraction(1.0, 0.0) == pytest.approx(1.0)
        assert overlay.clamp_leverage(3.0, 0.0) == pytest.approx(3.0)

    def test_budgeted_caps_fraction_at_envelope(self) -> None:
        assert budgeted().clamp_fraction(1.0, 0.0) == pytest.approx(0.35)

    def test_budgeted_caps_leverage_at_envelope(self) -> None:
        assert budgeted().clamp_leverage(5.0, 0.0) == pytest.approx(2.0)

    def test_unrestricted_still_caps_leverage_at_venue_ceiling(self) -> None:
        # Permissive is not disabled: the venue ceiling (75x, config.UNRESTRICTED)
        # still binds, so Track A cannot express infinite size.
        assert unrestricted().clamp_leverage(100.0, 0.0) == pytest.approx(75.0)


class TestDrawdownThrottle:
    def test_unrestricted_never_throttles(self) -> None:
        overlay = unrestricted()
        assert overlay.throttle_factor(0.0) == pytest.approx(1.0)
        assert overlay.throttle_factor(0.5) == pytest.approx(1.0)

    def test_budgeted_throttle_is_identity_at_zero_drawdown(self) -> None:
        assert budgeted().throttle_factor(0.0) == pytest.approx(1.0)

    def test_budgeted_throttle_halves_at_half_max_drawdown(self) -> None:
        assert budgeted().throttle_factor(0.10) == pytest.approx(0.5)

    def test_budgeted_throttle_is_zero_at_breaker(self) -> None:
        assert budgeted().throttle_factor(0.20) == pytest.approx(0.0)
        assert budgeted().throttle_factor(0.50) == pytest.approx(0.0)

    def test_throttle_scales_clamped_size(self) -> None:
        # At half the breaker the budgeted fraction request is halved again.
        assert budgeted().clamp_fraction(1.0, 0.10) == pytest.approx(0.175)

    def test_rejects_negative_drawdown(self) -> None:
        with pytest.raises(ValueError, match="drawdown"):
            budgeted().throttle_factor(-0.1)


class TestHaltAndCapacity:
    def test_budgeted_halts_at_breaker(self) -> None:
        assert budgeted().should_halt(0.20) is True
        assert budgeted().should_halt(0.10) is False

    def test_unrestricted_never_halts(self) -> None:
        assert unrestricted().should_halt(0.99) is False

    def test_budgeted_refuses_fourth_position(self) -> None:
        assert budgeted().allows_new_position(3) is False
        assert budgeted().allows_new_position(2) is True

    def test_unrestricted_allows_many_positions(self) -> None:
        assert unrestricted().allows_new_position(49) is True
        assert unrestricted().allows_new_position(50) is False


class TestClamps:
    def test_short_side_survives_clamp(self) -> None:
        assert budgeted().clamp_leverage(-5.0, 0.0) == pytest.approx(-2.0)

    def test_zero_request_stays_zero(self) -> None:
        assert budgeted().clamp_fraction(0.0, 0.0) == pytest.approx(0.0)
        assert budgeted().clamp_leverage(0.0, 0.0) == pytest.approx(0.0)

    def test_rejects_non_positive_price_free_fraction(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            budgeted().clamp_fraction(-0.5, 0.0)

    def test_deterministic(self) -> None:
        overlay = budgeted()
        first = (overlay.clamp_fraction(0.9, 0.05), overlay.clamp_leverage(4.0, 0.05))
        second = (overlay.clamp_fraction(0.9, 0.05), overlay.clamp_leverage(4.0, 0.05))
        assert first == second
