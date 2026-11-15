"""Tests for the two-track risk overlay.

The property worth locking is that the overlay cannot raise the ceiling. If a
vol-scaled track ever returns a higher best than the unrestricted one, the
overlay has started selecting rather than sizing, and every claim about the two
tracks differing only in risk would be false.
"""

from __future__ import annotations

import math

import two_tracks as tt


def test_vol_scale_caps_at_one() -> None:
    quiet = [0.001, -0.001, 0.002, -0.002]
    assert tt.vol_scale(quiet, 0.10) == 1.0


def test_vol_scale_reduces_exposure_when_vol_exceeds_budget() -> None:
    wild = [0.5, -0.4, 0.6, -0.5]
    scale = tt.vol_scale(wild, 0.10)
    assert 0.0 < scale < 1.0


def test_vol_scale_shrinks_with_tighter_budget() -> None:
    rets = [0.3, -0.2, 0.35, -0.3]
    loose = tt.vol_scale(rets, 0.20)
    tight = tt.vol_scale(rets, 0.05)
    assert tight < loose <= 1.0


def test_vol_scale_handles_degenerate_input() -> None:
    assert tt.vol_scale([0.01], 0.1) == 1.0
    assert tt.vol_scale([0.01] * 8, 0.1) == 1.0
    try:
        tt.vol_scale([0.01, 0.02], 0.0)
    except ValueError:
        return
    raise AssertionError("risk_budget=0 must raise")


# --- survivable leverage ---


def test_half_dip_allows_2x() -> None:
    # Low of 50 against a start of 100 is a 50% excursion -> 1/0.5 = 2x.
    assert tt.survivable_leverage([100.0, 50.0, 90.0], 100.0) == 2.0


def test_ninety_percent_dip_allows_about_1_11x() -> None:
    assert math.isclose(tt.survivable_leverage([10.0], 100.0), 1 / 0.9)


def test_total_loss_is_unholdable() -> None:
    assert tt.survivable_leverage([0.0], 100.0) == 0.0
    assert tt.survivable_leverage([], 100.0) == 0.0
    assert tt.survivable_leverage([5.0], 0.0) == 0.0


def test_survivable_leverage_uses_worst_low_not_first() -> None:
    good_then_bad = [200.0, 150.0, 10.0, 180.0]
    assert math.isclose(tt.survivable_leverage(good_then_bad, 100.0), 1 / 0.9)


# --- sizing semantics: scale the P&L, never the principal ---


def test_scale_window_scales_pnl_not_capital() -> None:
    # The bug this pins: scaling the principal, (1+r)*size, turned a 5% gain
    # into a 48% loss at size=0.5. Scaling the P&L leaves it a 2.5% gain.
    assert math.isclose(tt.scale_window(0.05, 0.5), 1.025)
    assert tt.scale_window(0.05, 0.5) > 1.0


def test_scale_window_shrinks_losses_toward_one() -> None:
    assert math.isclose(tt.scale_window(-0.80, 0.5), 0.60)
    assert tt.scale_window(-0.80, 0.5) > tt.scale_window(-0.80, 1.0)


def test_scale_window_is_identity_at_full_size() -> None:
    for r in (0.0, 0.5, -0.5, 29.2):
        assert math.isclose(tt.scale_window(r, 1.0), 1.0 + r)


def test_scale_window_is_flat_at_zero_size() -> None:
    assert tt.scale_window(29.2, 0.0) == 1.0
    assert tt.scale_window(-0.99, 0.0) == 1.0


def test_scale_window_rejects_negative_size() -> None:
    try:
        tt.scale_window(0.1, -0.5)
    except ValueError:
        return
    raise AssertionError("negative size must raise")


def test_scale_window_never_invents_a_worse_loss() -> None:
    """A size in (0,1) can only move a multiplier toward 1.0, never past it."""
    for r in (-0.95, -0.30, 0.0, 0.07, 38.2):
        full = 1.0 + r
        for k in (0.1, 0.5, 0.9):
            scaled = tt.scale_window(r, k)
            if full < 1.0:
                assert scaled >= full
            if full > 1.0:
                assert scaled <= full


# --- the asymmetry the module exists to show ---


def test_overlay_never_raises_the_ceiling() -> None:
    """The only hard invariant: sizing cannot manufacture a bigger best case.

    Deliberately asserted on `best` only. The median moves the other way on a
    losing sample - scaling exposure down shrinks a loss, so a 0.2x window
    becomes 0.4x - which is the overlay working, not a defect. Locking p50 here
    would encode a falsehood.
    """
    a = [298.0, 0.2, 0.1, 45.0, 0.05]
    b = [12.0, 0.4, 0.3, 2.0, 0.2]
    s = tt.compare_tracks(a, b)
    assert s["b_best"] <= s["a_best"]


def test_overlay_shrinks_losses_so_median_can_rise() -> None:
    a = [298.0, 0.2, 0.1, 45.0, 0.05]
    b = [12.0, 0.4, 0.3, 2.0, 0.2]
    s = tt.compare_tracks(a, b)
    assert s["b_p50"] > s["a_p50"]


def test_survivable_leverage_infinite_when_no_adverse_low() -> None:
    # A window that only went up imposes no leverage limit. This input used to
    # raise ZeroDivisionError, because the guard checked only the >= 1.0 end.
    assert tt.survivable_leverage([100.0, 150.0, 120.0], 100.0) == math.inf
    assert tt.survivable_leverage([100.0] * 6, 100.0) == math.inf


def test_compare_tracks_requires_pairing() -> None:
    try:
        tt.compare_tracks([1.0], [1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("mismatched lengths must raise")


def test_compare_tracks_handles_empty() -> None:
    assert tt.compare_tracks([], [])["windows"] == 0.0


def test_compare_tracks_reports_worst_loss() -> None:
    s = tt.compare_tracks([10.0, 0.0], [4.0, 0.5])
    assert s["a_worst_loss"] == 1.0
    assert math.isclose(s["b_worst_loss"], 0.5)
