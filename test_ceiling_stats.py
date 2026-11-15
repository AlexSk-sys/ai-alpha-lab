"""Tests for the multiple-testing corrections.

The two properties worth locking: effective_n must refuse to inflate a
correlated window count into independent observations, and pbo_cscv must
return ~0.5 on pure noise. If pbo_cscv cannot detect that the in-sample winner
is a coin flip, it is not measuring anything and the reported PBO is a number
without meaning.
"""

from __future__ import annotations

import math
import random

import ceiling_stats as cst


# --- effective_n: the cap that makes the rest honest ---


def test_effective_n_collapses_overlapping_windows() -> None:
    # 56 rows of 24-month windows over an 80-month span are ~3 independent draws.
    assert cst.effective_n(56, 24, 80) == 80 / 24


def test_effective_n_never_exceeds_row_count() -> None:
    assert cst.effective_n(2, 24, 800) == 2.0


def test_effective_n_floors_at_one() -> None:
    assert cst.effective_n(1, 24, 30) == 1.0
    assert cst.effective_n(10, 0, 80) == 1.0
    assert cst.effective_n(10, 24, 0) == 1.0


def test_effective_n_shrinks_as_hold_grows() -> None:
    short = cst.effective_n(56, 12, 80)
    long = cst.effective_n(56, 48, 80)
    assert long < short


# --- rate conversion ---


def test_monthly_equiv_compounds_back_to_total() -> None:
    monthly = cst.monthly_equiv([297.4], 12)  # the 298x 12-month window
    assert math.isclose((1.0 + monthly[0]) ** 12 - 1.0, 297.4, rel_tol=1e-9)


def test_monthly_equiv_preserves_sign() -> None:
    assert cst.monthly_equiv([-0.5], 12)[0] < 0.0
    assert cst.monthly_equiv([0.0], 12)[0] == 0.0


def test_monthly_equiv_rejects_total_loss_and_bad_hold() -> None:
    assert cst.monthly_equiv([-1.0], 12) == []
    try:
        cst.monthly_equiv([0.1], 0)
    except ValueError:
        return
    raise AssertionError("hold=0 must raise")


# --- sharpe / moments ---


def test_sharpe_annualizes_flat_series() -> None:
    # +1% every month -> 0.01 / 0 * sqrt(12); zero sd, so nan not inf.
    assert math.isnan(cst.sharpe([0.01] * 5))
    assert math.isnan(cst.sharpe([0.01]))


def test_sharpe_sign_follows_mean() -> None:
    up = cst.sharpe([0.05, 0.02, 0.03, 0.04])
    down = cst.sharpe([-0.05, -0.02, -0.03, -0.04])
    assert up > 0.0 > down


def test_skew_kurt_on_constant_series_is_neutral() -> None:
    g3, g4 = cst.skew_kurt([0.02] * 10)
    assert g3 == 0.0
    assert g4 == 3.0


def test_skew_kurt_uses_non_excess_kurtosis() -> None:
    _, g4 = cst.skew_kurt([-0.1, 0.1, -0.1, 0.1, 0.0])
    assert 1.0 <= g4 <= 6.0


# --- SR0 hurdle ---


def test_sr0_zero_for_single_trial() -> None:
    assert cst.sr0(1, [0.5]) == 0.0


def test_sr0_rises_with_more_trials() -> None:
    sharpes = [0.1 * math.sin(i) for i in range(20)]
    assert cst.sr0(10, sharpes) < cst.sr0(200, sharpes)


def test_sr0_zero_when_trials_identical() -> None:
    assert cst.sr0(5, [0.3] * 5) == 0.0


# --- PSR ---


def test_psr_monotone_in_observed_sharpe() -> None:
    low = cst.psr(0.2, 50, 0.0, 3.0, 0.3)
    high = cst.psr(0.8, 50, 0.0, 3.0, 0.3)
    assert 0.0 <= low < high <= 1.0


def test_psr_needs_at_least_two_observations() -> None:
    assert math.isnan(cst.psr(0.5, 1, 0.0, 3.0, 0.0))


def test_psr_rises_with_more_effective_observations() -> None:
    """More independent data at the same Sharpe means more confidence.

    The direction matters because effective_n is small here: a 24-month
    overlapping design yields ~3 independent observations, and the PSR at n=3 is
    barely above a coin flip. That weak reading is the honest one.
    """
    a = cst.psr(0.5, 100, 0.0, 3.0, 0.4)
    b = cst.psr(0.5, 3, 0.0, 3.0, 0.4)
    assert a > b


def test_psr_falls_as_the_reference_rises() -> None:
    """A higher bar is harder to clear, so P(true SR > bar) must go down."""
    low_bar = cst.psr(0.5, 20, 0.0, 3.0, 0.2)
    high_bar = cst.psr(0.5, 20, 0.0, 3.0, 0.6)
    assert 0.0 <= high_bar < low_bar <= 1.0


def test_dshr_reports_hurdle_and_edge() -> None:
    trials = [0.1 * math.cos(i) for i in range(12)]
    d = cst.dshr(1.0, trials, 3.0)
    assert d["observed"] == 1.0
    assert d["edge_over_sr0"] == 1.0 - d["sr0"]


# --- PBO via CSCV ---


def test_pbo_is_zero_when_one_strategy_truly_dominates() -> None:
    matrix = [[10.0] * 8 for _ in range(4)]
    for s in range(1, 4):
        matrix[s] = [0.0] * 8
    assert cst.pbo_cscv(matrix)["pbo"] == 0.0


def test_pbo_is_near_half_on_pure_noise() -> None:
    rng = random.Random(7)
    matrix = [[rng.gauss(0.0, 1.0) for _ in range(24)] for _ in range(6)]
    pbo = cst.pbo_cscv(matrix)["pbo"]
    assert 0.2 < pbo < 0.8, f"PBO should hover near 0.5 on noise, got {pbo}"


def test_pbo_detects_inverted_persistence() -> None:
    """Winner in-sample is the loser out-of-sample: PBO must approach 1."""
    n_time = 32
    matrix = []
    for s in range(4):
        row = []
        for t in range(n_time):
            row.append(1.0 if (t < n_time // 2) == (s == 0) else -1.0)
        matrix.append(row)
    assert cst.pbo_cscv(matrix)["pbo"] > 0.5


def test_pbo_rejects_degenerate_input() -> None:
    assert math.isnan(cst.pbo_cscv([])["pbo"])
    assert math.isnan(cst.pbo_cscv([[1.0, 2.0]])["pbo"])
    assert math.isnan(cst.pbo_cscv([[1.0] * 8, [1.0] * 8], n_splits=7)["pbo"])
