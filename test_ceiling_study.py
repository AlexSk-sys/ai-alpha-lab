"""Tests for the panel loader and the ceiling study.

The load-bearing test here is test_base_rates_honours_hold. A real bug shipped
through this project: _traj read the module-level HOLD instead of its `hold`
parameter, so every requested horizon was silently computed as a HOLD-month
window while only the eligibility filter changed. The symptom was subtle -
window counts and percentiles moved plausibly, but the best outcome was
byte-identical across 12/24/36/60-month horizons, which is impossible for real
price data. A test that only checked "returns a plausible number" would have
passed. These tests check the horizon is actually respected.
"""

from __future__ import annotations

from panel import Point, contiguous, month_index, yyyymm_from_index

import ceiling_study as cs


def make(
    closes: list[float], symbol: str = "TEST", start: int = 202001
) -> tuple[str, tuple[Point, ...]]:
    """Build a symbol from a close series, one month apart from `start`."""
    i0 = month_index(start)
    pts = tuple(
        Point(
            yyyymm=yyyymm_from_index(i0 + k),
            close=float(c),
            low=float(c),
        )
        for k, c in enumerate(closes)
    )
    return symbol, pts


# --- panel primitives ---


def test_month_index_round_trip() -> None:
    for yyyymm in (202001, 202012, 202101, 199912):
        assert yyyymm_from_index(month_index(yyyymm)) == yyyymm


def test_contiguous_detects_gap() -> None:
    _, clean = make([1.0] * 6)
    assert contiguous(clean)
    # Drop month index 2, leaving a 2-month hole.
    holed = (clean[0], clean[1], clean[3], clean[4], clean[5])
    assert not contiguous(holed)


# --- _traj: the horizon and the anchor ---


def test_traj_honours_hold_parameter() -> None:
    """A doubling series makes the window length directly observable."""
    _, h = make([2.0**k for k in range(40)])
    short = cs._traj(h, 0, 12)
    long = cs._traj(h, 0, 24)
    assert short is not None and long is not None
    assert short.ret == 2.0**12 - 1.0
    assert long.ret == 2.0**24 - 1.0


def test_traj_anchors_on_decision_index() -> None:
    """Flat, then 8x growth: anchoring at the decision must see the growth.

    The two windows must DISAGREE, otherwise this test cannot detect an anchor
    that silently fell back to index 0. Here the first six months are flat at
    1.0 and months 6..9 climb 2 -> 4 -> 8 -> 16, so a 3-month window from the
    decision month returns 16/2 - 1 = 7.0 while the same-length window from the
    listing date is flat at 0.0.
    """
    closes = [1.0] * 6 + [2.0, 4.0, 8.0, 16.0]
    _, h = make(closes)
    at_decision = cs._traj(h, 6, 3)
    from_listing = cs._traj(h, 0, 3)
    assert at_decision is not None and from_listing is not None
    assert at_decision.ret == 7.0
    assert from_listing.ret == 0.0
    # The anchor is the whole point: these cannot be equal.
    assert at_decision.ret != from_listing.ret


def test_traj_returns_none_when_window_incomplete() -> None:
    _, h = make([1.0] * 10)
    assert cs._traj(h, 0, 24) is None
    assert cs._traj(h, 5, 4) is not None
    # j + hold == len is exactly one short.
    assert cs._traj(h, 6, 4) is None


def test_traj_drawdown_sees_intrawindow_dip() -> None:
    """A close-to-close series would hide this; the intramonth low must not."""
    _, h = make([10.0, 2.0, 10.0])
    t = cs._traj(h, 0, 2)
    assert t is not None
    assert t.ret == 0.0  # flat start to finish
    assert t.max_dd == 0.8  # but -80% happened in between


def test_traj_rejects_nonpositive_start() -> None:
    """The guard is on the price AT the decision month, not inside the window.

    A contract that prints 0 at the decision month (delisted, wiped, or a bad
    archive row) has no computable close-to-close return, so the window must be
    rejected rather than divided by zero. A 0 *inside* the window is a real
    total loss instead: it stays in the sample and registers as max_dd = 1.0.
    """
    _, h = make([10.0, 12.0, 0.0, 5.0, 8.0])
    assert cs._traj(h, 2, 2) is None  # start is 0.0
    surviving = cs._traj(h, 0, 2)
    assert surviving is not None  # start is 10.0
    assert surviving.max_dd == 1.0  # the 0.0 inside is a -100% drawdown


# --- base_rates ---


def test_base_rates_honours_hold() -> None:
    """Regression: best must differ per horizon, not be pinned to HOLD."""
    _, h = make([2.0**k for k in range(40)])
    panel = {"D": h}
    by_hold = {hold: cs.base_rates(panel, hold=hold) for hold in (12, 24, 36)}
    assert by_hold[12]["best"] == 2.0**12 - 1.0
    assert by_hold[24]["best"] == 2.0**24 - 1.0
    assert by_hold[36]["best"] == 2.0**36 - 1.0


def test_base_rates_window_count_matches_arithmetic() -> None:
    _, h = make([1.0] * 20)
    panel = {"A": h}
    # len - hold windows per symbol, and only if the symbol is long enough.
    assert cs.base_rates(panel, hold=12)["windows"] == 20 - 12
    assert cs.base_rates(panel, hold=24)["windows"] == 0


def test_base_rates_is_selection_free() -> None:
    """Adding a dead symbol must not move the existing distribution."""
    _, h = make([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    one = cs.base_rates({"A": h}, hold=3)
    two = cs.base_rates({"A": h, "Z": make([1.0] * 7, "Z")[1]}, hold=3)
    assert one["best"] == two["best"]
    assert one["p50"] == two["p50"]


# --- rule_eval ---


def test_rules_cannot_see_the_future() -> None:
    """No rule's score may change when the future is hidden from it."""
    _, h = make([1.0 + 0.1 * k for k in range(30)])
    for j in (5, 12, 20):
        for name, rule in cs._RULES.items():
            assert rule(h, j) == rule(h[: j + 1], j), f"{name} peeked at j={j}"


def test_rule_eval_is_deterministic() -> None:
    _, h = make([1.0 + 0.05 * ((k * 7) % 5) for k in range(40)])
    panel = {"A": h, "B": make([2.0 - 0.03 * k for k in range(40)], "B")[1]}
    for name, rule in cs._RULES.items():
        a = cs.rule_eval(panel, rule, hold=6, non_overlapping=False)
        b = cs.rule_eval(panel, rule, hold=6, non_overlapping=False)
        assert a == b, f"{name} is not reproducible"


def test_non_overlapping_yields_fewer_windows() -> None:
    _, h = make([1.0 + 0.02 * k for k in range(48)])
    panel = {"A": h}
    overlapping = cs.rule_eval(panel, cs._mom(3), hold=6, non_overlapping=False)
    stepped = cs.rule_eval(panel, cs._mom(3), hold=6, non_overlapping=True)
    assert stepped["windows"] < overlapping["windows"]
    assert stepped["windows"] > 0


def test_rule_eval_reports_empty_panel() -> None:
    assert cs.rule_eval({}, cs._mom(1))["windows"] == 0
