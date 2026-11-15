"""Unconditional base rates, then ex-ante selection rules vs a random baseline.

The ordering is deliberate. Base rates come first and are selection-free, so
the selection rules are measured against a known prior instead of against each
other. Reporting rules without base rates invites the exact error this whole
project exists to avoid: noticing that some rule "worked" without knowing that
the target was rare to begin with.

Every rule sees only months <= t. No rule is fit, tuned, or selected inside the
measurement loop - the rule set is fixed in _RULES and evaluated as-is. That
still leaves a multiple-testing problem (we look at many rules and report the
best), which ceiling_stats.py corrects for; the raw numbers here are the
UNCORRECTED upper bound, and are labelled as such.

Data caveat, carried from panel.py: current listings only. Delisted contracts
are absent, so every rate here is computed on survivors and is optimistic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from panel import Point, load_panel, month_index

HOLD = 24  # months; the study horizon


@dataclass(frozen=True)
class Traj:
    """Forward path from a decision month, used for return and drawdown."""

    ret: float  # close-to-close total return over the window
    max_dd: float  # worst peak-to-trough within the window, 0..1


def _traj(history: tuple[Point, ...], j: int, hold: int) -> Traj | None:
    """Forward window of `hold` months starting at history[j]; None if short.

    `hold` is an explicit parameter, never the module constant. Reading it from
    a global here silently computes every requested horizon as HOLD-month
    windows while only the eligibility filter changes, which produces
    byte-identical "best" figures across horizons and looks plausible.
    """
    if j + hold >= len(history):
        return None
    start = history[j].close
    if start <= 0.0:
        return None
    window = history[j : j + hold + 1]
    peak = window[0].close
    worst = 0.0
    for p in window:
        peak = max(peak, p.close)
        if peak > 0.0:
            worst = max(worst, 1.0 - p.close / peak)
    return Traj(ret=window[-1].close / start - 1.0, max_dd=worst)


# --- selection rules: signature is (history, j) -> score; higher = better ---

Rule = Callable[[tuple[Point, ...], int], float]


def _mom(n: int) -> Rule:
    def rule(h: tuple[Point, ...], j: int) -> float:
        if j < n:
            return -1e18
        a, b = h[j - n].close, h[j].close
        return b / a - 1.0 if a > 0.0 else -1e18

    return rule


def _low_vol(n: int) -> Rule:
    """Lowest trailing volatility scores highest."""

    def rule(h: tuple[Point, ...], j: int) -> float:
        if j < n:
            return -1e18
        rets = [
            h[i].close / h[i - 1].close - 1.0
            for i in range(j - n + 1, j + 1)
            if h[i - 1].close > 0.0
        ]
        if len(rets) < 2:
            return -1e18
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
        return -(var**0.5)

    return rule


def _random(seed: int) -> Rule:
    """Baseline: no information. Any rule that cannot beat this has no edge."""

    def rule(h: tuple[Point, ...], j: int) -> float:
        return random.Random((seed << 32) ^ j ^ hash(h[0].close)).random()

    return rule


def _always(h: tuple[Point, ...], j: int) -> float:
    return 0.0


_RULES: dict[str, Rule] = {
    "mom_12": _mom(12),
    "mom_6": _mom(6),
    "mom_3": _mom(3),
    "mom_1": _mom(1),
    "lowvol_12": _low_vol(12),
    "random": _random(0),
    "all": _always,  # equal-weight every symbol
}


def base_rates(
    panel: dict[str, tuple[Point, ...]], *, hold: int = HOLD
) -> dict[str, object]:
    """Unconditional distribution of forward returns. No selection at all."""
    rets: list[float] = []
    dds: list[float] = []
    windows = 0
    for h in panel.values():
        for j in range(len(h)):
            if j + hold >= len(h):
                break
            t = _traj(h, j, hold)
            if t is not None:
                rets.append(t.ret)
                dds.append(t.max_dd)
                windows += 1
    rets.sort()

    def pct(q: float) -> float:
        if not rets:
            return float("nan")
        i = min(len(rets) - 1, max(0, int(q * len(rets))))
        return rets[i]

    def frac_at_least(x: float) -> float:
        if not rets:
            return float("nan")
        return sum(1 for r in rets if r >= x) / len(rets)

    return {
        "windows": windows,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "p999": pct(0.999),
        "best": rets[-1] if rets else float("nan"),
        "f2x": frac_at_least(2.0),
        "f10x": frac_at_least(10.0),
        "f100x": frac_at_least(100.0),
        "f250x": frac_at_least(250.0),
        "f1000x": frac_at_least(1000.0),
        "med_dd": sorted(dds)[len(dds) // 2] if dds else float("nan"),
        # Raw aligned window returns. ceiling_stats.py needs the series, not
        # just the summary, for CSCV; the percentiles above hide the pairing.
        "rets": list(rets),
    }


def rule_eval(
    panel: dict[str, tuple[Point, ...]],
    rule: Rule,
    *,
    hold: int = HOLD,
    top_n: int = 1,
    non_overlapping: bool = True,
) -> dict[str, object]:
    """Walk forward: at each decision month, rank on data <= t, take top_n, hold.

    non_overlapping steps the decision month by `hold` months so windows do not
    share returns, which keeps the sample honest at the cost of fewer points.
    """
    rets: list[float] = []
    dds: list[float] = []
    # (score, history, decision_index) - the index is required, because the
    # forward window starts at the DECISION month, not at the start of the
    # symbol's life. Dropping it silently measures each asset from its listing.
    by_month: dict[int, list[tuple[float, tuple[Point, ...], int]]] = {}
    for h in panel.values():
        for j in range(len(h)):
            if j + hold >= len(h):
                break
            by_month.setdefault(month_index(h[j].yyyymm), []).append((rule(h, j), h, j))
    months = sorted(by_month)
    prev_month: int | None = None
    for m in months:
        if non_overlapping:
            if prev_month is not None and m - prev_month < hold:
                continue
        prev_month = m
        ranked = sorted(by_month[m], key=lambda t: -t[0])
        chosen = [(h, j) for _, h, j in ranked[:top_n]]
        trajs = [t for t in (_traj(h, j, hold) for h, j in chosen) if t is not None]
        if not trajs:
            continue
        # Equal-weight the top_n picks, which is what a real book would do.
        rets.append(sum(t.ret for t in trajs) / len(trajs))
        dds.append(max(t.max_dd for t in trajs))
    if not rets:
        return {"windows": 0}
    ordered = sorted(rets)

    def pct(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]

    return {
        "windows": len(rets),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "best": ordered[-1],
        "med_dd": sorted(dds)[len(dds) // 2] if dds else float("nan"),
        # Aligned with base_rates' "rets" so CSCV can pair rules by window.
        "rets": list(rets),
    }
