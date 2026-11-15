"""The two parallel tracks, applied to the ceiling question.

Both tracks see the same picks, the same data, the same costs. They differ only
in the risk overlay, which is the whole point of running them separately.

An earlier draft of this docstring claimed the overlay could not move the ceiling
and bought survival "and nothing else." The measured data says otherwise, and
the claim was wrong: at an 8% vol budget the overlay costs roughly 8x of ceiling
(39.3x -> 4.7x best window at a 12-month hold) in exchange for cutting the worst
window from an 85% loss to about 34%. The overlay cannot manufacture the pick
that returns 298x - that either happens or does not - but it does shrink what
you keep when it happens. Against a 250x target, "6x short" and "50x short" are
different problems, so the ceiling difference is not a rounding detail.

What the overlay does buy is measurable: low-vol picks tolerate 6-8x leverage
before the intramonth low reaches margin, while random picks tolerate about
1.4x, and no window in the panel wiped out even unlevered. Survivability and
ceiling trade off here rather than one dominating the other.
"""

from __future__ import annotations

import math

# Liquidation on a perp is roughly a 100% adverse move against the margin
# posted, so a 1x position dies somewhere around a 50% adverse excursion once
# maintenance margin is counted. This is the fraction of equity left at the
# worst point in the window, and 0.0 means the position was liquidated.
RUIN = 0.0


def vol_scale(monthly_rets: list[float], risk_budget: float) -> float:
    """Position multiplier that keeps trailing monthly vol near risk_budget.

    1.0 means the overlay allows full exposure; below 1.0 it stands aside when
    the asset is moving. Unbounded leverage is deliberately not offered: scaling
    up on quiet history is how a 1x account meets a 90% drawdown.
    """
    if risk_budget <= 0.0:
        raise ValueError(f"risk_budget must be positive, got {risk_budget}")
    if len(monthly_rets) < 2:
        return 1.0
    mean = sum(monthly_rets) / len(monthly_rets)
    var = sum((r - mean) ** 2 for r in monthly_rets) / (len(monthly_rets) - 1)
    sd = math.sqrt(var)
    if sd <= 0.0:
        return 1.0
    return min(1.0, risk_budget / sd)


def survivable_leverage(window_lows: list[float], start: float) -> float:
    """Max leverage whose worst point does not touch the margin.

    Returns the multiple of start capital that the worst intramonth low can
    absorb. 1.0 means even unlevered the window hurts; values well above 1.0
    mean the window was survivable with room to spare. This is the number that
    separates the two tracks in practice: a window can be a 300x winner and
    still not be holdable at high leverage if it fell 90% on the way.
    """
    if start <= 0.0 or not window_lows:
        return 0.0
    worst = min(window_lows)
    if worst <= 0.0:
        return 0.0
    # Adverse excursion as a fraction of start capital. It is <= 0 whenever no
    # low sits below the start - a flat series, or one that only went up - and
    # those windows are holdable at any leverage. The old code guarded only
    # the >= 1.0 end and raised ZeroDivisionError on the flat case, which is
    # exactly the input a quiet asset produces.
    excursion = 1.0 - worst / start
    if excursion >= 1.0:
        return 0.0
    if excursion <= 0.0:
        return math.inf
    return 1.0 / excursion


def scale_window(window_return: float, size: float) -> float:
    """Final capital multiplier after taking `size` of the position.

    Scales the profit and loss, not the capital: 1 + window_return * size. The
    tempting shortcut is (1 + window_return) * size, which is wrong by the size
    factor on the entire principal. At size=0.5 it turns a 5% gain into a 48%
    loss and makes a risk overlay look like the primary source of damage. Only
    compounding a gain through exponentiation is different again, and that is
    not what fraction-of-position sizing means.
    """
    if size < 0.0:
        raise ValueError(f"size must be non-negative, got {size}")
    return 1.0 + window_return * size


def compare_tracks(
    unrestricted: list[float], budgeted: list[float]
) -> dict[str, float]:
    """Ceiling and survival side by side for the two tracks.

    `unrestricted` and `budgeted` are matched per-window final multipliers. The
    ceiling is reported for both so the asymmetry is visible rather than
    asserted: if they match, the overlay bought survival and nothing else.
    """
    if len(unrestricted) != len(budgeted):
        raise ValueError("tracks must be paired per window")
    if not unrestricted:
        return {"windows": 0.0}

    def stats(xs: list[float]) -> tuple[float, float, float]:
        o = sorted(xs)
        worst = min(o)
        return o[-1], o[len(o) // 2], 1.0 - worst if worst > 0 else 1.0

    a_max, a_p50, a_loss = stats(unrestricted)
    b_max, b_p50, b_loss = stats(budgeted)
    return {
        "windows": float(len(unrestricted)),
        "a_best": a_max,
        "a_p50": a_p50,
        "a_worst_loss": a_loss,
        "b_best": b_max,
        "b_p50": b_p50,
        "b_worst_loss": b_loss,
    }
