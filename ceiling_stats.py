"""Multiple-testing corrections for the ceiling study: DSR and PBO.

Nothing here improves results. It exists to answer one question honestly:
if we look at many rules and report the best, how much of the "best" is
selection noise rather than edge? The raw numbers in ceiling_study.py are the
uncorrected upper bound; these are the haircut.

THE OVERLAP PROBLEM. The panel spans ~80 months, so overlapping 24-month
windows give ~56 rows that share most of their price history with each other.
DSR and PBO both assume independent observations. Feeding them 56 correlated
rows would report far more significance than the data supports. The effective
sample size of an overlapping-window design is roughly span/hold - about 3 for
a 24-month hold - and that is what these functions use. It is the single most
consequential number in this module, and it is small enough to make most
inference impossible. That is a finding, not a nuisance.

References: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio";
Bailey, Borwein, Lopez de Prado, Zhu (2016) "The Probability of Backtest
Overfitting" (CSCV).
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

_N = NormalDist()
_EULER = 0.5772156649015329  # Euler-Mascheroni constant, for the SR0 tail term
_MONTHS_PER_YEAR = 12


def effective_n(n_windows: int, hold: int, span_months: int) -> float:
    """Independent observations available from an overlapping-window design.

    Windows are spaced one month apart but each covers `hold` months, so
    consecutive rows are not independent draws. Roughly span/hold of them are.
    Floored at 1.0 because a single window is still one observation, and
    returning 0 would make every downstream statistic undefined.
    """
    if hold <= 0 or span_months <= 0:
        return 1.0
    return max(1.0, min(float(n_windows), span_months / hold))


def monthly_equiv(total_returns: list[float], hold: int) -> list[float]:
    """Convert a hold-period total return into a per-month equivalent rate.

    Sharpe is only comparable across horizons on a common time scale. A 298x
    12-month total and a 49x 24-month total are not commensurate as written;
    both become "per month" before being annualized.
    """
    if hold <= 0:
        raise ValueError(f"hold must be positive, got {hold}")
    return [(1.0 + r) ** (1.0 / hold) - 1.0 for r in total_returns if r > -1.0]


def sharpe(rets: list[float], periods_per_year: int = _MONTHS_PER_YEAR) -> float:
    """Annualized Sharpe of a periodic return series, rf = 0."""
    if len(rets) < 2:
        return float("nan")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return float("nan")
    return mean / sd * math.sqrt(periods_per_year)


def skew_kurt(rets: list[float]) -> tuple[float, float]:
    """Fisher skewness and non-excess kurtosis, as DSR's formula wants them."""
    n = len(rets)
    if n < 3:
        return 0.0, 3.0
    # Guard on exact equality, not on `var <= 0.0`. Summing a repeated float
    # does not reliably land back on it, so a constant series can leave a tiny
    # positive variance; every deviation then has the same sign and magnitude,
    # (r - mean) / sd collapses to exactly -1, and skew comes out as a garbage
    # +/-1.0 that feeds straight into the PSR denominator. DSR is supposed to
    # shrink significance for degenerate input, not amplify it.
    if max(rets) == min(rets):
        return 0.0, 3.0
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    if var <= 0.0:
        return 0.0, 3.0
    sd = math.sqrt(var)
    g3 = sum(((r - mean) / sd) ** 3 for r in rets) / n
    g4 = sum(((r - mean) / sd) ** 4 for r in rets) / n
    return g3, g4


def sr0(n_trials: int, trial_sharpes: list[float]) -> float:
    """Expected maximum Sharpe across n_trials independent trials under the null.

    This is the hurdle the observed Sharpe has to clear. Looking at more rules
    raises it: the best of 7 noisier strategies beats the best of 1 even when
    none of them has any edge.
    """
    if n_trials < 2 or len(trial_sharpes) < 2:
        return 0.0
    n = len(trial_sharpes)
    mean = sum(trial_sharpes) / n
    var = sum((s - mean) ** 2 for s in trial_sharpes) / (n - 1)
    spread = math.sqrt(var)
    if spread == 0.0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return spread * ((1.0 - _EULER) * a + _EULER * b)


def psr(sharpe_obs: float, n_obs: float, g3: float, g4: float, sr_ref: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_ref), after non-normality.

    n_obs is the EFFECTIVE count, not the row count. Passing correlated rows as
    independent inflates sqrt(n_obs) and manufactures significance.
    """
    if not all(math.isfinite(x) for x in (sharpe_obs, g3, g4, sr_ref)):
        return float("nan")
    if n_obs < 2:
        return float("nan")
    denom = 1.0 - g3 * sharpe_obs + ((g4 - 1.0) / 4.0) * sharpe_obs**2
    if denom <= 0.0:
        return float("nan")
    return _N.cdf((sharpe_obs - sr_ref) * math.sqrt(n_obs - 1.0) / math.sqrt(denom))


def dshr(
    observed_sharpe: float, trial_sharpes: list[float], n_obs: float
) -> dict[str, float]:
    """Deflated Sharpe Ratio summary: the null hurdle, then the verdict."""
    hurdle = sr0(len(trial_sharpes), trial_sharpes)
    return {
        "observed": observed_sharpe,
        "sr0": hurdle,
        "edge_over_sr0": observed_sharpe - hurdle,
    }


def pbo_cscv(matrix: list[list[float]], n_splits: int = 8) -> dict[str, float]:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    `matrix` is [strategy][time], all strategies on a common time axis. Each
    of the C(n_splits, n_splits/2) ways to split the timeline into halves is
    scored: pick the winner in-sample, then ask where it lands out-of-sample.
    PBO is the share of splits where the in-sample winner lands in the
    out-of-sample bottom half. 0.5 means the selection carries no information.
    """
    n_strat = len(matrix)
    n_time = len(matrix[0]) if n_strat else 0
    if n_strat < 2 or n_time < n_splits or n_splits % 2:
        return {"pbo": float("nan"), "splits": 0.0, "n_strat": float(n_strat)}

    size = n_time // n_splits
    slices = [list(range(i * size, (i + 1) * size)) for i in range(n_splits)]
    half = n_splits // 2
    underperform = 0
    used = 0
    for combo in combinations(range(n_splits), half):
        is_idx = [i for i in range(n_splits) if i in combo]
        oos_idx = [i for i in range(n_splits) if i not in combo]
        is_cols = [c for i in is_idx for c in slices[i]]
        oos_cols = [c for i in oos_idx for c in slices[i]]
        if not oos_cols:
            continue
        is_scores = [sum(matrix[s][c] for c in is_cols) for s in range(n_strat)]
        best = max(range(n_strat), key=lambda s: is_scores[s])
        oos_scores = sorted(
            ((sum(matrix[s][c] for c in oos_cols), s) for s in range(n_strat)),
        )
        rank = [s for _, s in oos_scores].index(best)
        if rank < n_strat / 2:
            underperform += 1
        used += 1
    return {
        "pbo": underperform / used if used else float("nan"),
        "splits": float(used),
        "n_strat": float(n_strat),
    }
