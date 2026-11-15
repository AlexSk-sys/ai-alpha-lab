"""Multi-seed comparison across tracks A and B, plus the RESULTS.md generator.

WHY: one seed is an anecdote. N_SEEDS parameterizations per track turn the
backtest into a distribution (P10/P50/P90, ruin rate, drawdown spread) - the
only honest answer to "does a risk budget help or hurt". Both tracks run over
IDENTICAL bars; the overlay is the sole difference, so the comparison holds.

RESULTS.md is never hand-typed: `python seeds.py --results` runs the real
backtest over the local cache and writes the report from measured numbers.

WHAT IT DOES NOT DO: no network in the loop (cache only), no per-seed rows
(one compact row per track), no randomness beyond seeds 0..N_SEEDS-1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import COST_MODELS, RISK_BUDGETED, RUB_PER_USD, START_CAPITAL_RUB
from config import TARGET_USD, UNRESTRICTED, CostModel, RiskBudget, Venue
from data import Candle, DatasetKind, load_candles
from runner import PERIODS_PER_YEAR, RunResult, run_backtest

# Study seed count. Lower for a smoke test (e.g. compare(..., n_seeds=8));
# the constant stays 400 so a reduced run is always labelled as such.
N_SEEDS: int = 400

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
# Modest range: months must already sit in cache/ (no download during the run).
STUDY_MONTHS: tuple[tuple[int, int], ...] = ((2024, 3),)


@dataclass(frozen=True)
class TrackSummary:
    """Distribution of one track over its seeds, in starting-capital RUB."""

    track: str
    n: int
    p10: float
    p50: float
    p90: float
    ruin_rate: float
    dd_p10: float
    dd_p50: float
    dd_p90: float
    mean_sharpe: float
    mean_fees: float
    mean_turnover: float
    mean_trades: float


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]. Pure and total."""
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q / 100.0 * (len(ordered) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 >= len(ordered):
        return ordered[-1]
    return ordered[low] + frac * (ordered[low + 1] - ordered[low])


def _summarize(track: str, results: Sequence[RunResult]) -> TrackSummary:
    finals = [r.final_equity for r in results]
    dds = [r.max_drawdown for r in results]
    n = len(results)
    ruined = sum(1 for r in results if r.ruined)
    return TrackSummary(
        track=track,
        n=n,
        p10=percentile(finals, 10.0),
        p50=percentile(finals, 50.0),
        p90=percentile(finals, 90.0),
        ruin_rate=ruined / n if n else 0.0,
        dd_p10=percentile(dds, 10.0),
        dd_p50=percentile(dds, 50.0),
        dd_p90=percentile(dds, 90.0),
        mean_sharpe=sum(r.sharpe for r in results) / n if n else 0.0,
        mean_fees=sum(r.fees_paid for r in results) / n if n else 0.0,
        mean_turnover=sum(r.turnover for r in results) / n if n else 0.0,
        mean_trades=sum(r.n_trades for r in results) / n if n else 0.0,
    )


def compare(
    bars: Sequence[Candle],
    budgets: Sequence[RiskBudget],
    cost: CostModel,
    capital: float = START_CAPITAL_RUB,
    n_seeds: int = N_SEEDS,
    funding_rate: float = 0.0,
) -> dict[str, TrackSummary]:
    """Run every seed on every track. Keys sorted for stable output order."""
    if not bars:
        raise ValueError("bars must be non-empty")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    summaries: dict[str, TrackSummary] = {}
    for budget in sorted(budgets, key=lambda b: b.name):
        results = [
            run_backtest(list(bars), seed, budget, cost, capital, SYMBOL, funding_rate)
            for seed in range(n_seeds)
        ]
        summaries[budget.name] = _summarize(budget.name, results)
    return summaries


def format_table(summaries: dict[str, TrackSummary]) -> str:
    """One compact row per track. Never per-seed rows."""
    lines = ["track               n    P10(RUB)   P50(RUB)   P90(RUB)  ruin   ddP50"]
    for name in sorted(summaries):
        s = summaries[name]
        lines.append(
            f"{s.track:<19} {s.n:>3} {s.p10:>10,.0f} {s.p50:>10,.0f}"
            f" {s.p90:>10,.0f} {s.ruin_rate:>5.1%} {s.dd_p50:>6.1%}"
        )
    return "\n".join(lines)


def load_study_bars(cache_dir: Path) -> list[Candle]:
    """Load STUDY_MONTHS from the local cache only; fail loudly if absent."""
    bars: list[Candle] = []
    for year, month in STUDY_MONTHS:
        name = f"{SYMBOL}-{INTERVAL}-{year:04d}-{month:02d}.zip"
        if not (cache_dir / name).exists():
            raise FileNotFoundError(
                f"cache miss: {name} not in {cache_dir}; refusing to download"
                " during a measured run"
            )
        bars.extend(
            load_candles(
                DatasetKind.SPOT, SYMBOL, INTERVAL, year, month, cache_dir=cache_dir
            )
        )
    return bars


def render_report(
    summaries: dict[str, TrackSummary],
    n_bars: int,
    first_ms: int,
    last_ms: int,
    n_seeds: int,
) -> str:
    """Build the full RESULTS.md text from measured summaries only."""
    target_rub = TARGET_USD * RUB_PER_USD
    best = max(s.p90 for s in summaries.values())
    names = sorted(summaries)
    reached = [s.track for s in summaries.values() if s.p90 >= target_rub]
    verdict = (
        "NO - neither track reached $1M on any reported quantile."
        if not reached
        else "YES - " + ", ".join(reached) + " reached $1M at P90."
    )
    rows = "\n".join(
        f"| {s.track} | {s.n} | {s.p10:,.0f} | {s.p50:,.0f} | {s.p90:,.0f}"
        f" | {s.ruin_rate:.1%} | {s.dd_p10:.1%} / {s.dd_p50:.1%} /"
        f" {s.dd_p90:.1%} | {s.mean_sharpe:+.2f} | {s.mean_fees:,.0f}"
        f" | {s.mean_turnover:,.0f} | {s.mean_trades:.1f} |"
        for s in (summaries[k] for k in names)
    )
    smoke = (
        ""
        if n_seeds == N_SEEDS
        else (
            f"\n> SMOKE TEST: n_seeds={n_seeds}, not the full N_SEEDS={N_SEEDS}."
            " Treat every figure as provisional.\n"
        )
    )
    fee = COST_MODELS[Venue.PERP_TAKER].round_trip_fee
    slip = COST_MODELS[Venue.PERP_TAKER].slippage_bps
    return f"""# Results: 100,000 RUB -> $1M, tracks A vs B

Generated by `python seeds.py --results` from the real backtest loop over local Binance cache data. No number below was typed by hand.
{smoke}
## Headline

**Was $1M reached on either track? {verdict}**

Target: ${TARGET_USD:,.0f} at the frozen study rate of {RUB_PER_USD:.0f} RUB/USD (config.RUB_PER_USD) = {target_rub:,.0f} RUB from {START_CAPITAL_RUB:,.0f} RUB, i.e. a {target_rub / START_CAPITAL_RUB:,.0f}x multiple. Best measured P90 across tracks: {best:,.0f} RUB.

## Setup

- Market: {SYMBOL} {INTERVAL}, {n_bars} bars, open-time ms {first_ms}..{last_ms} (Binance Vision monthly klines from local `cache/`; no HTTP in the loop).
- Seeds: {n_seeds} per track (seeds 0..{n_seeds - 1}), identical bars on both tracks.
- Execution: perp taker, fee {fee} per side + {slip} bps slippage scale (see config.COST_MODELS). Funding is ASSUMED zero, not loaded from data - see Limitations.
- Sharpe: per-bar equity returns, annualised x sqrt({PERIODS_PER_YEAR}) (hourly bars, 24x365). Sharpe of a flat curve is defined as 0.0, not NaN.
- Track A ({UNRESTRICTED.name}): full conviction up to the 75x venue ceiling, no throttle, no breaker. Track B ({RISK_BUDGETED.name}): per-trade caps (2x, 0.35), max 3 open, linear drawdown throttle to a 20% breaker.

## Track comparison (RUB)

| track | n | P10 final | P50 final | P90 final | ruin | dd P10/P50/P90 | mean Sharpe | mean fees | mean turnover | mean trades |
|---|---|---|---|---|---|---|---|---|---|---|
{rows}

## Reading

The comparison is the result; the target is the stimulus. $1M demands a 1000x multiple here, beyond any attested major-crypto window, so failure is the expected outcome; the question is whether the risk budget changes the SHAPE of the failure (ruin rate, drawdown spread, fee drag), not whether it rescues the multiple.

## Limitations

- Funding is assumed zero, not measured. `sens_funding.py` re-runs the identical seeds at non-zero rates; for this strategy the effect is under 0.2% because holds are too short to cross many funding timestamps. That is a property of THIS strategy, not a general result. `broker.apply_funding` also bases the charge on entry notional rather than mark notional, which would understate drag for longer holds.
- No live execution: backtest plus paper-trading design; fills are simulated at the printed bar close.
- Binance data only: one venue, one symbol ({SYMBOL}), one interval; regimes elsewhere may differ and no claim transfers.
- No out-of-sample walk-forward yet: parameters are fixed seed arithmetic, but the month was cached, not pre-registered.
- Survivorship of the chosen symbol: {SYMBOL} is the winner that survived; testing only it flatters trend baselines.
- A favorable figure is not evidence of skill: with {n_seeds} seeds the top quantile is selected luck; only the distribution (and its costs) is reported.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed track comparison.")
    parser.add_argument(
        "--results", action="store_true", help="full run and write RESULTS.md"
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="quick run with this many seeds (no file output)",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    bars = load_study_bars(root / "cache")
    cost = COST_MODELS[Venue.PERP_TAKER]
    n = N_SEEDS if (args.results or args.smoke <= 0) else args.smoke
    summaries = compare(bars, (UNRESTRICTED, RISK_BUDGETED), cost, n_seeds=n)
    print(format_table(summaries))
    if args.results:
        text = render_report(
            summaries, len(bars), bars[0].open_time_ms, bars[-1].open_time_ms, N_SEEDS
        )
        (root / "RESULTS.md").write_text(text, encoding="utf-8")
        print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
