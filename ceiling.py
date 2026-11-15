"""Arithmetic ceiling for the $1M-from-$1.1k research question.

This module answers ONE question before any strategy is written:

    what multiplier is required, and is it historically attested?

Everything here is arithmetic on stated inputs. Market anchors are marked
UNVERIFIED because the data pipeline has not downloaded anything yet; they must
be re-checked against real OHLCV before they are cited in the dissertation.

Run:  python ceiling.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import (
    HORIZON_MONTHS,
    RUB_PER_USD,
    START_CAPITAL_RUB,
    TARGET_USD,
    Venue,
    required_multiple,
)

# Study parameters live in config.py. Re-exported here so this module stays
# runnable as a script and readable standalone, without a second copy drifting.
START_RUB = START_CAPITAL_RUB
TARGET_RUB = TARGET_USD * RUB_PER_USD


@dataclass(frozen=True)
class FeeSchedule:
    """Round-trip cost as a fraction of traded notional.

    Built from config.COST_MODELS so venue fees are stated exactly once.
    """

    label: str
    round_trip: float

    @classmethod
    def from_venue(cls, label: str, venue: Venue) -> "FeeSchedule":
        from config import COST_MODELS

        return cls(label=label, round_trip=2.0 * COST_MODELS[venue].round_trip_fee)


FEES: tuple[FeeSchedule, ...] = (
    FeeSchedule.from_venue("Binance spot, taker", Venue.SPOT_TAKER),
    FeeSchedule.from_venue("Binance spot, maker", Venue.SPOT_MAKER),
    FeeSchedule.from_venue("Binance perp, taker", Venue.PERP_TAKER),
    FeeSchedule.from_venue("Binance perp, maker", Venue.PERP_MAKER),
)

# UNVERIFIED - approximate anchors for the strongest 2-year crypto windows.
# src: widely reported monthly closes; must be replaced with downloaded OHLCV.
# These are the ONLY unverified market claims in the study; the arithmetic around
# them is exact.
ANCHORS: tuple[tuple[str, float, str, str], ...] = (
    ("BTC", 9.6, "2020-01 ~$7.2k -> 2021-11 ~$69k ATH close", "best 2y in history"),
    ("ETH", 37.0, "2020-01 ~$130 -> 2021-11 ~$4.8k", "best 2y in history"),
    ("SOL", 104.0, "2020-09 ~$2.5 -> 2021-11 ~$260", "from near-launch"),
    ("Low-cap alts (select)", 250.0, "various 2020-2021", "survivorship-selected"),
)


def _log2(x: float) -> float:
    return math.log2(x)


def annualised(multiple: float, months: int = HORIZON_MONTHS) -> float:
    """Compound annualised return implied by `multiple` over `months`."""
    if multiple <= 0.0:
        return -1.0
    return multiple ** (12.0 / months) - 1.0


def max_round_trips(fee: FeeSchedule, capital_fraction_lost: float = 0.30) -> int:
    """How many round trips before `capital_fraction_lost` of equity is gone.

    Compounding: each round trip multiplies equity by (1 - fee), so after n
    trips equity is (1 - fee)**n. Solve for the n where the retained fraction
    is (1 - capital_fraction_lost).
    """
    if fee.round_trip <= 0.0:
        return -1
    target = 1.0 - capital_fraction_lost
    if target <= 0.0:
        return 0
    # (1 - fee)^n = target  =>  n = ln(target) / ln(1 - fee)
    import math

    return int(math.log(target) / math.log(1.0 - fee.round_trip))


def leverage_survivable(leverage: float, monthly_vol: float) -> float:
    """Largest adverse monthly move an agent survives at `leverage`.

    Isolated-margin liquidation is assumed at a 100% loss of margin, so a move
    of size d against the position liquidates when leverage * d >= 1.0. The
    reported figure is the liquidation distance, i.e. d = 1 / leverage.
    """
    if leverage <= 0.0:
        return 0.0
    return 1.0 / leverage


def main() -> None:
    need = required_multiple()
    ann = annualised(need)

    print("=" * 74)
    print("1. WHAT THE TARGET REQUIRES")
    print("=" * 74)
    print(f"  start          {START_RUB:>14,.0f} RUB")
    print(f"  target         {TARGET_RUB:>14,.0f} RUB  ({TARGET_USD:,.0f} USD)")
    print(f"  horizon        {HORIZON_MONTHS:>14d} months")
    print(f"  REQUIRED       {need:>14.1f}x gross")
    print(f"  implied CAGR   {ann * 100:>13.0f}% per year")
    print()
    print(
        f"  No contributions. The whole {need:,.0f}x must come from capital"
        " appreciation."
    )
    months_per_double = HORIZON_MONTHS / (1.0 if need <= 1 else _log2(need))
    print(
        f"  A {need:,.0f}x outcome implies roughly a doubling every"
        f" {months_per_double:.1f} months, sustained, net of every cost."
    )
    print("  This is the number the rest of the study answers.")

    print()
    print("=" * 74)
    print(f"2. IS {need:,.0f}x ATTESTED? (anchors UNVERIFIED - recheck against OHLCV)")
    print("=" * 74)
    print(f"  {'asset':<24}{'2y best':>10}  {'CAGR':>8}  vs required {need:,.0f}x")
    print(f"  {'-' * 24}{'-' * 10}  {'-' * 8}  {'-' * 12}")
    for name, mult, window, note in ANCHORS:
        cagr = annualised(mult)
        ratio = mult / need
        flag = "MEETS" if mult >= need else "short"
        print(
            f"  {name:<24}{mult:>9.1f}x  {cagr * 100:>7.0f}%  "
            f"{ratio:>6.2f}x  {flag}  ({note})"
        )
    print()
    best = max(m for _, m, _, _ in ANCHORS)
    majors = [m for _, m, _, _ in ANCHORS if m <= 50.0]
    lo = min(majors)
    hi = max(majors)
    print(
        f"  Read: NOTHING in the table reaches {need:,.0f}x. The best anchor"
        f" ({best:,.0f}x) is still {need / best:.1f}x short."
    )
    print(
        f"  Liquid majors topped out at {lo:,.1f}x-{hi:,.1f}x over their best"
        f" two years, i.e. {need / hi:.0f}x-{need / lo:.0f}x short of the target."
    )
    print(
        "  Even the survivorship-selected low-cap bucket fails, and it fails by"
        " construction:"
    )
    print("  you would have had to name the winner in advance, with no lookahead.")

    print()
    print("=" * 74)
    print("3. FEE BUDGET AT THIS SIZE (unverified fees - confirm vs venue)")
    print("=" * 74)
    print("  30% of equity is assumed lost to costs; n = ln(0.7)/ln(1-fee)")
    print()
    print(f"  {'schedule':<32}{'RT cost':>9}{'max RT':>9}{'trades/24mo':>13}")
    print(f"  {'-' * 32}{'-' * 9}{'-' * 9}{'-' * 13}")
    for f in FEES:
        n = max_round_trips(f)
        per_month = n / (HORIZON_MONTHS / 12.0) if n > 0 else 0
        print(f"  {f.label:<32}{f.round_trip * 100:>8.3f}%{n:>9}{per_month:>12.1f}/mo")
    print()
    print("  Consequence: at $1.1k the strategy is constrained to LOW FREQUENCY.")
    print("  High-frequency / scalping is arithmetically impossible - costs would")
    print("  consume the account before any edge could express itself.")

    print()
    print("=" * 74)
    print("4. LEVERAGE: THE UNRESTRICTED TRACK'S STRUCTURAL FAILURE")
    print("=" * 74)
    print("  Isolated margin: a move of 1/leverage against the position liquidates.")
    print("  Crypto routinely produces 10%+ adverse monthly moves.")
    print()
    print(f"  {'leverage':>9}{'liq distance':>14}  verdict vs routine 10% monthly")
    print(f"  {'-' * 9}{'-' * 14}  {'-' * 34}")
    for lev in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
        d = leverage_survivable(lev, 0.0)
        if d >= 0.10:
            verdict = "survives a 10% adverse month"
        elif d >= 0.05:
            verdict = "dies to a 5% adverse month"
        else:
            verdict = "dies to routine noise"
        print(f"  {lev:>8.0f}x{d * 100:>13.1f}%  {verdict}")
    print()
    print("  This is the arithmetic behind the 2018 Dow -37% episode: the agent")
    print("  does not need bad luck, only ordinary volatility plus size.")

    print()
    print("=" * 74)
    print("5. CONCLUSION CARRIED INTO THE EXPERIMENT")
    print("=" * 74)
    print(
        f"  The ${TARGET_USD:,.0f} target is NOT reachable from"
        f" {START_RUB:,.0f} RUB by trading. It sits {need / hi:.0f}x beyond the"
        f" best two-year outcome ever printed by the large majors, and"
        f" {need / best:.0f}x beyond anything in the table at all."
    )
    print("  That is not a tuning gap. It is a different order of magnitude.")
    print()
    print("  The researchable question therefore becomes:")
    print("    (a) what multiplier is actually attainable, and")
    print("    (b) what does an unrestricted agent do when it fails?")
    print()
    print("  Track A (unrestricted) is expected to converge to ~0.")
    print("  Track B (risk-budgeted) is expected to survive with a modest return.")
    print("  The COMPARISON is the result; the target is the stimulus.")


if __name__ == "__main__":
    main()
