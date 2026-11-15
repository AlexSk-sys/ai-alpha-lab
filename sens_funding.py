"""Funding sensitivity: how much does zero funding flatter the high-leverage track?

The 400-seed study ran with funding_rate=0.0, which is not a neutral default for
perpetual futures - funding is a recurring real cost that scales with leverage.
This re-runs the identical seeds and bars at several rates to measure whether the
A-vs-B conclusion is robust to that omission.

Not part of the shipped study; a diagnostic. Read-only with respect to config.
"""

from __future__ import annotations

from pathlib import Path

from config import (
    COST_MODELS,
    START_CAPITAL_RUB,
    UNRESTRICTED,
    RISK_BUDGETED,
    Venue,
)
from seeds import compare, load_study_bars

# Typical Binance perp funding is ~0.01% per 8h; 0.03% is a plausible stress.
RATES = (0.0, 0.0001, 0.0003)


def main() -> None:
    bars = load_study_bars(Path("cache"))
    cost = COST_MODELS[Venue.PERP_TAKER]
    print(f"bars={len(bars)}  capital={START_CAPITAL_RUB:,.0f} RUB")
    print(f"leverage: A<=75x  B<=2x\n")
    for rate in RATES:
        s = compare(bars, [UNRESTRICTED, RISK_BUDGETED], cost, funding_rate=rate)
        a, b = s["A_unrestricted"], s["B_risk_budgeted"]
        print(
            f"funding={rate:<7} "
            f"A: P50={a.p50:>9,.0f} P90={a.p90:>9,.0f} dd50={a.dd_p50:>5.1%} "
            f"| B: P50={b.p50:>9,.0f} P90={b.p90:>9,.0f} dd50={b.dd_p50:>5.1%} "
            f"| B-A P50={b.p50 - a.p50:>+8,.0f}"
        )


if __name__ == "__main__":
    main()
