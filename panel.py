"""Load the monthly close panel and answer base-rate questions about it.

Kept separate from the study logic so the arithmetic can be checked against the
raw file independently. Nothing here selects, ranks, or filters on outcomes.

Row semantics: one (symbol, month) close and that month's intramonth low. The
low is kept because liquidation and ruin are path-dependent - a close-to-close
series hides a -90% month inside an otherwise positive year.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PANEL_CSV = Path("cache") / "monthly_close.csv"


@dataclass(frozen=True)
class Point:
    """One monthly observation for one symbol."""

    yyyymm: int
    close: float
    low: float


def month_index(yyyymm: int) -> int:
    """Months since year 0, so windows are plain integer differences."""
    y, m = divmod(yyyymm, 100)
    return y * 12 + (m - 1)


def yyyymm_from_index(idx: int) -> int:
    y, m = divmod(idx, 12)
    return y * 100 + (m + 1)


def load_panel(path: Path = PANEL_CSV) -> dict[str, tuple[Point, ...]]:
    """symbol -> chronologically sorted points. Symbols must be ASCII-safe."""
    raw: dict[str, list[Point]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw.setdefault(row["symbol"], []).append(
                Point(
                    yyyymm=int(row["yyyymm"]),
                    close=float(row["close"]),
                    low=float(row["low"]),
                )
            )
    out: dict[str, tuple[Point, ...]] = {}
    for sym, pts in raw.items():
        pts.sort(key=lambda p: p.yyyymm)
        # A symbol is only usable if its months are contiguous. A gap means
        # delisting or a missing archive; forward-filling across it would
        # manufacture a flat month and corrupt both return and drawdown.
        out[sym] = tuple(pts)
    return out


def contiguous(pts: tuple[Point, ...]) -> bool:
    """True when every consecutive pair is exactly one month apart."""
    return all(
        month_index(pts[i + 1].yyyymm) - month_index(pts[i].yyyymm) == 1
        for i in range(len(pts) - 1)
    )


def main() -> None:
    panel = load_panel()
    total_points = sum(len(v) for v in panel.values())
    clean = {s: p for s, p in panel.items() if contiguous(p)}
    print(f"symbols: {len(panel)}")
    print(f"monthly rows: {total_points}")
    print(
        f"contiguous symbols: {len(clean)} (non-contiguous dropped: "
        f"{len(panel) - len(clean)})"
    )

    spans = [(len(p), s) for s, p in clean.items()]
    spans.sort()
    print(f"shortest history: {spans[0][1]} {spans[0][0]} months")
    print(f"longest history:  {spans[-1][1]} {spans[-1][0]} months")
    longest = sorted(clean.values(), key=len)[-1]
    print(f"coverage: {longest[0].yyyymm} .. {longest[-1].yyyymm}")

    print()
    print("REMINDER: current listings only. Delisted contracts are absent, so")
    print("every rate below is computed on survivors and is an UPPER bound.")


if __name__ == "__main__":
    main()
