"""Bulk-download daily klines for the perp universe and compact to monthly closes.

Why daily: the monthly (1mo) and weekly (1w) Binance Vision series both stop
early - 1mo ends 2023-12, 1w ends 2024-01 - while 1d runs to 2026-08. Verified
2026-09-27 against the S3 bucket listing. A 24-month ceiling study needs data
through today, so 1d is the only interval that can carry it.

Why compact to monthly: the ceiling question is about multi-year horizons, and
a 522-symbol daily panel is ~2.5M rows. Monthly closes reduce that to ~15k rows
while preserving everything the study needs (returns, path, drawdown).

Resumable: data.fetch returns cached bytes when the file exists, so re-running
after an interruption costs nothing. 404 means "not listed yet" and is recorded
as absence, not as an error.
"""

from __future__ import annotations

import csv
import sys
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import data
from universe import Perp, fetch_universe

CACHE = Path("cache") / "daily"
OUT = Path("cache") / "monthly_close.csv"
INTERVAL = "1d"
WORKERS = 16

# Earliest perp data on the venue. Anything before this is a guaranteed 404 and
# is never requested, which cuts the request count roughly in half.
FIRST = (2019, 9)
LAST = (2026, 8)

_lock = threading.Lock()
_done = 0
_missing = 0
_errors: dict[str, int] = {}


def _months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = start
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _onboard_month(p: Perp) -> tuple[int, int]:
    if p.onboard_ms is None:
        return FIRST
    dt = datetime.fromtimestamp(p.onboard_ms / 1000, tz=timezone.utc)
    return (dt.year, dt.month)


def _one(symbol: str, year: int, month: int) -> tuple[str, float, float] | None:
    """Return (symbol, month_close, month_low) or None when not listed yet."""
    global _done, _missing
    try:
        rows = data.load_candles(
            data.DatasetKind.UM_PERP, symbol, INTERVAL, year, month, cache_dir=CACHE
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            with _lock:
                _missing += 1
            return None
        with _lock:
            _errors[str(exc.code)] = _errors.get(str(exc.code), 0) + 1
        return None
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
        with _lock:
            _errors[type(exc).__name__] = _errors.get(type(exc).__name__, 0) + 1
        return None
    with _lock:
        _done += 1
    if not rows:
        return None
    return (symbol, rows[-1].close, min(c.low for c in rows))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    perps = [p for p in fetch_universe() if p.symbol.isascii()]
    jobs: list[tuple[str, int, int]] = []
    for p in perps:
        for ym in _months(max(_onboard_month(p), FIRST), LAST):
            jobs.append((p.symbol, ym[0], ym[1]))
    print(f"symbols: {len(perps)}  planned requests: {len(jobs)}")

    # symbol -> {month_key: (close, low)}
    panel: dict[str, dict[str, tuple[float, float]]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_one, *j): j for j in jobs}
        for fut in as_completed(futures):
            done += 1
            got = fut.result()
            if got is None:
                continue
            sym, close, low = got
            key = futures[fut][1] * 100 + futures[fut][2]
            panel.setdefault(sym, {})[str(key)] = (close, low)
            if done % 2000 == 0:
                print(
                    f"  {done}/{len(jobs)}  files={_done}  absent={_missing}"
                    f"  symbols={len(panel)}  errors={sum(_errors.values())}",
                    flush=True,
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "yyyymm", "close", "low"])
        for sym in sorted(panel):
            for key in sorted(panel[sym]):
                close, low = panel[sym][key]
                w.writerow([sym, key, repr(close), repr(low)])

    months = sum(len(v) for v in panel.values())
    print()
    print(f"symbols with data: {len(panel)} / {len(perps)}")
    print(f"monthly rows: {months}")
    print(f"absent (not yet listed): {_missing}")
    print(f"errors: {_errors}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
