"""Fetch and cache the Binance USDT-M perpetual universe.

Why this exists: the ceiling question is "does any ex-ante rule pick the assets
that 1000x'd". That cannot be answered on a single symbol, so the universe has
to be enumerated from the exchange rather than assumed.

SURVIVORSHIP WARNING, read before using anything downstream:
Binance `/fapi/v1/exchangeInfo` lists only CURRENTLY listed perpetuals. Every
delisted, failed, or merged contract is absent by construction. Any selection
rule tested on this universe is therefore evaluated on winners only, and its
measured edge is an upper bound, never an achievable figure. There is no public
endpoint that returns the historical (incl. delisted) perp universe, so this
bias cannot be fixed with this source - it can only be stated.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FAPI = "https://fapi.binance.com/fapi/v1/exchangeInfo"
CACHE = Path("cache") / "universe_perp.json"
USER_AGENT = "Mozilla/5.0 (research; contact via local config)"


@dataclass(frozen=True)
class Perp:
    symbol: str
    base: str
    onboard_ms: int | None  # None when the venue omits it


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_universe(*, refresh: bool = False) -> tuple[Perp, ...]:
    """Current USDT-margined perpetuals, sorted by symbol for stable output."""
    if CACHE.exists() and not refresh:
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        payload = json.loads(_get(FAPI))
        raw = [
            {
                "symbol": s["symbol"],
                "base": s.get("baseAsset", s["symbol"].removesuffix("USDT")),
                "onboard_ms": s.get("onboardDate"),
            }
            for s in payload["symbols"]
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ]
        raw.sort(key=lambda r: r["symbol"])
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return tuple(Perp(**row) for row in raw)


def main() -> None:
    # Some Binance perp symbols contain non-ASCII code points, which crash a
    # cp1251 console. Not cosmetic: those symbols cannot be used as cache
    # filenames either, so they must be detected and excluded explicitly
    # rather than blowing up mid-run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    perps = fetch_universe()
    dated = [p for p in perps if p.onboard_ms is not None]
    ascii_only = [p for p in perps if p.symbol.isascii()]
    print(f"perpetuals (USDT, TRADING): {len(perps)}")
    print(f"with onboardDate: {len(dated)}")
    print(f"ASCII-safe symbols: {len(ascii_only)}")
    weird = [p.symbol for p in perps if not p.symbol.isascii()]
    if weird:
        print(f"non-ASCII symbols excluded ({len(weird)}): {weird[:8]}")
    if dated:
        earliest = min(dated, key=lambda p: p.onboard_ms or 0)
        print(f"earliest listing: {earliest.symbol}")
    print()
    print("REMINDER: current listings only - delisted contracts are absent.")
    print("Any edge measured here is an UPPER bound, not an achievable figure.")


if __name__ == "__main__":
    main()
