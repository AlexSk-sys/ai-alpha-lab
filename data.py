"""Binance Vision data layer.

Source: https://data.binance.vision (public, no key). Terms forbid redistribution
and commercial resale, so this module downloads and caches for local research
only; nothing here republishes exchange data.

Two details drive the design:

1. Timestamp units are NOT consistent. Spot klines are milliseconds before
   2025-01-01 and microseconds from 2025-01-01, in the same column of the same
   file family. `normalise_open_time_ms` collapses s/ms/us/ns to ms.
2. fundingRate is published MONTHLY ONLY. The daily prefix genuinely returns
   KeyCount 0 / 404, so it is not offered as an option.

Klines are the deliberate starting point. Raw `trades/` tick files measure
~1.4 GB compressed per month for BTCUSDT - roughly 100-150 GB for 2020-2026 on a
single pair. At a 100k RUB budget fees dominate, so bar-level research is the
correct fidelity and tick data is deferred.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# The bucket is addressed directly. Measured 2026-09-27 from this host:
#   s3-ap-northeast-1.amazonaws.com  -> 26.5 KB/s, 1.5 s for a 39 KB archive
#   data.binance.vision (CDN)        ->     0.09 KB/s, >180 s for the same file
# The CDN path is throttled by roughly 290x, which would turn a 6 MB dataset
# into a multi-hour job. The object paths are identical, so S3 is the primary
# and the CDN is kept only as a documented fallback.
BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
MIRROR = "https://data.binance.vision"

# Binance spot listings begin 2017-08; USD-M futures 2019-09. These bounds catch
# typo'd years. Extending them later is a deliberate act, not a side effect.
MIN_YEAR = 2017
MAX_YEAR = 2026

# Timestamp magnitude thresholds for unit detection. 1.7e12 ms = 2023-ish,
# 1.7e15 us, 1.7e18 ns. Anything below MIN_PLAUSIBLE_MS would be 1973 or earlier.
_MIN_PLAUSIBLE_MS = 10**11
_US_BOUND = 10**14
_NS_BOUND = 10**17


class DatasetKind(str, Enum):
    SPOT = "spot"
    UM_PERP = "um_perp"


class StreamKind(str, Enum):
    KLINES = "klines"
    FUNDING = "fundingRate"
    MARK_PRICE = "markPriceKlines"


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _root(kind: DatasetKind) -> str:
    return "data/spot" if kind is DatasetKind.SPOT else "data/futures/um"


def _validate(year: int, month: int) -> str:
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError(f"year {year} outside {MIN_YEAR}..{MAX_YEAR}")
    if not 1 <= month <= 12:
        raise ValueError(f"month {month} outside 1..12")
    return f"{year:04d}-{month:02d}"


def stream_url(
    kind: DatasetKind,
    stream: StreamKind,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> str:
    """Monthly archive URL for one symbol/stream.

    Klines and mark-price files embed the interval in the filename; funding
    files do not (there is exactly one funding cadence per contract).
    """
    period = _validate(year, month)
    if stream is StreamKind.FUNDING:
        if kind is not DatasetKind.UM_PERP:
            raise ValueError("funding is a perpetual-futures-only stream")
        # Funding has exactly one cadence per contract, so there is no interval
        # directory in the path - unlike klines and mark-price.
        return (
            f"{BASE}/{_root(kind)}/monthly/{stream.value}/{symbol}/"
            f"{symbol}-fundingRate-{period}.zip"
        )
    return (
        f"{BASE}/{_root(kind)}/monthly/{stream.value}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{period}.zip"
    )


def checksum_url(archive_url: str) -> str:
    """The sibling .CHECKSUM file Binance ships alongside every archive."""
    return f"{archive_url}.CHECKSUM"


def normalise_open_time_ms(value: int) -> int:
    """Collapse s/ms/us/ns epoch values to milliseconds.

    Required because Binance switched spot klines to microseconds on
    2025-01-01 without a format change. Left unhandled, every post-2025 row
    sorts ~500000 years in the future and silently breaks any date filter.
    """
    if value < _MIN_PLAUSIBLE_MS:
        raise ValueError(f"timestamp {value} too small to be a market epoch")
    if value < _US_BOUND:
        return value
    if value < _NS_BOUND:
        return value // 1_000
    return value // 1_000_000


KLINES_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def parse_kline_csv(text: str) -> list[Candle]:
    """Parse a klines CSV: ts,open,high,low,close,volume,...

    Verified against a real monthly archive (BTCUSDT 1h 2024-03): the file
    DOES carry a header row, which is skipped. The trailing columns (close
    time, quote volume, trade count, taker-buy volumes) are unused for bar
    research.
    """
    candles: list[Candle] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"kline row has {len(parts)} fields, expected >= 6")
        if parts[0] == "open_time":
            continue
        try:
            candles.append(
                Candle(
                    open_time_ms=normalise_open_time_ms(int(parts[0])),
                    open=float(parts[1]),
                    high=float(parts[2]),
                    low=float(parts[3]),
                    close=float(parts[4]),
                    volume=float(parts[5]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"malformed kline row: {line[:80]}") from exc
    if not candles:
        raise ValueError("kline payload is empty")
    return candles


def unzip_bytes(payload: bytes) -> bytes:
    """Extract the single CSV member from a Binance monthly archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError("archive is empty")
            return zf.read(names[0])
    except zipfile.BadZipFile as exc:
        raise ValueError(f"payload is not a readable zip: {exc}") from exc


def fetch(
    url: str,
    *,
    cache_dir: Path | None = None,
    verify_checksum: bool = True,
    timeout: float = 60.0,
) -> bytes:
    """Download one archive, verifying its .CHECKSUM when possible.

    Caching is by filename only. A cached file that fails verification is
    deleted and refetched once, so a truncated download cannot masquerade as data.
    """
    name = url.rsplit("/", 1)[-1]
    target = cache_dir / name if cache_dir else None
    if target is not None and target.exists():
        return target.read_bytes()

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read()

    if verify_checksum:
        with urllib.request.urlopen(checksum_url(url), timeout=timeout) as resp:
            expected = resp.read().decode().split()[0].strip().lower()
        import hashlib

        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"checksum mismatch for {name}")

    if target is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return payload


def load_candles(
    kind: DatasetKind,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    *,
    cache_dir: Path | None = None,
) -> list[Candle]:
    """Convenience: fetch one monthly klines archive and parse it."""
    url = stream_url(kind, StreamKind.KLINES, symbol, interval, year, month)
    return parse_kline_csv(unzip_bytes(fetch(url, cache_dir=cache_dir)).decode())
