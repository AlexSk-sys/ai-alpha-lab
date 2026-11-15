"""Invariants for the Binance Vision data layer.

The one genuinely dangerous detail here is the timestamp unit switch: Binance
spot klines are milliseconds before 2025-01-01 and MICROSECONDS from 2025-01-01
onward, in the same file family and same column. A parser that assumes one unit
produces plausible-looking prices and silently wrong chronology for half the
dataset, so the normalisation is pinned by test below.

URL construction is also pinned, because the path layout is not documented on
the bucket and a wrong guess fails at 404 rather than at import.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from data import (
    BASE,
    DatasetKind,
    StreamKind,
    checksum_url,
    normalise_open_time_ms,
    parse_kline_csv,
    stream_url,
    unzip_bytes,
)


class TestStreamUrl:
    def test_spot_monthly_klines(self) -> None:
        url = stream_url(DatasetKind.SPOT, StreamKind.KLINES, "BTCUSDT", "1h", 2024, 3)
        assert url == (
            f"{BASE}/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-03.zip"
        )

    def test_um_perp_monthly_klines(self) -> None:
        url = stream_url(
            DatasetKind.UM_PERP, StreamKind.KLINES, "BTCUSDT", "1h", 2024, 3
        )
        assert url == (
            f"{BASE}/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-03.zip"
        )

    def test_funding_rate_path(self) -> None:
        # Funding is a perps-only stream and, per the bucket, MONTHLY ONLY.
        # The daily variant genuinely 404s; this test exists to stop someone
        # "optimising" it into a daily path later.
        url = stream_url(
            DatasetKind.UM_PERP, StreamKind.FUNDING, "BTCUSDT", "8h", 2024, 3
        )
        assert url == (
            f"{BASE}/data/futures/um/monthly/fundingRate/"
            "BTCUSDT/BTCUSDT-fundingRate-2024-03.zip"
        )

    def test_mark_price_path(self) -> None:
        url = stream_url(
            DatasetKind.UM_PERP, StreamKind.MARK_PRICE, "BTCUSDT", "1h", 2024, 3
        )
        assert url == (
            f"{BASE}/data/futures/um/monthly/markPriceKlines/"
            "BTCUSDT/1h/BTCUSDT-1h-2024-03.zip"
        )

    def test_funding_rejected_for_spot(self) -> None:
        with pytest.raises(ValueError, match="funding"):
            stream_url(DatasetKind.SPOT, StreamKind.FUNDING, "BTCUSDT", "8h", 2024, 3)

    def test_checksum_url_is_sibling(self) -> None:
        assert checksum_url("https://x/BTCUSDT-1h-2024-03.zip") == (
            "https://x/BTCUSDT-1h-2024-03.zip.CHECKSUM"
        )

    @pytest.mark.parametrize("month", [0, 13, 99])
    def test_rejects_impossible_month(self, month: int) -> None:
        with pytest.raises(ValueError, match="month"):
            stream_url(
                DatasetKind.SPOT, StreamKind.KLINES, "BTCUSDT", "1h", 2024, month
            )

    @pytest.mark.parametrize("year", [1990, 2100])
    def test_rejects_impossible_year(self, year: int) -> None:
        with pytest.raises(ValueError, match="year"):
            stream_url(DatasetKind.SPOT, StreamKind.KLINES, "BTCUSDT", "1h", year, 3)


class TestTimestampNormalisation:
    def test_milliseconds_pass_through(self) -> None:
        assert normalise_open_time_ms(1_700_000_000_000) == 1_700_000_000_000

    def test_microseconds_are_scaled_down(self) -> None:
        # 1.7e15 is microseconds since epoch; as ms it would be year 55827.
        assert normalise_open_time_ms(1_700_000_000_000_000) == 1_700_000_000_000

    def test_nanoseconds_scaled_further(self) -> None:
        assert normalise_open_time_ms(1_700_000_000_000_000_000) == 1_700_000_000_000

    def test_rejects_implausibly_small_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            normalise_open_time_ms(1000)

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            normalise_open_time_ms(0)


class TestKlineParsing:
    RAW = (
        "1700000000000,42000.00,42500.00,41500.00,42300.00,1234.5,1700000059999,"
        "51800000.0,5000,600.0,25000000.0,0\n"
    )

    def test_parses_fields(self) -> None:
        rows = parse_kline_csv(self.RAW)
        assert len(rows) == 1
        candle = rows[0]
        assert candle.open_time_ms == 1_700_000_000_000
        assert candle.open == 42_000.00
        assert candle.high == 42_500.00
        assert candle.low == 41_500.00
        assert candle.close == 42_300.00
        assert candle.volume == 1_234.5

    def test_skips_blank_lines(self) -> None:
        rows = parse_kline_csv("\n" + self.RAW + "\n\n")
        assert len(rows) == 1

    def test_skips_real_binance_header_row(self) -> None:
        # Verified against an actual archive (BTCUSDT 1h 2024-03): the monthly
        # CSV DOES start with a header. Pinned so the assumption cannot return.
        header = (
            "open_time,open,high,low,close,volume,close_time,quote_volume,"
            "count,taker_buy_volume,taker_buy_quote_volume,ignore"
        )
        rows = parse_kline_csv(header + "\n" + self.RAW)
        assert len(rows) == 1
        assert rows[0].open_time_ms == 1_700_000_000_000

    def test_header_only_payload_is_rejected(self) -> None:
        header = "open_time,open,high,low,close,volume"
        with pytest.raises(ValueError, match="empty"):
            parse_kline_csv(header + "\n")

    def test_rejects_malformed_row(self) -> None:
        with pytest.raises(ValueError):
            parse_kline_csv("1700000000000,not_a_number\n")

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_kline_csv("\n\n")


class TestUnzip:
    def test_extracts_single_member(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("BTCUSDT-1h-2024-03.csv", "a,b\n")
        assert unzip_bytes(buf.getvalue()) == b"a,b\n"

    def test_rejects_non_zip(self) -> None:
        with pytest.raises(ValueError, match="zip"):
            unzip_bytes(b"not a zip file at all")

    def test_rejects_empty_archive(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass
        with pytest.raises(ValueError, match="empty"):
            unzip_bytes(buf.getvalue())
