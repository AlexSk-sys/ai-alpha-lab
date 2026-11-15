"""Invariants for the multi-seed comparison.

Four hundred seeds must collapse into a compact table - one row per track -
with ordered percentiles and bounded rates. These tests pin the constant, the
statistics, and the shape of the output on a tiny synthetic market.
"""

from __future__ import annotations

import pytest

from config import RISK_BUDGETED, UNRESTRICTED, CostModel, Venue
from data import Candle
from seeds import N_SEEDS, TrackSummary, compare, format_table, percentile


def perp() -> CostModel:
    return CostModel(Venue.PERP_TAKER, 0.00045, 0.0, True)


def bars(n: int = 32) -> list[Candle]:
    out: list[Candle] = []
    price = 50_000.0
    for i in range(n):
        price *= 1.001
        out.append(Candle(i * 3_600_000, price, price, price, price, 10.0))
    return out


class TestConstant:
    def test_study_seed_count_is_400(self) -> None:
        assert N_SEEDS == 400


class TestPercentile:
    def test_median_interpolates(self) -> None:
        assert percentile((1.0, 2.0, 3.0, 4.0), 50.0) == pytest.approx(2.5)

    def test_edges_are_min_and_max(self) -> None:
        assert percentile((3.0, 1.0, 2.0), 0.0) == pytest.approx(1.0)
        assert percentile((3.0, 1.0, 2.0), 100.0) == pytest.approx(3.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="values"):
            percentile((), 50.0)

    def test_rejects_out_of_range_q(self) -> None:
        with pytest.raises(ValueError, match="q"):
            percentile((1.0,), 101.0)
        with pytest.raises(ValueError, match="q"):
            percentile((1.0,), -1.0)


class TestCompare:
    def test_covers_both_tracks_with_ordered_percentiles(self) -> None:
        summaries = compare(bars(), (UNRESTRICTED, RISK_BUDGETED), perp(), n_seeds=4)
        assert set(summaries) == {"A_unrestricted", "B_risk_budgeted"}
        for summary in summaries.values():
            assert isinstance(summary, TrackSummary)
            assert summary.n == 4
            assert summary.p10 <= summary.p50 <= summary.p90
            assert 0.0 <= summary.ruin_rate <= 1.0
            assert 0.0 <= summary.dd_p50 <= 1.0

    def test_deterministic_across_runs(self) -> None:
        market = bars()
        first = compare(market, (RISK_BUDGETED,), perp(), n_seeds=4)
        second = compare(market, (RISK_BUDGETED,), perp(), n_seeds=4)
        assert first == second

    def test_key_order_is_stable(self) -> None:
        summaries = compare(bars(), (RISK_BUDGETED, UNRESTRICTED), perp(), n_seeds=2)
        assert list(summaries) == sorted(summaries)


class TestTable:
    def test_compact_one_row_per_track(self) -> None:
        summaries = compare(bars(), (UNRESTRICTED, RISK_BUDGETED), perp(), n_seeds=4)
        table = format_table(summaries)
        assert "A_unrestricted" in table
        assert "B_risk_budgeted" in table
        assert len(table.splitlines()) < 20
