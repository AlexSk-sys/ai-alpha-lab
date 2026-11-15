"""Invariants for the deterministic decision function.

The multi-seed distribution is meaningless unless decisions are pure (no
network, no clock, no globals), reproducible from bar data plus seed, blind to
the future, and genuinely seed-sensitive in kind - not just in prose.
"""

from __future__ import annotations

import pytest

from agent import Journal, SignalKind, decide, params_for_seed
from config import RISK_BUDGETED, UNRESTRICTED
from data import Candle
from risk import RiskOverlay


def trend(up: bool, n: int = 40, drift: float = 0.002) -> list[Candle]:
    bars: list[Candle] = []
    price = 50_000.0
    for i in range(n):
        price = price * (1.0 + drift) if up else price * (1.0 - drift)
        bars.append(Candle(i * 3_600_000, price, price, price, price, 10.0))
    return bars


def flat(n: int = 40) -> list[Candle]:
    return [
        Candle(i * 3_600_000, 50_000.0, 50_000.0, 50_000.0, 50_000.0, 10.0)
        for i in range(n)
    ]


class TestPurity:
    def test_same_inputs_give_identical_signal(self) -> None:
        bars = trend(True)
        risk = RiskOverlay(RISK_BUDGETED)
        first = decide(bars, len(bars) - 1, risk, 7)
        second = decide(bars, len(bars) - 1, risk, 7)
        assert first == second

    def test_justification_names_the_seed(self) -> None:
        bars = trend(True)
        signal = decide(bars, len(bars) - 1, RiskOverlay(RISK_BUDGETED), 7)
        assert "7" in signal.justification
        assert len(signal.justification) > 0

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            decide(trend(True), 39, RiskOverlay(RISK_BUDGETED), -1)

    def test_rejects_out_of_range_index(self) -> None:
        bars = trend(True)
        risk = RiskOverlay(RISK_BUDGETED)
        with pytest.raises(ValueError, match="index"):
            decide(bars, len(bars), risk, 0)
        with pytest.raises(ValueError, match="index"):
            decide(bars, -1, risk, 0)

    def test_rejects_empty_bars(self) -> None:
        with pytest.raises(ValueError, match="bars"):
            decide([], 0, RiskOverlay(RISK_BUDGETED), 0)


class TestNoLookahead:
    def test_future_bars_do_not_change_the_decision(self) -> None:
        bars = trend(True)
        risk = RiskOverlay(RISK_BUDGETED)
        before = decide(bars, 25, risk, 3)
        bars.extend(trend(False, n=10))
        bars.append(Candle(999, 1.0, 1.0, 1.0, 1.0, 1.0))
        after = decide(bars, 25, risk, 3)
        assert before == after

    def test_warmup_is_flat_before_enough_history(self) -> None:
        bars = trend(True, n=5)
        signal = decide(bars, 4, RiskOverlay(RISK_BUDGETED), 0)
        assert signal.kind is SignalKind.FLAT


class TestSeedSensitivity:
    def test_parameterizations_differ_across_seeds(self) -> None:
        params = {params_for_seed(seed) for seed in range(32)}
        assert len(params) > 1

    def test_kinds_differ_across_seeds_on_gentle_drift(self) -> None:
        # A 0.2% drift sits inside the threshold spread, so strict seeds stay
        # flat while eager seeds go long: the distribution has a shape.
        bars = trend(True, drift=0.002)
        risk = RiskOverlay(RISK_BUDGETED)
        kinds = {decide(bars, len(bars) - 1, risk, seed).kind for seed in range(64)}
        assert SignalKind.LONG in kinds
        assert SignalKind.FLAT in kinds

    def test_downtrend_can_emit_short(self) -> None:
        bars = trend(False)
        risk = RiskOverlay(RISK_BUDGETED)
        kinds = {decide(bars, len(bars) - 1, risk, seed).kind for seed in range(64)}
        assert SignalKind.SHORT in kinds


class TestEnvelope:
    def test_budgeted_signal_respects_envelope(self) -> None:
        bars = trend(True)
        signal = decide(bars, len(bars) - 1, RiskOverlay(RISK_BUDGETED), 0)
        assert signal.fraction <= 0.35 + 1e-12
        assert abs(signal.leverage) <= 2.0 + 1e-12

    def test_flat_signal_carries_no_size(self) -> None:
        signal = decide(flat(), len(flat()) - 1, RiskOverlay(UNRESTRICTED), 0)
        assert signal.kind is SignalKind.FLAT
        assert signal.fraction == pytest.approx(0.0)
        assert signal.leverage == pytest.approx(0.0)

    def test_short_leverage_is_negative(self) -> None:
        bars = trend(False)
        risk = RiskOverlay(RISK_BUDGETED)
        for seed in range(64):
            signal = decide(bars, len(bars) - 1, risk, seed)
            if signal.kind is SignalKind.SHORT:
                assert signal.leverage < 0.0
                return
        pytest.fail("no SHORT emitted on a straight downtrend")


class TestJournal:
    def test_journal_is_bounded(self) -> None:
        journal = Journal(capacity=8)
        for i in range(100):
            journal.append(f"entry {i}")
        assert len(journal.entries()) == 8

    def test_journal_keeps_the_latest(self) -> None:
        journal = Journal(capacity=4)
        for i in range(6):
            journal.append(f"entry {i}")
        assert journal.entries()[-1] == "entry 5"

    def test_journal_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            Journal(capacity=0)
