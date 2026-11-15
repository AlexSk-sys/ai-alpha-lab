"""Deterministic momentum agent: the strategy held constant across tracks.

WHY: the experiment isolates the risk envelope, so the decision function must
be IDENTICAL on both tracks - same bars, same seed, same signal. The tracks
diverge only when the runner passes the signal through different RiskOverlays.

Seed sensitivity (explicit): each seed explores a different parameterization -
a different fast/slow SMA window pair, trigger threshold, and conviction size -
derived by closed-form arithmetic on the seed. Different seeds therefore take
genuinely different stances on the same bars, which is what makes the
multi-seed final-equity distribution a distribution rather than a point. The
mapping seed -> params is a pure function, so it is reproducible forever.

WHAT IT DOES NOT DO: no network, no LLM, no wall clock, no `random` globals.
A decision at bar `i` reads only bars `0..i` inclusive - never the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from data import Candle
from risk import RiskOverlay


class SignalKind(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class Signal:
    """One stance: desired direction plus convicted size and its reason."""

    kind: SignalKind
    fraction: float
    leverage: float
    justification: str


@dataclass(frozen=True)
class Params:
    """The seed-parameterized SMA crossover the agent trades."""

    fast: int
    slow: int
    threshold: float
    base_fraction: float
    base_leverage: float


def params_for_seed(seed: int) -> Params:
    """Closed-form seed -> parameterization. Pure and total over seeds >= 0."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    fast = 3 + (seed % 5)
    slow = 12 + ((seed // 5) % 9)
    threshold = 0.001 * (1 + (seed % 7))
    base_fraction = 0.2 + 0.1 * ((seed // 35) % 5)
    base_leverage = 1.0 + ((seed // 175) % 3)
    return Params(fast, slow, threshold, base_fraction, base_leverage)


def _mean(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values) if values else 0.0


def decide(
    bars: Sequence[Candle],
    index: int,
    risk: RiskOverlay,
    seed: int,
) -> Signal:
    """Pure SMA-crossover stance from bars `0..index` plus `seed`."""
    if not bars:
        raise ValueError("bars must be non-empty")
    if not 0 <= index < len(bars):
        raise ValueError(f"index {index} outside 0..{len(bars) - 1}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    params = params_for_seed(seed)
    have = index + 1
    tag = f"seed={seed} fast={params.fast} slow={params.slow}"
    if have < params.slow:
        return Signal(SignalKind.FLAT, 0.0, 0.0, f"{tag} warmup -> FLAT")
    # Bounded tail: only the last `slow` closes ending at `index` are read,
    # so cost is O(slow) per decision and nothing past `index` is touched.
    start = index + 1 - params.slow
    closes = [bar.close for bar in bars[start : index + 1]]
    fast_avg = _mean(closes[-params.fast :])
    slow_avg = _mean(closes[-params.slow :])
    if slow_avg <= 0.0:
        return Signal(SignalKind.FLAT, 0.0, 0.0, f"{tag} bad-slow -> FLAT")
    edge = fast_avg / slow_avg - 1.0
    detail = f"{tag} edge={edge:+.4f} thr={params.threshold:.4f}"
    if edge > params.threshold:
        return Signal(
            SignalKind.LONG,
            risk.clamp_fraction(params.base_fraction, 0.0),
            risk.clamp_leverage(params.base_leverage, 0.0),
            f"{detail} -> LONG",
        )
    if edge < -params.threshold:
        return Signal(
            SignalKind.SHORT,
            risk.clamp_fraction(params.base_fraction, 0.0),
            risk.clamp_leverage(-params.base_leverage, 0.0),
            f"{detail} -> SHORT",
        )
    return Signal(SignalKind.FLAT, 0.0, 0.0, f"{detail} -> FLAT")


class Journal:
    """Bounded ring buffer of justification strings. Never grows past capacity."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buf: list[str] = []

    def append(self, entry: str) -> None:
        self._buf.append(entry)
        if len(self._buf) > self._capacity:
            del self._buf[: len(self._buf) - self._capacity]

    def entries(self) -> tuple[str, ...]:
        return tuple(self._buf)
