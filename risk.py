"""Risk overlay: the ONLY semantic difference between tracks A and B.

WHY: the study compares an unrestricted agent against an identical strategy
under a hard risk envelope. If the tracks also differed in strategy or
execution the comparison would be confounded and unpublishable, so every
difference lives here and nowhere else.

WHAT IT DOES NOT DO: it never looks at prices, never places orders, never
touches the network. It maps a requested size plus the current drawdown to an
allowed size, and answers halt/capacity questions from the budget.

Semantic difference (explicit):
    Track A (unrestricted) may size up to its full conviction: requests pass
    through unchanged except at the venue's own leverage ceiling, and no
    drawdown throttle or circuit breaker ever binds.
    Track B (budgeted) is capped three ways: per-trade size (max leverage and
    max position fraction), total open risk (max positions), and a
    max-drawdown throttle that scales allowed size linearly to zero at the
    breaker, where the breaker forces the agent flat.

Documented invariants (no arbitrary constants):
    VENUE_LEVERAGE_CEILING -- Track A inherits config.UNRESTRICTED.max_leverage
    (75x), the venue's own ceiling, so "permissive" never means "unbounded".
    ENVELOPE_CLAMP -- allowed size never exceeds budget.max_leverage or
    budget.max_position_fraction; clamps preserve the sign of leverage so a
    short request stays short.
    LINEAR_THROTTLE -- budgeted factor is 1 - drawdown / max_drawdown, the
    simplest monotone schedule with factor 1 at zero drawdown and 0 at the
    breaker; unrestricted has no breaker so its factor is identically 1.
    NONNEGATIVE_SIZE -- outputs are never negative and never amplify the
    request; throttle only shrinks.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import RiskBudget


@dataclass(frozen=True)
class RiskOverlay:
    """Pure size governor configured from a config.RiskBudget."""

    budget: RiskBudget

    def _checked_drawdown(self, drawdown: float) -> float:
        if drawdown < 0.0:
            raise ValueError(f"drawdown must be non-negative, got {drawdown}")
        return drawdown

    def throttle_factor(self, drawdown: float) -> float:
        """Linear drawdown throttle in [0, 1]."""
        self._checked_drawdown(drawdown)
        if self.budget.max_drawdown is None:
            return 1.0
        factor = 1.0 - drawdown / self.budget.max_drawdown
        return max(0.0, min(1.0, factor))

    def clamp_leverage(self, requested: float, drawdown: float) -> float:
        """Clamp leverage to the envelope, then scale by the throttle."""
        self._checked_drawdown(drawdown)
        if requested == 0.0:
            return 0.0
        cap = self.budget.max_leverage
        clamped = max(-cap, min(cap, requested))
        return clamped * self.throttle_factor(drawdown)

    def clamp_fraction(self, requested: float, drawdown: float) -> float:
        """Clamp cash fraction to the envelope, then scale by the throttle."""
        self._checked_drawdown(drawdown)
        if requested == 0.0:
            return 0.0
        if requested < 0.0:
            raise ValueError(f"fraction must be non-negative, got {requested}")
        clamped = min(requested, self.budget.max_position_fraction)
        return clamped * self.throttle_factor(drawdown)

    def allows_new_position(self, open_count: int) -> bool:
        """Whether total-open-risk capacity admits one more position."""
        if open_count < 0:
            raise ValueError(f"open_count must be non-negative, got {open_count}")
        return open_count < self.budget.max_positions

    def should_halt(self, drawdown: float) -> bool:
        """Whether the track's circuit breaker has tripped."""
        if drawdown < 0.0:
            raise ValueError(f"drawdown must be non-negative, got {drawdown}")
        return self.budget.is_halted(drawdown)
