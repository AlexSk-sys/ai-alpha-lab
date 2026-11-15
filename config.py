"""Canonical research parameters for the $1M autonomous-agent study.

Single source of truth. `ceiling.py` and the simulation engine both import from
here so that a number is stated exactly once in the codebase.

Study design
------------
Three tracks are run over identical market data, differing ONLY in the risk
constraint layer. That isolation is the whole point: if the tracks also differed
in strategy or execution, the comparison would be confounded and unpublishable.

    Track A  unrestricted  - carte blanche, may use any size/leverage
    Track B  risk-budgeted - identical strategy, hard risk envelope
    Control  conservative  - the pre-existing deposit/OFZ/index study

All tracks are analysis-only: backtest plus paper trading, no real capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --- Money and horizon ------------------------------------------------------
# Capital is the user's figure verbatim: 100_000 RUB, no monthly contributions.
START_CAPITAL_RUB: float = 100_000.0
TARGET_USD: float = 1_000_000.0
HORIZON_MONTHS: int = 24

# The RUB/USD rate is FROZEN so the target cannot drift with the market. It is a
# study constant chosen for round arithmetic, not a live quote. The study is
# therefore stated in two currencies and the USD multiple is rate-dependent:
# at 100 RUB/USD the target is 1000x, at 25 RUB/USD it would be 250x. Both are
# reported; the RUB multiple is the one that carries the claim.
RUB_PER_USD: float = 100.0

# Alternative rate retained so the sensitivity of the headline claim is visible
# rather than buried in a single number.
RUB_PER_USD_ALT: float = 25.0


class Venue(str, Enum):
    """Cost model flavour. Rates are UNVERIFIED study inputs."""

    SPOT_TAKER = "binance_spot_taker"
    SPOT_MAKER = "binance_spot_maker"
    PERP_TAKER = "binance_perp_taker"
    PERP_MAKER = "binance_perp_maker"


@dataclass(frozen=True)
class CostModel:
    """Per-venue execution costs.

    round_trip_fee is applied on each side, so a full round trip pays
    2 * round_trip_fee of the traded notional.
    """

    venue: Venue
    round_trip_fee: float
    slippage_bps: float
    supports_leverage: bool

    def __post_init__(self) -> None:
        if self.round_trip_fee < 0.0:
            raise ValueError(
                f"{self.venue}: round_trip_fee must be non-negative,"
                f" got {self.round_trip_fee}"
            )
        if self.slippage_bps < 0.0:
            raise ValueError(
                f"{self.venue}: slippage_bps must be non-negative,"
                f" got {self.slippage_bps}"
            )

    def fee(self, notional: float) -> float:
        if notional < 0.0:
            raise ValueError(f"notional must be non-negative, got {notional}")
        return notional * self.round_trip_fee

    def fill_price(self, mid: float, side_is_buy: bool) -> float:
        """Apply adverse slippage to a mid price. Buys pay up, sells give up."""
        if mid <= 0.0:
            raise ValueError(f"mid must be positive, got {mid}")
        drift = self.slippage_bps / 10_000.0
        return mid * (1.0 + drift) if side_is_buy else mid * (1.0 - drift)


# UNVERIFIED against live venue schedules before the dissertation is submitted.
COST_MODELS: dict[Venue, CostModel] = {
    Venue.SPOT_TAKER: CostModel(Venue.SPOT_TAKER, 0.0010, 5.0, False),
    Venue.SPOT_MAKER: CostModel(Venue.SPOT_MAKER, 0.0010, 2.0, False),
    Venue.PERP_TAKER: CostModel(Venue.PERP_TAKER, 0.00045, 4.0, True),
    Venue.PERP_MAKER: CostModel(Venue.PERP_MAKER, 0.00018, 1.5, True),
}


# --- Perpetual funding ------------------------------------------------------
# Binance-style perps fund every 8 hours. A long pays when the rate is positive.
FUNDING_INTERVAL_HOURS: int = 8
FUNDING_HOURS_PER_YEAR: int = 24 * 365
FUNDING_PERIODS_PER_YEAR: int = FUNDING_HOURS_PER_YEAR // FUNDING_INTERVAL_HOURS


@dataclass(frozen=True)
class RiskBudget:
    """The constraint envelope applied by Track B.

    max_leverage is the structural lever: at this size a leveraged position is
    liquidatable on an ordinary adverse move, so it is the primary risk knob.
    max_position_fraction caps concentration; max_drawdown is a hard circuit
    breaker that forces the agent flat, not merely smaller.
    """

    name: str
    max_leverage: float
    max_position_fraction: float
    max_positions: int
    # None means "no drawdown circuit breaker". This is a real state, not an
    # absent value: Track A and the buy-and-hold control must both run without a
    # breaker. A 0.0 sentinel would be ambiguous with "zero drawdown tolerance".
    max_drawdown: float | None = None

    def __post_init__(self) -> None:
        if self.max_leverage < 1.0:
            raise ValueError(
                f"{self.name}: max_leverage must be >= 1.0, got {self.max_leverage}"
            )
        if not 0.0 < self.max_position_fraction <= 1.0:
            raise ValueError(
                f"{self.name}: max_position_fraction must be in (0, 1],"
                f" got {self.max_position_fraction}"
            )
        if self.max_positions < 1:
            raise ValueError(
                f"{self.name}: max_positions must be >= 1, got {self.max_positions}"
            )
        if self.max_drawdown is not None and not 0.0 < self.max_drawdown < 1.0:
            raise ValueError(
                f"{self.name}: max_drawdown must be None or in (0, 1),"
                f" got {self.max_drawdown}"
            )

    def permits(self, leverage: float, position_fraction: float) -> bool:
        """Whether a proposed order clears this track's risk envelope."""
        if leverage > self.max_leverage:
            return False
        return position_fraction <= self.max_position_fraction

    def is_halted(self, current_drawdown: float) -> bool:
        """Whether a circuit breaker has tripped at this drawdown."""
        if self.max_drawdown is None:
            return False
        if current_drawdown < 0.0:
            raise ValueError(
                f"{self.name}: drawdown must be non-negative, got {current_drawdown}"
            )
        return current_drawdown >= self.max_drawdown


# Track A is intentionally NOT a tight envelope. It is the unconstrained arm of
# the experiment, so its only bound is the venue's own leverage ceiling.
UNRESTRICTED = RiskBudget(
    name="A_unrestricted",
    max_leverage=75.0,
    max_position_fraction=1.0,
    max_positions=50,
    max_drawdown=None,
)

RISK_BUDGETED = RiskBudget(
    name="B_risk_budgeted",
    max_leverage=2.0,
    max_position_fraction=0.35,
    max_positions=3,
    max_drawdown=0.20,
)

CONSERVATIVE_CONTROL = RiskBudget(
    name="C_conservative_control",
    max_leverage=1.0,
    max_position_fraction=1.0,
    max_positions=1,
    max_drawdown=None,
)


class Track(str, Enum):
    A_UNRESTRICTED = "A_unrestricted"
    B_RISK_BUDGETED = "B_risk_budgeted"
    C_CONTROL = "C_conservative_control"


TRACKS: dict[Track, RiskBudget] = {
    Track.A_UNRESTRICTED: UNRESTRICTED,
    Track.B_RISK_BUDGETED: RISK_BUDGETED,
    Track.C_CONTROL: CONSERVATIVE_CONTROL,
}


@dataclass(frozen=True)
class Study:
    """Aggregated immutable description of the experiment."""

    start_capital_rub: float = START_CAPITAL_RUB
    target_rub: float = TARGET_USD * RUB_PER_USD
    target_rub_alt_rate: float = TARGET_USD * RUB_PER_USD_ALT
    horizon_months: int = HORIZON_MONTHS
    seeds: int = 400
    costs: dict[Venue, CostModel] = field(default_factory=lambda: dict(COST_MODELS))
    tracks: dict[Track, RiskBudget] = field(default_factory=lambda: dict(TRACKS))


STUDY = Study()


def required_multiple(rub_per_usd: float = RUB_PER_USD) -> float:
    """Gross multiple the target demands at the given rate."""
    if rub_per_usd <= 0.0:
        raise ValueError(f"rub_per_usd must be positive, got {rub_per_usd}")
    return (TARGET_USD * rub_per_usd) / START_CAPITAL_RUB
