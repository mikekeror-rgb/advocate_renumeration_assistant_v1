"""
fee_calculator.py — Deterministic calculators for the Advocates Remuneration Order (Kenya).

Why this exists: several Schedules require tiered/cumulative percentage math
(e.g. Schedule 1) or a bracket lookup with an open-ended percentage tail
(e.g. Schedule 6, 7). No single sentence in the Order states these results —
they only exist after running the formula — so the LLM in rag_pipeline.py
correctly refuses to state them (per its system prompt) rather than guess.
This module computes them exactly instead, so the pipeline can call a
function and quote a verified number rather than ask the model to do math.

Currently implemented: Schedule 1 (First & Second Scale), Schedule 6 (Part A,
items 1(a)/1(b)), Schedule 7. Add more schedules by following the same
cumulative_tiered_fee() / bracket_fee() pattern — see the docstrings below.

NOT LEGAL ADVICE: this mechanically applies the schedule text as extracted
from the Order. Always confirm the amendment in force and any special-order
or discretionary adjustments (e.g. paragraph 79) with a qualified advocate
before relying on these figures.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------------

def cumulative_tiered_fee(value: float, tiers: list[tuple[float | None, float]], first_tier_minimum: float | None = None) -> float:
    """
    Schedule 1-style calculation: marginal rate applied cumulatively across bands.
    tiers: ordered list of (upper_bound, rate) — upper_bound is the cap for that band
           (None for the final open-ended band). Each rate applies only to the
           portion of `value` falling within that band.
    first_tier_minimum: the "...or Kshs X, whichever is higher" floor that applies
           to the first band's contribution only (per the Order's own wording).
    """
    if value <= 0:
        return first_tier_minimum or 0.0

    fee = 0.0
    prev_bound = 0.0
    for i, (upper, rate) in enumerate(tiers):
        if upper is None:
            band_amount = max(value - prev_bound, 0.0)
            contribution = band_amount * rate
        else:
            band_amount = max(min(value, upper) - prev_bound, 0.0)
            contribution = band_amount * rate

        if i == 0 and first_tier_minimum is not None:
            contribution = max(contribution, first_tier_minimum)

        fee += contribution
        if upper is not None:
            prev_bound = upper
        if upper is not None and value <= upper:
            break

    return round(fee, 2)


@dataclass
class Bracket:
    upper: float | None   # None marks the terminal open-ended bracket
    fixed_fee: float | None = None       # used for closed brackets with a flat fee
    base_from_upper: float | None = None  # for the open bracket: fee at the previous boundary
    excess_rate: float | None = None      # for the open bracket: % applied to the excess


def bracket_fee(value: float, brackets: list[Bracket]) -> float:
    """
    Schedule 6/7/9/11-style calculation: flat fee per bracket, with a final
    open-ended bracket computed as base_from_upper + excess_rate * (value - boundary).
    """
    prev_upper = 0.0
    for b in brackets:
        if b.upper is not None:
            if value <= b.upper:
                return b.fixed_fee
            prev_upper = b.upper
        else:
            excess = value - prev_upper
            return round(b.base_from_upper + excess * b.excess_rate, 2)
    raise ValueError("value did not match any bracket — check bracket list covers full range")


# ---------------------------------------------------------------------------
# Schedule 1 — First Scale: sales & purchases of immovable property (para 18(a))
# ---------------------------------------------------------------------------

SCHEDULE1_FIRST_SCALE_TIERS = [
    (5_000_000, 0.02),
    (100_000_000, 0.015),
    (250_000_000, 0.0125),
    (1_000_000_000, 0.01),
    (None, 0.001),
]
SCHEDULE1_FIRST_SCALE_MINIMUM = 35_000


def schedule1_first_scale_fee(value: float) -> float:
    """Vendor's/Purchaser's advocate fee for a sale/purchase of immovable property."""
    return cumulative_tiered_fee(value, SCHEDULE1_FIRST_SCALE_TIERS, SCHEDULE1_FIRST_SCALE_MINIMUM)


# ---------------------------------------------------------------------------
# Schedule 1 — Second Scale: mortgages/charges ("securities") (para 18(a))
# ---------------------------------------------------------------------------

SCHEDULE1_SECOND_SCALE_TIERS = [
    (2_500_000, 0.02),
    (5_000_000, 0.0175),
    (100_000_000, 0.01),
    (250_000_000, 0.0075),
    (1_000_000_000, 0.0015),
    (None, 0.001),
]
SCHEDULE1_SECOND_SCALE_MINIMUM = 28_000


def schedule1_second_scale_creation_fee(value: float, party: str = "grantee") -> float:
    """
    Fee for CREATING a security (mortgage/charge) of the given value.
    party: 'grantee' (lender/chargee) or 'grantor' (borrower/chargor, who pays 50% of grantee's fee).
    """
    grantee_fee = cumulative_tiered_fee(value, SCHEDULE1_SECOND_SCALE_TIERS, SCHEDULE1_SECOND_SCALE_MINIMUM)
    if party == "grantee":
        return grantee_fee
    elif party == "grantor":
        return round(grantee_fee * 0.5, 2)
    raise ValueError("party must be 'grantee' or 'grantor'")


def schedule1_second_scale_discharge_fee(value: float, party: str = "grantee", undertaking_required: bool = True) -> float:
    """
    Fee for DISCHARGING (reconveyance/reassignment) of a security of the given value.
    - grantee's advocate: 25% of the creation fee (min 15,000) if an undertaking is
      required for redemption; 15% (min 10,000) if not.
    - grantor's advocate: 25% of the (grantee's) creation fee under sub-para (b), min 15,000,
      regardless of the undertaking distinction (that distinction is only in (c), for the grantee).
    """
    creation_fee_grantee = cumulative_tiered_fee(value, SCHEDULE1_SECOND_SCALE_TIERS, SCHEDULE1_SECOND_SCALE_MINIMUM)

    if party == "grantee":
        if undertaking_required:
            return round(max(creation_fee_grantee * 0.25, 15_000), 2)
        else:
            return round(max(creation_fee_grantee * 0.15, 10_000), 2)
    elif party == "grantor":
        return round(max(creation_fee_grantee * 0.25, 15_000), 2)
    raise ValueError("party must be 'grantee' or 'grantor'")


# ---------------------------------------------------------------------------
# Schedule 6, Part A — High Court instruction fees, item 1(a)/(b)
# ---------------------------------------------------------------------------

def _schedule6_brackets(defended: bool) -> list[Bracket]:
    if not defended:
        # item 1(a): undefended / no denial of liability filed
        base_at_1m = 75_000
        return [
            Bracket(upper=500_000, fixed_fee=45_000),
            Bracket(upper=750_000, fixed_fee=65_000),
            Bracket(upper=1_000_000, fixed_fee=75_000),
            Bracket(upper=20_000_000, fixed_fee=None,
                    base_from_upper=base_at_1m, excess_rate=0.0175),
            Bracket(upper=None,
                    base_from_upper=base_at_1m + (20_000_000 - 1_000_000) * 0.0175,
                    excess_rate=0.015),
        ]
    else:
        # item 1(b): defended / denial of liability filed
        base_at_1m = 120_000
        return [
            Bracket(upper=500_000, fixed_fee=75_000),
            Bracket(upper=750_000, fixed_fee=90_000),
            Bracket(upper=1_000_000, fixed_fee=120_000),
            Bracket(upper=20_000_000, fixed_fee=None,
                    base_from_upper=base_at_1m, excess_rate=0.02),
            Bracket(upper=None,
                    base_from_upper=base_at_1m + (20_000_000 - 1_000_000) * 0.02,
                    excess_rate=0.015),
        ]


def schedule6_instruction_fee(value: float, defended: bool) -> float:
    """High Court instruction fee (party & party) for a suit of the given value."""
    # The 20,000,000 bracket's own excess_rate needs special handling since it's
    # itself a running total, not a flat fee — recompute directly instead of
    # reusing bracket_fee's single-step excess formula for that middle band.
    base_at_1m = 120_000 if defended else 75_000
    mid_rate = 0.02 if defended else 0.0175
    if value <= 500_000:
        return 75_000 if defended else 45_000
    elif value <= 750_000:
        return 90_000 if defended else 65_000
    elif value <= 1_000_000:
        return 120_000 if defended else 75_000
    elif value <= 20_000_000:
        return round(base_at_1m + (value - 1_000_000) * mid_rate, 2)
    else:
        fee_at_20m = base_at_1m + (20_000_000 - 1_000_000) * mid_rate
        return round(fee_at_20m + (value - 20_000_000) * 0.015, 2)


# ---------------------------------------------------------------------------
# Schedule 7 — Subordinate court instruction fees
# ---------------------------------------------------------------------------

_SCHEDULE7_TABLE = [
    (50_000, 10_000, 15_000),
    (100_000, 15_000, 30_000),
    (200_000, 30_000, 40_000),
    (500_000, 45_000, 65_000),
    (1_000_000, 65_000, 90_000),
    (2_000_000, 90_000, 120_000),
]


def schedule7_instruction_fee(value: float, scale: str = "lower") -> float:
    """
    Subordinate court instruction fee. scale: 'lower' (no denial of liability filed)
    or 'higher' (denial of liability filed / all other cases).
    """
    if scale not in ("lower", "higher"):
        raise ValueError("scale must be 'lower' or 'higher'")
    idx = 1 if scale == "lower" else 2

    for upper, lower_fee, higher_fee in _SCHEDULE7_TABLE:
        if value <= upper:
            return lower_fee if scale == "lower" else higher_fee

    # value > 2,000,000: fee as for 2,000,000 plus 2.5% of the excess
    base_at_2m = _SCHEDULE7_TABLE[-1][idx]
    return round(base_at_2m + (value - 2_000_000) * 0.025, 2)


# ---------------------------------------------------------------------------
# Self-test / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Schedule 1, Second Scale — discharge of charge over Kshs 2,800,000:")
    creation = schedule1_second_scale_creation_fee(2_800_000, party="grantee")
    print(f"  Creation fee (grantee's advocate): Kshs {creation:,.2f}")
    with_undertaking = schedule1_second_scale_discharge_fee(2_800_000, "grantee", undertaking_required=True)
    without_undertaking = schedule1_second_scale_discharge_fee(2_800_000, "grantee", undertaking_required=False)
    print(f"  Discharge fee (with undertaking):    Kshs {with_undertaking:,.2f}")
    print(f"  Discharge fee (without undertaking): Kshs {without_undertaking:,.2f}")

    print("\nSchedule 6 — High Court instruction fee for a Kshs 3,500,000 defended suit:")
    print(f"  Kshs {schedule6_instruction_fee(3_500_000, defended=True):,.2f}")

    print("\nSchedule 7 — Subordinate court instruction fee for Kshs 350,000, higher scale:")
    print(f"  Kshs {schedule7_instruction_fee(350_000, scale='higher'):,.2f}")
