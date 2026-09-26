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
# Full itemized bill of costs (Schedule 6 — High Court)
# ---------------------------------------------------------------------------

# Schedule 6, Part A rates (Advocates Remuneration Order, as revised to 2022)
PLEADING_BASE_RATE = 1_100      # item 4(a)(i): pleading of four folios or less
PLEADING_EXTRA_FOLIO = 150      # item 4(a)(ii): each folio after the first four
OTHER_DRAWING_FOLIO = 180       # item 4(d): all other necessary documents, per folio
PERUSAL_FOLIO = 50              # item 8(a): perusals, per folio
LETTER_RATE = 1_000             # item 6(a): letters before action / other necessary letters
MENTION_RATE = 1_000            # item 7(c): attendance at court on a fixed date / calling lists
HEARING_DAY_RATE = 10_000       # item 7(d): attendance before judge, whole day (ordinary scale)
ADVOCATE_CLIENT_UPLIFT = 0.50   # Part B: Part A fees increased by 50%


@dataclass
class BillLineItem:
    item: str
    calculation_note: str
    amount: float | None   # None displays as "—" — case-specific input not provided
    classification: str    # "Professional fee" | "Disbursement"
    basis: str = ""        # the Order provision (or other law) the item is charged under


@dataclass
class BillOfCosts:
    matter: str
    subject_matter_value: float
    applicable_scale: str
    line_items: list[BillLineItem]
    vat_rate: float

    @property
    def professional_fee_subtotal(self) -> float:
        return sum(li.amount or 0 for li in self.line_items if li.classification == "Professional fee")

    @property
    def disbursement_subtotal(self) -> float:
        return sum(li.amount or 0 for li in self.line_items if li.classification == "Disbursement")

    @property
    def subtotal(self) -> float:
        return round(self.professional_fee_subtotal + self.disbursement_subtotal, 2)

    @property
    def vat_amount(self) -> float:
        # VAT applies to taxable professional fees only, not disbursements —
        # disbursements are pass-through actual costs, not the advocate's own
        # taxable supply, under standard Kenyan VAT treatment of legal services.
        return round(self.professional_fee_subtotal * self.vat_rate, 2)

    @property
    def total(self) -> float:
        return round(self.subtotal + self.vat_amount, 2)


def pleading_drawing_fee(folios: int) -> float:
    """Schedule 6 item 4(a): Kshs 1,100 for a pleading of up to four folios,
    plus Kshs 150 for each folio beyond the first four."""
    if folios <= 0:
        return 0.0
    return PLEADING_BASE_RATE + max(folios - 4, 0) * PLEADING_EXTRA_FOLIO


def generate_high_court_bill(
    subject_matter_value: float,
    defended: bool,
    letters_to_advocate: int = 0,
    letters_to_client: int = 0,
    mentions: int = 0,
    hearings: int = 0,
    disbursements: dict[str, float] | None = None,
    vat_rate: float = 0.16,
    pleading_folios: list[int] | None = None,
    other_drawing_folios: int = 0,
    perusal_folios: int = 0,
    advocate_client: bool = False,
) -> BillOfCosts:
    """
    Assemble a full itemized High Court bill of costs under Schedule 6.

    Only the instruction fee and getting-up fee are pure formula on
    subject_matter_value. Everything else needs case-specific counts the
    user supplies — the Order prescribes a PER-ITEM rate but can't know how
    many letters, folios or hearings there were in a given matter. Items
    left at zero/None still appear in the table as "—" with their rate, so
    the bill shows what's still needed rather than silently dropping it.

    advocate_client=True applies Schedule 6, Part B: the Part A professional
    fees increased by 50% (disbursements are not uplifted).

    VAT defaults to 16% on professional fees only. VAT is set by the VAT Act,
    not the Order, and can change independently of it — confirm the rate.
    """
    instruction_fee = schedule6_instruction_fee(subject_matter_value, defended=defended)
    getting_up_fee = round(instruction_fee / 3, 2)

    PF = "Professional fee"
    line_items = [
        BillLineItem("Instruction fee", f"Tariff calculation on Kshs {subject_matter_value:,.0f}",
                     instruction_fee, PF, f"Sch. 6 Part A item 1({'b' if defended else 'a'})"),
        BillLineItem("Getting-up fee", "1/3 of instruction fee", getting_up_fee, PF,
                     "Sch. 6 Part A item 2"),
    ]

    # --- drawing and perusals (charged per folio of 100 words, para 17) ---
    if pleading_folios:
        docs = [f for f in pleading_folios if f > 0]
        amount = sum(pleading_drawing_fee(f) for f in docs)
        note = f"{len(docs)} pleading(s) of {', '.join(str(f) for f in docs)} folios — Kshs 1,100 up to 4 folios + Kshs 150/extra folio"
        line_items.append(BillLineItem("Drawing pleadings", note, amount, PF, "Sch. 6 Part A item 4(a)"))
    else:
        line_items.append(BillLineItem("Drawing pleadings",
                                       "folio counts not provided — Kshs 1,100 up to 4 folios + Kshs 150/extra folio",
                                       None, PF, "Sch. 6 Part A item 4(a)"))

    if other_drawing_folios:
        line_items.append(BillLineItem("Drawing other documents",
                                       f"{other_drawing_folios} folios × Kshs {OTHER_DRAWING_FOLIO}",
                                       other_drawing_folios * OTHER_DRAWING_FOLIO, PF, "Sch. 6 Part A item 4(d)"))
    else:
        line_items.append(BillLineItem("Drawing other documents",
                                       f"folio count not provided — Kshs {OTHER_DRAWING_FOLIO} per folio",
                                       None, PF, "Sch. 6 Part A item 4(d)"))

    if perusal_folios:
        line_items.append(BillLineItem("Perusals", f"{perusal_folios} folios × Kshs {PERUSAL_FOLIO}",
                                       perusal_folios * PERUSAL_FOLIO, PF, "Sch. 6 Part A item 8(a)"))
    else:
        line_items.append(BillLineItem("Perusals", f"folio count not provided — Kshs {PERUSAL_FOLIO} per folio",
                                       None, PF, "Sch. 6 Part A item 8(a)"))

    # --- correspondence ---
    for label, count in (("Letters to opposing advocate", letters_to_advocate),
                         ("Letters to client", letters_to_client)):
        if count:
            line_items.append(BillLineItem(label, f"{count} letters × Kshs {LETTER_RATE:,}",
                                           count * LETTER_RATE, PF, "Sch. 6 Part A item 6(a)"))
        else:
            line_items.append(BillLineItem(label, f"count not provided — Kshs {LETTER_RATE:,} per letter",
                                           None, PF, "Sch. 6 Part A item 6(a)"))

    # --- attendances ---
    if mentions or hearings:
        parts, amount = [], 0.0
        if mentions:
            parts.append(f"{mentions} mention(s) × Kshs {MENTION_RATE:,}")
            amount += mentions * MENTION_RATE
        if hearings:
            parts.append(f"{hearings} hearing day(s) × Kshs {HEARING_DAY_RATE:,}")
            amount += hearings * HEARING_DAY_RATE
        line_items.append(BillLineItem("Court attendances", " + ".join(parts), amount, PF,
                                       "Sch. 6 Part A items 7(c), 7(d)"))
    else:
        line_items.append(BillLineItem("Court attendances",
                                       f"count not provided — Kshs {MENTION_RATE:,} per mention / Kshs {HEARING_DAY_RATE:,} per hearing day",
                                       None, PF, "Sch. 6 Part A items 7(c), 7(d)"))

    # --- advocate-client uplift (computed on the Part A professional fees above) ---
    if advocate_client:
        part_a_fees = sum(li.amount or 0 for li in line_items if li.classification == PF)
        line_items.append(BillLineItem("Advocate-client uplift", "50% of Part A professional fees",
                                       round(part_a_fees * ADVOCATE_CLIENT_UPLIFT, 2), PF,
                                       "Sch. 6 Part B"))

    # --- disbursements (actual costs — vouchers producible on taxation, para 74) ---
    DISBURSEMENT_BASIS = {
        "Service of documents": "Sch. 6 item 9 (Kshs 1,400 within 3 km + Kshs 35/km + actual travel)",
        "Court filing fees": "Actual court fees paid (para 74)",
        "Photocopying/printing": "Sch. 6 item 5(d) (actual cost, vouched)",
    }
    disbursements = disbursements or {}
    for label, basis in DISBURSEMENT_BASIS.items():
        if label in disbursements:
            line_items.append(BillLineItem(label, "Actual allowable expense", disbursements[label], "Disbursement", basis))
        else:
            line_items.append(BillLineItem(label, "actual amount not provided", None, "Disbursement", basis))
    for label, amt in disbursements.items():
        if label not in DISBURSEMENT_BASIS:
            line_items.append(BillLineItem(label, "Actual allowable expense", amt, "Disbursement",
                                           "Actual cost, vouched (para 74)"))

    bill_type = "Advocate-client" if advocate_client else "Party and party"
    matter = f"{'Defended' if defended else 'Undefended'} High Court suit ({bill_type.lower()} costs)"
    scale_word = "defended contentious" if defended else "undefended"
    applicable_scale = f"High Court — {scale_word} matter, Schedule 6 Part {'A + B' if advocate_client else 'A'}"

    return BillOfCosts(
        matter=matter,
        subject_matter_value=subject_matter_value,
        applicable_scale=applicable_scale,
        line_items=line_items,
        vat_rate=vat_rate,
    )


def format_bill_as_markdown(bill: BillOfCosts) -> str:
    def fmt(amount):
        return f"{amount:,.2f}" if amount is not None else "—"

    lines = [
        f"**Matter:** {bill.matter}",
        f"**Subject matter:** KSh {bill.subject_matter_value:,.0f}",
        f"**Applicable scale:** {bill.applicable_scale}",
        "",
        "| Item | Calculation | Basis | Amount (KSh) | Classification |",
        "|---|---|---|---|---|",
    ]
    for li in bill.line_items:
        lines.append(f"| {li.item} | {li.calculation_note} | {li.basis} | {fmt(li.amount)} | {li.classification} |")

    lines.append(f"| **Professional fees** | | | **{fmt(bill.professional_fee_subtotal)}** | |")
    lines.append(f"| **Disbursements** | | | **{fmt(bill.disbursement_subtotal)}** | |")
    lines.append(f"| **Subtotal** | | | **{fmt(bill.subtotal)}** | |")
    lines.append(f"| VAT | {bill.vat_rate:.0%} on professional fees only | VAT Act (not set by the Order) | {fmt(bill.vat_amount)} | Tax |")
    lines.append(f"| **Total bill** | | | **{fmt(bill.total)}** | |")

    return "\n".join(lines)


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

    print("\nFull itemized bill — Kshs 26,000,000 defended High Court suit:")
    bill = generate_high_court_bill(
        26_000_000, defended=True,
        letters_to_advocate=10, letters_to_client=5, mentions=2, hearings=3,
        disbursements={
            "Service of documents": 8_000,
            "Court filing fees": 25_000,
            "Photocopying/printing": 12_000,
        },
    )
    print(format_bill_as_markdown(bill))
