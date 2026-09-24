"""
fee_router.py — Classifies a natural-language question, extracts the amount and any
modifiers (defended/undefended, grantee/grantor, undertaking), and dispatches to the
right function in fee_calculator.py.

This sits in front of rag_pipeline.py: if the router confidently classifies a query
as a computable fee question, the pipeline should use its result (an exact, verified
number) instead of asking the LLM to retrieve-and-compute or state a schedule figure.
"""

import re
from dataclasses import dataclass, field

import fee_calculator as fc


# ---------------------------------------------------------------------------
# Amount extraction
# ---------------------------------------------------------------------------

_AMOUNT_PATTERN = re.compile(
    r"(?:kshs?\.?|ksh\.?|shs\.?|shillings?)?\s*"
    r"([\d]{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(million|mn|m\b|thousand|k\b)?",
    re.IGNORECASE,
)

_MULTIPLIERS = {"million": 1_000_000, "mn": 1_000_000, "m": 1_000_000, "thousand": 1_000, "k": 1_000}


def extract_amount(text: str) -> float | None:
    """Pull the monetary amount out of a query. Requires either a comma-grouped
    number, an explicit m/million/k/thousand suffix, or a currency keyword nearby —
    otherwise small incidental numbers (e.g. 'Schedule 1') would false-match."""
    candidates = []
    for match in _AMOUNT_PATTERN.finditer(text):
        raw_number, suffix = match.groups()
        has_comma = "," in raw_number
        has_suffix = bool(suffix)
        preceding = text[max(0, match.start() - 12):match.start()].lower()
        has_currency_word = any(w in preceding for w in ("kshs", "ksh", "shs", "shilling"))

        if not (has_comma or has_suffix or has_currency_word):
            continue

        value = float(raw_number.replace(",", ""))
        if suffix:
            value *= _MULTIPLIERS[suffix.lower()]
        candidates.append(value)

    if not candidates:
        return None
    return max(candidates)  # the loan/consideration amount is usually the dominant figure


# ---------------------------------------------------------------------------
# Scenario classification
# ---------------------------------------------------------------------------

_DISCHARGE_WORDS = re.compile(r"\bdischarg|reconvey|reassign|redemption\b", re.IGNORECASE)
_SECURITY_WORDS = re.compile(r"\bmortgage|charge|security|debenture\b", re.IGNORECASE)
_SALE_WORDS = re.compile(r"\bsale|purchase|sell|buy|conveyanc|land|property\b", re.IGNORECASE)
_SUBORDINATE_WORDS = re.compile(r"\bsubordinate|magistrate\b", re.IGNORECASE)
_HIGHCOURT_INSTRUCTION_WORDS = re.compile(
    r"\binstruction fee|\bsue\b|\bsuit\b|\bplaint\b|\bdefend|\bhigh court\b", re.IGNORECASE
)


def classify_scenario(text: str) -> str | None:
    if _DISCHARGE_WORDS.search(text) and _SECURITY_WORDS.search(text):
        return "schedule1_second_scale_discharge"
    if _SUBORDINATE_WORDS.search(text):
        return "schedule7"
    if _SECURITY_WORDS.search(text):
        return "schedule1_second_scale_creation"
    if _SALE_WORDS.search(text):
        # checked before the High Court trigger: "instruction fee" alone isn't
        # exclusive to litigation — it's also Schedule 1's own term for a
        # conveyancing fee, so a bare "instruction fee for a land sale" was
        # previously misrouted to Schedule 6 before ever reaching this check.
        return "schedule1_first_scale"
    if _HIGHCOURT_INSTRUCTION_WORDS.search(text):
        return "schedule6"
    return None


# ---------------------------------------------------------------------------
# Modifier extraction
# ---------------------------------------------------------------------------

def _extract_party(text: str) -> str:
    if re.search(r"\bgrantor|borrower|mortgagor|chargor\b", text, re.IGNORECASE):
        return "grantor"
    return "grantee"  # statute's default framing (lender/chargee)


def _extract_undertaking(text: str) -> bool:
    if re.search(r"\bno undertaking|without (?:an )?undertaking\b", text, re.IGNORECASE):
        return False
    return True  # more common case; flagged as an assumption in the explanation


def _extract_defended(text: str) -> bool | None:
    if re.search(r"\bundefended|unopposed|no denial|not defended\b", text, re.IGNORECASE):
        return False
    if re.search(r"\bdefended|denial of liability|contested|opposed\b", text, re.IGNORECASE):
        return True
    return None  # ambiguous — caller should surface both figures


def _extract_scale(text: str) -> str | None:
    if re.search(r"\blower scale|undefended|no denial\b", text, re.IGNORECASE):
        return "lower"
    if re.search(r"\bhigher scale|defended|contested\b", text, re.IGNORECASE):
        return "higher"
    return None  # ambiguous


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    scenario: str
    amount: float
    modifiers: dict = field(default_factory=dict)
    fee: float | None = None          # single figure, when unambiguous
    fee_options: dict | None = None   # multiple figures, when a modifier was ambiguous
    explanation: str = ""
    schedule_citation: str = ""


def route(text: str) -> RouteResult | None:
    """Returns a RouteResult if the query is a computable fee question with a
    usable amount, else None (caller should fall back to normal RAG retrieval)."""
    scenario = classify_scenario(text)
    if scenario is None:
        return None

    amount = extract_amount(text)
    if amount is None:
        return None  # classified but no usable figure — let the LLM ask a clarifying question

    if scenario == "schedule1_second_scale_discharge":
        party = _extract_party(text)
        undertaking = _extract_undertaking(text)
        fee = fc.schedule1_second_scale_discharge_fee(amount, party=party, undertaking_required=undertaking)
        assumption = "" if re.search(r"undertaking", text, re.IGNORECASE) else \
            " (assumed: an undertaking is required for redemption — the more common case; say 'no undertaking' if that's not so)"
        return RouteResult(
            scenario=scenario, amount=amount,
            modifiers={"party": party, "undertaking_required": undertaking},
            fee=fee,
            explanation=(
                f"Discharge of security fee for {party}'s advocate on a Kshs {amount:,.0f} "
                f"security: Kshs {fee:,.0f}{assumption}."
            ),
            schedule_citation="Advocates Remuneration Order, Schedule 1, Second Scale, para 2(c)/(e)",
        )

    if scenario == "schedule1_second_scale_creation":
        party = _extract_party(text)
        fee = fc.schedule1_second_scale_creation_fee(amount, party=party)
        return RouteResult(
            scenario=scenario, amount=amount, modifiers={"party": party}, fee=fee,
            explanation=f"Creation-of-security fee for {party}'s advocate on a Kshs {amount:,.0f} security: Kshs {fee:,.0f}.",
            schedule_citation="Advocates Remuneration Order, Schedule 1, Second Scale, para 2(b)/(d)",
        )

    if scenario == "schedule1_first_scale":
        fee = fc.schedule1_first_scale_fee(amount)
        return RouteResult(
            scenario=scenario, amount=amount, fee=fee,
            explanation=f"Sale/purchase instruction fee on a Kshs {amount:,.0f} transaction: Kshs {fee:,.0f}.",
            schedule_citation="Advocates Remuneration Order, Schedule 1, First Scale, para 1",
        )

    if scenario == "schedule6":
        defended = _extract_defended(text)
        if defended is None:
            options = {
                "undefended": fc.schedule6_instruction_fee(amount, defended=False),
                "defended": fc.schedule6_instruction_fee(amount, defended=True),
            }
            return RouteResult(
                scenario=scenario, amount=amount, fee_options=options,
                explanation=(
                    f"High Court instruction fee on Kshs {amount:,.0f} depends on whether the suit is "
                    f"defended: undefended = Kshs {options['undefended']:,.0f}, "
                    f"defended = Kshs {options['defended']:,.0f}."
                ),
                schedule_citation="Advocates Remuneration Order, Schedule 6, Part A, item 1(a)/(b)",
            )
        fee = fc.schedule6_instruction_fee(amount, defended=defended)
        return RouteResult(
            scenario=scenario, amount=amount, modifiers={"defended": defended}, fee=fee,
            explanation=f"High Court instruction fee ({'defended' if defended else 'undefended'}) on Kshs {amount:,.0f}: Kshs {fee:,.0f}.",
            schedule_citation="Advocates Remuneration Order, Schedule 6, Part A, item 1(a)/(b)",
        )

    if scenario == "schedule7":
        scale = _extract_scale(text)
        if scale is None:
            options = {
                "lower": fc.schedule7_instruction_fee(amount, scale="lower"),
                "higher": fc.schedule7_instruction_fee(amount, scale="higher"),
            }
            return RouteResult(
                scenario=scenario, amount=amount, fee_options=options,
                explanation=(
                    f"Subordinate court instruction fee on Kshs {amount:,.0f}: "
                    f"lower scale = Kshs {options['lower']:,.0f}, higher scale = Kshs {options['higher']:,.0f} "
                    f"(higher scale applies unless the matter is undefended/ex parte)."
                ),
                schedule_citation="Advocates Remuneration Order, Schedule 7, Part A, item 1",
            )
        fee = fc.schedule7_instruction_fee(amount, scale=scale)
        return RouteResult(
            scenario=scenario, amount=amount, modifiers={"scale": scale}, fee=fee,
            explanation=f"Subordinate court instruction fee ({scale} scale) on Kshs {amount:,.0f}: Kshs {fee:,.0f}.",
            schedule_citation="Advocates Remuneration Order, Schedule 7, Part A, item 1",
        )

    return None


if __name__ == "__main__":
    test_queries = [
        "What are the legal fees for discharge of charge over a loan of 2.8m?",
        "What's the discharge fee for the grantor, no undertaking, on a 900,000 mortgage?",
        "Instruction fee to sue in a defended suit worth Kshs 3,500,000?",
        "Subordinate court fee for a 350,000 claim?",
        "What's the conveyancing fee on a Kshs 12,000,000 land sale?",
        "What is the capital of Kenya?",  # should return None
    ]
    for q in test_queries:
        result = route(q)
        print(f"\nQ: {q}")
        if result is None:
            print("  -> No route (falls back to normal RAG retrieval)")
        else:
            print(f"  -> {result.explanation}")
            print(f"     [{result.schedule_citation}]")
