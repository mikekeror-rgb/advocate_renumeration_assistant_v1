"""The fee calculator path: exact figures, amount parsing, and the calculation guard."""

import pytest

import case_matcher
import fee_router


@pytest.mark.parametrize("question, expected_fee", [
    # The calculator questions from eval/qa_pairs.jsonl
    ("What is the discharge fee for a charge over a Kshs 2,800,000 loan, where an undertaking "
     "is required for redemption?", 15000),
    ("What is the discharge fee for a charge over a Kshs 2,800,000 loan, with no undertaking "
     "required for redemption?", 10000),
    ("What is the cost of registering (creating) a mortgage over a loan of Kshs 15,000,000 for "
     "the grantee's advocate?", 193750),
    ("What is the fee for the grantor's advocate when creating a mortgage over a Kshs 15,000,000 loan?", 96875),
    ("What is the High Court instruction fee for a defended suit worth Kshs 3,500,000?", 170000),
    ("What is the subordinate court instruction fee for a Kshs 350,000 claim on the higher scale?", 65000),
    ("What is the instruction fee for a land sale worth Kshs 12,000,000?", 205000),
    # Regression: a bare number without commas used to skip the calculator entirely
    ("Calculate instruction fee for a contentious matter in the High Court with defense where "
     "the subject matter is 26000000", 590000),
    ("A defended High Court suit filed in 2024 worth 26000000", 590000),
])
def test_calculator_gives_exact_figure(question, expected_fee):
    result = fee_router.route(question)
    assert result is not None, "the question should reach the calculator"
    assert result.fee == pytest.approx(expected_fee)


def test_ambiguous_scale_gives_both_figures():
    result = fee_router.route("Subordinate court fee for a 350,000 claim?")
    assert result.fee is None
    assert result.fee_options == {"lower": 45000, "higher": 65000}


@pytest.mark.parametrize("text, amount", [
    ("the subject matter is 26000000", 26_000_000),
    ("worth Kshs 26,000,000", 26_000_000),
    ("a loan of 2.8m", 2_800_000),
    ("a claim of 26 million", 26_000_000),
    ("Kshs 900000", 900_000),
])
def test_amount_formats(text, amount):
    assert fee_router.extract_amount(text) == pytest.approx(amount)


@pytest.mark.parametrize("text", ["Schedule 6 of the Order", "a suit filed in 2024", "paragraph 11"])
def test_small_numbers_are_not_amounts(text):
    assert fee_router.extract_amount(text) is None


def test_guard_asks_for_missing_amount():
    message = fee_router.calculation_guard("Calculate the instruction fee for a defended High Court suit")
    assert message is not None and "couldn't find the amount" in message


def test_guard_refuses_unsupported_fee_type():
    message = fee_router.calculation_guard("Calculate the fee for drafting a will for a 5,000,000 estate")
    assert message is not None and "won't calculate" in message


@pytest.mark.parametrize("text", [
    # Regression (v3_seen_named_002): neutral citations were read as Kshs 13,078
    "In Mwangi v Njenga 2026KEHC13078(KLR) (Ruling), what was the original bill of costs?",
    "Mwendwa Mwinzi Associates Advocate v Mutie 2023KEELC16592(KLR)",
])
def test_case_citation_numbers_are_not_amounts(text):
    assert fee_router.extract_amount(text) is None
    assert fee_router.route(text) is None


def test_currency_prefix_without_space_still_parses():
    assert fee_router.extract_amount("a claim of Kshs26,000,000") == pytest.approx(26_000_000)


@pytest.mark.parametrize("question", [
    # Regressions from the v3 eval: questions about the rules, or about a ruling's
    # outcome, that the guard answered with a canned calculator message
    "Under the Advocates (Remuneration) Order, how is a fraction of KSh. 2,000 that is over "
    "KSh. 1,000 treated when calculating fees under scales 1, 2 or 3?",
    "When determining the value of the subject matter for the purpose of calculating legal fees, "
    "what is the hierarchy of precedence if a matter has been concluded by way of a settlement?",
    "In Cheruiyot v Ngeno 2026KEELC2320(KLR), how much did the court award in total instruction fees?",
])
def test_questions_about_calculation_rules_go_to_retrieval(question):
    assert fee_router.route(question) is None
    assert fee_router.calculation_guard(question) is None


@pytest.mark.parametrize("question", [
    "How much is the instruction fee for a defended High Court suit?",
    "Can you work out the instruction fee for a defended High Court suit?",
])
def test_guard_still_catches_calculation_requests(question):
    message = fee_router.calculation_guard(question)
    assert message is not None and "couldn't find the amount" in message


@pytest.mark.parametrize("question", [
    "How much does it cost to register a company in Uganda?",
    "How much is a single business permit in Mombasa?",
    "What is the capital of Kenya?",
])
def test_out_of_domain_questions_go_to_retrieval(question):
    assert fee_router.route(question) is None
    assert fee_router.calculation_guard(question) is None


_CASE_INDEX = case_matcher.build_case_index([
    "Mwananchi Credit Limited v Mwangi (Civil Appeal E064of2021) 2026KEHC1076(KLR) (5February2026) (Ruling)",
    "Brook Villas Estate Limited v Gatu another (Environment Land Case 25of2016) 2024KEELC5627(KLR) (25July2024) (Ruling)",
])


@pytest.mark.parametrize("question", [
    # Regressions (v3_unseen_named_022, v3_seen_named_011): questions about a named
    # ruling that quote its figures were sent to the calculator
    "In Mwananchi Credit Limited v Mwangi, what was the stated reason for the instruction "
    "fee of Kshs 120,000/= being considered manifestly excessive?",
    "In Brook Villas Estate Limited v Gatu another (Environment Land Case 25of2016), what value "
    "did the Applicant argue the subject matter of the reference was, as opposed to the whole "
    "value of Kshs.84,000,000?",
])
def test_questions_about_a_named_ruling_go_to_retrieval(question):
    assert fee_router.route(question) is not None, "route() alone still fires; route_question must not"
    assert fee_router.route_question(question, _CASE_INDEX) is None


def test_named_ruling_with_an_explicit_calculation_request_still_routes():
    question = ("In Mwananchi Credit Limited v Mwangi, calculate the High Court instruction fee "
                "for a defended suit worth Kshs 907,814")
    result = fee_router.route_question(question, _CASE_INDEX)
    assert result is not None and result.scenario == "schedule6"


def test_calculation_without_a_named_ruling_is_unchanged():
    question = "What is the High Court instruction fee for a defended suit worth Kshs 3,500,000?"
    assert fee_router.route_question(question, _CASE_INDEX).fee == pytest.approx(170000)
