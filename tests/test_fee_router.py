"""The fee calculator path: exact figures, amount parsing, and the calculation guard."""

import pytest

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


@pytest.mark.parametrize("question", [
    "How much does it cost to register a company in Uganda?",
    "How much is a single business permit in Mombasa?",
    "What is the capital of Kenya?",
])
def test_out_of_domain_questions_go_to_retrieval(question):
    assert fee_router.route(question) is None
    assert fee_router.calculation_guard(question) is None
