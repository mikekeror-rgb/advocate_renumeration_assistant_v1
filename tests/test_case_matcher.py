"""Case-scoped retrieval depends on spotting which ruling a question names."""

import case_matcher

TITLES = [
    "Kariithi v GN Thiongo Associates (Miscellaneous Application E262of2025) 2026KEHC11082(KLR) (22July2026) (Ruling)",
    "Cheruiyot v Ngeno  5 others (Environment and Land Miscellaneous Case E024of2025) 2026KEELC2320(KLR) (27April2026) (Ruling)",
    "Cheruiyot v Ngeno  5 others (Environment and Land Miscellaneous Case E001of2025) 2025KEELC4743(KLR) (26June2025) (Ruling)",
    "Kenya Sugar Board v Otieno (Miscellaneous Case E295of2024) 2025KEELRC731(KLR) (6March2025) (Ruling)",
]
INDEX = case_matcher.build_case_index(TITLES)


def test_punctuation_does_not_matter():
    for question in [
        "What was the outcome of the preliminary objection in Kariithi v G.N. Thiongo Associates?",
        "In Kariithi v G.N. Thiong'o Associates, did the court apply the Civil Procedure Rules?",
    ]:
        assert case_matcher.match_cases(question, INDEX) == [TITLES[0]]


def test_same_named_rulings_are_both_returned():
    matched = case_matcher.match_cases("What did the court remit to the taxing master in Cheruiyot v Ngeno?", INDEX)
    assert sorted(matched) == sorted(TITLES[1:3])


def test_fee_question_names_no_case():
    assert case_matcher.match_cases("Calculate the instruction fee for a defended suit worth 26,000,000", INDEX) == []
