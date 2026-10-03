"""
case_matcher.py: detect which ruling(s) a question names, so retrieval can search
inside that case.

Why this exists: chunk text rarely contains the case name (it lives only in the
metadata), so a question like "What was the outcome of the preliminary objection in
Kariithi v G.N. Thiong'o Associates?" cannot pull that ruling's decisive chunk by
similarity alone. Eval diagnostics showed the answer-bearing chunk absent even from
the top 32 retrieved results (ruling_02, ruling_05).

Matching rule: a case is named when the first significant word of EACH party
(e.g. "kariithi" and "thiongo") appears in the question. Punctuation is ignored,
so "G.N. Thiong'o" matches the title "GN Thiongo".
"""

import re
from typing import Iterable

# Words that don't identify a party on their own.
_STOPWORDS = {
    "the", "and", "another", "others", "other", "ltd", "limited", "co", "company",
    "inc", "plc", "llp", "advocates", "advocate", "associates", "of", "in", "re",
    "ex", "parte", "exparte", "republic", "attorney", "general", "county",
    "government", "kenya", "trustees", "registered", "estate", "late", "deceased",
}
_PARTY_SPLIT = re.compile(r"\s+(?:v|vs|versus)\.?\s+", re.IGNORECASE)


def _normalise(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", "", text)  # "G.N." -> "gn", "Thiong'o" -> "thiongo"
    return re.sub(r"\s+", " ", text).strip()


def case_key_tokens(doc_title: str) -> list[str]:
    """'Kariithi v GN Thiongo Associates (Misc...)' -> ['kariithi', 'thiongo']"""
    short_name = doc_title.split("(")[0]
    keys = []
    for party in _PARTY_SPLIT.split(short_name, maxsplit=1):
        for token in _normalise(party).split():
            if len(token) > 2 and token not in _STOPWORDS and not token.isdigit():
                keys.append(token)
                break
    return keys


def build_case_index(doc_titles: Iterable[str]) -> dict[str, list[str]]:
    """Map each distinct ruling title to its party key tokens."""
    index = {}
    for title in set(doc_titles):
        keys = case_key_tokens(title)
        if keys:
            index[title] = keys
    return index


def match_cases(query: str, case_index: dict[str, list[str]]) -> list[str]:
    """Titles of every ruling the query names. Two different rulings can share a
    name (e.g. two 'Cheruiyot v Ngeno' cases), in which case both are returned."""
    words = set(_normalise(query).split())
    matched = []
    for title, keys in case_index.items():
        if len(keys) == 1 and len(keys[0]) < 5:
            continue  # a single short party word is too likely to match by accident
        if all(key in words for key in keys):
            matched.append(title)
    return sorted(matched)


if __name__ == "__main__":
    titles = [
        "Kariithi v GN Thiongo Associates (Miscellaneous Application E262of2025) 2026KEHC11082(KLR) (22July2026) (Ruling)",
        "Cheruiyot v Ngeno  5 others (Environment and Land Miscellaneous Case E024of2025) 2026KEELC2320(KLR) (27April2026) (Ruling)",
        "Cheruiyot v Ngeno  5 others (Environment and Land Miscellaneous Case E001of2025) 2025KEELC4743(KLR) (26June2025) (Ruling)",
        "Kenya Sugar Board v Otieno (Miscellaneous Case E295of2024) 2025KEELRC731(KLR) (6March2025) (Ruling)",
    ]
    index = build_case_index(titles)
    for q in [
        "What was the outcome of the preliminary objection in Kariithi v G.N. Thiongo Associates?",
        "In Kariithi v G.N. Thiong'o Associates, did the court hold that the Civil Procedure Rules apply?",
        "What did the court remit to the taxing master in Cheruiyot v Ngeno, and why?",
        "Calculate instruction fee for a defended High Court suit worth 26,000,000",
    ]:
        print(f"{q[:70]}\n  -> {match_cases(q, index)}\n")
