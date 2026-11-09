"""
citations.py: check the [chunk_id: ...] citations in an answer.

Two separate questions, because they fail for different reasons:

- verifiable: does every cited chunk ID name a chunk that was retrieved for this
  question? (Was it in the model's context?)
- invented: does any cited chunk ID point at a section that doesn't exist in the case
  it names? (e.g. "__s10__c1" for a ruling whose chunks end at s2.)

Models rarely copy IDs character for character, so a reference names a chunk when
either its whole ID matches after normalising case, spacing and punctuation, or its
neutral citation (e.g. "2016KEHC2577") and section match. Real examples an exact
match rejected: the corpus misspells the Order as "Advocates Renumeration Order" and
models write "Remuneration"; case IDs keep a double space where "&" was stripped; and
models shorten or misspell party names ("Gatumjia" for "Gatumu Kamanja").
"""

import json
import re
from pathlib import Path

CITED_CHUNK_ID_PATTERN = re.compile(r"chunk_id:\s*([^\]】]+)")
# The end of a chunk ID: __s<section>__c<chunk>, or a fee-table chunk of the Order.
# (?!\d) stops "__c1" matching inside "__c10".
_ID_SUFFIX = re.compile(r"__(?:s\d+__c\d+|table\d+__schedule_\d+)(?!\d)")
_NEUTRAL_CITATION = re.compile(r"\d{4}ke[a-z]+\d+")  # "2016KEHC2577" after normalising
CHUNKS_PATH = Path(__file__).resolve().parent / "data" / "chunks.jsonl"


def normalise_id(text: str) -> str:
    """Lowercase, fix the corpus's 'Renumeration' spelling, keep only [a-z0-9_]."""
    text = text.lower().replace("renumeration", "remuneration")
    return re.sub(r"[^a-z0-9_]", "", text)


class ChunkIndex:
    """A set of chunk IDs that cited references can be looked up in."""

    def __init__(self, chunk_ids):
        self.full_ids, self.titles = set(), set()
        self.by_citation = set()  # (neutral citation, suffix)
        for chunk_id in chunk_ids:
            norm = normalise_id(chunk_id)
            self.full_ids.add(norm)
            suffix = _ID_SUFFIX.search(norm)
            if not suffix:
                continue
            self.titles.add(norm[:suffix.start()])
            citation = _NEUTRAL_CITATION.search(norm)
            if citation:
                self.by_citation.add((citation.group(0), suffix.group(0)))
        self.citations = {c for c, _ in self.by_citation}

    def names(self, reference: str, suffix: str) -> bool:
        """Does this normalised reference (ending in suffix) name a chunk in the index?"""
        if any(reference.endswith(full) for full in self.full_ids):
            return True
        citation = _NEUTRAL_CITATION.findall(reference)
        return bool(citation) and (citation[-1], suffix) in self.by_citation

    def knows_case(self, reference: str, suffix: str) -> bool:
        """Does the reference name a document in the index, by its exact title or (for
        rulings) its neutral citation?"""
        title = reference[:-len(suffix)]
        if any(title.endswith(t) for t in self.titles):
            return True
        citation = _NEUTRAL_CITATION.findall(title)
        return bool(citation) and citation[-1] in self.citations


def load_corpus(path: Path = CHUNKS_PATH) -> ChunkIndex:
    with open(path, encoding="utf-8") as f:
        return ChunkIndex(json.loads(line)["id"] for line in f)


def _references(citation: str) -> list[tuple[str, str]]:
    """Split one citation string into (normalised reference, suffix) pairs. One string
    can hold several IDs ('[chunk_id: A, chunk_id: B]') or trailing text
    ('[chunk_id: A, Kshs 17,750]'). Suffixes are found before normalising, so a comma
    keeps '__c2, 2026' from becoming '__c22026'."""
    text = citation.lower().replace("renumeration", "remuneration")
    refs, start = [], 0
    for match in _ID_SUFFIX.finditer(text):
        refs.append((normalise_id(text[start:match.end()]), match.group(0)))
        start = match.end()
    return refs


def citations_verifiable(answer_text: str, retrieved_ids: list[str]) -> bool | None:
    """True if every cited chunk ID names a chunk retrieved for this question. None if
    the answer cites no chunk_id at all (e.g. a correct 'Not covered', or a calculator
    answer citing only '[source: ...]')."""
    citations = CITED_CHUNK_ID_PATTERN.findall(answer_text)
    if not citations:
        return None
    retrieved = ChunkIndex(retrieved_ids)
    for citation in citations:
        refs = _references(citation)
        if not refs or not all(retrieved.names(ref, suffix) for ref, suffix in refs):
            return False
    return True


def citations_invented(answer_text: str, corpus: ChunkIndex) -> bool | None:
    """True if any cited chunk ID names a real case but a section that case doesn't
    have. None if the answer cites no chunk_id. References whose case can't be
    identified are not counted (they are still unverifiable)."""
    citations = CITED_CHUNK_ID_PATTERN.findall(answer_text)
    if not citations:
        return None
    return any(
        corpus.knows_case(ref, suffix) and not corpus.names(ref, suffix)
        for citation in citations for ref, suffix in _references(citation)
    )
