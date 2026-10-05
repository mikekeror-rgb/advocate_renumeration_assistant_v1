"""The eval harness's deterministic checks, which every reported metric relies on."""

CHUNKS = [
    {"chunk_id": "Kariithi v GN Thiongo Associates (Misc E262of2025)__s3__c0",
     "text": "30. The preliminary objection was merited/ allowed. 31. The Notice of Motion is dismissed."},
    {"chunk_id": "Brookshill Limited v County Government of Kwale__s3__c13", "text": "x" * 1200 + " Mukisa"},
]


def test_all_citations_must_be_retrieved(eval_module):
    good = "Allowed [chunk_id: Kariithi v GN Thiongo Associates (Misc E262of2025)__s3__c0, Kariithi]."
    fabricated = good + " Also [chunk_id: Made Up v Nobody__s9__c9]."
    assert eval_module.check_citation_verifiable(good, CHUNKS) is True
    assert eval_module.check_citation_verifiable(fabricated, CHUNKS) is False
    assert eval_module.check_citation_verifiable("Not covered in the provided documents.", CHUNKS) is None


def test_evidence_visible_truncated_or_missing(eval_module):
    assert eval_module.check_evidence_retrieved({"evidence_hint": "preliminary objection was merited"}, CHUNKS) == "visible"
    assert eval_module.check_evidence_retrieved({"evidence_hint": "Mukisa"}, CHUNKS) == "truncated"
    assert eval_module.check_evidence_retrieved({"evidence_hint": "never appears"}, CHUNKS) == "missing"


def test_cites_the_case_asked_about(eval_module):
    qa = {"expected_source": "ruling", "expected_type": "factual", "doc_hint": "Kariithi"}
    right = "Mukisa [chunk_id: Kariithi v GN Thiongo Associates (Misc E262of2025)__s3__c0]"
    wrong = "Mukisa [chunk_id: Brookshill Limited v County Government of Kwale__s3__c13]"
    assert eval_module.check_cites_named_case(qa, right) is True
    assert eval_module.check_cites_named_case(qa, wrong) is False


def test_guard_rows_need_the_guard_message(eval_module):
    qa = {"expected_type": "guard", "expected_value": "couldn't find the amount"}
    assert eval_module.check_guard(qa, "I can calculate this exactly, but I couldn't find the amount.", None) is True
    assert eval_module.check_guard(qa, "The fee is Kshs 590,000.", None) is False
