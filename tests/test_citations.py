"""Citation checks: formatting differences pass, missing or invented chunks don't."""

import citations

ORDER = "Advocates Renumeration Order__s93__c1"  # the corpus's own spelling
CASE = "Daniel Otieno Ogonjo  2 others v Pendulum Entertainment Limited 2021KEELRC1108(KLR)__s2__c3"
RETRIEVED = [ORDER, CASE]
CORPUS = citations.ChunkIndex(RETRIEVED + ["Bemuta Agencies Limited v Jatomy 2017KEHC6574(KLR)__s2__c0"])


def test_formatting_differences_still_verify():
    # Regressions (v3_calc_007, v3_unseen_named_007): real retrieved chunks, rejected by exact match
    assert citations.citations_verifiable("Kshs 17,750 [chunk_id: Advocates Remuneration Order__s93__c1, Kshs 17,750].", RETRIEVED)
    assert citations.citations_verifiable(
        "[chunk_id: Daniel Otieno Ogonjo 2 others v Pendulum Entertainment Limited 2021KEELRC1108(KLR)__s2__c3]", RETRIEVED)


def test_every_cited_id_must_be_retrieved():
    both = "[chunk_id: Advocates Remuneration Order__s93__c1, chunk_id: Advocates Remuneration Order__s94__c0]"
    assert citations.citations_verifiable(both, RETRIEVED) is False
    assert citations.citations_verifiable("Not covered in the provided documents.", RETRIEVED) is None


def test_chunk_number_must_match_exactly():
    assert citations.citations_verifiable("[chunk_id: Advocates Renumeration Order__s93__c10]", RETRIEVED) is False


def test_case_name_without_chunk_id_is_unverifiable_but_not_invented():
    answer = "[chunk_id: Daniel Otieno Ogonjo 2 others v Pendulum Entertainment Limited]"
    assert citations.citations_verifiable(answer, RETRIEVED) is False
    assert citations.citations_invented(answer, CORPUS) is False


def test_nonexistent_section_is_invented():
    # Regression (v3_unseen_named_016): that ruling's chunks end at s2
    invented = "[chunk_id: Bemuta Agencies Limited v Jatomy 2017KEHC6574(KLR)__s10__c1]"
    assert citations.citations_invented(invented, CORPUS) is True
    assert citations.citations_invented("[chunk_id: Advocates Remuneration Order__s93__c1]", CORPUS) is False
    assert citations.citations_invented("No citation here.", CORPUS) is None


def test_shortened_case_title_is_identified_by_neutral_citation():
    # Regression (v3_unseen_named_026): misspelled party name, real chunk
    retrieved = ["In re Estate of Gatumu Kamanja (Deceased) 2016KEHC2577(KLR)__s2__c0"]
    answer = "[chunk_id: In re Estate of Gatumjia (Deceased) 2016KEHC2577(KLR)__s2__c0]"
    assert citations.citations_verifiable(answer, retrieved) is True
    assert citations.citations_invented(answer, citations.ChunkIndex(retrieved)) is False


def test_text_after_an_id_is_not_glued_onto_the_chunk_number():
    # Regression (v3_unseen_named_022): "__c2, 2026..." was read as "__c22026"
    retrieved = ["Mwananchi Credit Limited v Mwangi 2026KEHC1076(KLR)__s2__c2"]
    answer = "[chunk_id: Mwananchi Credit Limited v Mwangi 2026KEHC1076(KLR)__s2__c2, 2026KEHC1076(KLR)]"
    assert citations.citations_verifiable(answer, retrieved) is True


def test_fee_table_chunks_and_invented_statute_sections():
    order = ["Advocates Renumeration Order__table0__SCHEDULE_6", "Advocates Renumeration Order__s93__c1"]
    assert citations.citations_verifiable("[chunk_id: Advocates Remuneration Order__table0__SCHEDULE_6]", order) is True
    assert citations.citations_invented("[chunk_id: Advocates Remuneration Order__s200__c0]", citations.ChunkIndex(order)) is True
