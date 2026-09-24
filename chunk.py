"""
Step 2: Clean + chunk Kenyan advocate-remuneration case-law PDFs into data/chunks.jsonl

Output schema per line:
{
  "id": "<doc_title>__<section_idx>__<chunk_idx>",
  "source_url": "<file path>",
  "doc_title": "<filename stem>",
  "case_number": "<e.g. MISC. APPLICATION NO. E262 OF 2025>",
  "court": "<e.g. HIGH COURT OF KENYA AT NAIROBI, FAMILY DIVISION>",
  "case_citation": "<e.g. [2026] KEHC 11082 (KLR)>",
  "ruling_date": "<e.g. 22nd July, 2026>",
  "section": "<detected heading text, or 'body' if none found>",
  "text": "<chunk text>"
}
"""

import json
import re
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("./data/raw")          # where your Kenya Law ruling PDFs live
OUTPUT_PATH = Path("./data/chunks.jsonl")

CHUNK_SIZE_CHARS = 1800   # ~350-450 tokens, matches the 300-500 token target
CHUNK_OVERLAP_CHARS = 200  # ~50 tokens

# Legal-ruling headings look different from policy-doc headings:
# - ALL CAPS court/party blocks: "REPUBLIC OF KENYA", "RULING"
# - Short title-case section labels on their own line: "Analysis and determination.",
#   "Disposition.", "Applicant's submission", "Respondent's Submissions"
# Numbered paragraphs ("1. Before the Court is...") are body text, NOT headings —
# excluded by the length/word-count cap below so they don't get misclassified.
HEADING_PATTERN = re.compile(
    r"^(?:"
    r"[A-Z][A-Z0-9 ,\-:'.]{3,80}"                              # ALL CAPS lines (incl. SCHEDULE n)
    r"|Appendix\s+[A-Z0-9]+[^\n]{0,60}"                        # Appendix headers
    r"|(?:Analysis and [Dd]etermination|Disposition|Orders? accordingly"
    r"|Applicant'?s?\s+[Ss]ubmissions?|Respondent'?s?\s+[Ss]ubmissions?"
    r"|Analysis\s*(?:and\s*)?[Dd]etermination|Background|Introduction|Ruling)\.?"
    r"|Part\s+[IVXLCD]+\b[^\n]{0,80}"                          # "Part I – GENERAL MATTERS"
    r"|\d{1,2}[A-Z]?\.\s+[A-Z][^\n]{2,80}"                     # "1. Citation", "13A. Taxation..."
    r")$"
)

CASE_NUMBER_PATTERN = re.compile(
    r"(MISC\.?\s*APPLICATION\s*NO\.?\s*[A-Z0-9]+\s*OF\s*\d{4}"
    r"|ELC\s*MISC\.?\s*[A-Z0-9]+\s*OF\s*\d{4}"
    r"|[A-Z]{2,6}\s*MISC(?:ELLANEOUS)?\s*(?:APPLICATION|CASE)?\s*NO\.?\s*[A-Z0-9]+\s*OF\s*\d{4})",
    re.IGNORECASE,
)

COURT_PATTERN = re.compile(r"IN THE [A-Z .,'&]+ COURT[A-Z .,'&]*")

RULING_DATE_PATTERN = re.compile(
    r"Dated(?:,\s*Signed and delivered[^.\n]*)?\s*(?:at\s+[A-Za-z]+\s*)?this\s+"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,?\s*\d{4})",
    re.IGNORECASE,
)

# Filenames from kenyalaw.org embed a neutral citation like "2026KEHC11082_KLR"
CITATION_IN_FILENAME_PATTERN = re.compile(r"(20\d{2})KE([A-Z]+)(\d+)_KLR", re.IGNORECASE)

# Matches the ARO itself (and any other Legal Notice / subsidiary legislation you add)
STATUTE_CITATION_PATTERN = re.compile(
    r"LEGAL\s+NOTICE\s*(?:NO\.?\s*)?(\d+)\s+OF\s+(\d{4})", re.IGNORECASE
)


CELL_NUMERIC_RE = re.compile(r"^[—\-]?\s*(Kshs?\.?\s*)?[\d,]+(\.\d+)?\s*%?\s*$", re.IGNORECASE)


def is_real_fee_table(table: list) -> bool:
    """Filter pdfplumber's extract_tables() output: it also returns false positives
    where wrapped prose paragraphs get mistaken for a 2-column 'table' (line-wrap
    artifacts). A real fee table has short, mostly-numeric cells and/or a
    'Kshs...Exceed' header — verified against the actual ARO PDF's Schedules 6/7/9/10/11."""
    if len(table) < 2:
        return False
    rows = [r for r in table if r]
    multi_col_rows = [r for r in rows if len(r) >= 2]
    if len(multi_col_rows) < len(rows) * 0.6:
        return False

    header_text = " ".join(str(c) for c in table[0] if c).lower()
    header_hit = "exceed" in header_text  # "Exceeds Kshs." / "But does not exceed Kshs."

    body_cells = [str(c).strip() for row in table[1:] for c in row if c and str(c).strip()]
    if not body_cells:
        return False
    numeric_cells = sum(1 for c in body_cells if CELL_NUMERIC_RE.match(c))
    numeric_ratio = numeric_cells / len(body_cells)

    avg_len = sum(len(c) for c in body_cells) / len(body_cells)
    if avg_len > 25:  # prose cells run long; fee-table cells are short numbers/labels
        return False

    return header_hit or numeric_ratio > 0.5


def table_to_markdown(table: list) -> str:
    rows = [[str(c).replace("\n", " ").strip() if c else "" for c in row] for row in table]
    header, *body = rows
    md = "| " + " | ".join(header) + " |\n"
    md += "|" + "|".join(["---"] * len(header)) + "|\n"
    for row in body:
        row = row + [""] * (len(header) - len(row))  # pad ragged rows
        md += "| " + " | ".join(row[:len(header)]) + " |\n"
    return md


def extract_fee_tables(pdf_path: Path) -> list[dict]:
    """Pull clean fee-schedule tables out of the statute PDF as their own chunks,
    since pdfplumber's extract_text() mashes multi-column fee tables into unreadable
    runs of numbers. Each returned dict is ready to become a standalone chunk."""
    tables_out = []
    current_schedule = "Unknown Schedule"
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            schedule_match = re.search(r"SCHEDULE\s+\d+[A-Z]?", page_text)
            if schedule_match:
                current_schedule = schedule_match.group(0)

            for table_idx, table in enumerate(page.extract_tables()):
                if is_real_fee_table(table):
                    tables_out.append({
                        "page": page_num,
                        "schedule": current_schedule,
                        "markdown": table_to_markdown(table),
                    })
    return tables_out


def clean_pdf_text(text: str) -> str:
    """Strip common PDF extraction artifacts."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^\s*(Page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", "", text)
    text = re.sub(r"(?m)^\s*P a g e \d+\s*\|\s*\d+\s*$", "", text)  # "P a g e 1 | 8" style footers
    text = re.sub(r"(?m)^\s*©.*$", "", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from all pages of a PDF, page breaks preserved as newlines."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    return "\n".join(pages)


def extract_case_metadata(text: str, filename: str) -> dict:
    """Pull citation/court/date metadata for legal citation in downstream RAG answers.
    Distinguishes the ARO statute itself (doc_type='statute') from court rulings
    (doc_type='ruling') since they need different fields and different citation formats.
    Falls back to 'Unknown' where a pattern doesn't match — spot-check chunks.jsonl and
    tighten these regexes against your actual corpus."""
    case_number_match = CASE_NUMBER_PATTERN.search(text)
    statute_match = STATUTE_CITATION_PATTERN.search(text)

    if statute_match and not case_number_match:
        number, year = statute_match.groups()
        return {
            "doc_type": "statute",
            "case_number": "N/A (primary/subsidiary legislation)",
            "court": "N/A (not a court ruling)",
            "ruling_date": "Unknown",
            "case_citation": f"Legal Notice {number} of {year} (Advocates (Remuneration) Order)",
        }

    court_match = COURT_PATTERN.search(text)
    date_match = RULING_DATE_PATTERN.search(text)
    citation_match = CITATION_IN_FILENAME_PATTERN.search(filename)

    case_citation = "Unknown"
    if citation_match:
        year, court_code, number = citation_match.groups()
        case_citation = f"[{year}] KE{court_code.upper()} {number} (KLR)"

    return {
        "doc_type": "ruling",
        "case_number": case_number_match.group(0).strip() if case_number_match else "Unknown",
        "court": court_match.group(0).strip() if court_match else "Unknown",
        "ruling_date": date_match.group(1).strip() if date_match else "Unknown",
        "case_citation": case_citation,
    }


RULING_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"[A-Z][A-Z0-9 ,\-:'.]{3,80}"                              # ALL CAPS lines
    r"|Appendix\s+[A-Z0-9]+[^\n]{0,60}"                        # Appendix headers
    r"|(?:Analysis and [Dd]etermination|Disposition|Orders? accordingly"
    r"|Applicant'?s?\s+[Ss]ubmissions?|Respondent'?s?\s+[Ss]ubmissions?"
    r"|Analysis\s*(?:and\s*)?[Dd]etermination|Background|Introduction|Ruling)\.?"
    r")$"
    # Deliberately NO numbered-short-line clause here (unlike HEADING_PATTERN above).
    # Rulings don't have "rule titles" — every numbered line is a substantive judgment
    # paragraph, however short. Applying the statute's numbered-heading pattern to
    # rulings was siphoning short disposition paragraphs (e.g. "31. The Notice of
    # Motion is therefore incompetent and is subsequently dismissed.") into the
    # 'section' label instead of the searchable chunk text — leaving only a
    # leftover fragment like "dismissed." as the actual retrievable content.
    # Confirmed against the real corpus: this was why ruling_02's eval question
    # about the case's disposition could never be retrieved.
)


def split_into_sections(text: str, is_statute: bool = False) -> list[tuple[str, str]]:
    """
    Split cleaned text into (heading, body) pairs using heading heuristics.
    Text before the first detected heading is returned under heading 'body'.
    is_statute selects the heading pattern: only the statute has genuine short
    numbered rule titles ("1. Citation") that should be pulled out as headings;
    rulings' numbered paragraphs are always body content, never headings.
    """
    pattern = HEADING_PATTERN if is_statute else RULING_HEADING_PATTERN
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = "body"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and pattern.match(stripped) and len(stripped) < 90:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [(heading, "\n".join(body_lines).strip()) for heading, body_lines in sections if "\n".join(body_lines).strip()]


def chunk_documents(data_dir: Path) -> list[dict]:
    pdf_paths = sorted(data_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDF files under {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    skipped_files = 0

    for pdf_path in pdf_paths:
        try:
            raw_text = extract_pdf_text(pdf_path)
        except Exception as e:
            print(f"  ! Failed to read {pdf_path.name}: {e}")
            skipped_files += 1
            continue

        cleaned_text = clean_pdf_text(raw_text)
        if len(cleaned_text) < 50:
            print(f"  ! Skipping near-empty (likely scanned/no OCR): {pdf_path.name}")
            skipped_files += 1
            continue

        doc_title = pdf_path.stem
        case_metadata = extract_case_metadata(cleaned_text, pdf_path.name)
        sections = split_into_sections(cleaned_text, is_statute=(case_metadata.get("doc_type") == "statute"))

        MIN_CHUNK_CHARS = 30  # below this, chunks are reliably page-break/header debris
                               # (e.g. "(BEFORE D. K. N. MARETE)", "AND", "18.08.16") rather
                               # than real content — confirmed by spot-checking a full corpus run

        for section_idx, (heading, section_text) in enumerate(sections):
            sub_chunks = splitter.split_text(section_text)
            for chunk_idx, chunk_text in enumerate(sub_chunks):
                if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
                    continue
                all_chunks.append({
                    "id": f"{doc_title}__s{section_idx}__c{chunk_idx}",
                    "source_url": str(pdf_path),
                    "doc_title": doc_title,
                    "section": heading,
                    "text": chunk_text,
                    **case_metadata,
                })

        # For the statute, also pull clean fee tables directly — extract_text() mashes
        # multi-column Schedule tables into unreadable digit runs; extract_tables() doesn't.
        if case_metadata.get("doc_type") == "statute":
            fee_tables = extract_fee_tables(pdf_path)
            print(f"  Extracted {len(fee_tables)} clean fee table(s) from {pdf_path.name}")
            for tbl_idx, tbl in enumerate(fee_tables):
                all_chunks.append({
                    "id": f"{doc_title}__table{tbl_idx}__{tbl['schedule'].replace(' ', '_')}",
                    "source_url": str(pdf_path),
                    "doc_title": doc_title,
                    "section": f"{tbl['schedule']} (fee table, PDF page {tbl['page'] + 1})",
                    "text": tbl["markdown"],
                    **case_metadata,
                })

    print(f"Chunked {len(pdf_paths) - skipped_files} PDFs into {len(all_chunks)} chunks "
          f"({skipped_files} files skipped)")
    return all_chunks


def write_jsonl(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    chunks = chunk_documents(DATA_DIR)
    write_jsonl(chunks, OUTPUT_PATH)
