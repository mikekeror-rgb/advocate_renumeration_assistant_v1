# Advocate Remuneration Assistant: Technical Documentation

**Author:** Mike Keror · **Repository:** [advocate_renumeration_assistant_v1](https://github.com/mikekeror-rgb/advocate_renumeration_assistant_v1)

## Contents

1. [Overview](#1-overview)
2. [Data](#2-data)
3. [Ingestion pipeline](#3-ingestion-pipeline)
4. [Retrieval](#4-retrieval)
5. [Fee calculator and router](#5-fee-calculator-and-router)
6. [Generation](#6-generation)
7. [Evaluation harness](#7-evaluation-harness)
8. [What the evaluation caught](#8-what-the-evaluation-caught)
9. [The application](#9-the-application)
10. [Configuration and deployment](#10-configuration-and-deployment)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. Overview

Kenyan advocates' fees are governed by the **Advocates (Remuneration) Order**
(Legal Notice 64 of 1962, as revised to 2022) and by a large body of court
rulings on the *taxation* of bills of costs. Answering a fee question well
needs two different kinds of work:

- **Legal lookup:** finding what the Order or a ruling actually says, and
  citing it.
- **Arithmetic:** many fees exist only after running the Order's formulas.
  Schedule 1, for example, applies cumulative percentage bands, so no sentence
  in the Order states "the fee on a Kshs 15,000,000 charge is Kshs 193,750".

Language models are unreliable at the second kind of work. This project
separates the two: a **retrieval-augmented generation (RAG) pipeline** handles
lookup, and a **deterministic calculator** handles arithmetic, with a router
deciding which applies.

> This is a portfolio project, not legal advice. Figures follow the Order as
> revised to 2022; taxing officers retain discretion (paragraph 16), and rates
> may have changed since.

---

## 2. Data

| Source | Documents | Role |
|---|---|---|
| Advocates (Remuneration) Order, via Kenya Law | 1 | The authoritative rules and fee schedules |
| Kenya Law rulings on taxation of costs | ~209 | How courts interpret and apply the Order |

After chunking, the corpus holds roughly 3,900 chunks: about 3,700 from rulings
and about 215 from the Order. This imbalance turned out to matter a great deal
for retrieval (see [section 4](#4-retrieval)).

Ruling file names from Kenya Law embed a neutral citation (e.g.
`2026KEHC11082_KLR`), which the ingestion step converts into a readable
citation, `[2026] KEHC 11082 (KLR)`.

---

## 3. Ingestion pipeline

**`chunk.py`** turns PDFs into `data/chunks.jsonl`:

1. **Text extraction and cleaning** with `pdfplumber`, removing page numbers,
   running footers (e.g. `P a g e 1 | 8`) and copyright lines.
2. **Document typing.** A document is tagged `statute` if it carries a Legal
   Notice citation and no case number, otherwise `ruling`. Rulings also get
   their case number, court, ruling date and neutral citation.
3. **Section-aware splitting.** Text is split at headings before chunking, so
   chunks don't straddle sections. Statutes and rulings use **different heading
   patterns**: the Order has short numbered rule titles ("1. Citation",
   "13A. Powers of taxing officer"), but in rulings every numbered line is a
   substantive paragraph. Using one pattern for both caused a real bug (see
   [section 8](#8-what-the-evaluation-caught)).
4. **Chunking** with `RecursiveCharacterTextSplitter` at 1,800 characters with
   200 characters of overlap (roughly 350–450 tokens per chunk). Fragments under
   30 characters (stray headers, dates, "AND") are discarded.
5. **Fee tables.** `pdfplumber.extract_text()` flattens the Order's multi-column
   fee tables into unreadable runs of numbers, so tables are also extracted with
   `extract_tables()`, filtered to reject wrapped prose that merely looks
   tabular, and stored as Markdown-table chunks. Nine clean tables were
   recovered, from Schedules 6, 7, 9, 10 and 11.

**`embed_index.py`** embeds every chunk with `BAAI/bge-small-en-v1.5`
(normalised vectors) and stores them, with their metadata, in a persistent
ChromaDB collection.

---

## 4. Retrieval

Retrieval is **hybrid**: three ranked lists are fused into one.

| List | Method | Size |
|---|---|---|
| General dense | Embedding similarity over all chunks | top 5 |
| Statute dense | Embedding similarity over `doc_type = statute` only | top 2 |
| Sparse | BM25 keyword scoring (`rank-bm25`) | top 3 |

The lists are merged with **Reciprocal Rank Fusion (RRF)**: each chunk scores
`Σ 1 / (60 + rank)` over the lists it appears in, and the top 7 go to the model.
RRF uses only *ranks*, which sidesteps comparing BM25 scores (unbounded) with
cosine similarities (bounded).

Each list exists because of a failure observed during evaluation:

- **Dense search** handles paraphrase: a question about a "guest pass" finds a
  passage about "visitor permits".
- **BM25** was added because dense search under-ranked short, exact phrases
  such as "fourteen days" once the corpus was large.
- **The statute-only list** was added because the Order's ~215 chunks were being
  drowned out by thousands of ruling chunks that *discuss* the Order. A question
  about the Order's own rules would retrieve five rulings quoting it in passing,
  and none of the Order itself.

Queries are embedded with bge's required instruction prefix (*"Represent this
sentence for searching relevant passages: "*); documents are embedded without it.

### 4.1 Relationship to Dense Passage Retrieval (Karpukhin et al., 2020)

The dense component follows the approach introduced in *Dense Passage Retrieval
for Open-Domain Question Answering* (Karpukhin et al., EMNLP 2020). DPR showed
that retrieval can work well with dense representations alone: a **dual
encoder** maps questions and passages into the same vector space, and passages
are ranked by vector similarity. On open-domain QA benchmarks its dense
retriever outperformed a strong BM25 baseline by 9–19% absolute in top-20
passage retrieval accuracy.

**What this project replicates:**

- **Dual-encoder retrieval.** Questions and passages are encoded separately into
  one embedding space and ranked by inner product (cosine similarity, since the
  vectors are normalised), with passages pre-indexed offline, as in DPR.
- **Comparison against BM25.** DPR's central evaluation measures dense retrieval
  against BM25. This project does the same on a small scale in its companion
  recipe, [hybrid-retrieval-eval](https://github.com/mikekeror-rgb/hybrid-retrieval-eval),
  which scores BM25, dense and hybrid retrieval on the same questions.
- **Combining sparse and dense.** DPR also reports a combined BM25 + DPR
  retriever. This project likewise combines the two.

**What it does not replicate:**

- **No encoder training.** DPR trains its question and passage encoders on
  question–passage pairs with in-batch negatives. This project uses an
  off-the-shelf, general-purpose embedding model (`bge-small-en-v1.5`) with no
  fine-tuning on legal data.
- **Different fusion.** DPR's combined retriever mixes the two scores linearly;
  this project fuses rankings with RRF instead, and adds a third,
  metadata-filtered list.
- **Different reader.** DPR pairs retrieval with an extractive reader that
  selects an answer span. Here, a generative LLM writes the answer from the
  retrieved passages.
- **Scale.** DPR indexes about 21 million Wikipedia passages; this corpus has
  about 3,900 chunks.

In short, the project adopts DPR's retrieval paradigm and its dense-versus-BM25
comparison, not its training procedure or reader.

---

## 5. Fee calculator and router

### 5.1 Calculator (`fee_calculator.py`)

Pure Python functions implementing the Order's formulas:

| Function | Covers | Example |
|---|---|---|
| Schedule 1, First Scale | Sale or purchase of land | Kshs 12,000,000 sale → Kshs 205,000 |
| Schedule 1, Second Scale (creation) | Creating a charge or mortgage | Kshs 15,000,000 charge → Kshs 193,750 |
| Schedule 1, Second Scale (discharge) | Discharging a charge, with or without an undertaking | Kshs 2,800,000 → Kshs 15,000 / Kshs 10,000 |
| Schedule 6 | High Court instruction fees, defended or undefended | Kshs 26,000,000 defended → Kshs 590,000 |
| Schedule 7 | Subordinate court instruction fees, lower and higher scale | |
| Full High Court bill | Itemised bill of costs (below) | |

A full bill includes the instruction fee, a getting-up fee (one third of the
instruction fee), drawing pleadings and other documents per folio, perusals,
letters, court attendances, an optional 50% advocate–client uplift (Schedule 6,
Part B), and disbursements. VAT (16%) applies to professional fees only. Every
line records the provision it is charged under, and any item whose count was
not supplied appears as "—" with its rate, rather than being silently dropped.

### 5.2 Router (`fee_router.py`)

The router classifies a question with keyword patterns, extracts the monetary
amount, and reads modifiers such as defended/undefended, grantee/grantor and
whether an undertaking is required. When a modifier is ambiguous, it returns
both figures instead of guessing.

A routed result is passed to the model as a **verified calculation** that it
must state exactly. Full bills skip the model entirely: the table is already
complete, and asking a model to restate a large table only adds risk.

---

## 6. Generation

Answers are generated by `openai/gpt-oss-20b` on Groq's hosted API (temperature
0.1). The system prompt requires the model to:

- answer only from the supplied context, and reply exactly *"Not covered in the
  provided documents"* otherwise;
- cite the chunk ID and case citation behind each claim;
- treat a verified calculation as authoritative;
- distinguish figures quoted from the Order itself from figures a court applied
  in a particular case.

`gpt-oss-20b` is a reasoning model: hidden reasoning tokens come out of the
same budget as the visible answer. The pipeline therefore sets
`reasoning_effort="low"` and `max_completion_tokens=4096`, and raises an
explicit error if the model still returns empty content, rather than passing
an empty answer to the user. Rate-limit errors are retried with increasing
back-off.

The project originally generated locally with `qwen2.5:7b` via Ollama, and
moved to Groq so it could be deployed on Streamlit Community Cloud.

---

## 7. Evaluation harness

`eval/qa_pairs.jsonl` holds 22 questions in four categories:

| Category | Count | Tests |
|---|---|---|
| Calculator | 7 | Routing and exact figures |
| Statute lookup | 6 | Retrieval and faithful quoting of the Order |
| Case law | 6 | Finding the right ruling among many similar ones |
| No answer | 3 | Refusing out-of-scope questions, including a near-miss (UK trade mark law vs Kenya's Schedule 4) |

`eval/eval.py` scores each answer on:

| Metric | How it's measured | Deterministic? |
|---|---|---|
| Retrieval hit | A retrieved chunk contains the expected phrase (statutes) or comes from the expected case (rulings) | Yes |
| Numeric exact match | The calculator produced the expected figure | Yes |
| Answer states figure | The visible answer text contains the expected figure | Yes |
| Calculator routing | The router fired exactly when it should | Yes |
| Citation verifiable | Every cited chunk ID was actually retrieved for that question | Yes |
| Faithfulness (1–5) | An LLM judge rates support by the retrieved context | No |

Two design choices matter:

- **Content-based matching.** Expected phrases survive re-chunking; hard-coded
  chunk IDs would break every time the corpus changes.
- **Deterministic metrics first.** The LLM judge proved unreliable. A 7B local
  judge (`qwen2.5:7b`) scored correct refusals and correct answers as failures.
  Correct refusals are now scored deterministically, and a stronger judge
  (`gpt-5.4-mini`) can be selected with the `JUDGE_MODEL` environment variable.
  Faithfulness is reported as a secondary signal only.

**Latest results** (from `eval/results.csv`, also shown in the app's Eval tab):

| Metric | Result |
|---|---|
| Numeric accuracy | [FILL IN] |
| Answer states figure | [FILL IN] |
| Retrieval hit rate | [FILL IN] |
| Citation accuracy | [FILL IN] |
| Avg. faithfulness | [FILL IN] |

---

## 8. What the evaluation caught

The harness was run ten times while developing the pipeline. Each run exposed a
different problem:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Calculator questions answered "Not covered" | The prompt's refusal rule didn't recognise the verified calculation as valid context | Prompt rule making verified calculations authoritative |
| 2 | Calculator questions answered with unrelated case text | Ollama's default 2,048-token context silently cut off the start of the prompt | Explicit context size; question and calculation repeated at both ends of the prompt |
| 3 | Every lookup question answered "Not covered" | A new 800-character snippet limit cut chunks built at 1,800 characters | Limit raised above chunk size |
| 4 | Statute questions retrieved rulings instead of the Order | The Order was outnumbered roughly 17:1 in a growing corpus | Statute-only retrieval list |
| 5 | Exact phrases ("fourteen days") missed | Dense search under-ranks short exact matches | BM25 added, fused with RRF |
| 6 | A correct-sounding answer with a fabricated citation | The model attached one ruling's text to another ruling's name | Citation-verifiability metric and a stricter citation prompt |
| 7 | Case-law questions scored against the wrong ruling | 12 first-party names appear in several unrelated rulings | Eval hints use case numbers, not surnames |
| 8 | A ruling's disposition could never be retrieved | The statute heading pattern mistook short ruling paragraphs for headings, leaving only the word "dismissed." searchable | Separate heading patterns for rulings |
| 9 | Correct answers and refusals scored 1/5 | Unreliable 7B judge | Deterministic scoring for refusals; optional stronger judge |
| 10 | Blank answers in production | Reasoning tokens used up the whole token budget | Lower reasoning effort, larger budget, explicit error |

The harness itself also had bugs: its numeric check first read the *loan
amount* instead of the fee, and its judge was shown only section labels
instead of chunk text. Both were fixed, and a separate "answer states figure"
metric was added once a correct calculation turned out to be invisible in the
final answer text.

---

## 9. The application

`app.py` is a Streamlit app with three tabs:

- **Ask a question:** answers with sources, and a badge when the calculator
  produced the figure. Answers are cached per session, so interacting with the
  page doesn't trigger repeat API calls. When a question isn't covered, users can
  suggest a document to add; suggestions are filed as GitHub issues labelled
  `document-request` (at most three per session, with user text quoted safely).
- **Bill of costs:** a form for a full High Court bill: subject-matter value,
  defended or not, party-and-party or advocate–client, folio counts, letters,
  attendances and disbursements. It downloads as CSV or Markdown and shows the
  supporting chunks of the Order.
- **Eval results:** metrics and charts from `eval/results.csv`, including any
  fabricated citations.

---

## 10. Configuration and deployment

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Generation |
| `GITHUB_TOKEN`, `GITHUB_REPO` | No | Document-suggestion issues (fine-grained token, Issues read/write, this repo only) |
| `JUDGE_MODEL` | No | Evaluation judge; defaults to local `qwen2.5:7b` via Ollama |
| `OPENAI_API_KEY` | Only for an OpenAI judge | Used when `JUDGE_MODEL` names a `gpt-` model |

Locally, variables go in `.env` (loaded by `python-dotenv`) or
`.streamlit/secrets.toml`; on Streamlit Community Cloud, in the app's Secrets
settings. Both files are git-ignored.

The deployed app reads the pre-built `chroma_db/` from the repository, since
Streamlit Community Cloud cannot run the ingestion step. The embedding model
runs on CPU.

---

## 11. Limitations

- **Case-law retrieval is the weakest area.** With many similar rulings,
  retrieving the *specific* case asked about is still unreliable.
- **Small models can answer from memory.** The Order is public, so the model
  sometimes gives a correct figure while citing an unrelated chunk. The
  citation-verifiability metric detects this; the prompt reduces it but doesn't
  eliminate it.
- **The router is keyword-based.** Unusual phrasing can miss a trigger; the Bill
  of costs form avoids this by taking structured input.
- **Calculator coverage.** Schedules 1, 6 and 7 are implemented; the bill form
  covers the High Court (Schedule 6) only.
- **Currency of the law.** Rates follow the 2022 revision; VAT is set by the VAT
  Act, not the Order.
- **Small eval set.** 22 questions, so single questions move the metrics
  substantially.

---

## 12. References

- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., &
  Yih, W. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
  Proceedings of EMNLP 2020. <https://arxiv.org/abs/2004.04906>
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank
  Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.
- Advocates (Remuneration) Order, Legal Notice 64 of 1962 (revised to 31
  December 2022). Kenya Law. <https://new.kenyalaw.org/akn/ke/act/ln/1962/64/eng@2022-12-31>
- BAAI. *bge-small-en-v1.5.* <https://huggingface.co/BAAI/bge-small-en-v1.5>
