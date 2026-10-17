#!/usr/bin/env bash
# Answer latency per pipeline setting, on the same mixed-strata sample, no judge.
# Needs LLM_BASE_URL and LLM_API_KEY exported and the vLLM server up.
set -euo pipefail
: "${LLM_BASE_URL:?set LLM_BASE_URL}"
: "${LLM_API_KEY:?set LLM_API_KEY}"
export LLM_MODEL=base QA_FILE=eval/qa_pairs_v3.jsonl
: > eval/timing.txt
run() { local tag=$1; shift; env "$@" TAG="$tag" python eval/time_pipeline.py 2>/dev/null | tail -1 | tee -a eval/timing.txt; }
run full
run nocase   DISABLE_CASE_SCOPE=1
run snip1000 CONTEXT_SNIPPET_CHARS=1000
run nocalc   DISABLE_CALCULATOR=1
