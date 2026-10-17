#!/usr/bin/env bash
# v3 held-out evaluation. Pre-registered in eval/PREREG.md.
# Needs the vLLM multi-LoRA server up, serving: base, advocate-ep1, advocate-dpo
#
#   export LLM_BASE_URL=https://<POD-ID>-8000.proxy.runpod.net/v1
#   export LLM_API_KEY=<VLLM_API_KEY>
#   ./run_v3_eval.sh            # all runs; safe to rerun, finished runs are skipped
#
# Results go to eval/results_ev3_<tag>.csv. Old results_v3_* files are never touched.
set -euo pipefail

: "${LLM_BASE_URL:?set LLM_BASE_URL}"
: "${LLM_API_KEY:?set LLM_API_KEY}"

# The pre-registered judge. Do not change.
export JUDGE_MODEL=gpt-5.4-mini
export QA_FILE=eval/qa_pairs_v3.jsonl

want=95ddd72d65bffeca
got=$(shasum -a 256 "$QA_FILE" | cut -c1-16)
if [ "$got" != "$want" ]; then
  echo "ABORT: $QA_FILE hash is $got, pre-registered hash is $want" >&2
  exit 1
fi
echo "eval set hash OK ($got); judge=$JUDGE_MODEL"

run() {
  local tag=$1 model=$2; shift 2
  local out="eval/results_ev3_${tag}.csv"
  if [ -s "$out" ]; then echo "skip ev3_$tag (exists)"; return; fi
  echo "=== ev3_$tag  model=$model  ($*) ==="
  env "$@" LLM_MODEL="$model" EVAL_TAG="ev3_${tag}" python eval/eval_v2.py
}

# Systems under test, full pipeline
run base        base
run ep1         advocate-ep1
run dpo         advocate-dpo

# Run-to-run noise on the base model
run base_rerun  base

# Pipeline ablations, on the base model (matches the earlier study)
run nocase      base DISABLE_CASE_SCOPE=1
run snip1000    base CONTEXT_SNIPPET_CHARS=1000
run nocalc      base DISABLE_CALCULATOR=1
run stripped    base CONTEXT_SNIPPET_CHARS=1000 DISABLE_CASE_SCOPE=1 DISABLE_CALCULATOR=1

echo; echo "Done. Results: eval/results_ev3_*.csv"
