#!/usr/bin/env bash
# Pipeline-component ablation on the BASE model, 62 questions, new pipeline (v2).
# Compares each run against the existing full-pipeline run eval/results_v3_base.csv.
# Needs the vLLM multi-LoRA server up (serves the model name "base").
#
#   export LLM_BASE_URL=https://<POD-ID>-8000.proxy.runpod.net/v1
#   export LLM_API_KEY=<VLLM_API_KEY>
#   export JUDGE_MODEL=gpt-5.4-mini
#   ./run_pipeline_ablation.sh            # snippet, case-scope, and a noise rerun
#   WITH_CALC=1 ./run_pipeline_ablation.sh   # also calculator-off and all-off runs
set -euo pipefail

: "${LLM_BASE_URL:?set LLM_BASE_URL}"
: "${LLM_API_KEY:?set LLM_API_KEY}"
export QA_FILE=eval/qa_pairs_v2.jsonl
export LLM_MODEL=base

run() {
  local tag=$1; shift
  local out="eval/results_${tag}.csv"
  if [ -s "$out" ]; then echo "skip $tag (exists)"; return; fi
  echo "=== $tag  ($*) ==="
  env "$@" EVAL_TAG="$tag" python eval/eval_v2.py
}

run abl_fullrerun                      # same config as v3_base: measures run-to-run noise
run abl_snip1000   CONTEXT_SNIPPET_CHARS=1000
run abl_nocase     DISABLE_CASE_SCOPE=1
if [ "${WITH_CALC:-0}" = "1" ]; then
  run abl_nocalc   DISABLE_CALCULATOR=1
  run abl_stripped CONTEXT_SNIPPET_CHARS=1000 DISABLE_CASE_SCOPE=1 DISABLE_CALCULATOR=1
fi

echo; echo "Each ablation vs full pipeline (B - A; negative = the component helped):"
for t in abl_fullrerun abl_snip1000 abl_nocase abl_nocalc abl_stripped; do
  f="eval/results_${t}.csv"; [ -s "$f" ] || continue
  echo "-- $t"; python eval/summarize_ablation.py --pair eval/results_v3_base.csv "$f"
done
echo; echo "Fine-tuning effects, full pipeline (already run):"
echo "-- SFT vs base";      python eval/summarize_ablation.py --pair eval/results_v3_base.csv eval/results_v3_advocate-ep1.csv
echo "-- DPO vs SFT";       python eval/summarize_ablation.py --pair eval/results_v3_advocate-ep1.csv eval/results_v3_advocate-dpo.csv
