"""Per-question answer latency for ONE pipeline setting. No judge calls.
Settings come from env vars, exactly as in eval_v2.py (DISABLE_CASE_SCOPE, etc.).
Run via time_pipeline.sh, which runs each setting in its own process."""
import importlib, json, os, statistics, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mod = importlib.import_module(os.environ.get("PIPELINE_MODULE", "rag_pipeline_v2"))
pipeline = mod.RagPipeline()

qa = [json.loads(l) for l in open(os.environ.get("QA_FILE", "eval/qa_pairs_v3.jsonl"))]
sample = qa[::6]                      # every 6th question: mixes all strata
for q in sample[:3]:                  # warm-up, not timed
    pipeline.answer(q["question"])

times = []
for q in sample:
    t0 = time.perf_counter()
    pipeline.answer(q["question"])
    times.append(time.perf_counter() - t0)

times.sort()
p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
print(f"{os.environ.get('TAG','run')}: n={len(times)} median={statistics.median(times):.2f}s "
      f"mean={statistics.mean(times):.2f}s p95={p95:.2f}s")
