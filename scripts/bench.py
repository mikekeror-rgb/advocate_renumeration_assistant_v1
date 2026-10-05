"""
bench.py: load-test an OpenAI-compatible LLM server (vLLM, Ollama, TGI) with REAL RAG prompts.

Prompts are the system + user messages from training/sft_train.jsonl: the same ~4k-token
"retrieved context + question" inputs the assistant sees in production, not toy prompts.

For each concurrency level it sends a batch of streamed requests and measures:
  TTFT        time to first token (what a user waits before text appears)
  latency     time to the full answer
  tok/s/req   decode speed one user sees
  agg tok/s   total output tokens per second across all users (server throughput)
  req/s       requests completed per second

Run it ON the pod against localhost, so internet latency doesn't pollute the numbers:
  set -a; source .env; set +a
  python scripts/bench.py --model base --label vllm_bf16_base
  python scripts/bench.py --model advocate-ep1 --label vllm_bf16_lora_ep1

Outputs: results/bench_<label>.json and one row per level appended to results/bench_summary.csv.
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def load_prompts(path: Path) -> list[list[dict]]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["messages"][:-1])  # system + user, no answer
    if not prompts:
        raise RuntimeError(f"No prompts in {path}")
    return prompts


async def one_request(client: AsyncOpenAI, model: str, messages: list[dict], max_tokens: int,
                      ignore_eos: bool) -> dict:
    start = time.perf_counter()
    first = None
    chunks = 0
    completion_tokens = prompt_tokens = None
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True} if ignore_eos else None,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            chunks += 1
        usage = getattr(chunk, "usage", None)
        if usage:
            completion_tokens, prompt_tokens = usage.completion_tokens, usage.prompt_tokens
    end = time.perf_counter()
    first = first or end
    tokens = completion_tokens if completion_tokens is not None else chunks
    decode_time = end - first
    return {
        "ttft": first - start,
        "latency": end - start,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "tok_s": (tokens - 1) / decode_time if decode_time > 0 and tokens > 1 else float("nan"),
    }


async def run_level(client: AsyncOpenAI, args, prompts: list, concurrency: int, offset: int) -> dict:
    """offset: index of the first prompt to use. Every request in the whole run gets a
    different prompt, so vLLM's prefix cache can't serve repeats and flatter the numbers."""
    total = args.requests or max(16, 2 * concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(total):
        queue.put_nowait(prompts[(offset + i) % len(prompts)])
    results, errors = [], []

    async def worker():
        while True:
            try:
                messages = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(await one_request(client, args.model, messages, args.max_tokens, args.ignore_eos))
            except Exception as error:  # record and keep going, so one failure doesn't void the level
                errors.append(f"{type(error).__name__}: {str(error)[:150]}")

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started

    ttft = [r["ttft"] for r in results]
    latency = [r["latency"] for r in results]
    tok_s = [r["tok_s"] for r in results if r["tok_s"] == r["tok_s"]]  # drop NaN
    prompt_tokens = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
    output_tokens = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests_ok": len(results),
        "errors": len(errors),
        "error_examples": errors[:3],
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens)) if prompt_tokens else None,
        "ttft_p50_ms": round(percentile(ttft, 0.5) * 1000),
        "ttft_p95_ms": round(percentile(ttft, 0.95) * 1000),
        "latency_p50_s": round(percentile(latency, 0.5), 2),
        "latency_p95_s": round(percentile(latency, 0.95), 2),
        "tok_s_per_request_p50": round(statistics.median(tok_s), 1) if tok_s else None,
        "aggregate_tok_s": round(output_tokens / wall, 1),
        "requests_per_s": round(len(results) / wall, 2),
        "wall_s": round(wall, 1),
    }


async def main_async(args) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600, max_retries=0)
    prompts = load_prompts(Path(args.prompts))
    print(f"Server: {args.base_url} | model: {args.model} | {len(prompts)} real prompts | "
          f"max_tokens={args.max_tokens} | ignore_eos={args.ignore_eos}")

    print("Warm-up (2 requests)...")
    for messages in prompts[-2:]:  # the last prompts, never reused below
        await one_request(client, args.model, messages, 32, False)

    levels, offset = [], 0
    for concurrency in [int(c) for c in args.concurrency.split(",")]:
        print(f"Concurrency {concurrency}...", flush=True)
        level = await run_level(client, args, prompts, concurrency, offset)
        offset += level["requests_ok"] + level["errors"]
        levels.append(level)
        if level["errors"]:
            print(f"  ⚠ {level['errors']} errors, e.g. {level['error_examples'][0]}")

    header = f"{'conc':>4} {'ok':>4} {'err':>3} {'TTFT p50':>9} {'TTFT p95':>9} {'lat p50':>8} {'lat p95':>8} {'tok/s/req':>9} {'agg tok/s':>10} {'req/s':>6}"
    print(f"\n=== {args.label} ===\n{header}")
    for l in levels:
        print(f"{l['concurrency']:>4} {l['requests_ok']:>4} {l['errors']:>3} {l['ttft_p50_ms']:>7}ms {l['ttft_p95_ms']:>7}ms "
              f"{l['latency_p50_s']:>7}s {l['latency_p95_s']:>7}s {str(l['tok_s_per_request_p50']):>9} "
              f"{l['aggregate_tok_s']:>10} {l['requests_per_s']:>6}")
    if levels and levels[0]["mean_prompt_tokens"]:
        print(f"Mean prompt length: {levels[0]['mean_prompt_tokens']} tokens")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    meta = {"label": args.label, "model": args.model, "base_url": args.base_url,
            "max_tokens": args.max_tokens, "ignore_eos": args.ignore_eos}
    (out_dir / f"bench_{args.label}.json").write_text(json.dumps({**meta, "levels": levels}, indent=2))
    summary_path = out_dir / "bench_summary.csv"
    new_file = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        fields = ["label", "model"] + [k for k in levels[0] if k != "error_examples"]
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for l in levels:
            writer.writerow({"label": args.label, "model": args.model,
                             **{k: v for k, v in l.items() if k != "error_examples"}})
    print(f"\nSaved results/bench_{args.label}.json and appended to results/bench_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY") or os.environ.get("LLM_API_KEY", "none"))
    parser.add_argument("--model", required=True, help="served model name, e.g. base or advocate-ep1")
    parser.add_argument("--label", required=True, help="name for this run, e.g. vllm_bf16_base")
    parser.add_argument("--concurrency", default="1,8,32", help="comma-separated levels")
    parser.add_argument("--requests", type=int, default=0, help="requests per level (default max(16, 2x concurrency))")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompts", default=str(ROOT / "training" / "sft_train.jsonl"),
                        help="real RAG prompts; the train file has enough (507) that no prompt repeats")
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                        help="let answers stop naturally (needed for servers that reject ignore_eos)")
    parser.set_defaults(ignore_eos=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
