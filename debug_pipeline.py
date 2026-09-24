# Run this in your project folder to isolate where the calculator result gets lost.
from rag_pipeline import RagPipeline
import fee_router

query = "cost of registering a mortgage over a loan of Kshs15m"

pipeline = RagPipeline()

# Step 1: confirm the router fires inside answer()
route_result = fee_router.route(query)
print("1. route() result:", route_result)
print()

# Step 2: call answer() and inspect what came back
result = pipeline.answer(query)
print("2. result['calculator_result']:", result["calculator_result"])
print()

# Step 3: the actual prompt sent to the model — check calc_block is really in there
retrieved = pipeline.retrieve(query)
context_block = pipeline.build_context_block(retrieved)
if result["calculator_result"] is not None:
    calc_block = (
        f"VERIFIED CALCULATION (computed exactly from the Order's schedule formula — "
        f"state this figure precisely; do not recompute, round differently, or alter it):\n"
        f"{result['calculator_result'].explanation}\n"
        f"[source: {result['calculator_result'].schedule_citation}]"
    )
    print("3. calc_block that should be in the prompt:")
    print(calc_block)
print()

print("4. Final LLM answer:")
print(result["answer"])
