import json

from evaluate_answers import apply_regression_checks

with open("evals/generation_results.json", encoding="utf-8") as f:
    results = json.load(f)

q03 = next(item for item in results if item["id"] == "Q03")

retrieved_chunks = [
    {
        "chunk_id": chunk.get("chunk_id"),
        "timestamp": chunk.get("timestamp"),
        "content": chunk.get("content", "")
    }
    for chunk in q03.get("retrieved_chunks", [])
]

test_scores = {
    "correctness": 5,
    "faithfulness": 5,
    "relevance": 5,
    "abstention_score": None,
    "claim_checks": [],
    "reason": "Test"
}

result = apply_regression_checks(
    q03,
    test_scores,
    retrieved_chunks
)

print("Faithfulness:", result["faithfulness"])
print("Claim checks:", result["claim_checks"])
print("Reason:", result["reason"])