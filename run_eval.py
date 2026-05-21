"""
Re-run full evaluation with improved context formatting
"""
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from rag_graph import build_rag_graph

# Load test cases
with open("questions/test_questions_simple.json") as f:
    simple_data = json.load(f)

test_cases = []
for q in simple_data["questions"]:
    # Handle both original/rephrased pairs and standalone out-of-scope questions
    if "rephrased" in q:
        # Test both original and rephrased
        test_cases.append({
            "question": q["original"],
            "id": q["id"] + "_orig",
            "expected_scope": "in_scope",
        })
        test_cases.append({
            "question": q["rephrased"],
            "id": q["id"] + "_reph",
            "expected_scope": "in_scope",
        })
    else:
        # Out-of-scope questions
        test_cases.append({
            "question": q["question"],
            "id": q.get("id"),
            "expected_scope": "out_of_scope" if q.get("id", "").startswith("oos_") else "in_scope",
        })

graph = build_rag_graph()

print(f"Running evaluation on {len(test_cases)} questions...")
results = []

for i, test in enumerate(test_cases, 1):
    question = test["question"]
    expected_scope = test.get("expected_scope")

    t_start = time.time()

    # Run graph with initial question
    final_state = graph.invoke({
        "original_question": question,
        "question": question,
        "context": "",
        "answer": "",
        "attempts": 0,
    })

    t_elapsed = time.time() - t_start
    answer = final_state["answer"]
    attempts = final_state["attempts"]

    is_outofscope = any(phrase in answer.lower() for phrase in [
        "don't have information",
        "don't know",
        "i'm not",
        "not available",
        "not mentioned",
        "not provided",
    ])

    scope_correct = (
        (expected_scope == "out_of_scope" and is_outofscope) or
        (expected_scope == "in_scope" and not is_outofscope)
    )

    results.append({
        "id": test.get("id"),
        "question": question,
        "answer": answer,
        "expected_scope": expected_scope,
        "is_outofscope": is_outofscope,
        "scope_correct": scope_correct,
        "answer_length": len(answer),
        "elapsed_seconds": t_elapsed,
        "attempts": attempts,
    })

    print(f"[{i}/{len(test_cases)}] {expected_scope}: {scope_correct} - {t_elapsed:.1f}s (attempts: {attempts})")

# Save and analyze
with open("eval_results_v2.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary
correct = sum(1 for r in results if r["scope_correct"])
print(f"\n{'='*70}")
print(f"IMPROVED RESULTS: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
print(f"{'='*70}")

in_scope_correct = sum(1 for r in results if r["expected_scope"] == "in_scope" and not r["is_outofscope"])
in_scope_total = sum(1 for r in results if r["expected_scope"] == "in_scope")
out_scope_correct = sum(1 for r in results if r["expected_scope"] == "out_of_scope" and r["is_outofscope"])
out_scope_total = sum(1 for r in results if r["expected_scope"] == "out_of_scope")

print(f"In-scope accuracy: {in_scope_correct}/{in_scope_total} ({100*in_scope_correct/in_scope_total:.1f}%)")
print(f"Out-of-scope accuracy: {out_scope_correct}/{out_scope_total} ({100*out_scope_correct/out_scope_total:.1f}%)")
print(f"\nPrevious: In-scope 40%, Out-of-scope 100%, Total 52%")
