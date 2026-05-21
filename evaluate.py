"""
Quick evaluation of existing results.json without re-running the pipeline.
"""
import json


def analyze_results(results_file="eval_results.json"):
    """Analyze pre-computed results from eval_results.json"""

    with open(results_file) as f:
        results = json.load(f)

    print(f"\n{'='*70}")
    print("RAG EVALUATION ANALYSIS")
    print(f"{'='*70}\n")

    print(f"Total questions evaluated: {len(results)}")

    # Scope classification
    correct_scope = sum(
        1 for r in results
        if r.get("expected_scope") and (
            (r["expected_scope"] == "out_of_scope" and r["is_outofscope"]) or
            (r["expected_scope"] == "in_scope" and not r["is_outofscope"])
        )
    )
    total_with_expected = sum(1 for r in results if r.get("expected_scope"))

    print(f"\nSCOPE CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Accuracy: {correct_scope}/{total_with_expected} ({100*correct_scope/total_with_expected:.1f}%)")

    in_scope_correct = sum(
        1 for r in results
        if r.get("expected_scope") == "in_scope" and not r["is_outofscope"]
    )
    in_scope_total = sum(1 for r in results if r.get("expected_scope") == "in_scope")

    out_of_scope_correct = sum(
        1 for r in results
        if r.get("expected_scope") == "out_of_scope" and r["is_outofscope"]
    )
    out_of_scope_total = sum(1 for r in results if r.get("expected_scope") == "out_of_scope")

    print(f"In-scope: {in_scope_correct}/{in_scope_total} ({100*in_scope_correct/in_scope_total:.1f}%)")
    print(f"Out-of-scope: {out_of_scope_correct}/{out_of_scope_total} ({100*out_of_scope_correct/out_of_scope_total:.1f}%)")

    # Answer quality
    print(f"\nANSWER QUALITY")
    print(f"{'='*70}")
    avg_length = sum(r["answer_length"] for r in results) / len(results)
    print(f"Average answer length: {avg_length:.0f} chars")

    grounded_count = sum(1 for r in results if r.get("grounded"))
    print(f"Grounded answers: {grounded_count}/{len(results)} ({100*grounded_count/len(results):.1f}%)")

    # Speed
    if "elapsed_seconds" in results[0]:
        print(f"\nPERFORMANCE")
        print(f"{'='*70}")
        total_time = sum(r["elapsed_seconds"] for r in results)
        avg_time = total_time / len(results)
        print(f"Total time: {total_time:.1f}s")
        print(f"Average per question: {avg_time:.2f}s")

    # False negatives (in-scope but marked out-of-scope)
    false_negatives = [
        r for r in results
        if r.get("expected_scope") == "in_scope" and r["is_outofscope"]
    ]

    if false_negatives:
        print(f"\nFALSE NEGATIVES ({len(false_negatives)}):")
        print(f"{'='*70}")
        for i, r in enumerate(false_negatives, 1):
            print(f"\n[{i}] {r['question'][:100]}...")
            print(f"    Answer: {r['answer']}")
            print(f"    Expected: in_scope, Got: out_of_scope")

    # Hallucination tests (should all be out-of-scope)
    hallucination_tests = [
        r for r in results
        if r.get("expected_scope") == "out_of_scope"
    ]
    hallucination_correct = sum(1 for r in hallucination_tests if r["is_outofscope"])
    print(f"\nHALLUCINATION RESISTANCE")
    print(f"{'='*70}")
    print(f"Correctly refused: {hallucination_correct}/{len(hallucination_tests)}")
    print(f"(Lower is better - system should refuse out-of-scope questions)")

    return results


if __name__ == "__main__":
    analyze_results()
