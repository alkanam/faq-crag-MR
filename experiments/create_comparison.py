"""
Create comparison table from all experiment results.
"""
import json
from pathlib import Path

results_dir = Path("results")
results_files = sorted(results_dir.glob("results_*.json"))

print("\n" + "="*100)
print("FAQ-CRAG OPTIMIZATION: FINAL RESULTS COMPARISON")
print("="*100)

if not results_files:
    print("\nNo results found yet. Waiting for experiments to complete...")
else:
    comparison = {}

    for results_file in results_files:
        with open(results_file) as f:
            results = json.load(f)

        name = results_file.stem.replace("results_", "").replace("_", " ").title()

        total = len(results)
        correct = sum(1 for r in results if r["scope_correct"])
        in_scope_correct = sum(1 for r in results if r["expected_scope"] == "in_scope" and not r["is_outofscope"])
        in_scope_total = sum(1 for r in results if r["expected_scope"] == "in_scope")
        out_scope_correct = sum(1 for r in results if r["expected_scope"] == "out_of_scope" and r["is_outofscope"])
        out_scope_total = sum(1 for r in results if r["expected_scope"] == "out_of_scope")
        avg_latency = sum(r["elapsed_seconds"] for r in results) / total if total > 0 else 0

        in_scope_acc = in_scope_correct / in_scope_total if in_scope_total > 0 else 0
        out_scope_acc = out_scope_correct / out_scope_total if out_scope_total > 0 else 0

        comparison[name] = {
            "total_acc": 100 * correct / total,
            "in_scope": 100 * in_scope_acc,
            "out_scope": 100 * out_scope_acc,
            "latency": avg_latency,
            "correct": f"{correct}/{total}",
        }

    # Print table
    print(f"\n{'Method':<40} {'Accuracy':<12} {'In-Scope':<12} {'Out-Scope':<12} {'Latency':<10}")
    print("-"*100)

    for name, metrics in sorted(comparison.items(), key=lambda x: x[1]["total_acc"], reverse=True):
        print(
            f"{name:<40} "
            f"{metrics['total_acc']:.1f}%{'':<8} "
            f"{metrics['in_scope']:.1f}%{'':<8} "
            f"{metrics['out_scope']:.1f}%{'':<8} "
            f"{metrics['latency']:.2f}s"
        )

    # Find best performer
    best_method = max(comparison.items(), key=lambda x: x[1]["total_acc"])
    print(f"\n{'='*100}")
    print(f"BEST PERFORMER: {best_method[0]} ({best_method[1]['total_acc']:.1f}%)")
    print(f"{'='*100}\n")

    # Save summary
    with open(results_dir / "final_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: results/final_comparison.json")
