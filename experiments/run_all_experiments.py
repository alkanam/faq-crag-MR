"""
Master script to run all experiments and create comparison summary.
"""
import os
import env_loader
import json
import subprocess
from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("="*70)
print("FAQ-CRAG OPTIMIZATION: COMPREHENSIVE EXPERIMENTS")
print("="*70)
print()

# Track all results
all_results = {}

# Test 1: Baseline (dense only)
print("\n[1/4] Running BASELINE (dense-only retriever)...")
print("-"*70)
try:
    result = subprocess.run(["python", "test_baseline.py"], capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: Hybrid retriever
print("\n[2/4] Running HYBRID RETRIEVER (dense + BM25)...")
print("-"*70)
try:
    result = subprocess.run(["python", "test_hybrid.py"], capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: Query rewriting strategies
print("\n[3/4] Running QUERY REWRITING STRATEGIES...")
print("-"*70)
try:
    result = subprocess.run(["python", "test_rewrite_strategies.py"], capture_output=True, text=True, timeout=1800)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Load and compare all results
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

results_files = list(results_dir.glob("results_*.json"))

if results_files:
    comparison_data = {}

    for results_file in sorted(results_files):
        with open(results_file) as f:
            results = json.load(f)

        name = results_file.stem.replace("results_", "")

        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r["scope_correct"])
        in_scope_correct = sum(1 for r in results if r["expected_scope"] == "in_scope" and not r["is_outofscope"])
        in_scope_total = sum(1 for r in results if r["expected_scope"] == "in_scope")
        out_scope_correct = sum(1 for r in results if r["expected_scope"] == "out_of_scope" and r["is_outofscope"])
        out_scope_total = sum(1 for r in results if r["expected_scope"] == "out_of_scope")
        avg_latency = sum(r["elapsed_seconds"] for r in results) / total if total > 0 else 0

        in_scope_acc = in_scope_correct / in_scope_total if in_scope_total > 0 else 0
        out_scope_acc = out_scope_correct / out_scope_total if out_scope_total > 0 else 0

        comparison_data[name] = {
            "total_accuracy": f"{100*correct/total:.1f}%",
            "in_scope_accuracy": f"{100*in_scope_acc:.1f}%",
            "out_scope_accuracy": f"{100*out_scope_acc:.1f}%",
            "avg_latency_s": f"{avg_latency:.2f}",
            "correct": f"{correct}/{total}",
        }

    # Print table
    print(f"\n{'Method':<40} {'Accuracy':<12} {'In-Scope':<12} {'Out-Scope':<12} {'Latency'}")
    print("-"*90)

    for name, metrics in sorted(comparison_data.items()):
        print(f"{name:<40} {metrics['total_accuracy']:<12} {metrics['in_scope_accuracy']:<12} {metrics['out_scope_accuracy']:<12} {metrics['avg_latency_s']}s")

    # Save comparison
    with open(results_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    print("\nComparison saved to: results/comparison_summary.json")
else:
    print("No results files found!")

print("\n" + "="*70)
print("Experiment complete. Check results/ folder for detailed results.")
print("="*70)
