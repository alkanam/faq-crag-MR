"""
Test baseline retriever (existing dense-only retriever).
"""
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from retriever import setup_multi_vector_retriever
from eval_runner import run_evaluation

retriever = setup_multi_vector_retriever()
results = run_evaluation(retriever, "baseline_dense")

print(f"\nBaseline (Dense Only) Summary:")
print(f"  Accuracy: {results['total_correct']}/{results['total']} ({100*results['total_accuracy']:.1f}%)")
print(f"  In-scope: {100*results['in_scope_accuracy']:.1f}%")
print(f"  Out-of-scope: {100*results['out_scope_accuracy']:.1f}%")
print(f"  Avg latency: {results['avg_latency']:.2f}s")
