"""
Test different query rewriting strategies with baseline retriever.
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from eval_runner import run_evaluation
from rewriting.query_rewrite_strategies import (
    simple_rewrite,
    step_back_rewrite,
    multi_query_rewrite
)

from retriever import setup_multi_vector_retriever

retriever = setup_multi_vector_retriever()

print("\n" + "="*70)
print("TEST 1: Simple Query Rewrite (baseline + simple rewrite)")
print("="*70)
results_simple = run_evaluation(
    retriever,
    "baseline_simple_rewrite",
    use_rewrite_strategy=simple_rewrite
)

print(f"\nSimple Rewrite Summary:")
print(f"  Accuracy: {results_simple['total_correct']}/{results_simple['total']} ({100*results_simple['total_accuracy']:.1f}%)")
print(f"  In-scope: {100*results_simple['in_scope_accuracy']:.1f}%")
print(f"  Out-of-scope: {100*results_simple['out_scope_accuracy']:.1f}%")
print(f"  Avg latency: {results_simple['avg_latency']:.2f}s")

print("\n" + "="*70)
print("TEST 2: Step-back Query Rewrite (baseline + step-back)")
print("="*70)
results_stepback = run_evaluation(
    retriever,
    "baseline_stepback_rewrite",
    use_rewrite_strategy=step_back_rewrite
)

print(f"\nStep-back Rewrite Summary:")
print(f"  Accuracy: {results_stepback['total_correct']}/{results_stepback['total']} ({100*results_stepback['total_accuracy']:.1f}%)")
print(f"  In-scope: {100*results_stepback['in_scope_accuracy']:.1f}%")
print(f"  Out-of-scope: {100*results_stepback['out_scope_accuracy']:.1f}%")
print(f"  Avg latency: {results_stepback['avg_latency']:.2f}s")

print("\n" + "="*70)
print("TEST 3: Multi-query Rewrite (baseline + multi-query - first variant only)")
print("="*70)

def multi_query_first(question):
    """Use only first variant from multi-query for speed."""
    variants = multi_query_rewrite(question, num_variants=2)
    return variants[0] if variants else question

results_multiquery = run_evaluation(
    retriever,
    "baseline_multiquery_rewrite",
    use_rewrite_strategy=multi_query_first
)

print(f"\nMulti-query Rewrite Summary:")
print(f"  Accuracy: {results_multiquery['total_correct']}/{results_multiquery['total']} ({100*results_multiquery['total_accuracy']:.1f}%)")
print(f"  In-scope: {100*results_multiquery['in_scope_accuracy']:.1f}%")
print(f"  Out-of-scope: {100*results_multiquery['out_scope_accuracy']:.1f}%")
print(f"  Avg latency: {results_multiquery['avg_latency']:.2f}s")
