"""
Test hybrid retriever (dense + BM25).
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from eval_runner import run_evaluation
from embeddings.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(alpha=0.5)
results = run_evaluation(retriever, "hybrid_dense_bm25")

print(f"\nHybrid (Dense + BM25) Summary:")
print(f"  Accuracy: {results['total_correct']}/{results['total']} ({100*results['total_accuracy']:.1f}%)")
print(f"  In-scope: {100*results['in_scope_accuracy']:.1f}%")
print(f"  Out-of-scope: {100*results['out_scope_accuracy']:.1f}%")
print(f"  Avg latency: {results['avg_latency']:.2f}s")
