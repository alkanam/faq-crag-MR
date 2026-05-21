"""
Test single question with LangSmith tracing.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from rag_graph import build_rag_graph
import json

# Build graph
graph = build_rag_graph()

# Test with a challenging question that might trigger rewrite
test_question = "If a company operates a forest management activity that partially involves afforestation on degraded land, but also uses some synthetic fertilizers in other sections of the same project, does the entire activity fail DNSH criteria or can they claim alignment for the compliant portions?"

print(f"Question: {test_question}\n")
print("="*70)

# Run graph
initial_state = {
    "original_question": test_question,
    "question": test_question,
    "context": "",
    "answer": "",
    "attempts": 0,
}

final_state = graph.invoke(initial_state)

print(f"\nAttempts: {final_state['attempts']}")
print(f"\nFinal Question: {final_state['question']}")
print(f"\nAnswer:\n{final_state['answer']}")
print(f"\nContext length: {len(final_state['context'])} chars")
print("\n" + "="*70)
print("Check LangSmith dashboard for full trace:")
print("https://smith.langchain.com/")
