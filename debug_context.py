"""
Debug: see what context is actually being sent to the LLM
"""
import os
import env_loader

from retriever import setup_multi_vector_retriever

retriever = setup_multi_vector_retriever()

question = "What is the Climate Delegated Act?"
docs = retriever.invoke(question)

print(f"Question: {question}\n")
print("="*70)
print("RAW CONTEXT BEING SENT TO LLM:")
print("="*70)

context = "\n\n".join([doc.page_content for doc in docs])
print(context)
print("\n" + "="*70)
print(f"Total context length: {len(context)} chars")
