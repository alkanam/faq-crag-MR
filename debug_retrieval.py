"""
Debug retrieval to see what documents are being returned.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from retriever import setup_multi_vector_retriever

retriever = setup_multi_vector_retriever()

test_question = "If a company operates a forest management activity that partially involves afforestation on degraded land, but also uses some synthetic fertilizers in other sections of the same project, does the entire activity fail DNSH criteria or can they claim alignment for the compliant portions?"

print(f"Question: {test_question}\n")
print("="*70)
print("RETRIEVED DOCUMENTS:")
print("="*70)

docs = retriever.invoke(test_question)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:500])
    print("...")
