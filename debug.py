from retriever import setup_multi_vector_retriever
from sentence_transformers import SentenceTransformer

# Test retrieval
retriever = setup_multi_vector_retriever()
model = SentenceTransformer("all-MiniLM-L6-v2")

question = "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?"

print(f"Question: {question}\n")

# Test retrieval directly
docs = retriever.invoke(question)

print(f"Retrieved {len(docs)} documents:\n")
for i, doc in enumerate(docs, 1):
    print(f"Doc {i}:")
    print(f"  Content (first 200 chars): {doc.page_content[:200]}")
    print(f"  Metadata: {doc.metadata}")
    print()

# Show formatted context
def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])

context = format_context(docs)
print(f"Formatted context length: {len(context)} chars")
print(f"Formatted context (first 500 chars):\n{context[:500]}")
