from retriever import setup_multi_vector_retriever

retriever = setup_multi_vector_retriever()

question = "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?"

print(f"Testing retriever with question:\n{question}\n")

docs = retriever.invoke(question)

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs, 1):
    print(f"Doc {i}:")
    print(f"Content (first 300 chars):\n{doc.page_content[:300]}\n")
