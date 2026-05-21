"""
Test with simple, straightforward FAQ questions
"""
import os
import env_loader
import json

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

# Setup
retriever = setup_multi_vector_retriever()
llm = OllamaLLM(model="qwen2.5:3b")

template = """Based on the context below, answer the question. Only use information from the context. If the answer is not in the context, say "I don't have information about this."

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

simple_questions = [
    "What are the environmental objectives?",
    "What is DNSH?",
    "What is the Climate Delegated Act?",
    "What are technical screening criteria?",
    "How many environmental objectives are there?",
]

print("Testing simple FAQ questions:\n")
print("="*70)

for i, question in enumerate(simple_questions, 1):
    print(f"\n[{i}] {question}")
    print("-"*70)

    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} docs:")
    for j, doc in enumerate(docs, 1):
        print(f"  Doc {j}: {doc.page_content[:80]}...")

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Check scope
    is_outofscope = any(phrase in answer.lower() for phrase in [
        "don't have information",
        "don't know",
        "i'm not",
        "not available",
    ])

    print(f"\nScope: {'OUT-OF-SCOPE' if is_outofscope else 'IN-SCOPE'}")
    print(f"Answer: {answer[:200]}...")

print("\n" + "="*70)
print("Done. Check LangSmith traces at:")
print("https://smith.langchain.com/projects/faq-rag-evaluation")
