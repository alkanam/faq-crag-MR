"""
Test just the first in-scope question
"""
import os
import env_loader
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

# Setup LangSmith

question = "What are the environmental objectives under EU Taxonomy?"

print(f"Question: {question}\n")

# Setup
retriever = setup_multi_vector_retriever()
llm = OllamaLLM(model="qwen2.5:3b")

template = """Based on the context below, answer the question. Only use information from the context. If the answer is not in the context, say "I don't have information about this."

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Measure timing
t_start = time.time()

print("Retrieving documents...")
t_ret_start = time.time()
docs = retriever.invoke(question)
t_ret_elapsed = time.time() - t_ret_start

print(f"Retrieved {len(docs)} docs in {t_ret_elapsed:.2f}s\n")

for i, doc in enumerate(docs, 1):
    print(f"[Doc {i}]")
    print(doc.page_content[:250])
    print()

print("Generating answer...")
t_gen_start = time.time()
context = "\n\n".join([doc.page_content for doc in docs])
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})
t_gen_elapsed = time.time() - t_gen_start

t_total = time.time() - t_start

print(f"ANSWER:\n{answer}\n")

print("=" * 70)
print("TIMING")
print("=" * 70)
print(f"Retrieval: {t_ret_elapsed:.2f}s")
print(f"Generation: {t_gen_elapsed:.2f}s")
print(f"Total: {t_total:.2f}s")
