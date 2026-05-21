"""
Quick test to verify LangSmith tracing is working
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

print("Testing LangSmith tracing...")
print(f"API Key set: {'LANGSMITH_API_KEY' in os.environ}")
print(f"Tracing enabled: {os.environ.get('LANGSMITH_TRACING_V2')}")
print(f"Project: {os.environ.get('LANGSMITH_PROJECT')}")

# Setup
retriever = setup_multi_vector_retriever()
llm = OllamaLLM(model="qwen2.5:3b")

template = """Based on the context below, answer the question. Only use information from the context. If the answer is not in the context, say "I don't have information about this."

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Run one question
question = "What are the environmental objectives under EU Taxonomy?"
print(f"\nQuestion: {question}\n")

docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in docs])
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})

print(f"Answer: {answer[:200]}...\n")
print("Check LangSmith at: https://smith.langchain.com/projects/faq-rag-evaluation")
print("The trace should appear in the project")
