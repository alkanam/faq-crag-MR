"""
Test with original code structure (no LangGraph) to see if it produces better answer.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

retriever = setup_multi_vector_retriever()
llm = OllamaLLM(model="qwen2.5:3b")

template = """You are a FAQ expert. Answer the user's question using ONLY the provided FAQ context.

The context contains Q&A pairs from the FAQ. Extract and synthesize information from these Q&A pairs to answer the question.

If the Q&A context contains relevant information to answer the question, provide the answer.
If the information is NOT in the Q&A context at all, respond: "I don't have information about this."

Context (FAQ Q&A pairs):
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

test_question = "If a company operates a forest management activity that partially involves afforestation on degraded land, but also uses some synthetic fertilizers in other sections of the same project, does the entire activity fail DNSH criteria or can they claim alignment for the compliant portions?"

print(f"Question: {test_question}\n")
print("="*70)

# Original code path
docs = retriever.invoke(test_question)
context = "\n\n".join([doc.page_content for doc in docs])

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": test_question})

print("ANSWER:")
print("="*70)
print(answer)
