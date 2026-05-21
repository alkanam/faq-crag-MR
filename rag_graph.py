"""
LangGraph-based RAG with query rewriting on retrieval failure.
Single retry loop: if model says "I don't know", rewrite query once and retry.
"""
import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

load_dotenv()


class RAGState(TypedDict):
    """State for RAG graph."""
    original_question: str
    question: str
    context: str
    answer: str
    attempts: int


# Define the LLM and templates once
llm = OllamaLLM(model="qwen2.5:3b")
retriever = setup_multi_vector_retriever()

answer_template = """You are a FAQ expert. Answer the user's question using ONLY the provided FAQ context.

The context contains Q&A pairs from the FAQ. Extract and synthesize information from these Q&A pairs to answer the question.

If the Q&A context contains relevant information to answer the question, provide the answer.
If the information is NOT in the Q&A context at all, respond: "I don't have information about this."

Context (FAQ Q&A pairs):
{context}

Question: {question}

Answer:"""

rewrite_template = """Given this question that didn't yield results in the FAQ, rephrase it using different terminology while keeping the core meaning. Keep it concise.

Original question: {question}

Rephrased question:"""


def retrieve_and_answer(state: RAGState) -> RAGState:
    """Retrieve documents and generate answer."""
    # Retrieve context
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate answer
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    chain = answer_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})

    # Update state
    return {
        **state,
        "context": context,
        "answer": answer,
        "attempts": state["attempts"] + 1,
    }


def rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the query for better retrieval."""
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": state["original_question"]})

    # Update state with rewritten question
    return {
        **state,
        "question": rewritten.strip(),
    }


def should_continue(state: RAGState) -> str:
    """Decide whether to rewrite and retry, or exit."""
    answer = state["answer"].lower()

    # Check if model said "don't know"
    no_knowledge_phrases = [
        "I don't have information"
    ]
    has_no_knowledge = any(phrase in answer for phrase in no_knowledge_phrases)

    # Only rewrite if: model said "don't know" AND this is first attempt
    if has_no_knowledge and state["attempts"] == 1:
        return "rewrite"

    return "end"


def build_rag_graph():
    """Build and return the RAG graph."""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve_answer", retrieve_and_answer)
    graph.add_node("rewrite_query", rewrite_query)

    # Set entry point
    graph.set_entry_point("retrieve_answer")

    # Add conditional edges from retrieve_answer
    graph.add_conditional_edges(
        "retrieve_answer",
        should_continue,
        {
            "rewrite": "rewrite_query",
            "end": END,
        },
    )

    # After rewrite, go back to retrieve_answer
    graph.add_edge("rewrite_query", "retrieve_answer")

    return graph.compile()


# For backwards compatibility
def create_rag_chain():
    """Create a runnable chain from the graph."""
    graph = build_rag_graph()

    def chain_invoke(question: str) -> dict:
        initial_state = {
            "original_question": question,
            "question": question,
            "context": "",
            "answer": "",
            "attempts": 0,
        }
        final_state = graph.invoke(initial_state)
        return final_state

    return chain_invoke
