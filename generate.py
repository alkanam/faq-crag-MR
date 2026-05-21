"""
Answer generation with multi-query retrieval enhancement.

This module provides functions for generating answers to user questions using
a language model with optional multi-query retrieval. When the initial retrieval
returns sparse context, it can generate alternative phrasings of the question
and retry retrieval with better results.

The multi-query approach improves recall by covering different semantic angles:
- Original phrasing may miss relevant documents
- Alternative phrasings often retrieve complementary information
- Best context is selected automatically for answer generation
"""
import os
from dotenv import load_dotenv
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

load_dotenv()


def generate_multi_query_variants(question: str, llm) -> list:
    """
    Generate 2 alternative phrasings of a question for retrieval.

    Asks the LLM to rephrase the question from different angles without changing
    the meaning. This helps retrieve documents that might be missed by the original
    phrasing but contain relevant information using different terminology.

    Args:
        question (str): The original user question
        llm: Language model to use for generating variants (OllamaLLM)

    Returns:
        list: List of up to 2 question variants (strings), empty if generation fails
    """
    multi_query_template = """Generate 2 different reformulations of this question that ask the same thing but from different angles:

Original question: {question}

Reformulations (one per line):"""

    multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)
    chain = multi_query_prompt | llm | StrOutputParser()

    variants_text = chain.invoke({"question": question})

    # Parse variants (each line is one)
    variants = [
        v.strip().lstrip('0123456789. ').strip()
        for v in variants_text.split('\n')
        if v.strip()
    ]

    return variants[:2]  # Return up to 2 variants


def generate(question: str, retriever, llm, prompt, use_multi_query: bool = True):
    """
    Generate a streaming answer with optional multi-query retrieval fallback.

    Retrieves context using the user's question, then optionally generates question
    variants and retries if initial retrieval yields sparse context (<1000 chars).
    Streams the answer token-by-token to the caller with timing information.

    The process:
    1. Retrieve documents for the original question
    2. If context is sparse and multi_query enabled:
       - Generate 2 question variants
       - Retrieve with first variant
       - Use variant results if they contain more context
    3. Generate answer by streaming from the LLM
    4. Yield answer chunks plus metadata (query used, variants attempted)

    Args:
        question (str): The user's original question
        retriever: Document retriever instance (SimpleMultiVectorRetriever)
        llm: Language model instance (OllamaLLM)
        prompt: Prompt template (ChatPromptTemplate)
        use_multi_query (bool): Enable multi-query variant fallback (default True)

    Yields:
        str: Answer text chunks (for streaming) followed by metadata lines
            - Query used for retrieval
            - Number of variants attempted (if applicable)
    """
    t_start = time.time()

    # Retrieve with primary question
    t1 = time.time()
    docs = retriever.invoke(question)
    t2 = time.time()
    context = "\n\n".join([doc.page_content for doc in docs])

    retrieval_time = (t2 - t1) * 1000
    query_used = question
    variants_tried = []

    print(f"[TIMING] Retrieval (primary): {retrieval_time:.0f}ms, Docs: {len(docs)}")

    # Check if answer is insufficient (first attempt)
    no_knowledge_phrases = [
        "don't have information",
        "don't know",
        "i'm not",
        "not available",
        "not mentioned",
        "not provided",
    ]

    # Try retrieving with variants if enabled and initial retrieval seems weak
    if use_multi_query and len(context) < 1000:  # If context is sparse
        try:
            t_variant = time.time()
            variants = generate_multi_query_variants(question, llm)
            variants_tried = variants

            # Try first variant
            if variants:
                variant_docs = retriever.invoke(variants[0])
                variant_context = "\n\n".join([doc.page_content for doc in variant_docs])

                # Use variant if it gives more context
                if len(variant_context) > len(context):
                    docs = variant_docs
                    context = variant_context
                    query_used = variants[0]
                    print(f"[TIMING] Retrieval (variant): {(time.time() - t_variant)*1000:.0f}ms, Using variant query")
                    print(f"[QUERY] Variant: {variants[0][:80]}...")
        except Exception as e:
            print(f"[WARNING] Multi-query failed: {e}, using original query")

    # Generate with streaming
    t3 = time.time()
    chain = prompt | llm | StrOutputParser()
    stream = chain.stream({"context": context, "question": question})

    def timed_stream():
        first_token = True
        total_time = 0
        for chunk in stream:
            if first_token:
                t4 = time.time()
                ttft = (t4 - t3) * 1000
                total_time = (t4 - t_start) * 1000
                print(f"[TIMING] First token: {ttft:.0f}ms, Total so far: {total_time:.0f}ms")
                first_token = False
            yield chunk

        # Store metadata for UI
        t_end = time.time()
        yield f"\n\n---\n*Query used: {query_used[:100]}{'...' if len(query_used) > 100 else ''}*"
        if variants_tried:
            yield f"\n*Variants tried: {len(variants_tried)}*"

    return timed_stream()
