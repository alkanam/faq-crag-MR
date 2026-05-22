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
    Generate 3 alternative phrasings of a question for retrieval.

    Asks the LLM to rephrase the question from different angles without changing
    the meaning. This helps retrieve documents that might be missed by the original
    phrasing but contain relevant information using different terminology.

    Args:
        question (str): The original user question
        llm: Language model to use for generating variants (OllamaLLM)

    Returns:
        list: List of up to 3 question variants (strings), empty if generation fails
    """
    multi_query_template = """Generate 3 different reformulations of this question that ask the same thing but from different angles:

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

    return variants[:3]  # Return up to 3 variants


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

    # Generate initial answer
    t3 = time.time()
    chain = prompt | llm | StrOutputParser()
    print(f"\n[GENERATION] Initial retrieval: {len(docs)} docs, {len(context)} chars")
    initial_answer = chain.invoke({"context": context, "question": question})

    # Check if answer is a hallucination
    hallucination_phrases = [
        "I don't have information about this"
    ]

    is_hallucinating = any(phrase.lower() in initial_answer.lower() for phrase in hallucination_phrases)
    print(f"[GENERATION] Hallucination detected: {is_hallucinating}")

    # If hallucinating and multi-query enabled, try variants
    if is_hallucinating and use_multi_query:
        try:
            print(f"\n[MULTI-QUERY] Starting variant retrieval...")
            t_variant = time.time()
            variants = generate_multi_query_variants(question, llm)
            variants_tried = variants
            print(f"[MULTI-QUERY] Generated {len(variants)} variants")

            # Re-retrieve with all variants
            all_variant_docs = []
            variant_results = {}

            for i, variant in enumerate(variants):
                variant_docs = retriever.invoke(variant)
                variant_context = "\n\n".join([doc.page_content for doc in variant_docs])
                all_variant_docs.extend(variant_docs)
                variant_results[variant] = variant_docs
                print(f"[MULTI-QUERY] Variant {i+1}: {len(variant_docs)} docs, {len(variant_context)} chars - {variant[:60]}...")

            # Rerank: combine original + variant results, score by relevance
            combined_docs = docs + all_variant_docs

            # Score docs by embedding similarity to original question
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            question_embedding = embeddings.embed_query(question)

            # Simple reranking: score by average word overlap + position
            def score_doc(doc, position):
                q_words = set(question.lower().split())
                doc_words = set(doc.page_content.lower().split())
                overlap = len(q_words & doc_words) / (len(q_words) + 1)
                position_score = 1 / (position + 1)
                return overlap * 0.7 + position_score * 0.3

            scored_docs = [(doc, score_doc(doc, i)) for i, doc in enumerate(combined_docs)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Use top docs for regeneration
            best_docs = [doc for doc, _ in scored_docs[:3]]
            context = "\n\n".join([doc.page_content for doc in best_docs])
            query_used = question

            print(f"[MULTI-QUERY] Reranked {len(combined_docs)} docs, using top 3")
            print(f"[TIMING] Retrieval (multi-query): {(time.time() - t_variant)*1000:.0f}ms")

            # Regenerate answer with reranked context
            print(f"[MULTI-QUERY] Regenerating answer with reranked context...")
            initial_answer = chain.invoke({"context": context, "question": question})
        except Exception as e:
            print(f"[WARNING] Multi-query failed: {e}, using original answer")

    def timed_stream():
        first_token = True
        total_time = 0
        for chunk in initial_answer:
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
