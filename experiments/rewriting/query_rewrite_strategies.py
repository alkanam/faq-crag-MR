"""
Different query rewriting strategies for improving retrieval.
"""
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="qwen2.5:3b")


def step_back_rewrite(question: str) -> str:
    """
    Step-back prompting: ask LLM to identify the underlying concept first,
    then reformulate the question based on that concept.
    """
    step_back_template = """For this question about EU Taxonomy, identify the core concept or principle being asked about.

Question: {question}

Core concept:"""

    step_back_prompt = ChatPromptTemplate.from_template(step_back_template)
    chain = step_back_prompt | llm | StrOutputParser()
    concept = chain.invoke({"question": question})

    # Now rewrite based on concept
    rewrite_template = """Based on this core concept, rephrase the original question in a more general way that focuses on the underlying principle.

Original question: {question}
Core concept: {concept}

Rephrased question:"""

    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question, "concept": concept})

    return rewritten.strip()


def multi_query_rewrite(question: str, num_variants: int = 3) -> list:
    """
    Multi-query: generate multiple reformulations of the question
    to retrieve from different angles.
    """
    multi_query_template = """Generate {num_variants} different reformulations of this question that ask the same thing but from different angles:

Original question: {question}

Reformulations (one per line):"""

    multi_query_prompt = ChatPromptTemplate.from_template(multi_query_template)
    chain = multi_query_prompt | llm | StrOutputParser()

    variants_text = chain.invoke({
        "question": question,
        "num_variants": num_variants
    })

    # Parse variants (each line is one)
    variants = [
        v.strip().lstrip('0123456789. ').strip()
        for v in variants_text.split('\n')
        if v.strip()
    ]

    return variants[:num_variants]


def simple_rewrite(question: str) -> str:
    """
    Simple rewrite: just rephrase using different terminology.
    (This is the default from rag_graph.py)
    """
    template = """Given this question about EU Taxonomy, rephrase it using different terminology while keeping the core meaning.

Original question: {question}

Rephrased question:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})

    return rewritten.strip()
