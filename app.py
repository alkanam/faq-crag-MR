"""
Streamlit web UI for EU Taxonomy FAQ Assistant.

This module provides an interactive interface for asking questions about the EU Taxonomy
Regulation and Climate Delegated Act. It uses a RAG (Retrieval-Augmented Generation) system
with multi-query retrieval to improve answer accuracy and relevance.

The system:
- Retrieves relevant Q&A pairs from ChromaDB using semantic similarity
- Generates variants of user questions for better coverage
- Streams answers in real-time with performance metrics
- Shows retrieved context documents for transparency
"""
import os
from dotenv import load_dotenv
import streamlit as st
from generate import generate
from retriever import setup_multi_vector_retriever
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


@st.cache_resource
def init_components():
    """
    Initialize and cache RAG components (retriever, LLM, prompt).

    Creates a multi-vector retriever from ChromaDB, initializes the Qwen 2.5 (3B)
    language model via Ollama, and prepares the answer generation prompt template.
    Performs a warmup call to load the model into memory on first access.

    Returns:
        tuple: (retriever, llm, prompt_template) cached for the session
            - retriever: SimpleMultiVectorRetriever instance
            - llm: OllamaLLM with qwen2.5:3b model
            - prompt: ChatPromptTemplate for answer generation
    """
    retriever = setup_multi_vector_retriever()
    llm = OllamaLLM(model="qwen2.5:3b")
    template = """Based on the context below, answer the question. Only use information from the context. If the answer is not in the context, say "I don't have information about this."

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Warmup the model on first load
    with st.spinner("Loading model for first time (this takes ~30s)..."):
        from langchain_core.output_parsers import StrOutputParser
        chain = prompt | llm | StrOutputParser()
        chain.invoke({"context": "test", "question": "test"})

    return retriever, llm, prompt


def main():
    """
    Run the main Streamlit application.

    Configures the page layout, displays the UI components, and handles user
    interactions. When a user submits a question, this function:
    1. Initializes RAG components if not already cached
    2. Retrieves relevant context documents
    3. Generates an answer with multi-query retrieval fallback
    4. Displays the answer, source context, and performance metrics
    """
    st.set_page_config(
        page_title="FAQ RAG System",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 EU Taxonomy FAQ Assistant")
    st.markdown("""
    Ask questions about the EU Taxonomy Regulation and Climate Delegated Act.

    **Enhanced with multi-query retrieval** for better accuracy (72.4% on test set).
    The system generates question variants and retrieves relevant Q&A pairs using Qwen.
    """)

    # Preload model on app startup
    init_components()

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **FAQ RAG System (Optimized)**

        - **Questions indexed**: 328
        - **Model**: Qwen 2.5 (3B)
        - **Vector DB**: ChromaDB
        - **Tracing**: LangSmith

        **Retrieval Strategy:**
        - Multi-query rewriting (generates question variants)
        - Dense + semantic matching
        - Estimated accuracy: **72.4%**

        **Features:**
        - Context-aware answers
        - Scope-aware (refuses out-of-scope)
        - Full traceability via LangSmith
        - Multi-angle retrieval for better coverage
        """)

    # Main interface
    question = st.text_input(
        "Ask your question:",
        placeholder="e.g., Will the technical screening criteria be made stricter over time?"
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and question:
        try:
            retriever, llm, prompt = init_components()

            # Display answer with multi-query enabled
            st.markdown("### Answer")
            answer_container = st.empty()
            full_answer = ""

            for chunk in generate(question, retriever, llm, prompt, use_multi_query=True):
                full_answer += chunk
                answer_container.markdown(full_answer)

            # Expander for context
            with st.expander("📖 View Retrieved Context"):
                docs = retriever.invoke(question)

                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(doc.page_content)
                    st.divider()

            # LangSmith link
            st.info("""
            💡 **Tip:** Check your LangSmith project (`faq-rag`) to see:
            - Original question + generated variants
            - Which query variant was used for retrieval
            - Retrieved documents and scoring
            - Token generation timeline
            """)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif search_btn:
        st.warning("Please enter a question.")



if __name__ == "__main__":
    main()

