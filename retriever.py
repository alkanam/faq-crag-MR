"""
Document retrieval using ChromaDB and semantic embeddings.

This module provides a multi-vector retriever that searches for questions in a
vector database and returns the corresponding Q&A pairs. It uses HuggingFace
embeddings (all-MiniLM-L6-v2) for semantic similarity matching.

The retriever stores questions and answers separately to support:
- Searching by question embeddings (better recall for question-like inputs)
- Returning complete Q&A pairs as context for the LLM
- Efficient updates and indexing

Architecture:
- Questions collection: indexed and searchable by semantic similarity
- Answers collection: keyed by question ID, retrieved in matching pairs
"""
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SimpleMultiVectorRetriever(Runnable):
    """
    Multi-vector retriever for semantic Q&A search in ChromaDB.

    Searches the questions collection by semantic similarity using HuggingFace
    embeddings, then retrieves the corresponding answers. Returns complete Q&A
    pairs as LangChain Document objects for use in RAG pipelines.

    Attributes:
        q_vectorstore (Chroma): Langchain wrapper around questions collection
        q_collection: Raw ChromaDB collection for questions
        a_collection: Raw ChromaDB collection for answers
        embeddings (HuggingFaceEmbeddings): Embedding model (all-MiniLM-L6-v2)
    """

    def __init__(self):
        """
        Initialize the retriever with ChromaDB and HuggingFace embeddings.

        Loads persistent ChromaDB data from chroma_db/ directory in the project
        root. Initializes the all-MiniLM-L6-v2 embedding model for semantic
        similarity matching. Expects two collections to exist: 'questions' and
        'answers', both keyed by the same question ID.

        Raises:
            RuntimeError: If ChromaDB collections are not found
        """
        import os
        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.q_vectorstore = Chroma(
            client=client,
            collection_name="questions",
            embedding_function=embeddings
        )

        self.q_collection = client.get_collection("questions")
        self.a_collection = client.get_collection("answers")
        self.embeddings = embeddings

    def invoke(self, input, config=None):
        """
        Search for documents matching the input question.

        Embeds the input question and performs semantic similarity search in the
        questions collection. Returns the top-k matching Q&A pairs as Document
        objects with source IDs in metadata.

        Args:
            input (str or dict): User's question
                - If str: used directly as the question
                - If dict: extracts 'question' key
            config (dict, optional): Runnable config (unused)

        Returns:
            list[Document]: Top-3 matching Q&A pairs formatted as:
                [Source: {id}]\nQ: {question}\n\nA: {answer}
                Each Document includes the question ID in metadata for tracing.
        """
        question = input if isinstance(input, str) else input.get("question", input)
        k = 3

        # Embed with the same model used for indexing
        query_embedding = self.embeddings.embed_query(question)

        results = self.q_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # For each question result, fetch the corresponding answer
        qa_docs = []
        for i, qa_id in enumerate(results["ids"][0]):
            q_text = results["documents"][0][i]
            a_result = self.a_collection.get(ids=[qa_id], include=["documents"])

            if a_result["documents"]:
                a_text = a_result["documents"][0]
                qa_docs.append(Document(
                    page_content=f"[Source: {qa_id}]\nQ: {q_text}\n\nA: {a_text}",
                    metadata={"id": qa_id}
                ))

        return qa_docs


def setup_multi_vector_retriever():
    """
    Factory function to create a configured multi-vector retriever.

    Returns:
        SimpleMultiVectorRetriever: Initialized retriever connected to ChromaDB
            with pre-indexed FAQ Q&A pairs. Ready for use in RAG pipelines.
    """
    return SimpleMultiVectorRetriever()

