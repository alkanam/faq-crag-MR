"""
Hybrid retriever combining dense (vector similarity) and sparse (BM25) retrieval.
"""
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class HybridRetriever(Runnable):
    """Hybrid retriever: combines dense vector search + BM25 sparse search."""

    def __init__(self, alpha=0.5):
        """
        alpha: weight for dense retrieval (1-alpha for BM25).
        0.5 = equal weight. 0.7 = favor dense. 0.3 = favor BM25.
        """
        import os
        self.alpha = alpha

        # Dense retrieval setup
        chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "chroma_db"
        )
        client = chromadb.PersistentClient(path=chroma_path)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.dense_vectorstore = Chroma(
            client=client,
            collection_name="questions",
            embedding_function=embeddings
        )

        self.q_collection = client.get_collection("questions")
        self.a_collection = client.get_collection("answers")
        self.embeddings = embeddings

        # Sparse retrieval setup (BM25)
        # Get all documents for BM25
        all_qa_ids = self.q_collection.get()["ids"]
        bm25_docs = []

        for qa_id in all_qa_ids:
            q_text = self.q_collection.get(ids=[qa_id])["documents"][0]
            a_result = self.a_collection.get(ids=[qa_id], include=["documents"])
            if a_result["documents"]:
                a_text = a_result["documents"][0]
                bm25_docs.append(Document(
                    page_content=f"Q: {q_text}\n\nA: {a_text}",
                    metadata={"id": qa_id}
                ))

        self.bm25 = BM25Retriever.from_documents(bm25_docs)

    def invoke(self, input, config=None):
        """Retrieve using both dense and sparse methods, combine results."""
        question = input if isinstance(input, str) else input.get("question", input)
        k = 3

        # Dense retrieval
        dense_results = self.dense_vectorstore.similarity_search(question, k=k)

        # Sparse retrieval (BM25)
        sparse_results = self.bm25.invoke(question)[:k]

        # Combine results, preferring dense but including sparse
        seen_ids = set()
        combined = []

        for doc in dense_results:
            doc_id = doc.metadata.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append(doc)

        for doc in sparse_results:
            doc_id = doc.metadata.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append(doc)

        # Format as Q&A pairs
        qa_docs = []
        for doc in combined[:k]:
            qa_id = doc.metadata.get("id")
            if qa_id:
                q_result = self.q_collection.get(ids=[qa_id], include=["documents"])
                a_result = self.a_collection.get(ids=[qa_id], include=["documents"])

                if q_result["documents"] and a_result["documents"]:
                    q_text = q_result["documents"][0]
                    a_text = a_result["documents"][0]
                    qa_docs.append(Document(
                        page_content=f"[Source: {qa_id}]\nQ: {q_text}\n\nA: {a_text}",
                        metadata={"id": qa_id}
                    ))

        return qa_docs
