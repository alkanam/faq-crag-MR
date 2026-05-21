import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load ChromaDB directly
client = chromadb.PersistentClient(path="./chroma_db")
q_collection = client.get_collection("questions")

print(f"Questions collection count: {q_collection.count()}")
print(f"Total IDs: {len(q_collection.get()['ids'])}\n")

# Try vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
q_vectorstore = Chroma(
    client=client,
    collection_name="questions",
    embedding_function=embeddings
)

question = "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?"

print(f"Vectorstore count: {q_vectorstore._collection.count()}")

results = q_vectorstore.similarity_search(question, k=3)
print(f"\nSimilarity search results: {len(results)}")

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Score: {result.metadata}")
    print(f"Content: {result.page_content[:200]}")
