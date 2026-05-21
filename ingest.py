import re
import chromadb
from sentence_transformers import SentenceTransformer


def parse_faqs(md_path: str):
    """Extract questions and answers from markdown."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'^###\s+(.+?)$\n\n(.*?)(?=\n###\s+|$)'
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

    questions = []
    answers = []

    for idx, match in enumerate(matches, 1):
        qa_id = f"qa_{idx:05d}"
        question = match.group(1).strip()
        answer = match.group(2).strip()

        if question and answer:
            questions.append({"id": qa_id, "text": question})
            answers.append({"id": qa_id, "text": answer})

    return questions, answers


def main():
    # Parse
    md_file = "./doc/taxonomy_faqs_cleaned.md"
    print(f"Parsing {md_file}...")
    questions, answers = parse_faqs(md_file)
    print(f"Found {len(questions)} Q&A pairs\n")

    # Initialize
    client = chromadb.PersistentClient(path="./chroma_db")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Store questions
    print("Storing questions...")
    q_collection = client.get_or_create_collection(name="questions")
    for q in questions:
        embedding = model.encode(q["text"]).tolist()
        q_collection.upsert(
            ids=[q["id"]],
            embeddings=[embedding],
            documents=[q["text"]]
        )

    # Store answers
    print("Storing answers...")
    a_collection = client.get_or_create_collection(name="answers")
    for a in answers:
        embedding = model.encode(a["text"]).tolist()
        a_collection.upsert(
            ids=[a["id"]],
            embeddings=[embedding],
            documents=[a["text"]]
        )

    print(f"\nDone:")
    print(f"  Questions: {q_collection.count()}")
    print(f"  Answers: {a_collection.count()}")


if __name__ == "__main__":
    main()
