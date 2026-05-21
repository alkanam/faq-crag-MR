"""
Generic evaluation runner that works with any retriever implementation.
Compares against baseline and saves results.
"""
import os
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def run_evaluation(retriever, retriever_name: str, use_rewrite_strategy=None):
    """
    Run evaluation on test_questions_simple with given retriever.

    Args:
        retriever: Retriever instance with .invoke(question) method
        retriever_name: Name for this retriever (used in filenames)
        use_rewrite_strategy: Optional function(question) -> str for query rewriting

    Returns:
        results dict with stats
    """
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

    # Load test cases - use absolute path
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_path, "questions", "test_questions_simple.json")

    with open(test_file) as f:
        simple_data = json.load(f)

    test_cases = []
    for q in simple_data["questions"]:
        if "rephrased" in q:
            test_cases.append({
                "question": q["original"],
                "id": q["id"] + "_orig",
                "expected_scope": "in_scope",
            })
            test_cases.append({
                "question": q["rephrased"],
                "id": q["id"] + "_reph",
                "expected_scope": "in_scope",
            })
        else:
            test_cases.append({
                "question": q["question"],
                "id": q.get("id"),
                "expected_scope": "out_of_scope" if q.get("id", "").startswith("oos_") else "in_scope",
            })

    print(f"Running evaluation: {retriever_name}")
    print(f"Using rewrite strategy: {use_rewrite_strategy.__name__ if use_rewrite_strategy else 'None'}")
    print(f"Running on {len(test_cases)} questions...\n")

    results = []

    for i, test in enumerate(test_cases, 1):
        question = test["question"]
        expected_scope = test.get("expected_scope")

        t_start = time.time()

        # Retrieve
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # If using rewrite and answer says "don't know", try rewrite
        no_knowledge_phrases = [
            "don't have information",
            "don't know",
            "i'm not",
            "not available",
            "not mentioned",
            "not provided",
        ]
        has_no_knowledge = any(phrase in answer.lower() for phrase in no_knowledge_phrases)

        attempts = 1
        if has_no_knowledge and use_rewrite_strategy:
            # Rewrite and retry
            rewritten = use_rewrite_strategy(question)
            docs = retriever.invoke(rewritten)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = chain.invoke({"context": context, "question": rewritten})
            attempts = 2

        t_elapsed = time.time() - t_start

        is_outofscope = any(phrase in answer.lower() for phrase in no_knowledge_phrases)

        scope_correct = (
            (expected_scope == "out_of_scope" and is_outofscope) or
            (expected_scope == "in_scope" and not is_outofscope)
        )

        results.append({
            "id": test.get("id"),
            "question": question,
            "answer": answer,
            "expected_scope": expected_scope,
            "is_outofscope": is_outofscope,
            "scope_correct": scope_correct,
            "answer_length": len(answer),
            "elapsed_seconds": t_elapsed,
            "attempts": attempts,
        })

        print(f"[{i}/{len(test_cases)}] {expected_scope}: {scope_correct} - {t_elapsed:.1f}s (attempts: {attempts})")

    # Summary
    print(f"\n{'='*70}")
    correct = sum(1 for r in results if r["scope_correct"])
    print(f"RESULTS ({retriever_name}): {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"{'='*70}")

    in_scope_correct = sum(1 for r in results if r["expected_scope"] == "in_scope" and not r["is_outofscope"])
    in_scope_total = sum(1 for r in results if r["expected_scope"] == "in_scope")
    out_scope_correct = sum(1 for r in results if r["expected_scope"] == "out_of_scope" and r["is_outofscope"])
    out_scope_total = sum(1 for r in results if r["expected_scope"] == "out_of_scope")

    in_scope_acc = 100*in_scope_correct/in_scope_total if in_scope_total > 0 else 0
    out_scope_acc = 100*out_scope_correct/out_scope_total if out_scope_total > 0 else 0

    print(f"In-scope accuracy: {in_scope_correct}/{in_scope_total} ({in_scope_acc:.1f}%)")
    print(f"Out-of-scope accuracy: {out_scope_correct}/{out_scope_total} ({out_scope_acc:.1f}%)")

    avg_latency = sum(r["elapsed_seconds"] for r in results) / len(results)
    print(f"Avg latency: {avg_latency:.2f}s")

    # Save results - use absolute path
    results_dir = Path(base_path) / "experiments" / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    results_file = results_dir / f"results_{retriever_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return {
        "name": retriever_name,
        "total_accuracy": correct / len(results),
        "in_scope_accuracy": in_scope_correct / in_scope_total if in_scope_total > 0 else 0,
        "out_scope_accuracy": out_scope_correct / out_scope_total if out_scope_total > 0 else 0,
        "avg_latency": avg_latency,
        "total_correct": correct,
        "total": len(results),
    }
