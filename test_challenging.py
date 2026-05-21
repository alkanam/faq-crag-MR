"""
Test a challenging question: retrieval + generation
"""
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import setup_multi_vector_retriever

load_dotenv()

question = "If a company operates a forest management activity that partially involves afforestation on degraded land, but also uses some synthetic fertilizers in other sections of the same project, does the entire activity fail DNSH criteria or can they claim alignment for the compliant portions?"

print(f"QUESTION:\n{question}\n")

# Setup
retriever = setup_multi_vector_retriever()
llm = OllamaLLM(model="qwen2.5:3b")

template = """Based on the context below, answer the question. Only use information from the context. If the answer is not in the context, say "I don't have information about this."

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# ============================================================================
# STEP 1: RETRIEVAL
# ============================================================================
print("=" * 70)
print("STEP 1: RETRIEVAL (What docs are relevant?)")
print("=" * 70)

docs = retriever.invoke(question)
print(f"\nRetrieved {len(docs)} documents:\n")

for i, doc in enumerate(docs, 1):
    metadata = doc.metadata.get("id", "unknown")
    print(f"[Doc {i} - ID: {metadata}]")
    print(doc.page_content[:280])
    print()

# ============================================================================
# STEP 2: GENERATION
# ============================================================================
print("=" * 70)
print("STEP 2: GENERATION (What did the model answer?)")
print("=" * 70)

context = "\n\n".join([doc.page_content for doc in docs])
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})

print(f"\nANSWER:\n{answer}\n")

# ============================================================================
# MANUAL EVALUATION
# ============================================================================
print("=" * 70)
print("MANUAL EVALUATION")
print("=" * 70)

# Check if this is an out-of-scope answer
is_outofscope = any(phrase in answer.lower() for phrase in [
    "don't have information",
    "don't know",
    "i'm not",
    "not available",
    "not mentioned",
    "not provided",
])

print(f"\nScope: {'OUT-OF-SCOPE [OK]' if is_outofscope else 'IN-SCOPE (Generated answer)'}")

# Evaluation observations
print("\nRETRIEVAL QUALITY OBSERVATIONS:")
print("-" * 70)
for i, doc in enumerate(docs, 1):
    content = doc.page_content.lower()
    has_dnsh = "dnsh" in content
    has_afforestation = "afforestation" in content or "forest" in content
    has_fertilizer = "fertilizer" in content or "synthetic" in content
    has_compliance = "compliance" or "compliant" in content

    print(f"Doc {i}: ", end="")
    relevance_signals = [
        "DNSH" if has_dnsh else "",
        "afforestation" if has_afforestation else "",
        "fertilizer" if has_fertilizer else "",
        "compliance" if has_compliance else "",
    ]
    signals = [s for s in relevance_signals if s]
    print(", ".join(signals) if signals else "Generic content")

print("\nGENERATION QUALITY OBSERVATIONS:")
print("-" * 70)
print(f"- Out-of-scope detected: {is_outofscope} (good if question really is out-of-scope)")
print(f"- Answer length: {len(answer)} chars")
print(f"- Contains specific DNSH language: {'Yes' if 'DNSH' in answer else 'No'}")
print(f"- Makes claims about 'partial compliance': {'Yes' if 'partial' in answer.lower() else 'No'}")

print("\nSUMMARY:")
print("-" * 70)
if is_outofscope:
    print("[OK] System correctly identified this as out-of-scope")
    print("[OK] Did not hallucinate an answer")
else:
    print("[!] System attempted to answer")
    print("    Check if answer is grounded in retrieved docs above")
    print("    Does it cite specific policies or make unfounded claims?")
