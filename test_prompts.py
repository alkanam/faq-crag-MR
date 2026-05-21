from generate import generate

test_questions = [
    "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?",
    "How should GHG emissions be calculated for technical screening criteria?",
    "What is the best pizza topping?",  # Out of scope
]

print("Testing improved prompt with source citations and scope limits\n")
print("=" * 80)

for question in test_questions:
    print(f"\nQuestion: {question}\n")
    answer = generate(question)
    print(f"Answer: {answer}")
    print("-" * 80)
