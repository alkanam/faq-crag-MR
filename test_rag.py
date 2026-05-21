from generate import generate

question = "Will the technical screening criteria set out in the Climate Delegated Act be made stricter and updated over time?"

print(f"Question: {question}\n")
print("Generating answer...\n")

answer = generate(question)
print(f"Answer:\n{answer}")
