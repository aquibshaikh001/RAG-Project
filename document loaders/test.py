
import ollama

print("Testing Ollama embedding...")

try:
    response = ollama.embed(
        model="nomic-embed-text",
        input="Who is Theo Faber?"
    )

    print("SUCCESS!")
    print("Embedding dimensions:",
        len(response["embeddings"][0]))

except Exception as e:
    print("FAILED!")
    print("Error:", e)
