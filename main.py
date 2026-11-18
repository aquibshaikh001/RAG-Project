
from dotenv import load_dotenv
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------
# 1. Load Embedding Model
# --------------------------------

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# --------------------------------
# 2. Connect to Chroma Database
# --------------------------------

DB_DIR = str(Path(__file__).parent / "chroma_db_test")

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
    collection_name="silent_patient"
)

print("Chroma database connected!")

print("Database path:", DB_DIR)

print("Total documents in database:",
    vectorstore._collection.count())

# --------------------------------
# 3. Create Retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# --------------------------------
# 4. Load LLM
# --------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# --------------------------------
# 5. Create Prompt Template
# --------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

            Answer the question using only the
            provided context.

            If the answer is not present in the
            context, say:
            I could not find the answer in the document."""
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}"""
        )
    ]
)

print("\nRAG system created successfully!")
print("Ask questions about The Silent Patient.")
print("Type 0 to exit.\n")

# --------------------------------
# 6. User Query Loop
# --------------------------------

while True:

    query = input("You: ")

    if query.strip() == "0":
        print("Exiting RAG system...")
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: No relevant documents found.")
        continue

    # Display retrieved chunks
    print("\nRetrieved documents:", len(docs))

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}: Page {doc.metadata.get('page', 'Unknown')}")

    # Create context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Generate final prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Generate answer
    response = llm.invoke(final_prompt)

    print("\nAI:", response.content)
    print("\n" + "-" * 50)
