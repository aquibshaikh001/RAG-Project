
# Load PDF
# Split into chunks
# Create embeddings
# Store in Chroma

from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# PDF loading
pdf_path = r"C:\Users\Asus\Desktop\RAG Project\document loaders\The Silent Patient{Michaelides, Alex}{6079872} libgen.li.pdf"

data = PyPDFLoader(pdf_path)

docs = data.load()

print("PDF loaded successfully!")
print("Total pages:", len(docs))

# Text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Total chunks created:", len(chunks))


# Create embeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

from pathlib import Path
from langchain_chroma import Chroma

DB_DIR = str(Path(__file__).parent / "chroma_db_test")

# Connect to Chroma
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="silent_patient"
)

print("Chroma database connected!")

# Process chunks in small batches

batch_size = 5

for i in range(0, len(chunks), batch_size):

    batch = chunks[i:i + batch_size]

    try:
        vectorstore.add_documents(batch)

        print(
            f"Processed {min(i + batch_size, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    except Exception as e:
        print("Error processing batch:", i)
        print(e)
        raise

print("All embeddings created successfully!")
print("PDF stored in Chroma database!")
