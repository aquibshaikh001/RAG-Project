
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
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Store in Chroma
DB_DIR = str(Path(__file__).parent / "chroma_db")

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
    collection_name="silent_patient"
)

# Process 5 chunks at a time
batch_size = 5

for i in range(0, len(chunks), batch_size):

    batch = chunks[i:i + batch_size]

    vectorstore.add_documents(batch)

    print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

print("All embeddings stored successfully!")

print("Embeddings created successfully!")
print("Documents stored in Chroma database!")
