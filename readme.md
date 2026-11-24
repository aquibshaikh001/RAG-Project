# 📚 Retrieval-Augmented Generation (RAG) Project

A modular, interactive **Retrieval-Augmented Generation (RAG)** application built with Python, **ChromaDB**, and **Streamlit**. This project enables users to upload custom documents, generate dense vector embeddings, store them in a persistent local vector database, and query their knowledge base using natural language to receive context-aware, grounded AI responses.

---

## 🌟 Key Features

* **Interactive Streamlit Web UI:** Clean, user-friendly interface (`app.py`) for uploading files, managing knowledge bases, and chatting with your documents in real time.
* **Persistent Vector Storage:** Uses **ChromaDB** (`chroma_db/` and `chroma_db_test/`) to persist document embeddings locally, avoiding the need to re-index documents across sessions.
* **Modular Ingestion & Retrieval Pipeline:** Clean separation of concerns across document loading (`document loaders/`), database initialization (`create_database.py`), vector store management (`Vector store/`), and core execution logic (`main.py`).
* **Environment-Driven Configuration:** Secure API key and model configuration managed via `.env` variables (`.env.example` provided).
* **Local Document Staging:** Dedicated `uploads/` directory to manage and process user-provided source files.

---

## 🏗️ Repository Architecture & Structure

```text
RAG-Project/
├── .streamlit/             # Streamlit configuration and UI theming settings
├── Vector store/           # Vector database management, indexing, and retrieval helpers
├── chroma_db/              # Primary persistent ChromaDB vector storage directory
├── chroma_db_test/         # Isolated ChromaDB instance for testing and experimentation
├── document loaders/       # Custom loaders and text-splitting/chunking utilities
├── uploads/                # Staging directory for uploaded user documents
├── .env.example            # Template for required environment variables and API keys
├── .gitignore              # Git ignore rules (excludes .env, local caches, etc.)
├── app.py                  # Streamlit frontend application entry point
├── create_database.py      # Standalone script to ingest documents and build the vector store
├── main.py                 # Core RAG pipeline orchestration (retrieval + LLM generation)
└── requirements.txt        # Python package dependencies
```

---

## ⚙️ How It Works (RAG Workflow)

1. **Document Ingestion (`document loaders/` & `uploads/`):** Source documents are loaded, parsed, and split into semantically meaningful text chunks with configurable overlap to preserve context.
2. **Embedding & Indexing (`create_database.py` & `Vector store/`):** Text chunks are converted into high-dimensional vector representations using an embedding model and stored inside **ChromaDB** (`chroma_db/`).
3. **Semantic Retrieval (`main.py`):** When a user submits a query, the question is embedded into the same vector space, and ChromaDB performs a similarity search to retrieve the top-$k$ most relevant document chunks.
4. **Augmented Generation (`app.py` & `main.py`):** The retrieved chunks are injected into a structured prompt alongside the user's query and passed to a Large Language Model (LLM) to generate an accurate, source-grounded response.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.9+** installed on your system
* **Git**
* API Key(s) for your configured LLM / Embedding provider (e.g., OpenAI, HuggingFace, Groq, or Google Gemini — check `.env.example` for exact variable names)

### 1. Clone the Repository

```bash
git clone https://github.com/aquibshaikh001/RAG-Project.git
cd RAG-Project
```

### 2. Create and Activate a Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and populate it with your credentials:

```bash
cp .env.example .env
```

Open `.env` and add your required API keys and configuration parameters.

---

## 💻 Usage

### Option A: Run the Interactive Streamlit Web App (Recommended)

Launch the web interface to upload documents and interact with the RAG assistant via your browser:

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

### Option B: Build the Vector Database via CLI

If you want to pre-populate or rebuild the ChromaDB vector store directly from your source documents before launching the app:

```bash
python create_database.py
```

### Option C: Run Core CLI Pipeline

To test or run the core retrieval-and-generation logic directly from the terminal:

```bash
python main.py
```

---

## 🧪 Testing & Development

* **`chroma_db_test/`:** Use this directory when experimenting with new chunk sizes, embedding models, or document loaders so you do not overwrite your primary production index in `chroma_db/`.
* **`.streamlit/`:** Customize `config.toml` inside `.streamlit/` to adjust server ports, maximum upload file sizes, or UI color themes.

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Language:** Python
* **Environment Management:** `python-dotenv`

---

## 📝 Best Practices & Notes

* **Do not commit `.env`:** Ensure your `.env` file remains listed in `.gitignore` so API keys are never exposed publicly.
* **Re-indexing:** Whenever you change your embedding model or chunking strategy, clear or rebuild `chroma_db/` using `create_database.py` to prevent vector dimension mismatches.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.