
import streamlit as st
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------
# 1. Basic Configuration
# --------------------------------

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------
# 2. Custom CSS
# --------------------------------


st.markdown("""
<style>

/* -------------------------------
   MAIN APPLICATION BACKGROUND
-------------------------------- */

.stApp {
    background: #F5F7FC;
}

[data-testid="stAppViewContainer"] {
    background: #F5F7FC;
}

.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

/* -------------------------------
   MAIN PAGE TEXT
-------------------------------- */

.main [data-testid="stMarkdownContainer"] {
    color: #202B46 !important;
}

.main [data-testid="stMarkdownContainer"] p {
    color: #526078 !important;
    font-size: 15px;
    line-height: 1.7;
}

.main h1, .main h2, .main h3,
.main h4, .main h5 {
    color: #202B46 !important;
    font-weight: 700;
}

.main label,
.main [data-testid="stWidgetLabel"] {
    color: #263653 !important;
}

/* -------------------------------
   HEADER
-------------------------------- */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #24345B !important;
    letter-spacing: -1px;
}

.subtitle {
    color: #65738B !important;
    font-size: 16px;
    margin-top: 8px;
}

/* -------------------------------
   SIDEBAR
-------------------------------- */

section[data-testid="stSidebar"] {
    background: #17233D;
}

section[data-testid="stSidebar"] * {
    color: #F1F5FF;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D0D9EA !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] input {
    color: #202B46 !important;
}

/* -------------------------------
   BUTTONS
-------------------------------- */

.stButton > button {
    background: linear-gradient(
        135deg,
        #536DFE,
        #7657E8
    );

    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 15px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #4054D6;
    color: #FFFFFF !important;
    border: none;
    box-shadow: 0 4px 14px rgba(83,109,254,0.25);
}

/* -------------------------------
   FILE UPLOADER
-------------------------------- */

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #AAB8D2;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: #F8FAFF;
    border-radius: 10px;
}

/* -------------------------------
   ALERT MESSAGES
-------------------------------- */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* -------------------------------
   CHAT MESSAGES
-------------------------------- */

[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E5EAF3;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 0 3px 10px rgba(30,50,90,0.04);
}

[data-testid="stChatMessage"] p {
    color: #263653 !important;
}

/* -------------------------------
   CHAT INPUT
-------------------------------- */

[data-testid="stChatInput"] {
    background: #FFFFFF;
    border: 1px solid #DCE3EF;
    border-radius: 14px;
}

[data-testid="stChatInput"] textarea {
    color: #202B46 !important;
}

/* -------------------------------
   CODE BLOCKS
-------------------------------- */

.stCodeBlock {
    border-radius: 10px;
}

/* -------------------------------
   DIVIDERS
-------------------------------- */

hr {
    border-color: #E0E6F0;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# 3. Load Models
# --------------------------------

@st.cache_resource
def load_embedding_model():

    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )


@st.cache_resource
def load_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


embedding_model = load_embedding_model()
llm = load_llm()

# --------------------------------
# 4. Prompt Template
# --------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI document assistant.

        Answer questions using only the provided context.

        If the answer is not present in the context,
        say:
        I could not find the answer in the document.
        """
    ),
    (
        "human",
        """Context:
        {context}

        Question:
        {question}"""
    )
])

# --------------------------------
# 5. Session State
# --------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

# --------------------------------
# 6. Header
# --------------------------------

st.markdown(
    '<div class="main-title">🧠 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent AI assistant. Upload a document '
    'and start asking questions.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# --------------------------------
# 7. Sidebar - Document Upload
# --------------------------------

with st.sidebar:

    st.title("📁 Document Manager")

    st.write("Upload your PDF document below.")

    uploaded_file = st.file_uploader(
        "Choose your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(f"Selected: {uploaded_file.name}")

        if st.button("🚀 Process Document", use_container_width=True):

            try:

                with st.spinner("Processing your document..."):

                    # Save uploaded PDF
                    file_bytes = uploaded_file.getvalue()

                    file_hash = hashlib.sha256(
                        file_bytes
                    ).hexdigest()

                    upload_dir = Path("uploads")
                    upload_dir.mkdir(exist_ok=True)

                    pdf_path = upload_dir / (
                        file_hash + ".pdf"
                    )

                    with open(pdf_path, "wb") as file:
                        file.write(file_bytes)

                    # Load PDF
                    loader = PyPDFLoader(str(pdf_path))
                    documents = loader.load()

                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(documents)

                    if not chunks:
                        st.error("No readable text found in this PDF.")
                        st.stop()

                    # Create unique Chroma collection
                    collection_name = "doc_" + file_hash[:20]

                    db_dir = str(
                        Path("chroma_db_test").resolve()
                    )

                    vectorstore = Chroma(
                        persist_directory=db_dir,
                        embedding_function=embedding_model,
                        collection_name=collection_name
                    )

                    # Avoid incomplete old collections
                    current_count = vectorstore._collection.count()

                    if current_count != len(chunks):

                        vectorstore.delete_collection()

                        vectorstore = Chroma(
                            persist_directory=db_dir,
                            embedding_function=embedding_model,
                            collection_name=collection_name
                        )

                        # Create embeddings in batches
                        batch_size = 5

                        progress = st.progress(0)

                        for i in range(
                            0, len(chunks), batch_size
                        ):

                            batch = chunks[
                                i:i + batch_size
                            ]

                            vectorstore.add_documents(batch)

                            completed = min(
                                i + batch_size,
                                len(chunks)
                            )

                            progress.progress(
                                completed / len(chunks)
                            )

                    # Save in session
                    st.session_state.vectorstore = vectorstore

                    st.session_state.document_name = uploaded_file.name

                    st.session_state.processed_hash = file_hash

                    st.session_state.chat_history = []

                st.success("Document processed successfully!")

            except Exception as e:

                st.error(f"Error: {str(e)}")

    st.divider()

    st.subheader("⚙️ Model Information")

    st.caption("Embedding Model")
    st.code("nomic-embed-text")

    st.caption("Vector Database")
    st.code("ChromaDB")

    st.caption("LLM")
    st.code("Groq - gpt-oss-20b")

# --------------------------------
# 8. Main Chat Interface
# --------------------------------

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a PDF document from the sidebar "
        "and click Process Document to begin."
    )

    
    st.markdown("## ✨ What can you do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
    <div style="
        background:white;
        padding:25px;
        border-radius:16px;
        border:1px solid #E1E7F0;
        min-height:170px;
        box-shadow:0 4px 15px rgba(30,50,90,0.05);
    ">
        <div style="font-size:35px;">📄</div>
        <h3 style="color:#24345B;">Upload Documents</h3>
        <p style="color:#65738B;">
        Upload your own PDF documents and
        let AI understand their content.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div style="
        background:white;
        padding:25px;
        border-radius:16px;
        border:1px solid #E1E7F0;
        min-height:170px;
        box-shadow:0 4px 15px rgba(30,50,90,0.05);
    ">
        <div style="font-size:35px;">🔍</div>
        <h3 style="color:#24345B;">Find Information</h3>
        <p style="color:#65738B;">
        Search through your document
        and find relevant information quickly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
    <div style="
        background:white;
        padding:25px;
        border-radius:16px;
        border:1px solid #E1E7F0;
        min-height:170px;
        box-shadow:0 4px 15px rgba(30,50,90,0.05);
    ">
        <div style="font-size:35px;">💬</div>
        <h3 style="color:#24345B;">Ask Questions</h3>
        <p style="color:#65738B;">
        Ask questions in natural language
        and receive AI-generated answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:

    st.success(
        f"Active Document: {st.session_state.document_name}"
    )

    st.markdown("## 💬 Chat with your document")

    # Display previous conversation
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):
                st.caption(
                    "Sources: " + message["sources"]
                )

    # User input
    query = st.chat_input(
        "Ask anything about your document..."
    )

    if query:

        # Display user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:

                    retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 4,
                                "fetch_k": 10,
                                "lambda_mult": 0.5
                            }
                        )
                    )

                    # Retrieve relevant chunks
                    docs = retriever.invoke(query)

                    if not docs:

                        answer = (
                            "I could not find the answer "
                            "in the document."
                        )

                        sources = ""

                    else:

                        # Create context
                        context = "\n\n".join(
                            doc.page_content
                            for doc in docs
                        )

                        # Generate prompt
                        final_prompt = prompt.invoke({
                            "context": context,
                            "question": query
                        })

                        # Get LLM response
                        response = llm.invoke(final_prompt)

                        answer = response.content

                        # Get source pages
                        pages = sorted(set(
                            doc.metadata.get("page", "Unknown")
                            for doc in docs
                        ))

                        sources = ", ".join(
                            f"Page {page + 1}"
                            if isinstance(page, int)
                            else f"Page {page}"
                            for page in pages
                        )

                    st.markdown(answer)

                    if sources:
                        st.caption("📚 Sources: " + sources)

                    # Save assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(f"Something went wrong: {str(e)}")

