from langchain_community.vectorstores import Chroma 
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document 

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for Data Analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = OllamaEmbeddings( model = "nomic-embed-text")

# vectors = embeddings.embed_documents(chunks)

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory= "chroma-db"
)

result = vectorstore.similarity_search("What is used for data anlysis?",k=2)

for i in result:
    print(i.page_content)
    print(i.metadata)
    
retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)