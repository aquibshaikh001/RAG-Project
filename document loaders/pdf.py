# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter


# data = PyPDFLoader("C:/Users/Asus/Desktop/RAG Project/document loaders/Agentic_AI_Research_Paper_2026.pdf")
# docs = data.load()

# splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

# chunks = splitter.split_documents(docs)

# print(chunks[0].page_content)

# for i in chunks:
#     print(i.page_content)
#     print()
#     print()
#     print()

#-----------------------------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("C:/Users/Asus/Desktop/RAG Project/document loaders/Agentic_AI_Research_Paper_2026.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)