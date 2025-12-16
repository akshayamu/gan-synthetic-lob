from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

loader = PyPDFLoader("data/aave_v2.pdf")
docs = loader.load()
texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

db = FAISS.from_documents(texts, HuggingFaceEmbeddings())
db.save_local("faiss_index")
print("Index saved.")
