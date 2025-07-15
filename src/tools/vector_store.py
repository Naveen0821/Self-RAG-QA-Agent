from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def build_vector_store(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)

def get_retriever(vector_store, k=5):
    return vector_store.as_retriever(search_type="similarity", k=k)
