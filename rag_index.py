from llama_index.core import VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding

def build_index(documents, index_path="rag_index"):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    vector_index.storage_context.persist(index_path)
    return vector_index
