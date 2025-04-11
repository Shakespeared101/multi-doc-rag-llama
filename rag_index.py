# rag_index.py
from llama_index import VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding

def build_index(documents, index_path="rag_index"):
    """
    Builds a vector index from provided documents and persists it to disk.
    Returns the built index.
    """
    # Use a small, efficient sentence-transformer model for embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Build index from documents.
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    
    # Persist the storage context
    index.storage_context.persist(index_path)
    return index

if __name__ == '__main__':
    # Quick test: load documents and build index
    from doc_loader import load_documents
    docs = load_documents("documents")
    if docs:
        build_index(docs)
        print("Index built and persisted.")
    else:
        print("No documents loaded.")
