from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

def build_or_load_index(documents, index_path="index"):
    # Initialize the embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Initialize the LLM
    llm = Ollama(model="llama3.1", request_timeout=120.0)

    # Create a service context
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, chunk_size=512
    )

    # Check if the index already exists
    if os.path.exists(index_path):
        print("[Info] Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context, service_context=service_context)
    else:
        print("[Info] Building new index...")
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=index_path)

    return index
