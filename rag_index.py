from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

def build_or_load_index(documents, index_path="index"):
    # Configure global settings
    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512

    # Check if the index directory exists and contains the necessary files
    if os.path.exists(index_path) and os.path.isfile(os.path.join(index_path, "docstore.json")):
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
    else:
        # Build a new index from documents
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        # Persist the index to disk
        index.storage_context.persist(persist_dir=index_path)

    return index
