from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext
import os

def load_query_engine(index_path="index"):
    # Configure global settings
    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512

    # Load the index from disk
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index directory '{index_path}' does not exist.")

    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    # Create a query engine from the index
    query_engine = index.as_query_engine()
    return query_engine

def query_index(query_engine, query_str):
    response = query_engine.query(query_str)
    return str(response)
