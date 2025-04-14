import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import FaithfulnessEvaluator

def load_query_engine(index_or_path="index"):
    """
    Loads a query engine.
    If a persist directory path (str) is passed as index_or_path, it loads the index from storage.
    If a VectorStoreIndex instance is passed, it creates a query engine directly from it.
    """
    # Configure global settings using open source components:
    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512

    # Determine if the parameter is a directory or an index object.
    if isinstance(index_or_path, str):
        if not os.path.exists(index_or_path):
            raise FileNotFoundError(f"Index directory '{index_or_path}' does not exist.")
        storage_context = StorageContext.from_defaults(persist_dir=index_or_path)
        index = load_index_from_storage(storage_context)
    elif isinstance(index_or_path, VectorStoreIndex):
        index = index_or_path
    else:
        raise ValueError("Parameter must be a persist directory path (str) or a VectorStoreIndex instance.")

    # Create and return a query engine from the index.
    query_engine = index.as_query_engine()
    return query_engine

def query_index(query_engine, query_str: str) -> str:
    """
    Executes a query on the provided query engine and returns the response as a string.
    """
    response = query_engine.query(query_str)
    return str(response)