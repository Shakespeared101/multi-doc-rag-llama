from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext
import os

def load_query_engine(index_path="index"):
    # Initialize the embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Initialize the LLM
    llm = Ollama(model="llama3.1", request_timeout=120.0)

    # Create a service context
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, chunk_size=512
    )

    # Load the index from disk
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index directory '{index_path}' does not exist.")

    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context, service_context=service_context)

    # Create a query engine from the index
    query_engine = index.as_query_engine()
    return query_engine

def query_index(query_engine, query_str):
    response = query_engine.query(query_str)
    return str(response)
