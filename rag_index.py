import os
import shutil
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import sys

if sys.platform.startswith("win"):
    import types
    sys.modules['resource'] = types.SimpleNamespace()

def build_or_load_index(documents, index_path="index"):
    """
    Builds or loads a vector store index from documents.
    """
    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 256
    Settings.chunk_overlap = 100

    if documents:
        print("[Info] Overwriting existing index with newly uploaded documents...")
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)

        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir=index_path)
        print(f"[Debug] Index files: {os.listdir(index_path)}")
        return index
    else:
        if os.path.exists(index_path) and os.path.isfile(os.path.join(index_path, "docstore.json")):
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            return index
        else:
            raise ValueError("No documents provided and no persisted index exists.")