# rag_index.py

from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def build_or_load_index():
    # Load documents
    documents = SimpleDirectoryReader("data").load_data()

    # Configure embedding model
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Set global settings
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=30)

    # Build index
    index = VectorStoreIndex.from_documents(documents)
    return index
