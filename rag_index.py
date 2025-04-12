import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import Document

PERSIST_DIR = "./storage"

def build_or_load_index(documents: List[Document]) -> VectorStoreIndex:
    if os.path.exists(PERSIST_DIR):
        print("[Info] Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("[Info] Building new index...")
        parser = SimpleNodeParser(text_splitter=SentenceSplitter(chunk_size=512))
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index
