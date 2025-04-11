from llama_index.core import load_index_from_storage, StorageContext
from llama_index.llms.ollama import Ollama

def query_index(query, index_path="rag_index"):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    llm = Ollama(model="llama3")
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=10, response_mode="tree_summarize")
    return query_engine.query(query)
