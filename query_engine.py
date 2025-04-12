from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex

def query_rag(index: VectorStoreIndex, query: str) -> str:
    try:
        llm = Ollama(model="llama3")  # Ensure `ollama run llama3` is active
        engine = RetrieverQueryEngine.from_args(index.as_retriever(), llm=llm)
        response = engine.query(query)
        return str(response)
    except Exception as e:
        return f"[Error] Query failed: {e}"
