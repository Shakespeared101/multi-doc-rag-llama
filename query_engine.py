# query_engine.py

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine

def query_rag(index, query):
    retriever = index.as_retriever()
    response_synthesizer = CompactAndRefine()
    engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
    response = engine.query(query)
    return response.response
