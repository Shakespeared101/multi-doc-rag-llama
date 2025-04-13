from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import CompactAndRefine

def get_query_engine(index):
    llm = Ollama(model="llama3")
    retriever = index.as_retriever(similarity_top_k=5)
    response_synthesizer = CompactAndRefine(llm=llm)
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
