# query_engine.py
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama

def query_index(query, index_path="rag_index"):
    """
    Loads the index from storage, sets up the query engine with LLaMA 3,
    and returns the response for a given query.
    """
    # Load persisted index context from disk
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    
    # Configure Ollama LLM with LLaMA 3 (customize parameters as needed)
    llm = Ollama(model="llama3")
    
    # Set up query engine; using tree-summarize response mode to combine info from multiple docs.
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=10,
        response_mode="tree_summarize"
    )
    
    response = query_engine.query(query)
    return response

if __name__ == '__main__':
    # Quick test: prompt user for a query and print the response.
    user_query = input("Enter your query: ")
    answer = query_index(user_query)
    print("\nResponse:\n", answer)
