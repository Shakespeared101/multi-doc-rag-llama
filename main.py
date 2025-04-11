# main.py
from doc_loader import load_documents
from rag_index import build_index
from query_engine import query_index

def main():
    # Specify the folder that holds your documents.
    documents_path = "documents"  # Adjust this path as needed.
    
    print("Loading documents...")
    docs = load_documents(documents_path)
    if not docs:
        print(f"No documents found in folder: {documents_path}")
        return
    print(f"Loaded {len(docs)} documents.")
    
    print("Building index...")
    build_index(docs)
    print("Index built and persisted.")
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = query_index(question)
        print("\nResponse:\n", response)

if __name__ == '__main__':
    main()
