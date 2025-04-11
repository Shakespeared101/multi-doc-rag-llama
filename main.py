from doc_loader import load_documents
from rag_index import build_index
from query_engine import query_index

# Step 1: Load and parse documents from your chosen folder
docs = load_documents("your_folder_path")  # Replace with the path where your documents are stored
print("Documents loaded.")

# Step 2: Build the index
build_index(docs)
print("Index created.")

# Step 3: Accept a query from the user and display the response
question = input("Ask a question: ")
response = query_index(question)
print("\nResponse:\n", response)
