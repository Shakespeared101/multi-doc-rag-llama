import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        documents.extend(loader.load())
    return documents
