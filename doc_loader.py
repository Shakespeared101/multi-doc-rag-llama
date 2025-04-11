# doc_loader.py
from pathlib import Path
from llama_index.readers.file import PDFReader, DocxReader, PptxReader

def load_documents(folder_path):
    """
    Loads documents from the given folder path.
    Returns a list of Document objects.
    """
    documents = []
    folder = Path(folder_path)
    
    # Process each file in the directory based on file type.
    for file in folder.glob("*"):
        if file.suffix.lower() == ".pdf":
            documents.extend(PDFReader().load_data(str(file)))
        elif file.suffix.lower() == ".docx":
            documents.extend(DocxReader().load_data(str(file)))
        elif file.suffix.lower() == ".pptx":
            documents.extend(PptxReader().load_data(str(file)))
    return documents

if __name__ == '__main__':
    # Quick test
    docs = load_documents("documents")
    print(f"Loaded {len(docs)} documents.")
