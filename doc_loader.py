from pathlib import Path
from llama_index.readers.file import PDFReader, DocxReader, PptxReader

def load_documents(folder_path):
    documents = []
    for file in Path(folder_path).glob("*"):
        if file.suffix == ".pdf":
            documents.extend(PDFReader().load_data(file))
        elif file.suffix == ".docx":
            documents.extend(DocxReader().load_data(file))
        elif file.suffix == ".pptx":
            documents.extend(PptxReader().load_data(file))
    return documents
