from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    PptxReader,
    MarkdownReader,
    HTMLTagReader,
    EpubReader,
    CSVReader,
    RTFReader,
    ImageReader,
    IPYNBReader
)

def load_documents(directory_path="docs"):
    file_extractor = {
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
        ".pptx": PptxReader(),
        ".md": MarkdownReader(),
        ".html": HTMLTagReader(),
        ".epub": EpubReader(),
        ".csv": CSVReader(),
        ".rtf": RTFReader(),
        ".png": ImageReader(),
        ".jpg": ImageReader(),
        ".jpeg": ImageReader(),
        ".ipynb": IPYNBReader()
    }

    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        recursive=True,
        file_extractor=file_extractor
    )
    documents = reader.load_data()
    return documents
