import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_pdf_chunks(file_path: str, chunk_size=750, overlap=100) -> list[Document]:
    doc = fitz.open(file_path)
    pages = [page.get_text() for page in doc]
    text = "\n\n".join(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]
