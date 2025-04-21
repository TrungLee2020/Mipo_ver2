from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

TOKENIZE_MODEL = os.getenv("TOKENIZE_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def chunking_documents(file_path, chunk_size=512, chunk_overlap=0):
    """Split documents into chunks.
    Chunks to 512 tokens with langchain RecursiveCharacter to vector embedding store"""

    # Tokenize with Phobert-base
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZE_MODEL)

    # Markdown splitter
    headers_to_split_on = [(f"{'#' * i}", f"{'#' * i}") for i in range(1, 7)]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # text split from RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(tokenizer.encode(x)),
        separators=["\n\n", "\n", " ", ""]
    )

    # Đọc file
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Phân đoạn markdown trước
    md_header_splits = markdown_splitter.split_text(text_content)

    # Tiếp tục phân đoạn dựa trên kích thước token
    from langchain.schema import Document
    documents = [Document(page_content=split) for split in md_header_splits]
    chunks = text_splitter.split_documents(documents)

    # Thêm metadata
    for chunk in chunks:
        chunk.metadata["source"] = file_path
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    return chunks


def process_document_with_metadata(file_path, metadata_df, chunk_size=512, chunk_overlap=50):
    """Process a document with its metadata and split into chunks"""

    # Lọc metadata cho file hiện tại
    relative_path = os.path.relpath(file_path, base_folder)
    doc_metadata = metadata_df[metadata_df['filepath'] == relative_path]

    if len(doc_metadata) == 0:
        print(f"Warning: No metadata found for {file_path}")
        return []

    doc_metadata = doc_metadata.iloc[0].to_dict()

    # Phân đoạn tài liệu
    chunks = chunking_documents(file_path, chunk_size, chunk_overlap)

    # Thêm metadata vào từng chunk
    for i, chunk in enumerate(chunks):
        # Thêm metadata từ tài liệu gốc
        chunk.metadata.update({
            "document_id": doc_metadata['document_id'],
            "chunk_index": i,
            "title": doc_metadata['title'],
            "author": doc_metadata['author'],
            "source": doc_metadata['source'],
            "category": doc_metadata['category'],
            "date": doc_metadata['date'],
            "parent_document": file_path,
        })

    return chunks

