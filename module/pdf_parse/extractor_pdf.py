import os
import pymupdf
from pymupdf4llm.helpers.get_text_lines import get_text_lines
import pymupdf4llm


def extract_pdf_text(pdf_path):
    """
    Extract text from all pages of a PDF preserving top-to-bottom reading order.
    """
    if not os.path.exists(pdf_path):
        return f"Error: File {pdf_path} not found."

    try:
        doc = pymupdf.open(pdf_path)
        all_text = []

        for page in doc:
            text_lines = get_text_lines(page)
            all_text.append(text_lines)

        doc.close()
        return all_text

    except Exception as e:
        return f"Error processing PDF: {e}"

def extract_pdf_markdown(pdf_path):
    """ Extract markdown from a PDF"""
    md_text = pymupdf4llm.to_markdown(pdf_path)
    # llama_reader = pymupdf4llm.LlamaMarkdownReader()
    # md_text = llama_reader.load_data(pdf_path)
    return md_text

