import os
import re
import hashlib
import tempfile
import urllib.parse
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader, PdfWriter
from config.settings import settings


def extract_single_page_as_pdf(pdf_path: str, page_num: int) -> bytes:
    """Extracts a single page from a PDF and returns it as bytes."""
    reader = PdfReader(pdf_path)
    if page_num < 1 or page_num > len(reader.pages):
        raise ValueError(f"Page {page_num} out of range (1-{len(reader.pages)})")
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num - 1])
    buf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        writer.write(buf)
        buf.seek(0)
        return buf.read()
    finally:
        buf.close()
        os.unlink(buf.name)


def split_pdf_to_pages(pdf_path: str, output_dir: str) -> List[str]:
    """Splits a PDF into individual page-wise files.

    Args:
        pdf_path: Path to the source PDF
        output_dir: Directory to write page-wise PDFs

    Returns:
        List of written file paths
    """
    path = Path(pdf_path)
    stem = path.stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(path)
    written = []
    for i, page in enumerate(reader.pages, start=1):
        page_filename = f"{stem}_page_{i}.pdf"
        page_path = output_path / page_filename
        writer = PdfWriter()
        writer.add_page(page)
        with open(page_path, "wb") as f:
            writer.write(f)
        written.append(str(page_path))
    return written


def get_pagewise_url(filename: str, page_num: int) -> str:
    """Constructs a URL pointing to the page-wise PDF file.

    Example:
        get_pagewise_url("v1_rag_cnc.pdf", 5)
        -> "http://localhost:8003/v1_rag_cnc_page_5.pdf"
    """
    stem = Path(filename).stem
    page_filename = f"{stem}_page_{page_num}.pdf"
    encoded = urllib.parse.quote(page_filename)
    return f"{settings.cleaned_server_url}/{encoded}"


def get_pagewise_filepath(filename: str, page_num: int) -> str:
    """Returns the local filesystem path for a page-wise PDF.

    Example:
        get_pagewise_filepath("v1_rag_cnc.pdf", 5)
        -> "data/cleaned/v1_rag_cnc_page_5.pdf"
    """
    stem = Path(filename).stem
    page_filename = f"{stem}_page_{page_num}.pdf"
    return str(Path(settings.cleaned_dir) / page_filename)


def annotate_pdf_page(
    pdf_bytes: bytes,
    query: str,
    page_num: int = None,
) -> bytes:
    """Annotates a PDF page using annotateai.

    Args:
        pdf_bytes: Raw PDF bytes (single page)
        query: User's question to derive annotation keywords
        page_num: Optional page number for metadata

    Returns:
        Annotated PDF bytes
    """
    model = settings.annotation_model or settings.ollama_llm_model
    if model and not model.startswith("ollama/"):
        model = f"ollama/{model}"
    keywords = _extract_keywords(query)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.pdf")
        output_path = os.path.join(tmpdir, "annotated.pdf")
        with open(input_path, "wb") as f:
            f.write(pdf_bytes)

        from annotateai import Annotate
        annotate = Annotate(model)
        annotate(input_path, output=output_path, keywords=keywords, progress=False)

        annotated = Path(output_path).read_bytes()
        if not annotated:
            return pdf_bytes
        return annotated


def annotate_page_from_file(
    filepath: str,
    page_num: int,
    query: str,
) -> bytes:
    """Extracts a single page from a PDF file and annotates it."""
    page_bytes = extract_single_page_as_pdf(filepath, page_num)
    return annotate_pdf_page(page_bytes, query, page_num)


def _extract_keywords(query: str) -> List[str]:
    """Extracts meaningful keywords from the user's query.

    Strips stopwords and short words. Always returns user's words —
    never falls back to LLM auto-generation.
    """
    stopwords = {
        "what", "why", "how", "does", "the", "is", "are", "can", "do",
        "would", "could", "should", "will", "has", "have", "been", "was",
        "this", "that", "with", "for", "and", "not", "but", "you", "your",
        "its", "all", "tell", "give", "list", "show", "find", "need", "help",
    }
    words = []
    for w in re.split(r"[?.,!;:\s]+", query):
        w = w.strip().lower()
        if w and w not in stopwords and len(w) > 2:
            words.append(w)

    return words if words else query.strip().lower().split()[:3]
