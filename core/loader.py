import re
from typing import List, Tuple
from pypdf import PdfReader
from pathlib import Path

class PDFLoader:
    """Handles loading and cleaning text from PDF files."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans extracted text by removing null characters and normalizing whitespace."""
        text = text.replace("\x00", " ")
        return re.sub(r"\s+", " ", text).strip()

    def load(self, pdf_path: str) -> List[Tuple[str, dict]]:
        """
        Loads a PDF and returns a list of (page_text, metadata) tuples.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        reader = PdfReader(path)
        results = []

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text:
                continue

            metadata = {
                "filename": str(path.absolute()),
                "page": page_num,
                "source": "pdf",
            }
            results.append((cleaned_text, metadata))

        return results
