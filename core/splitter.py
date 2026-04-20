from typing import List

class TextSplitter:
    """Handles splitting large text into smaller chunks with overlap."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """
        Splits text into chunks of chunk_size with overlap.
        """
        chunks = []
        if not text:
            return chunks

        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= text_len:
                break
            
            # Move start back by overlap to ensure continuity
            start = end - self.chunk_overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if start >= end:
                start = end

        return chunks
