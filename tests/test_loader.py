import pytest
from core.loader import PDFLoader
from unittest.mock import MagicMock, patch

def test_clean_text():
    loader = PDFLoader()
    text = "Hello\x00World   New\nLine"
    cleaned = loader.clean_text(text)
    assert cleaned == "Hello World New Line"

@patch("core.loader.PdfReader")
def test_load_empty_pdf(mock_reader_class):
    mock_reader = MagicMock()
    mock_reader.pages = []
    mock_reader_class.return_value = mock_reader
    
    loader = PDFLoader()
    # Using a fake path since it's mocked
    with patch("core.loader.Path.exists", return_value=True):
        results = loader.load("fake.pdf")
        assert len(results) == 0

def test_loader_file_not_found():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.pdf")
