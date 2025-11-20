"""
DataOrchestra - A comprehensive data processing and extraction toolkit.

This package provides tools for:
- Web scraping and crawling
- Document processing (PDF, DOCX, etc.)
- Text cleaning and normalization
- Data extraction and transformation
"""

from .core import (
    BaseProcessor,
    FileProcessor,
    ProcessResult,
    FileProcessingResult,
    get_config,
    set_config,
    reset_config,
)
from .extract import (
    process_pdf,
    process_docx,
    process_web,
)
from .utils import (
    clean_text,
    normalize_text,
    split_text_by_tokens,
)

__version__ = "0.1.0"
__all__ = [
    "BaseProcessor",
    "FileProcessor",
    "ProcessResult",
    "FileProcessingResult",
    "get_config",
    "set_config",
    "reset_config",
    "process_pdf",
    "process_docx",
    "process_web",
    "clean_text",
    "normalize_text",
    "split_text_by_tokens",
]