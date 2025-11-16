"""
DataOrchestra - A toolkit for cleaning and preparing text datasets for LLM training and finetuning ,
This module provides utilities for:
- Text cleaning and normalization
- fetching content from various sources
- crawling web pages
- extracting text from files
- Filtering meaningful content
- Processing files in bulk
"""

from .cleaner import (  
    normalize_unicode,
    remove_urls_and_emails,
    remove_noise,
    filter_meaningful_words,
    remove_repeated_phrases,
    get_word_count,
    clean_text,
    clean_file,
    process_directory,
)

__version__ = "0.0.1"
