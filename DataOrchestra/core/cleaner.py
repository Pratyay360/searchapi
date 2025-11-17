"""
Advanced text cleaning and processing utilities for LLM datasets.
"""
from __future__ import annotations
import os
import re
import unicodedata
from pathlib import Path
from typing import Pattern, Set
import time

from docx import Document

from .exceptions import FileProcessingError, ValidationError
from .base import FileProcessor, FileProcessingResult
from .config import get_config, ProcessingConfig
from .logging_utils import get_logger, LogContext


class TextCleaner(FileProcessor):
    """Advanced text cleaning processor with configuration support."""
    
    def __init__(self):
        super().__init__(
            name="TextCleaner",
            supported_extensions={'.txt', '.md', '.json', '.csv', '.html'}
        )
        self.config = get_config().processing
        self._noise_patterns: list[Pattern[str]] = []
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        self._noise_patterns = [
            re.compile(r"https?://\S*"),  # URLs
            re.compile(r"ftp://\S*"),     # FTP URLs
            re.compile(r"www\.\S*"),      # WWW URLs
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Emails
            re.compile(r"\S+\.(com|org|net|edu|gov|io|co|uk)\S*"),  # Domain-like
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),  # IP addresses
            re.compile(r"[\x00-\x1F\x7F-\x9F]"),  # Control characters
            re.compile(r"([!?.,])\1+"),  # Repeated punctuation
            re.compile(r"\s+"),  # Multiple whitespace
        ]
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize("NFKC", text)
    
    def remove_urls_and_emails(self, text: str) -> str:
        """Remove URLs and email addresses from text."""
        for pattern in self._noise_patterns[:4]:  # URL/email patterns
            text = pattern.sub("", text)
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove general noise from text."""
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        
        # Apply noise patterns
        for pattern in self._noise_patterns:
            text = pattern.sub(" ", text)
        
        # Remove standalone numbers
        text = re.sub(r"\b\d+\b", " ", text)
        
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def filter_meaningful_words(self, text: str) -> str:
        """Filter out noise words and keep meaningful content."""
        words = text.split()
        filtered_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Skip noise words
            if word_lower in self.config.noise_words:
                continue
            
            # Skip pure numbers
            if re.match(r"^\d+$", word):
                continue
            
            # Skip short alphanumeric mixed words
            if re.match(r"^\w*\d\w*$", word) and len(word) <= 4:
                continue
            
            # Skip words with repeated characters
            if re.search(r"(.)\1{2,}", word):
                continue
            
            # Check minimum length
            if len(word) >= self.config.min_word_length:
                filtered_words.append(word)
        
        return " ".join(filtered_words)
    
    def remove_repeated_phrases(self, text: str) -> str:
        """Remove repeated sentences and phrases."""
        sentences = re.split(r"[.!?]+", text)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            clean_sentence = sentence.strip().lower()
            
            # Only process substantial sentences
            if (
                clean_sentence
                and len(clean_sentence) > self.config.min_sentence_length
                and clean_sentence not in seen_sentences
            ):
                seen_sentences.add(clean_sentence)
                unique_sentences.append(sentence.strip())
        
        result = ". ".join(unique_sentences).strip()
        
        # Ensure proper sentence ending
        if result and result[-1] not in ".!?":
            result += "."
        
        return result
    
    def clean_text(self, text: str) -> str:
        """Apply full text cleaning pipeline."""
        with LogContext(self.logger, "clean_text", text_length=len(text)):
            # Step 1: Normalize Unicode
            text = self.normalize_unicode(text)
            
            # Step 2: Remove noise
            text = self.remove_noise(text)
            
            # Step 3: Filter meaningful words
            text = self.filter_meaningful_words(text)
            
            # Step 4: Remove repeated phrases
            text = self.remove_repeated_phrases(text)
            
            # Step 5: Final normalization
            text = text.lower()
            text = re.sub(r"\s+", " ", text).strip()
            
            return text
    
    def process_file(self, file_path: Path) -> FileProcessingResult:
        """Process a text file with comprehensive cleaning."""
        try:
            # Try different encodings
            text = None
            for encoding in self.config.encoding_fallbacks:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise FileProcessingError(f"Cannot decode {file_path.name} with any supported encoding")
            
            if len(text.strip()) < 50:
                return FileProcessingResult(
                    success=False,
                    error="Original content too short (< 50 characters)",
                    input_path=file_path,
                    metadata={"processor": self.name}
                )
            
            # Clean the text
            original_text = text
            cleaned_text = self.clean_text(text)
            
            # Check if cleaned text has meaningful content
            if len(cleaned_text.strip()) < 20:
                return FileProcessingResult(
                    success=False,
                    error="No meaningful content after cleaning",
                    input_path=file_path,
                    metadata={"processor": self.name, "original_length": len(text)}
                )
            
            # Write cleaned text back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            
            # Calculate statistics
            original_word_count = len(original_text.split())
            final_word_count = len(cleaned_text.split())
            final_size_kb = file_path.stat().st_size / 1024
            original_size = len(original_text.encode("utf-8")) / 1024
            
            reduction = (
                ((original_size - final_size_kb) / original_size * 100)
                if original_size > 0
                else 0
            )
            
            # Create metadata
            metadata = {
                "processor": self.name,
                "original_length": len(original_text),
                "cleaned_length": len(cleaned_text),
                "original_word_count": original_word_count,
                "final_word_count": final_word_count,
                "size_reduction_percent": reduction,
                "encodings_tried": self.config.encoding_fallbacks
            }
            
            if final_word_count < 10:
                self.logger.warning(
                    f"{file_path.name}: Very small after cleaning "
                    f"(words: {final_word_count}, size: {final_size_kb:.2f}KB)"
                )
            
            return FileProcessingResult(
                success=True,
                input_path=file_path,
                metadata=metadata
            )
            
        except Exception as e:
            raise FileProcessingError(f"Failed to process {file_path.name}: {str(e)}") from e


class DocumentConverter(FileProcessor):
    """Convert various document formats to text."""
    
    def __init__(self):
        super().__init__(
            name="DocumentConverter",
            supported_extensions={'.docx', '.odt', '.doc'}
        )
    
    def convert_to_text(self, source_path: Path, target_path: Path) -> None:
        """Convert document to text format."""
        if source_path.suffix.lower() == ".docx":
            self._convert_docx(source_path, target_path)
        else:
            raise ValidationError(f"Unsupported document format: {source_path.suffix}")
    
    def _convert_docx(self, source_path: Path, target_path: Path) -> None:
        """Convert DOCX to text."""
        try:
            doc = Document(source_path)
            with open(target_path, "w", encoding="utf-8") as f:
                for para in doc.paragraphs:
                    f.write(para.text + "\n")
        except Exception as e:
            raise FileProcessingError(f"Failed to convert DOCX {source_path.name}: {str(e)}") from e
    
    def process_file(self, file_path: Path) -> FileProcessingResult:
        """Convert document file to text."""
        try:
            txt_output_path = file_path.with_suffix(".txt")
            self.convert_to_text(file_path, txt_output_path)
            
            return FileProcessingResult(
                success=True,
                input_path=file_path,
                output_path=txt_output_path,
                metadata={"processor": self.name}
            )
            
        except Exception as e:
            return FileProcessingResult(
                success=False,
                error=str(e),
                input_path=file_path,
                metadata={"processor": self.name}
            )


class TextProcessor:
    """Main text processing orchestrator."""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.converter = DocumentConverter()
        self.logger = get_logger(self.__class__.__name__)
    
    def process_file(self, file_path: Path) -> FileProcessingResult:
        """Process a single file with appropriate processor."""
        with LogContext(self.logger, "process_file", file_path=str(file_path)):
            if file_path.suffix.lower() in self.converter.supported_extensions:
                # Convert document first
                conv_result = self.converter.process_file(file_path)
                if conv_result.success and conv_result.output_path:
                    # Then clean the converted text
                    clean_result = self.cleaner.process_file(conv_result.output_path)
                    
                    # Merge results
                    clean_result.input_path = file_path
                    return clean_result
                else:
                    return conv_result
            else:
                # Direct text cleaning
                return self.cleaner.process_file(file_path)
    
    def process_directory(self, directory: Path, recursive: bool = True) -> list[FileProcessingResult]:
        """Process all supported files in a directory."""
        with LogContext(self.logger, "process_directory", 
                       directory=str(directory), recursive=recursive):
            
            results = []
            supported_extensions = (self.cleaner.supported_extensions | 
                                  self.converter.supported_extensions)
            
            # Find all files
            if recursive:
                files = directory.rglob("*")
            else:
                files = directory.glob("*")
            
            for file_path in files:
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        result = self.process_file(file_path)
                        results.append(result)
                        
                        if result.success:
                            self.logger.info(
                                f"Cleaned: {file_path.name} "
                                f"(words: {result.metadata.get('final_word_count', 0)}, "
                                f"size: {result.file_size_after/1024:.2f}KB)"
                            )
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path.name}: {e}")
                        results.append(FileProcessingResult(
                            success=False,
                            error=str(e),
                            input_path=file_path
                        ))
            
            self.logger.info(f"Processing complete: {len(results)} files processed")
            return results


# Legacy functions for backward compatibility
def normalize_unicode(text: str) -> str:
    """Legacy function for Unicode normalization."""
    return unicodedata.normalize("NFKC", text)

def clean_text(text: str) -> str:
    """Legacy function for text cleaning."""
    processor = TextCleaner()
    return processor.clean_text(text)

def clean_file(file_path: Path) -> None:
    """Legacy function for file cleaning."""
    processor = TextProcessor()
    result = processor.process_file(file_path)
    if not result.success:
        raise FileProcessingError(f"Failed to clean {file_path}: {result.error}")

def process_directory(directory: str | Path, recursive: bool = True) -> list[FileProcessingResult]:
    """Legacy function for directory processing."""
    processor = TextProcessor()
    return processor.process_directory(Path(directory), recursive)
