"""
Processing stages for the text processing pipeline.
"""
from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.config import ProcessingConfig
from ..core.logging_utils import get_logger


class ProcessingStage(ABC):
    """
    Abstract base class for text processing stages.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, text: str, config: ProcessingConfig) -> str:
        """
        Process text through this stage.
        
        Args:
            text: Input text to process
            config: Processing configuration
            
        Returns:
            Processed text
        """
        pass
    
    def __str__(self) -> str:
        return self.name


class UnicodeNormalizationStage(ProcessingStage):
    """Stage for Unicode normalization."""
    
    def __init__(self):
        super().__init__("UnicodeNormalization")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Normalize Unicode characters to NFKC form."""
        return unicodedata.normalize("NFKC", text)


class NoiseRemovalStage(ProcessingStage):
    """Stage for removing noise from text."""
    
    def __init__(self):
        super().__init__("NoiseRemoval")
        # Pre-compile patterns for performance
        self._patterns = {
            'urls': re.compile(r"https?://\S+"),
            'ftp_urls': re.compile(r"ftp://\S+"),
            'www_urls': re.compile(r"www\.\S+"),
            'emails': re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            'domains': re.compile(r"\S+\.(com|org|net|edu|gov|io|co|uk)\S*"),
            'ip_addresses': re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            'control_chars': re.compile(r"[\x00-\x1F\x7F-\x9F]"),
            'repeated_punct': re.compile(r"([!?.,])\1+"),
            'standalone_numbers': re.compile(r"\b\d+\b"),
            'whitespace': re.compile(r"\s+")
        }
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Remove various types of noise from text."""
        # Normalize line endings and tabs
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        
        # Apply noise patterns
        text = self._patterns['urls'].sub("", text)
        text = self._patterns['ftp_urls'].sub("", text)
        text = self._patterns['www_urls'].sub("", text)
        text = self._patterns['emails'].sub("", text)
        text = self._patterns['domains'].sub("", text)
        text = self._patterns['ip_addresses'].sub("", text)
        text = self._patterns['control_chars'].sub(" ", text)
        text = self._patterns['standalone_numbers'].sub(" ", text)
        text = self._patterns['repeated_punct'].sub(r"\1", text)
        
        # Clean up whitespace
        text = self._patterns['whitespace'].sub(" ", text).strip()
        
        return text


class ContentFilteringStage(ProcessingStage):
    """Stage for filtering content based on configurable rules."""
    
    def __init__(self):
        super().__init__("ContentFiltering")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Filter content based on word length and noise words."""
        words = text.split()
        filtered_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Skip noise words
            if word_lower in config.noise_words:
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
            if len(word) >= config.min_word_length:
                filtered_words.append(word)
        
        return " ".join(filtered_words)


class StructureAnalysisStage(ProcessingStage):
    """Stage for analyzing and improving text structure."""
    
    def __init__(self):
        super().__init__("StructureAnalysis")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Analyze and improve text structure by removing repeated phrases."""
        sentences = re.split(r"[.!?]+", text)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            clean_sentence = sentence.strip().lower()
            
            # Only process substantial sentences
            if (
                clean_sentence
                and len(clean_sentence) > config.min_sentence_length
                and clean_sentence not in seen_sentences
            ):
                seen_sentences.add(clean_sentence)
                unique_sentences.append(sentence.strip())
        
        result = ". ".join(unique_sentences).strip()
        
        # Ensure proper sentence ending
        if result and result[-1] not in ".!?":
            result += "."
        
        return result


class FormatStandardizationStage(ProcessingStage):
    """Stage for final format standardization."""
    
    def __init__(self):
        super().__init__("FormatStandardization")
        self._whitespace_pattern = re.compile(r"\s+")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Apply final format standardization."""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = self._whitespace_pattern.sub(" ", text).strip()
        
        return text


class LanguageDetectionStage(ProcessingStage):
    """Stage for detecting language and applying language-specific rules."""
    
    def __init__(self):
        super().__init__("LanguageDetection")
        self._lang_detector = None
        self._language_processors = {}
        self._initialize_language_detector()
    
    def _initialize_language_detector(self):
        """Initialize language detector."""
        try:
            from langdetect import detect
            self._lang_detector = detect
        except ImportError:
            self.logger.warning("langdetect not available, language detection disabled")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Detect language and apply language-specific processing."""
        if not self._lang_detector:
            return text
        
        try:
            # Detect language
            lang = self._lang_detector(text)
            self.logger.debug(f"Detected language: {lang}")
            
            # Apply language-specific processing if available
            if lang in self._language_processors:
                return self._language_processors[lang](text)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return text
    
    def register_language_processor(self, lang_code: str, processor_func):
        """Register a language-specific processor."""
        self._language_processors[lang_code] = processor_func
        self.logger.info(f"Registered processor for language: {lang_code}")


class QualityAssessmentStage(ProcessingStage):
    """Stage for assessing text quality and applying improvements."""
    
    def __init__(self):
        super().__init__("QualityAssessment")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        """Assess text quality and apply improvements."""
        # Calculate quality metrics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # Log quality metrics
        self.logger.debug(
            f"Text quality: words={word_count}, "
            f"sentences={sentence_count}, "
            f"avg_word_length={avg_word_length:.1f}"
        )
        
        # Apply quality improvements based on metrics
        if avg_word_length < 3:
            self.logger.debug("Text has very short words, applying gentle processing")
        
        return text