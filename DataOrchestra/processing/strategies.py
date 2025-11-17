"""
Cleaning strategies for different text processing approaches.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Dict

from ..core.config import ProcessingConfig
from ..core.logging_utils import get_logger


class CleaningStrategy(ABC):
    """
    Abstract base class for text cleaning strategies.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def apply(self, text: str, config: ProcessingConfig) -> str:
        """
        Apply the cleaning strategy to text.

        Args:
            text: Input text to clean
            config: Processing configuration

        Returns:
            Cleaned text
        """
        pass

    def __str__(self) -> str:
        return self.name


class AggressiveCleaningStrategy(CleaningStrategy):
    """
    Aggressive cleaning strategy that removes most noise and formatting.
    """

    def __init__(self):
        super().__init__("AggressiveCleaning")

    def apply(self, text: str, config: ProcessingConfig) -> str:
        """Apply aggressive cleaning to text."""
        import re

        # Remove all remaining URLs, emails, and special patterns
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # Remove all numbers and special characters
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)

        # Normalize whitespace aggressively
        text = re.sub(r"\s+", " ", text).strip()

        # Filter very short words
        words = [word for word in text.split() if len(word) >=
                 config.min_word_length]

        return " ".join(words).lower()


class GentleCleaningStrategy(CleaningStrategy):
    """
    Gentle cleaning strategy that preserves more original formatting.
    """

    def __init__(self):
        super().__init__("GentleCleaning")

    def apply(self, text: str, config: ProcessingConfig) -> str:
        """Apply gentle cleaning to text."""
        import re

        # Only remove obvious noise
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Preserve some formatting while cleaning
        text = re.sub(r"\s+", " ", text).strip()

        # Filter only obvious noise words
        noise_words = {"http", "https", "www", "com", "org"}
        words = [word for word in text.split() if word.lower()
                 not in noise_words]

        return " ".join(words)


class DomainSpecificStrategy(CleaningStrategy):
    """
    Domain-specific cleaning strategy for specialized content.
    """

    def __init__(self, domain: str = "general"):
        super().__init__(f"DomainSpecific({domain})")
        self.domain = domain
        self._domain_rules = self._load_domain_rules(domain)

    def _load_domain_rules(self, domain: str) -> Dict[str, Any]:
        """Load domain-specific cleaning rules."""
        rules = {
            "academic": {
                "preserve_references": True,
                "preserve_citations": True,
                "min_word_length": 3,
                "noise_words": {"http", "https", "www", "et", "al", "fig"}
            },
            "legal": {
                "preserve_case_numbers": True,
                "preserve_legal_terms": True,
                "min_word_length": 2,
                "noise_words": {"http", "https", "www", "section", "subsection"}
            },
            "medical": {
                "preserve_medical_terms": True,
                "preserve_dosages": True,
                "min_word_length": 3,
                "noise_words": {"http", "https", "www", "mg", "ml", "mcg"}
            },
            "code": {
                "preserve_code_blocks": True,
                "preserve_function_names": True,
                "min_word_length": 1,
                "noise_words": {"http", "https", "www", "var", "let", "const"}
            },
            "general": {
                "min_word_length": 2,
                "noise_words": {"http", "https", "www", "com", "org", "contact", "follow"}
            }
        }
        return rules.get(domain, rules["general"])

    def apply(self, text: str, config: ProcessingConfig) -> str:
        """Apply domain-specific cleaning to text."""
        import re

        rules = self._domain_rules

        # Apply domain-specific rules
        if self.domain == "academic":
            text = self._clean_academic_text(text, rules)
        elif self.domain == "legal":
            text = self._clean_legal_text(text, rules)
        elif self.domain == "medical":
            text = self._clean_medical_text(text, rules)
        elif self.domain == "code":
            text = self._clean_code_text(text, rules)
        else:
            text = self._clean_general_text(text, rules)

        return text

    def _clean_academic_text(self, text: str, rules: Dict[str, Any]) -> str:
        """Clean academic text while preserving references."""
        # Preserve citation patterns like [1], (Smith, 2020)
        citations = re.findall(r'\[\d+\]|\([^)]+\d{4}[^)]*\)', text)

        # Clean the text
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Restore citations
        for citation in citations:
            text = text[:100] + citation + text[100 + len(citation):]

        return text

    def _clean_legal_text(self, text: str, rules: Dict[str, Any]) -> str:
        """Clean legal text while preserving legal terms."""
        # Preserve case numbers and legal citations
        case_numbers = re.findall(r'\b(?:No\.|Case\s+)\d+', text)

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Restore case numbers
        for case_num in case_numbers:
            text = text[:100] + case_num + text[100 + len(case_num):]

        return text

    def _clean_medical_text(self, text: str, rules: Dict[str, Any]) -> str:
        """Clean medical text while preserving medical terms."""
        # Preserve dosage patterns
        dosages = re.findall(r'\b\d+(?:mg|ml|g|mcg|µg)\b', text)

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Restore dosages
        for dosage in dosages:
            text = text[:100] + dosage + text[100 + len(dosage):]

        return text

    def _clean_code_text(self, text: str, rules: Dict[str, Any]) -> str:
        """Clean code-related text while preserving code elements."""
        # Preserve code blocks
        code_blocks = re.findall(r'```[^`]*```|`[^`]*`', text)

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Restore code blocks
        for code_block in code_blocks:
            text = text[:100] + code_block + text[100 + len(code_block):]

        return text

    def _clean_general_text(self, text: str, rules: Dict[str, Any]) -> str:
        """Clean general text with standard rules."""
        import re

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Filter noise words
        noise_words = rules["noise_words"]
        words = [word for word in text.split() if word.lower()
                 not in noise_words]

        return " ".join(words)


class LanguageAwareStrategy(CleaningStrategy):
    """
    Language-aware cleaning strategy that adapts based on detected language.
    """

    def __init__(self):
        super().__init__("LanguageAware")
        self._lang_detector = None
        self._language_strategies = {}
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize language detector."""
        try:
            from langdetect import detect
            self._lang_detector = detect
        except ImportError:
            self.logger.warning(
                "langdetect not available, falling back to English rules")

    def apply(self, text: str, config: ProcessingConfig) -> str:
        """Apply language-aware cleaning to text."""
        if not self._lang_detector:
            return self._apply_english_rules(text, config)

        try:
            # Detect language
            lang = self._lang_detector(text)
            self.logger.debug(f"Detected language: {lang}")

            # Apply language-specific rules
            if lang == 'en':
                return self._apply_english_rules(text, config)
            elif lang == 'zh':
                return self._apply_chinese_rules(text, config)
            elif lang == 'es':
                return self._apply_spanish_rules(text, config)
            elif lang == 'fr':
                return self._apply_french_rules(text, config)
            else:
                return self._apply_default_rules(text, config)

        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return self._apply_default_rules(text, config)

    def _apply_english_rules(self, text: str, config: ProcessingConfig) -> str:
        """Apply English-specific cleaning rules."""
        import re

        # English-specific patterns
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # Preserve contractions
        contractions = {"can't", "won't", "don't",
                        "it's", "that's", "there's", "here's"}
        words = text.split()
        filtered_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower in contractions:
                filtered_words.append(word)
            elif word_lower not in config.noise_words and len(word) >= config.min_word_length:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def _apply_chinese_rules(self, text: str, config: ProcessingConfig) -> str:
        """Apply Chinese-specific cleaning rules."""
        import re

        # Remove URLs and emails
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Chinese text typically doesn't use spaces between words
        # Preserve Chinese characters while removing noise
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_text = ''.join(chinese_pattern.findall(text))

        return chinese_text.strip()

    def _apply_spanish_rules(self, text: str, config: ProcessingConfig) -> str:
        """Apply Spanish-specific cleaning rules."""
        import re

        # Spanish-specific patterns (including accented characters)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Preserve Spanish-specific characters and patterns
        spanish_noise = ["http", "https", "www", "página", "sitio"]
        words = text.split()
        filtered_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower not in spanish_noise and len(word) >= config.min_word_length:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def _apply_french_rules(self, text: str, config: ProcessingConfig) -> str:
        """Apply French-specific cleaning rules."""
        import re

        # French-specific patterns
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Preserve French-specific characters and patterns
        french_noise = ["http", "https", "www", "page", "site"]
        words = text.split()
        filtered_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower not in french_noise and len(word) >= config.min_word_length:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def _apply_default_rules(self, text: str, config: ProcessingConfig) -> str:
        """Apply default cleaning rules for unknown languages."""
        import re

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = [word for word in text.split() if len(word) >=
                 config.min_word_length]
        return " ".join(words)


class AdaptiveStrategy(CleaningStrategy):
    """
    Adaptive cleaning strategy that adjusts based on text characteristics.
    """

    def __init__(self):
        super().__init__("Adaptive")

    def apply(self, text: str, config: ProcessingConfig) -> str:
        """Apply adaptive cleaning based on text analysis."""
        import re

        # Analyze text characteristics
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()
                              ) / word_count if word_count > 0 else 0
        url_count = len(re.findall(r"https?://\S+", text))
        email_count = len(re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text))

        self.logger.debug(
            f"Text analysis: words={word_count}, "
            f"avg_length={avg_word_length:.1f}, "
            f"urls={url_count}, emails={email_count}"
        )

        # Choose strategy based on characteristics
        if url_count > 2 or email_count > 2:
            # Likely web content, use aggressive cleaning
            strategy = AggressiveCleaningStrategy()
        elif avg_word_length > 8:
            # Likely formal content, use gentle cleaning
            strategy = GentleCleaningStrategy()
        else:
            # Use domain-specific strategy
            strategy = DomainSpecificStrategy("general")

        return strategy.apply(text, config)
