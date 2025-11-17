"""
Text quality assessment and scoring utilities.
"""
from __future__ import annotations

import re
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from ..core.logging_utils import get_logger


@dataclass
class QualityMetrics:
    """Data class for text quality metrics."""
    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    readability_score: float
    coherence_score: float
    completeness_score: float
    overall_score: float
    issues: List[str]


class TextQualityAssessor:
    """
    Assesses text quality using multiple metrics.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def assess(self, text: str) -> float:
        """
        Assess text quality and return overall score.
        
        Args:
            text: Text to assess
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        metrics = self.calculate_metrics(text)
        self.logger.debug(f"Quality metrics: {metrics}")
        
        # Calculate overall score
        overall_score = (
            metrics.readability_score * 0.3 +
            metrics.coherence_score * 0.3 +
            metrics.completeness_score * 0.2 +
            self._calculate_structure_score(metrics) * 0.2
        )
        
        metrics.overall_score = min(1.0, max(0.0, overall_score))
        
        # Log quality assessment
        if metrics.overall_score < 0.3:
            self.logger.warning(f"Low quality text detected: {metrics.overall_score:.3f}")
        elif metrics.overall_score < 0.6:
            self.logger.info(f"Medium quality text: {metrics.overall_score:.3f}")
        else:
            self.logger.debug(f"High quality text: {metrics.overall_score:.3f}")
        
        return metrics.overall_score
    
    def calculate_metrics(self, text: str) -> QualityMetrics:
        """
        Calculate detailed quality metrics for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            QualityMetrics object with detailed scores
        """
        if not text or not text.strip():
            return QualityMetrics(
                word_count=0,
                sentence_count=0,
                avg_word_length=0.0,
                avg_sentence_length=0.0,
                readability_score=0.0,
                coherence_score=0.0,
                completeness_score=0.0,
                overall_score=0.0,
                issues=["Empty or whitespace-only text"]
            )
        
        # Basic metrics
        words = text.split()
        sentences = self._split_sentences(text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = sum(len(sent) for sent in sentences) / sentence_count if sentence_count > 0 else 0
        
        # Quality scores
        readability_score = self._calculate_readability(text, words, sentences)
        coherence_score = self._calculate_coherence(words, sentences)
        completeness_score = self._calculate_completeness(text, words, sentences)
        
        # Identify issues
        issues = self._identify_issues(text, words, sentences)
        
        return QualityMetrics(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            overall_score=0.0,  # Will be calculated in assess()
            issues=issues
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with NLP libraries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """
        Calculate readability score based on various factors.
        
        Args:
            text: Original text
            words: List of words
            sentences: List of sentences
            
        Returns:
            Readability score (0.0 to 1.0)
        """
        if not words or not sentences:
            return 0.0
        
        # Factors affecting readability
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
        
        # Score based on optimal ranges
        word_length_score = 1.0 - abs(avg_word_length - 5.0) / 10.0  # Optimal around 5 chars
        sentence_length_score = 1.0 - abs(avg_sentence_length - 15.0) / 30.0  # Optimal around 15 chars
        
        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(word.lower() for word in words))
        vocab_diversity = unique_words / len(words)
        
        # Punctuation and capitalization
        punctuation_ratio = len(re.findall(r'[.!?]', text)) / len(sentences) if sentences else 0
        capitalization_score = self._calculate_capitalization_score(text)
        
        # Combine scores
        readability = (
            word_length_score * 0.3 +
            sentence_length_score * 0.3 +
            vocab_diversity * 0.2 +
            min(1.0, punctuation_ratio) * 0.1 +
            capitalization_score * 0.1
        )
        
        return min(1.0, max(0.0, readability))
    
    def _calculate_capitalization_score(self, text: str) -> float:
        """Calculate score based on proper capitalization."""
        if not text:
            return 0.0
        
        # Check for proper sentence capitalization
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.5
        
        properly_capitalized = 0
        for sentence in sentences:
            if sentence and sentence[0].isupper():
                properly_capitalized += 1
        
        return properly_capitalized / len(sentences)
    
    def _calculate_coherence(self, words: List[str], sentences: List[str]) -> float:
        """
        Calculate text coherence score.
        
        Args:
            words: List of words
            sentences: List of sentences
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not words or not sentences:
            return 0.0
        
        # Factors for coherence
        sentence_length_variance = self._calculate_variance([len(s) for s in sentences])
        word_length_variance = self._calculate_variance([len(w) for w in words])
        
        # Lower variance is better for coherence
        length_score = 1.0 - min(1.0, sentence_length_variance / 100.0)
        word_variance_score = 1.0 - min(1.0, word_length_variance / 10.0)
        
        # Repetition detection
        repetition_score = self._calculate_repetition_score(words)
        
        # Transition words (conjunctions, prepositions)
        transition_words = {'and', 'but', 'or', 'however', 'therefore', 'because', 'so', 'then'}
        transition_count = sum(1 for word in words if word.lower() in transition_words)
        transition_score = min(1.0, transition_count / len(words) * 5)
        
        coherence = (
            length_score * 0.3 +
            word_variance_score * 0.2 +
            repetition_score * 0.3 +
            transition_score * 0.2
        )
        
        return min(1.0, max(0.0, coherence))
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_repetition_score(self, words: List[str]) -> float:
        """Calculate repetition score (lower is better)."""
        if len(words) < 2:
            return 1.0
        
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # Calculate repetition ratio
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(words)
        
        # Lower repetition is better
        return 1.0 - repetition_ratio
    
    def _calculate_completeness(self, text: str, words: List[str], sentences: List[str]) -> float:
        """
        Calculate text completeness score.
        
        Args:
            text: Original text
            words: List of words
            sentences: List of sentences
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Factors for completeness
        has_start = bool(text.strip())
        has_end = text.strip()[-1] in '.!?' if text.strip() else False
        has_content = len(words) > 0
        has_structure = len(sentences) > 1
        
        # Length appropriateness
        length_score = min(1.0, len(text) / 50.0)  # Assume 50 chars is minimum
        
        completeness = (
            (1.0 if has_start else 0.0) * 0.3 +
            (1.0 if has_end else 0.0) * 0.2 +
            (1.0 if has_content else 0.0) * 0.3 +
            (1.0 if has_structure else 0.0) * 0.1 +
            length_score * 0.1
        )
        
        return min(1.0, completeness)
    
    def _calculate_structure_score(self, metrics: QualityMetrics) -> float:
        """Calculate structure score from metrics."""
        if metrics.word_count == 0:
            return 0.0
        
        # Ideal ratios
        ideal_words_per_sentence = 15.0
        ideal_avg_word_length = 5.0
        
        words_per_sentence_score = 1.0 - abs(metrics.word_count / metrics.sentence_count - ideal_words_per_sentence) / ideal_words_per_sentence
        word_length_score = 1.0 - abs(metrics.avg_word_length - ideal_avg_word_length) / ideal_avg_word_length
        
        return (words_per_sentence_score + word_length_score) / 2.0
    
    def _identify_issues(self, text: str, words: List[str], sentences: List[str]) -> List[str]:
        """Identify quality issues in text."""
        issues = []
        
        if not text or not text.strip():
            issues.append("Empty or whitespace-only text")
        
        if len(words) < 3:
            issues.append("Very short text (less than 3 words)")
        
        if len(sentences) < 1:
            issues.append("No complete sentences found")
        
        if len(sentences) == 1 and len(words) > 20:
            issues.append("Long run-on sentence")
        
        # Check for excessive repetition
        if self._calculate_repetition_score(words) < 0.7:
            issues.append("Excessive word repetition")
        
        # Check for very short or very long words
        short_words = [w for w in words if len(w) < 2]
        long_words = [w for w in words if len(w) > 20]
        
        if len(short_words) > len(words) * 0.3:
            issues.append("Many very short words")
        
        if len(long_words) > 0:
            issues.append(f"Very long words present: {len(long_words)}")
        
        # Check for unusual characters
        unusual_chars = len(re.findall(r'[^\w\s.,!?;:\'"()-]', text))
        if unusual_chars > len(text) * 0.1:
            issues.append("Many unusual characters")
        
        return issues
    
    def get_quality_report(self, text: str) -> Dict[str, Any]:
        """
        Get detailed quality report for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detailed quality information
        """
        metrics = self.calculate_metrics(text)
        
        return {
            "overall_score": metrics.overall_score,
            "readability_score": metrics.readability_score,
            "coherence_score": metrics.coherence_score,
            "completeness_score": metrics.completeness_score,
            "word_count": metrics.word_count,
            "sentence_count": metrics.sentence_count,
            "avg_word_length": metrics.avg_word_length,
            "avg_sentence_length": metrics.avg_sentence_length,
            "issues": metrics.issues,
            "quality_level": self._get_quality_level(metrics.overall_score),
            "recommendations": self._get_recommendations(metrics)
        }
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description from score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Get improvement recommendations based on metrics."""
        recommendations = []
        
        if metrics.readability_score < 0.5:
            recommendations.append("Improve sentence structure and punctuation")
        
        if metrics.coherence_score < 0.5:
            recommendations.append("Reduce repetition and improve transitions")
        
        if metrics.completeness_score < 0.5:
            recommendations.append("Add more content and proper sentence endings")
        
        if metrics.avg_word_length < 3:
            recommendations.append("Use more descriptive language")
        elif metrics.avg_word_length > 8:
            recommendations.append("Use simpler language")
        
        if metrics.word_count < 10:
            recommendations.append("Add more content to better assess quality")
        
        return recommendations