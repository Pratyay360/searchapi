"""
Test suite for processing functionality of DataOrchestra.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import asyncio

from DataOrchestra.processing.pipeline import TextProcessingPipeline
from DataOrchestra.core.base import ProcessResult
from DataOrchestra.core.config import ProcessingConfig
from DataOrchestra.processing.stages import (
    ProcessingStage, UnicodeNormalizationStage, NoiseRemovalStage,
    ContentFilteringStage, StructureAnalysisStage, FormatStandardizationStage
)
from DataOrchestra.processing.strategies import (
    CleaningStrategy, AggressiveCleaningStrategy, GentleCleaningStrategy,
    DomainSpecificStrategy, LanguageAwareStrategy, AdaptiveStrategy
)
from DataOrchestra.processing.quality import TextQualityAssessor
from DataOrchestra.processing.cache import CacheManager, CacheConfig


class ConcreteProcessingStage(ProcessingStage):
    """Concrete implementation of ProcessingStage for testing."""
    
    def __init__(self):
        super().__init__("TestStage")
    
    def process(self, text: str, config: ProcessingConfig) -> str:
        return text + " processed"


class ConcreteCleaningStrategy(CleaningStrategy):
    """Concrete implementation of CleaningStrategy for testing."""
    
    def __init__(self):
        super().__init__("TestStrategy")
    
    def apply(self, text: str, config: ProcessingConfig) -> str:
        return text + " cleaned"


class TestProcessingPipeline:
    """Test cases for the text processing pipeline."""
    
    def test_pipeline_initialization(self):
        """Test TextProcessingPipeline initialization."""
        pipeline = TextProcessingPipeline()
        
        assert pipeline is not None
        assert len(pipeline.stages) > 0  # Should have default stages
        assert pipeline.quality_assessor is not None
        assert pipeline.cache_manager is not None
        assert pipeline.config is not None
        assert pipeline.name == "TextProcessingPipeline"
    
    def test_pipeline_with_custom_stages(self):
        """Test TextProcessingPipeline with custom stages."""
        mock_stage = ConcreteProcessingStage()
        pipeline = TextProcessingPipeline(stages=[mock_stage])
        
        assert len(pipeline.stages) == 1
        assert isinstance(pipeline.stages[0], ConcreteProcessingStage)
    
    def test_pipeline_with_custom_strategy(self):
        """Test TextProcessingPipeline with custom strategy."""
        mock_strategy = ConcreteCleaningStrategy()
        pipeline = TextProcessingPipeline(strategy=mock_strategy)
        
        assert pipeline.strategy is mock_strategy
    
    def test_pipeline_process_method(self):
        """Test the process method of TextProcessingPipeline."""
        pipeline = TextProcessingPipeline()
        
        input_text = "Hello, this is a test text with some noise!"
        result = pipeline.process(input_text)
        
        # The result should be different from input but still contain meaningful content
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != input_text  # Processing should modify the text
    
    def test_pipeline_async_process_method(self):
        """Test the async process method of TextProcessingPipeline."""
        pipeline = TextProcessingPipeline()
        
        async def run_async_test():
            input_text = "Hello, this is a test text for async processing!"
            result = await pipeline.process_async(input_text)
            return result
        
        # Run the async test
        result = asyncio.run(run_async_test())
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_pipeline_add_remove_stages(self):
        """Test adding and removing stages from the pipeline."""
        pipeline = TextProcessingPipeline()
        
        initial_stage_count = len(pipeline.stages)
        
        mock_stage = ConcreteProcessingStage()
        
        # Add stage
        pipeline.add_stage(mock_stage)
        assert len(pipeline.stages) == initial_stage_count + 1
        
        # Remove stage
        removed = pipeline.remove_stage(ConcreteProcessingStage)
        assert removed is True
        assert len(pipeline.stages) == initial_stage_count
        
        # Try to remove non-existent stage
        removed = pipeline.remove_stage(ConcreteProcessingStage)
        assert removed is False
    
    def test_pipeline_set_strategy(self):
        """Test setting a strategy for the pipeline."""
        pipeline = TextProcessingPipeline()
        
        mock_strategy = ConcreteCleaningStrategy()
        pipeline.set_strategy(mock_strategy)
        
        assert pipeline.strategy is mock_strategy
    
    def test_pipeline_get_info(self):
        """Test getting pipeline information."""
        pipeline = TextProcessingPipeline()
        
        info = pipeline.get_pipeline_info()
        
        assert "stages" in info
        assert "strategy" in info
        assert "quality_assessor" in info
        assert "cache_enabled" in info
        assert "config" in info
        
        assert isinstance(info["stages"], list)
        assert len(info["stages"]) > 0
    
    def test_pipeline_process_file(self):
        """Test processing a text file with the pipeline."""
        pipeline = TextProcessingPipeline()
        
        # Create a temporary file with content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("Hello, this is test content for file processing!")
            tmp_path = Path(tmp.name)
        
        try:
            # Process the file
            result = pipeline.process_file(tmp_path)
            
            assert isinstance(result, ProcessResult)
            assert result.success is True
            assert result.error is None
        finally:
            # Clean up
            os.unlink(tmp_path)


class TestProcessingStages:
    """Test cases for processing stages."""
    
    def test_unicode_normalization_stage(self):
        """Test Unicode normalization stage."""
        stage = UnicodeNormalizationStage()
        config = ProcessingConfig()
        
        # Test with text containing various Unicode characters
        input_text = "café résumé naïve"
        result = stage.process(input_text, config)
        
        assert isinstance(result, str)
        assert result == input_text  # Should be normalized but text remains the same in appearance
    
    def test_noise_removal_stage(self):
        """Test noise removal stage."""
        stage = NoiseRemovalStage()
        config = ProcessingConfig()
        
        # Test with text containing noise
        input_text = "Visit https://example.com or email test@example.com for more info!"
        result = stage.process(input_text, config)
        
        assert isinstance(result, str)
        # Should not contain URLs or emails
        assert "https://example.com" not in result
        assert "test@example.com" not in result
    
    def test_content_filtering_stage(self):
        """Test content filtering stage."""
        stage = ContentFilteringStage()
        config = ProcessingConfig()
        
        # Test with text containing noise words
        input_text = "This is a test with http www and other noise words"
        result = stage.process(input_text, config)
        
        assert isinstance(result, str)
        # Should not contain noise words
        assert "http" not in result
        assert "www" not in result
    
    def test_structure_analysis_stage(self):
        """Test structure analysis stage."""
        stage = StructureAnalysisStage()
        config = ProcessingConfig()
        
        # Test with text containing repeated sentences
        input_text = "This is a sentence. This is a sentence. This is another sentence."
        result = stage.process(input_text, config)
        
        assert isinstance(result, str)
        # Should have removed duplicate sentences
        assert result.count("This is a sentence") == 1
    
    def test_format_standardization_stage(self):
        """Test format standardization stage."""
        stage = FormatStandardizationStage()
        config = ProcessingConfig()
        
        # Test with text containing mixed case and extra whitespace
        input_text = "  This   has   extra   whitespace  AND   MIXED   CASE  "
        result = stage.process(input_text, config)
        
        assert isinstance(result, str)
        assert result == "this has extra whitespace and mixed case"


class TestCleaningStrategies:
    """Test cases for cleaning strategies."""
    
    def test_aggressive_cleaning_strategy(self):
        """Test aggressive cleaning strategy."""
        strategy = AggressiveCleaningStrategy()
        config = ProcessingConfig()
        
        input_text = "This has URLs https://example.com and numbers 12345 and symbols !@#$%"
        result = strategy.apply(input_text, config)
        
        assert isinstance(result, str)
        # Should be cleaned of URLs, numbers, and symbols
        assert "https://example.com" not in result
        assert not any(c.isdigit() for c in result if c != ' ')
    
    def test_gentle_cleaning_strategy(self):
        """Test gentle cleaning strategy."""
        strategy = GentleCleaningStrategy()
        config = ProcessingConfig()
        
        input_text = "This has URLs https://example.com but preserves some formatting"
        result = strategy.apply(input_text, config)
        
        assert isinstance(result, str)
        # Should remove URLs but preserve more content
        assert "https://example.com" not in result
        assert "preserves" in result
    
    def test_domain_specific_strategy(self):
        """Test domain-specific cleaning strategy."""
        strategy = DomainSpecificStrategy("academic")
        config = ProcessingConfig()
        
        input_text = "This is academic text with http references [1] and citations (Smith, 2020)"
        result = strategy.apply(input_text, config)
        
        assert isinstance(result, str)
        # Should clean noise but preserve academic elements
        assert "http" not in result
    
    def test_language_aware_strategy(self):
        """Test language-aware cleaning strategy."""
        strategy = LanguageAwareStrategy()
        config = ProcessingConfig()
        
        input_text = "This is English text with some noise http://example.com"
        result = strategy.apply(input_text, config)
        
        assert isinstance(result, str)
        # Should clean noise while preserving English text
        assert "http://" not in result
    
    def test_adaptive_strategy(self):
        """Test adaptive cleaning strategy."""
        strategy = AdaptiveStrategy()
        config = ProcessingConfig()
        
        input_text = "This text has https://example.com URLs and should trigger aggressive cleaning"
        result = strategy.apply(input_text, config)
        
        assert isinstance(result, str)
        # Should adapt based on content characteristics
        assert "https://" not in result


class TestTextQualityAssessor:
    """Test cases for text quality assessment."""
    
    def test_quality_assessor_initialization(self):
        """Test TextQualityAssessor initialization."""
        assessor = TextQualityAssessor()
        
        assert assessor is not None
    
    def test_quality_assessor_assess(self):
        """Test quality assessment of text."""
        assessor = TextQualityAssessor()
        
        # Test with good text
        good_text = "This is a well-formed, meaningful sentence."
        good_score = assessor.assess(good_text)
        
        assert isinstance(good_score, float)
        assert 0.0 <= good_score <= 1.0
        
        # Test with poor text
        poor_text = "asdasdasdasd !@#$%^&*()"
        poor_score = assessor.assess(poor_text)
        
        assert isinstance(poor_score, float)
        assert 0.0 <= poor_score <= 1.0


class TestCacheManager:
    """Test cases for cache management."""
    
    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        cache = CacheManager()
        
        assert cache is not None
        assert cache.config is not None
    
    def test_cache_manager_get_set(self):
        """Test cache get/set operations."""
        cache = CacheManager()
        
        # Test setting and getting a value
        cache.set("key1", "value1")
        retrieved = cache.get("key1")
        
        assert retrieved == "value1"
        
        # Test getting a non-existent value
        retrieved = cache.get("nonexistent")
        assert retrieved is None
    
    def test_cache_manager_clear(self):
        """Test clearing the cache."""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Verify items are in cache
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # Clear cache
        cache.clear()
        
        # Verify items are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_manager_config(self):
        """Test cache manager with custom config."""
        config = CacheConfig(enabled=True, memory_max_size=500, ttl_seconds=1800)
        cache = CacheManager(config)
        
        assert cache.config.memory_max_size == 500
        assert cache.config.ttl_seconds == 1800


class TestTextUtilities:
    """Test cases for text utilities."""
    
    def test_clean_text(self):
        """Test text cleaning utility."""
        from DataOrchestra.utils.text_utils import clean_text
        
        # Test with text containing URLs, emails, etc.
        dirty_text = "Visit https://example.com or email test@example.com for more info!"
        clean_result = clean_text(dirty_text)
        
        assert isinstance(clean_result, str)
        # Should not contain URLs or emails
        assert "https://example.com" not in clean_result
        assert "test@example.com" not in clean_result
        assert clean_result != dirty_text  # Should be modified
    
    def test_normalize_text(self):
        """Test text normalization utility."""
        from DataOrchestra.utils.text_utils import normalize_text
        
        # Test with text containing extra whitespace and mixed case
        text_with_issues = "  This   has   extra   whitespace  and   MiXeD   CaSe  "
        normalized = normalize_text(text_with_issues)
        
        assert isinstance(normalized, str)
        assert normalized == "this has extra whitespace and mixed case"
    
    def test_split_text_by_tokens(self):
        """Test text splitting by tokens."""
        from DataOrchestra.utils.text_utils import split_text_by_tokens
        
        long_text = " ".join([f"word{i}" for i in range(100)])  # 100 words
        
        # Split into chunks of 10 tokens
        chunks = list(split_text_by_tokens(long_text, token_limit=10))
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(len(chunk.split()) <= 10 for chunk in chunks)  # Each chunk should respect limit


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])
