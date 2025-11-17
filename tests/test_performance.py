"""
Performance benchmarks for DataOrchestra modules.
"""
from __future__ import annotations

import pytest
import time
import tempfile
import os
from pathlib import Path
from typing import Callable

from DataOrchestra.processing.pipeline import TextProcessingPipeline
from DataOrchestra.utils.text_utils import clean_text, normalize_text, split_text_by_tokens
from DataOrchestra.extract.pdf_extractor import process_pdf
from DataOrchestra.extract.docx_extractor import process_docx
from DataOrchestra.extract.markdown_extractor import process_markdown
from DataOrchestra.extract.web_extractor import process_web


class TestPerformanceBenchmarks:
    """Performance benchmarks for DataOrchestra modules."""
    
    def test_text_processing_pipeline_performance(self):
        """Benchmark text processing pipeline performance."""
        pipeline = TextProcessingPipeline()
        
        # Create a moderately large text sample
        sample_text = "This is a test sentence. " * 100  # 100 sentences
        
        start_time = time.time()
        result = pipeline.process(sample_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify the result
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Performance requirement: should process 100 sentences in under 5 seconds
        assert processing_time < 5.0, f"Pipeline took {processing_time:.2f}s, expected < 5.0s"
        
        print(f"Pipeline processed {len(sample_text)} chars in {processing_time:.3f}s")
        print(f"Throughput: {len(sample_text) / processing_time:.0f} chars/sec")
    
    def test_text_cleaning_performance(self):
        """Benchmark text cleaning utility performance."""
        # Create a text with various issues to clean
        dirty_text = "Visit https://example.com or email test@example.com! " * 50
        dirty_text += " EXTRA   WHITESPACE and MiXeD CaSe! " * 500
        
        # Benchmark clean_text
        start_time = time.time()
        cleaned = clean_text(dirty_text)
        clean_time = time.time() - start_time
        
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        assert clean_time < 2.0, f"Text cleaning took {clean_time:.2f}s, expected < 2.0s"
        
        # Benchmark normalize_text
        start_time = time.time()
        normalized = normalize_text(dirty_text)
        normalize_time = time.time() - start_time
        
        assert isinstance(normalized, str)
        assert normalize_time < 1.0, f"Text normalization took {normalize_time:.2f}s, expected < 1.0s"
        
        print(f"Text cleaning: {clean_time:.3f}s for {len(dirty_text)} chars")
        print(f"Text normalization: {normalize_time:.3f}s for {len(dirty_text)} chars")
    
    def test_text_splitting_performance(self):
        """Benchmark text splitting performance."""
        long_text = " ".join([f"word{i}" for i in range(1000)])  # 1000 words
        
        start_time = time.time()
        chunks = list(split_text_by_tokens(long_text, token_limit=100))
        split_time = time.time() - start_time
        
        assert len(chunks) > 0
        assert split_time < 1.0, f"Text splitting took {split_time:.2f}s, expected < 1.0s"
        
        print(f"Split {len(long_text)} chars into {len(chunks)} chunks in {split_time:.3f}s")
    
    def test_extraction_performance(self):
        """Benchmark extraction performance with mocked dependencies."""
        # Test markdown extraction performance
        markdown_content = "# Test Document\n\n"
        markdown_content += "This is paragraph " + str(list(range(100))) + "\n\n"  # Create content
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(markdown_content)
            tmp_path = Path(tmp.name)
        
        try:
            start_time = time.time()
            result = process_markdown(tmp_path)
            extraction_time = time.time() - start_time
            
            assert result.success is True
            assert extraction_time < 1.0, f"Markdown extraction took {extraction_time:.2f}s, expected < 1.0s"
            
            print(f"Markdown extraction: {extraction_time:.3f}s for {len(markdown_content)} chars")
        finally:
            os.unlink(tmp_path)
    
    def test_pipeline_throughput(self):
        """Test pipeline throughput with multiple inputs."""
        pipeline = TextProcessingPipeline()
        
        # Create multiple test texts
        test_texts = [f"Test text {i} with some content for processing." for i in range(50)]
        
        start_time = time.time()
        results = [pipeline.process(text) for text in test_texts]
        total_time = time.time() - start_time
        
        assert len(results) == 50
        assert all(isinstance(r, str) for r in results)
        
        avg_time_per_text = total_time / len(test_texts)
        assert avg_time_per_text < 0.1, f"Avg processing time {avg_time_per_text:.3f}s, expected < 0.1s"
        
        print(f"Processed {len(test_texts)} texts in {total_time:.3f}s")
        print(f"Average: {avg_time_per_text:.3f}s per text")
        print(f"Throughput: {len(test_texts) / total_time:.1f} texts/sec")
    
    def test_memory_usage_stability(self):
        """Test that processing doesn't cause memory leaks over time."""
        pipeline = TextProcessingPipeline()
        
        # Process many texts to check for memory stability
        sample_text = "This is a test sentence that we will process many times." * 10
        
        # Get initial memory info (conceptual - real implementation would use memory profiling)
        import gc
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Process multiple times
        for i in range(100):
            result = pipeline.process(sample_text)
            assert isinstance(result, str)
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Objects should not grow significantly (allowing for some variation)
        growth = final_objects - initial_objects
        assert growth < 1000, f"Memory grew by {growth} objects, indicating possible leak"
        
        print(f"Memory growth: {growth} objects after 100 operations")


class TestScalability:
    """Scalability tests for different input sizes."""
    
    def run_performance_test(self, func: Callable, input_size: int) -> float:
        """Helper to run performance test and return execution time."""
        start_time = time.time()
        func(input_size)
        return time.time() - start_time
    
    def test_scaling_with_text_size(self):
        """Test how performance scales with text size."""
        pipeline = TextProcessingPipeline()
        
        sizes = [100, 500, 1000, 500, 10000]  # Character counts
        times = []
        
        for size in sizes:
            text = "Test text. " * (size // 10)  # Approximate the size
            start_time = time.time()
            result = pipeline.process(text)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            assert isinstance(result, str)
            print(f"Size {size:5d}: {elapsed:.4f}s")
        
        # Performance should scale reasonably (not exponentially)
        if len(times) > 1:
            growth_rate = times[-1] / times[0] if times[0] > 0 else 1
            # Even with 100x larger text, should not take 1000x longer time
            assert growth_rate < 100, f"Performance scaling too poor: {growth_rate}x time for 100x size"


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])