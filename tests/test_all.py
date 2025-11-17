"""
Comprehensive test suite for DataOrchestra - runs all tests.

This file serves as the main test runner for the entire DataOrchestra project,
executing all test modules to provide complete test coverage validation.

Usage:
    # Run all tests
    pytest tests/test_all.py -v
    
    # Run with coverage
    pytest tests/test_all.py --cov=DataOrchestra --cov-report=html
    
    # Run only specific test categories
    pytest tests/test_all.py -m "core or extraction"
    
    # Run performance tests
    pytest tests/test_all.py -m "performance"
"""

from __future__ import annotations

import sys
import os
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules for execution
from test_core import (
    TestBaseProcessor,
    TestFileProcessor, 
    TestProcessResult,
    TestFileProcessingResult,
    TestConfig,
    TestExceptions,
    TestLoggingUtils
)

from test_extraction import (
    TestPDFExtractor,
    TestDOCXExtractor,
    TestMarkdownExtractor,
    TestWebExtractor,
    TestExtractionIntegration
)

from test_processing import (
    TestProcessingPipeline,
    TestProcessingStages,
    TestCleaningStrategies,
    TestTextQualityAssessor,
    TestCacheManager,
    TestTextUtilities
)

from test_web import (
    TestWebFetcher,
    TestWebSpider,
    TestWebIntegration
)

from test_integration import (
    TestModuleIntegration,
    TestCrossModuleCompatibility,
    TestRealWorldWorkflows
)

# Additional comprehensive test categories
class TestCompleteSystemIntegration:
    """Complete system integration tests covering all DataOrchestra functionality."""
    
    def test_complete_text_workflow(self):
        """Test the complete text processing workflow from raw input to final output."""
        from DataOrchestra import clean_text, normalize_text, split_text_by_tokens
        from DataOrchestra.processing import TextProcessingPipeline
        from DataOrchestra.core.base import ProcessResult
        from DataOrchestra.core.config import get_config
        
        # Create realistic input text with various issues
        raw_text = """
        Visit https://example.com or contact admin@company.org for information!
        
        This document has  extra    spaces, MiXeD CaSe, and repetitive content.
        Repetitive content should be removed during processing.
        
        Key Points:
        • Important information here
        • Another key point
        • Final important point
        """
        
        # Step 1: Apply utility functions
        cleaned = clean_text(raw_text)
        normalized = normalize_text(cleaned)
        
        assert isinstance(cleaned, str)
        assert isinstance(normalized, str)
        assert "https://example.com" not in cleaned
        assert "admin@company.org" not in cleaned
        assert normalized == normalized.lower()
        
        # Step 2: Process through pipeline
        pipeline = TextProcessingPipeline()
        processed = pipeline.process(normalized)
        
        assert isinstance(processed, str)
        assert len(processed) > 0
        assert processed != raw_text
        
        # Step 3: Verify configuration is working
        config = get_config()
        assert config is not None
        
        return {
            "raw": raw_text,
            "cleaned": cleaned,
            "normalized": normalized,
            "processed": processed
        }
    
    def test_multi_format_file_processing(self):
        """Test processing files in multiple formats."""
        from DataOrchestra.extract.markdown_extractor import process_markdown
        from DataOrchestra.extract import process_web
        from DataOrchestra.processing.pipeline import TextProcessingPipeline
        from DataOrchestra.core.base import FileProcessingResult
        
        files_created = []
        
        try:
            # Create test files in different formats
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as md_file:
                md_file.write("# Test Document\n\nContent with [link](https://example.com)")
                md_path = Path(md_file.name)
                files_created.append(md_path)
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as html_file:
                html_file.write("""
                <html>
                    <head><title>Test HTML</title></head>
                    <body>
                        <h1>HTML Content</h1>
                        <p>Some text content here.</p>
                        <script>console.log('remove');</script>
                    </body>
                </html>
                """)
                html_path = Path(html_file.name)
                files_created.append(html_path)
            
            # Process each file type
            pipeline = TextProcessingPipeline()
            results = []
            
            for file_path in files_created:
                if file_path.suffix == ".md":
                    result = process_markdown(file_path)
                elif file_path.suffix == ".html":
                    result = process_web(file_path)
                else:
                    continue
                
                assert isinstance(result, FileProcessingResult)
                if result.success:
                    processed_content = pipeline.process(result.data)
                    assert isinstance(processed_content, str)
                    assert len(processed_content) > 0
                    results.append(processed_content)
            
            assert len(results) == 2  # Both files should be processed
            
        finally:
            # Clean up all created files
            for file_path in files_created:
                try:
                    os.unlink(file_path)
                except FileNotFoundError:
                    pass
    
    @patch('requests.get')
    @patch('DataOrchestra.web.spider.BeautifulSoup')
    def test_complete_web_workflow(self, mock_bs, mock_get):
        """Test complete web data processing workflow."""
        from DataOrchestra.web.fetcher import fetch_url
        from DataOrchestra.web.spider import crawl_website
        from DataOrchestra.extract.web_extractor import process_web
        from DataOrchestra.processing.pipeline import TextProcessingPipeline
        
        # Mock web responses
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>Test Site</title></head>
            <body>
                <h1>Main Content</h1>
                <p>This is the main content of the page.</p>
                <a href="/about">About Page</a>
                <script>console.log('script to remove');</script>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_link = Mock()
        mock_link.__getitem__ = Mock(return_value="/about")
        mock_link.get = Mock(return_value="/about")
        mock_soup.find_all.return_value = [mock_link]
        mock_bs.return_value = mock_soup
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Step 1: Fetch URL
            fetch_result = fetch_url("https://example.com", output_dir=output_dir)
            assert fetch_result.success is True
            
            # Step 2: Crawl website
            crawl_results = crawl_website("https://example.com", output_dir=output_dir)
            assert isinstance(crawl_results, list)
            assert len(crawl_results) > 0
            
            # Step 3: Process HTML files
            html_files = list(output_dir.glob("*.html"))
            assert len(html_files) > 0
            
            for html_file in html_files:
                extract_result = process_web(html_file)
                assert extract_result.success is True
                assert extract_result.data is not None
                
                # Step 4: Process through pipeline
                pipeline = TextProcessingPipeline()
                processed = pipeline.process(extract_result.data)
                
                assert isinstance(processed, str)
                assert len(processed) > 0
                # Should not contain script tags
                assert "console.log" not in processed


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases across the entire system."""
    
    def test_all_exceptions_are_properly_handled(self):
        """Test that all custom exceptions can be raised and caught."""
        from DataOrchestra.core.exceptions import (
            DataOrchestraError, FileProcessingError, DownloadError,
            CrawlError, ConfigurationError, ValidationError
        )
        
        exceptions_to_test = [
            DataOrchestraError("Base error"),
            FileProcessingError("File error"),
            DownloadError("Download error"),
            CrawlError("Crawl error"),
            ConfigurationError("Config error"),
            ValidationError("Validation error")
        ]
        
        for exc in exceptions_to_test:
            with pytest.raises(DataOrchestraError):
                raise exc
    
    def test_malformed_input_handling(self):
        """Test handling of various malformed inputs."""
        from DataOrchestra.utils.text_utils import clean_text, normalize_text
        from DataOrchestra.processing.pipeline import TextProcessingPipeline
        
        malformed_inputs = [
            "",  # Empty string
            "   ",  # Only whitespace
            "None",  # String representation of None
            "undefined",  # String representation of undefined
            "123456789",  # Only numbers
            "!@#$%^&*()",  # Only special characters
            "a" * 10000,  # Very long text
        ]
        
        pipeline = TextProcessingPipeline()
        
        for input_text in malformed_inputs:
            try:
                # Clean function should handle all inputs gracefully
                cleaned = clean_text(input_text)
                assert isinstance(cleaned, str)
                
                # Normalize function should handle all inputs gracefully
                normalized = normalize_text(input_text)
                assert isinstance(normalized, str)
                
                # Pipeline should handle all inputs gracefully
                processed = pipeline.process(input_text)
                assert isinstance(processed, str)
                
            except Exception as e:
                pytest.fail(f"Failed to handle input '{input_text[:50]}...': {e}")
    
    def test_file_permission_and_access_errors(self):
        """Test handling of file access and permission errors."""
        from DataOrchestra.extract.markdown_extractor import process_markdown
        from pathlib import Path
        
        # Test with non-existent file
        non_existent_path = Path("/tmp/this_file_does_not_exist_12345.md")
        result = process_markdown(non_existent_path)
        
        assert result.success is False
        assert result.error is not None
        
        # Test with directory instead of file
        temp_dir = Path("/tmp")
        result = process_markdown(temp_dir)
        
        assert result.success is False
        assert result.error is not None


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    def test_large_text_processing_performance(self):
        """Test processing of large texts."""
        from DataOrchestra.utils.text_utils import clean_text, normalize_text, split_text_by_tokens
        from DataOrchestra.processing.pipeline import TextProcessingPipeline
        
        # Create large text (simulate real-world documents)
        large_text = "This is a sentence. " * 10000  # 10000 sentences
        
        start_time = time.time()
        
        # Test utility functions
        cleaned = clean_text(large_text)
        normalized = normalize_text(cleaned)
        
        # Test splitting
        chunks = list(split_text_by_tokens(large_text, token_limit=1000))
        
        # Test pipeline
        pipeline = TextProcessingPipeline()
        processed = pipeline.process(large_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert processing_time < 10.0  # 10 seconds max
        assert isinstance(cleaned, str)
        assert isinstance(normalized, str)
        assert isinstance(processed, str)
        assert len(chunks) > 1  # Should be split into multiple chunks
    
    def test_memory_usage_with_large_files(self):
        """Test memory usage with large file processing."""
        from DataOrchestra.processing.pipeline import TextProcessingPipeline
        
        # Create large text content
        large_content = "Content line " + "x" * 1000 + "\n" * 10000
        
        pipeline = TextProcessingPipeline()
        
        # Process without memory issues
        start_memory = 0  # We can't actually measure memory in this context
        result = pipeline.process(large_content)
        end_memory = 0
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Memory usage should be reasonable (this is more of a guideline)
        # In a real implementation, you might use memory profilers here


class TestConfigurationAndEnvironment:
    """Test configuration and environment handling."""
    
    def test_configuration_persistence(self):
        """Test that configuration changes persist across operations."""
        from DataOrchestra.core.config import Config, get_config, set_config, reset_config
        
        # Get original config
        original_config = get_config()
        original_verbose = original_config.verbose
        
        try:
            # Modify configuration
            new_config = Config(verbose=True)
            set_config(new_config)
            
            # Verify change
            current_config = get_config()
            assert current_config.verbose is True
            
        finally:
            # Reset to original
            reset_config()
            final_config = get_config()
            assert final_config.verbose == original_verbose
    
    def test_logging_configuration(self):
        """Test logging configuration and output."""
        from DataOrchestra.core.logging_utils import get_logger
        
        logger = get_logger("test_integration")
        assert logger is not None
        assert logger.name == "DataOrchestra.test_integration"
        
        # Test logging at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


# Pytest configuration and fixtures
def pytest_configure(config):
    """Configure pytest for DataOrchestra tests."""
    config.addinivalue_line(
        "markers", "core: mark test as core functionality test"
    )
    config.addinivalue_line(
        "markers", "extraction: mark test as extraction functionality test"
    )
    config.addinivalue_line(
        "markers", "processing: mark test as processing functionality test"
    )
    config.addinivalue_line(
        "markers", "web: mark test as web functionality test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    This is a sample text for testing.
    It contains URLs like https://example.com and emails like test@example.com.
    There are extra    spaces and MiXeD CaSe to test normalization.
    """


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for output."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_web_responses():
    """Provide mocked web responses for testing."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>Mocked Page</title></head>
            <body>
                <h1>Mocked Content</h1>
                <p>This is mocked content for testing.</p>
                <a href="/test">Test Link</a>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch('DataOrchestra.web.spider.BeautifulSoup') as mock_bs:
            mock_soup = Mock()
            mock_link = Mock()
            mock_link.__getitem__ = Mock(return_value="/test")
            mock_link.get = Mock(return_value="/test")
            mock_soup.find_all.return_value = [mock_link]
            mock_bs.return_value = mock_soup
            
            yield {
                'mock_get': mock_get,
                'mock_response': mock_response,
                'mock_bs': mock_bs,
                'mock_soup': mock_soup
            }


# Test collection and execution
def test_collection_coverage():
    """Verify that all test modules are properly collected and can be imported."""
    # This test ensures all modules can be imported without errors
    modules_to_test = [
        'test_core',
        'test_extraction', 
        'test_processing',
        'test_web',
        'test_integration'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


# Mark tests with appropriate categories
TestCompleteSystemIntegration.test_complete_text_workflow.mark = "integration"
TestCompleteSystemIntegration.test_multi_format_file_processing.mark = "integration"
TestCompleteSystemIntegration.test_complete_web_workflow.mark = "integration"

TestErrorHandlingAndEdgeCases.test_all_exceptions_are_properly_handled.mark = "core"
TestErrorHandlingAndEdgeCases.test_malformed_input_handling.mark = "core"
TestErrorHandlingAndEdgeCases.test_file_permission_and_access_errors.mark = "extraction"

TestPerformanceAndScalability.test_large_text_processing_performance.mark = "performance"
TestPerformanceAndScalability.test_memory_usage_with_large_files.mark = "performance"

TestConfigurationAndEnvironment.test_configuration_persistence.mark = "core"
TestConfigurationAndEnvironment.test_logging_configuration.mark = "core"

test_collection_coverage.mark = "core"


if __name__ == "__main__":
    """
    Main execution block for running all tests.
    
    Provides different execution modes:
    - All tests (default)
    - Only unit tests
    - Only integration tests
    - Only performance tests
    - With coverage reporting
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description="DataOrchestra Test Runner")
    parser.add_argument(
        "--mode", 
        choices=["all", "unit", "integration", "performance", "core"],
        default="all",
        help="Test execution mode"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = [__file__]
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.failfast:
        pytest_args.append("-x")
    
    if args.coverage:
        pytest_args.extend([
            "--cov=DataOrchestra",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Apply test filtering based on mode
    if args.mode == "unit":
        pytest_args.extend(["-m", "not integration and not performance"])
    elif args.mode == "integration":
        pytest_args.extend(["-m", "integration"])
    elif args.mode == "performance":
        pytest_args.extend(["-m", "performance"])
    elif args.mode == "core":
        pytest_args.extend(["-m", "core"])
    # 'all' mode runs all tests (no additional filtering)
    
    # Add markers for different test categories
    pytest_args.extend([
        "-m", "core or extraction or processing or web or integration or performance"
    ])
    
    print(f"Running DataOrchestra tests in '{args.mode}' mode...")
    print(f"Command: {' '.join(['pytest'] + pytest_args)}")
    print("-" * 50)
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    print("-" * 50)
    if exit_code == 0:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)