"""
Integration tests for DataOrchestra.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import time

from DataOrchestra.extract import process_pdf, process_docx, process_web, process_markdown
from DataOrchestra.utils import clean_text, normalize_text, split_text_by_tokens
from DataOrchestra.web import fetch_url, crawl_website
from DataOrchestra.core.base import FileProcessingResult
from DataOrchestra.processing.pipeline import TextProcessingPipeline
from DataOrchestra.extract.pdf_extractor import process_pdf as extract_pdf
from DataOrchestra.extract.docx_extractor import process_docx as extract_docx
from DataOrchestra.extract.web_extractor import process_web as extract_web
from DataOrchestra.extract.markdown_extractor import process_markdown as extract_markdown
from DataOrchestra.utils.text_utils import clean_text as util_clean_text
from DataOrchestra.utils.text_utils import normalize_text as util_normalize_text
from DataOrchestra.utils.text_utils import split_text_by_tokens as util_split_text
from DataOrchestra.web.fetcher import fetch_url as web_fetch_url
from DataOrchestra.web.spider import crawl_website as web_crawl_website


class TestModuleIntegration:
    """Integration tests for different modules working together."""
    
    def test_end_to_end_text_processing(self):
        """Test end-to-end text processing workflow."""
        # Create a sample text with various issues
        raw_text = """
        Visit https://example.com or email test@example.com for more info!
        This has extra    whitespace and MiXeD CaSe.
        Repetitive sentence. Repetitive sentence. Another sentence here.
        """
        
        # Step 1: Clean the text using utility functions
        cleaned_text = util_clean_text(raw_text)
        normalized_text = util_normalize_text(cleaned_text)
        
        # Step 2: Process through the pipeline
        pipeline = TextProcessingPipeline()
        processed_text = pipeline.process(normalized_text)
        
        # Verify the text has been processed
        assert isinstance(processed_text, str)
        assert len(processed_text) > 0
        # Should be different from original due to cleaning
        assert processed_text != raw_text
        # Should not contain URLs or emails
        assert "https://example.com" not in processed_text
        assert "test@example.com" not in processed_text
        # Should be in lowercase
        assert processed_text == processed_text.lower()
    
    def test_file_processing_pipeline(self):
        """Test processing files through the complete pipeline."""
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write("# Test Document\n\nThis is a test document with [link](https://example.com) and other content.")
            tmp_path = Path(tmp.name)
        
        try:
            # Process the markdown file
            result = extract_markdown(tmp_path)
            
            assert isinstance(result, FileProcessingResult)
            assert result.success is True
            assert result.data is not None
            
            # Process the extracted content through the pipeline
            pipeline = TextProcessingPipeline()
            processed_content = pipeline.process(result.data)
            
            assert isinstance(processed_content, str)
            assert len(processed_content) > 0
        finally:
            os.unlink(tmp_path)
    
    @patch('requests.get')
    @patch('DataOrchestra.web.spider.BeautifulSoup')
    def test_web_data_processing_pipeline(self, mock_bs, mock_get):
        """Test processing web data from fetching to extraction to processing."""
        # Mock web response
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Web Content</h1>
                <p>This is web content with <a href="https://example.com">a link</a>.</p>
                <script>console.log('remove this');</script>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_link = Mock()
        mock_link.__getitem__ = Mock(return_value="/test")
        mock_link.get = Mock(return_value="/test")
        mock_soup.find_all.return_value = [mock_link]
        mock_bs.return_value = mock_soup
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Fetch the web page
            fetch_result = web_fetch_url("https://example.com", output_dir=output_dir)
            assert fetch_result.success is True
            
            # Process the fetched HTML file
            html_files = list(output_dir.glob("*.html"))
            if html_files:
                process_result = extract_web(html_files[0])
                assert process_result.success is True
                assert process_result.data is not None
                
                # Process through pipeline
                pipeline = TextProcessingPipeline()
                final_result = pipeline.process(process_result.data)
                assert isinstance(final_result, str)
                assert len(final_result) > 0


class TestCrossModuleCompatibility:
    """Test compatibility between different modules."""
    
    def test_pipeline_accepts_utility_output(self):
        """Test that pipeline can process output from utility functions."""
        original_text = "This is a TEST with extra   SPACES and UPPERCASE."
        
        # Process with utility functions
        cleaned = util_clean_text(original_text)
        normalized = util_normalize_text(cleaned)
        
        # Process with pipeline
        pipeline = TextProcessingPipeline()
        result = pipeline.process(normalized)
        
        assert isinstance(result, str)
        assert result != original_text  # Should be different due to processing
        assert result == result.lower() # Should be lowercase
    
    def test_split_then_process(self):
        """Test splitting text and then processing each chunk."""
        long_text = " ".join([f"This is sentence {i} with some content." for i in range(100)])
        
        # Split into chunks
        chunks = list(util_split_text(long_text, token_limit=20))
        assert len(chunks) > 1
        
        # Process each chunk
        pipeline = TextProcessingPipeline()
        processed_chunks = [pipeline.process(chunk) for chunk in chunks]
        
        # Verify all chunks were processed
        assert len(processed_chunks) == len(chunks)
        for chunk in processed_chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0


class TestRealWorldWorkflows:
    """Test real-world usage workflows."""
    
    def test_document_analysis_workflow(self):
        """Test a document analysis workflow."""
        # Create a sample document with various content types
        sample_content = """
        # Project Report
        This report analyzes the performance of our systems.
        
        ## Key Findings
        - Performance increased by 25%
        - Error rate decreased significantly
        - User satisfaction improved
        
        Contact: admin@example.com
        Visit: https://company.com/reports
        """
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(sample_content)
            tmp_path = Path(tmp.name)
        
        try:
            # Extract content
            extract_result = extract_markdown(tmp_path)
            assert extract_result.success is True
            
            # Process through pipeline
            pipeline = TextProcessingPipeline()
            processed = pipeline.process(extract_result.data)
            
            # Verify processing worked
            assert isinstance(processed, str)
            assert "performance" in processed.lower()
            assert "contact" not in processed.lower()  # Should be filtered out
            assert "admin@example.com" not in processed  # Should be removed
        finally:
            os.unlink(tmp_path)
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple files."""
        # Create multiple temporary files
        file_paths = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
                    tmp.write(f"# Document {i}\n\nContent for document {i} with some text.")
                    file_paths.append(Path(tmp.name))
            
            # Process all files
            results = []
            for path in file_paths:
                result = extract_markdown(path)
                results.append(result)
            
            # Verify all were processed
            assert len(results) == 3
            for result in results:
                assert isinstance(result, FileProcessingResult)
                assert result.success is True
                assert result.data is not None
            
            # Process all results through pipeline
            pipeline = TextProcessingPipeline()
            processed_results = [pipeline.process(result.data) for result in results]
            
            # Verify all were processed
            assert len(processed_results) == 3
            for processed in processed_results:
                assert isinstance(processed, str)
                assert len(processed) > 0
        finally:
            # Clean up
            for path in file_paths:
                os.unlink(path)


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])