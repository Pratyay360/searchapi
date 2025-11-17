"""
Test suite for extraction functionality of DataOrchestra.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from DataOrchestra.extract.pdf_extractor import process_pdf
from DataOrchestra.extract.docx_extractor import process_docx
from DataOrchestra.extract.markdown_extractor import process_markdown
from DataOrchestra.extract.web_extractor import process_web
from DataOrchestra.core.base import FileProcessingResult
from DataOrchestra.core.exceptions import FileProcessingError


class TestPDFExtractor:
    """Test cases for PDF extraction functionality."""
    
    def test_process_pdf_with_mock(self):
        """Test PDF processing with mocked pdfplumber."""
        with patch('DataOrchestra.extract.pdf_extractor.pdfplumber.open') as mock_pdf:
            # Create a mock PDF object
            mock_pdf_instance = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample PDF text content"
            mock_pdf_instance.__enter__.return_value = mock_pdf_instance
            mock_pdf_instance.__exit__.return_value = None
            mock_pdf.return_value = mock_pdf_instance
            mock_pdf_instance.pages = [mock_page]
            
            # Create a temporary file to simulate PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                
                try:
                    result = process_pdf(tmp_path)
                    
                    assert isinstance(result, FileProcessingResult)
                    assert result.success is True
                    assert result.error is None
                    assert result.data is not None
                    assert "Sample PDF text content" in result.data
                finally:
                    # Clean up
                    os.unlink(tmp_path)
    
    @patch('DataOrchestra.extract.pdf_extractor.pdfplumber.open')
    def test_process_pdf_with_error(self, mock_pdf):
        """Test PDF processing with error."""
        mock_pdf.side_effect = Exception("PDF processing error")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
            try:
                result = process_pdf(tmp_path)
                
                assert isinstance(result, FileProcessingResult)
                assert result.success is False
                assert result.error is not None
                assert "PDF processing error" in result.error
            finally:
                # Clean up
                os.unlink(tmp_path)


class TestDOCXExtractor:
    """Test cases for DOCX extraction functionality."""
    
    def test_process_docx_with_mock(self):
        """Test DOCX processing with mocked python-docx."""
        with patch('DataOrchestra.extract.docx_extractor.Document') as mock_docx:
            # Create a mock DOCX object
            mock_docx_instance = Mock()
            mock_paragraph1 = Mock()
            mock_paragraph1.text = "First paragraph text"
            mock_paragraph2 = Mock()
            mock_paragraph2.text = "Second paragraph text"
            mock_docx_instance.paragraphs = [mock_paragraph1, mock_paragraph2]
            mock_docx.return_value = mock_docx_instance
            
            # Create a temporary file to simulate DOCX
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                
                try:
                    result = process_docx(tmp_path)
                    
                    assert isinstance(result, FileProcessingResult)
                    assert result.success is True
                    assert result.error is None
                    assert result.data is not None
                    assert "First paragraph text" in result.data
                    assert "Second paragraph text" in result.data
                finally:
                    # Clean up
                    os.unlink(tmp_path)
    
    @patch('DataOrchestra.extract.docx_extractor.Document')
    def test_process_docx_with_error(self, mock_docx):
        """Test DOCX processing with error."""
        mock_docx.side_effect = Exception("DOCX processing error")
        
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
            try:
                result = process_docx(tmp_path)
                
                assert isinstance(result, FileProcessingResult)
                assert result.success is False
                assert result.error is not None
                assert "DOCX processing error" in result.error
            finally:
                # Clean up
                os.unlink(tmp_path)


class TestMarkdownExtractor:
    """Test cases for Markdown extraction functionality."""
    
    def test_process_markdown_success(self):
        """Test successful Markdown processing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write("# Test Markdown\n\nThis is **bold** text and *italic* text.\n\n- Item 1\n- Item 2")
            tmp_path = Path(tmp.name)
        
        try:
            result = process_markdown(tmp_path)
            
            assert isinstance(result, FileProcessingResult)
            assert result.success is True
            assert result.error is None
            assert result.data is not None
            assert "Test Markdown" in result.data
            assert "bold" in result.data
            assert "italic" in result.data
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_process_markdown_with_error(self):
        """Test Markdown processing with error."""
        # Try to process a non-existent file
        result = process_markdown(Path("/nonexistent/file.md"))
        
        assert isinstance(result, FileProcessingResult)
        assert result.success is False
        assert result.error is not None


class TestWebExtractor:
    """Test cases for Web extraction functionality."""
    
    def test_process_web_success(self):
        """Test successful web content processing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            tmp.write("""
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <h1>Main Title</h1>
                    <p>This is a test paragraph with some content.</p>
                    <p>Another paragraph for testing.</p>
                    <script>console.log('test');</script>
                    <style>body { color: red; }</style>
                </body>
            </html>
            """)
            tmp_path = Path(tmp.name)
        
        try:
            result = process_web(tmp_path)
            
            assert isinstance(result, FileProcessingResult)
            assert result.success is True
            assert result.error is None
            assert result.data is not None
            assert "Test Page" in result.data
            assert "Main Title" in result.data
            assert "test paragraph" in result.data
            # Should not contain script or style content
            assert "console.log" not in result.data
            assert "color: red" not in result.data
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_process_web_with_error(self):
        """Test web processing with error."""
        # Try to process a non-existent file
        result = process_web(Path("/nonexistent/file.html"))
        
        assert isinstance(result, FileProcessingResult)
        assert result.success is False
        assert result.error is not None


class TestExtractionIntegration:
    """Integration tests for extraction functionality."""
    
    def test_extraction_output_formats(self):
        """Test that extraction functions return consistent output formats."""
        # Create test files for each type
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create markdown file
            md_path = temp_path / "test.md"
            with open(md_path, "w") as f:
                f.write("# Test\n\nContent")
            
            # All extraction functions should return FileProcessingResult
            markdown_result = process_markdown(md_path)
            assert isinstance(markdown_result, FileProcessingResult)
            
            # For PDF and DOCX, we use mocks since we don't have real files
            with patch('DataOrchestra.extract.pdf_extractor.pdfplumber.open') as mock_pdf:
                mock_pdf_instance = Mock()
                mock_page = Mock()
                mock_page.extract_text.return_value = "Test PDF content"
                mock_pdf_instance.__enter__.return_value = mock_pdf_instance
                mock_pdf_instance.__exit__.return_value = None
                mock_pdf.return_value = mock_pdf_instance
                mock_pdf_instance.pages = [mock_page]
                
                pdf_path = temp_path / "test.pdf"
                with open(pdf_path, "w") as f:
                    f.write("dummy")  # Just to create the file
                
                pdf_result = process_pdf(pdf_path)
                assert isinstance(pdf_result, FileProcessingResult)
            
            with patch('DataOrchestra.extract.docx_extractor.Document') as mock_docx:
                mock_docx_instance = Mock()
                mock_paragraph = Mock()
                mock_paragraph.text = "Test DOCX content"
                mock_docx_instance.paragraphs = [mock_paragraph]
                mock_docx.return_value = mock_docx_instance
                
                docx_path = temp_path / "test.docx"
                with open(docx_path, "w") as f:
                    f.write("dummy")  # Just to create the file
                
                docx_result = process_docx(docx_path)
                assert isinstance(docx_result, FileProcessingResult)
            
            # Create HTML file
            html_path = temp_path / "test.html"
            with open(html_path, "w") as f:
                f.write("<html><body>Test HTML content</body></html>")
            
            html_result = process_web(html_path)
            assert isinstance(html_result, FileProcessingResult)


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])
