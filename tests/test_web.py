"""
Test suite for web functionality of DataOrchestra.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import requests

from DataOrchestra.web.fetcher import fetch_url
from DataOrchestra.web.spider import crawl_website, WebSpider
from DataOrchestra.core.base import FileProcessingResult
from DataOrchestra.core.exceptions import DownloadError, CrawlError
from DataOrchestra.core.config import get_config


class TestWebFetcher:
    """Test cases for web fetching functionality."""
    
    @patch('requests.get')
    def test_fetch_url_success(self, mock_get):
        """Test successful URL fetching."""
        # Mock the response
        mock_response = Mock()
        mock_response.text = "Mocked HTML content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = fetch_url("https://example.com", output_dir=output_dir)
            
            assert isinstance(result, FileProcessingResult)
            assert result.success is True
            assert result.error is None
            assert result.output_path is not None
            assert result.output_path.exists()
    
    @patch('requests.get')
    def test_fetch_url_with_error(self, mock_get):
        """Test URL fetching with request error."""
        # Mock a request exception
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(DownloadError):
            fetch_url("https://invalid-url.com")
    
    @patch('requests.get')
    def test_fetch_url_http_error(self, mock_get):
        """Test URL fetching with HTTP error."""
        # Mock an HTTP error response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(DownloadError):
            fetch_url("https://example.com/404")
    
    def test_fetch_url_invalid_url(self):
        """Test fetching with invalid URL."""
        with pytest.raises(DownloadError):
            fetch_url("invalid-url")


class TestWebSpider:
    """Test cases for web spider functionality."""
    
    @patch('requests.get')
    @patch('DataOrchestra.web.spider.BeautifulSoup')
    def test_crawl_website_success(self, mock_bs, mock_get):
        """Test successful website crawling."""
        # Mock the response
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="https://other-site.com">Other Site</a>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_link1 = Mock()
        mock_link1.__getitem__ = Mock(return_value="/page1")
        mock_link1.get = Mock(return_value="/page1")
        mock_link2 = Mock()
        mock_link2.__getitem__ = Mock(return_value="/page2")
        mock_link2.get = Mock(return_value="/page2")
        mock_link3 = Mock()
        mock_link3.__getitem__ = Mock(return_value="https://other-site.com")
        mock_link3.get = Mock(return_value="https://other-site.com")
        
        mock_soup.find_all.return_value = [mock_link1, mock_link2, mock_link3]
        mock_bs.return_value = mock_soup
        
        # Test crawling
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            results = crawl_website("https://example.com", output_dir=output_dir)
            
            assert isinstance(results, list)
            # At least the base page should be processed
            assert len(results) >= 1
            for result in results:
                assert isinstance(result, FileProcessingResult)
    
    @patch('requests.get')
    def test_web_spider_initialization(self, mock_get):
        """Test WebSpider initialization."""
        spider = WebSpider("https://example.com")
        
        assert spider.base_url == "https://example.com"
        assert spider.domain == "example.com"
        assert isinstance(spider.visited_urls, set)
        assert spider.config is not None
    
    @patch('requests.get')
    def test_crawl_with_limits(self, mock_get):
        """Test crawling with page and depth limits."""
        # Mock the response
        mock_response = Mock()
        mock_response.text = "<html><body><a href='/page2'>Page 2</a></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        spider = WebSpider("https://example.com")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            spider.output_dir = output_dir
            
            # Test with limits
            results = spider.crawl(max_pages=2, max_depth=1)
            
            assert isinstance(results, list)
            assert len(results) <= 2  # Respects page limit
    
    @patch('requests.get')
    def test_crawl_with_request_error(self, mock_get):
        """Test crawling with request errors."""
        # Mock a request exception
        mock_get.side_effect = requests.RequestException("Connection error")
        
        spider = WebSpider("https://example.com")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            spider.output_dir = output_dir
            
            with pytest.raises(CrawlError):
                spider.crawl(max_pages=1, max_depth=1)
    
    def test_crawl_with_invalid_base_url(self):
        """Test crawling with invalid base URL."""
        with pytest.raises(CrawlError):
            crawl_website("invalid-url")


class TestWebIntegration:
    """Integration tests for web functionality."""
    
    @patch('requests.get')
    @patch('DataOrchestra.web.spider.BeautifulSoup')
    def test_fetch_and_crawl_integration(self, mock_bs, mock_get):
        """Test integration between fetch and crawl functionality."""
        # Mock responses for both fetch and crawl
        mock_response = Mock()
        mock_response.text = "<html><body><a href='/test'>Test</a></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_link = Mock()
        mock_link.__getitem__ = Mock(return_value="/test")
        mock_link.get = Mock(return_value="/test")
        mock_soup.find_all.return_value = [mock_link]
        mock_bs.return_value = mock_soup
        
        # Test fetching
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Fetch single URL
            fetch_result = fetch_url("https://example.com", output_dir=output_dir)
            assert fetch_result.success is True
            
            # Crawl website
            crawl_results = crawl_website("https://example.com", output_dir=output_dir)
            assert isinstance(crawl_results, list)
            assert len(crawl_results) >= 1


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])