"""
Comprehensive testing suite for web scraping components.
"""
from __future__ import annotations

import asyncio
import pytest
import time
from unittest.mock import Mock, patch
from typing import Any, Dict, List
from pathlib import Path

from DataOrchestra.web.rate_limiter import (
    RateLimiter, RateLimitConfig, RateLimitStrategy, 
    get_rate_limiter, reset_rate_limiter
)
from DataOrchestra.web.browser_renderer import (
    BrowserRenderer, BrowserConfig, BrowserType, RenderMode
)
from DataOrchestra.web.search_fallbacks import (
    FallbackSearchManager, FallbackStrategy, test_all_providers
)
from DataOrchestra.web.input_sanitizer import (
    InputSanitizer, SanitizationConfig, ValidationLevel, InputType
)
from DataOrchestra.web.retry_manager import (
    RetryManager, RetryConfig, RetryStrategy, execute_with_retry
)
from DataOrchestra.core.exceptions import CrawlError, ValidationError
from DataOrchestra.core.errors import NetworkError, SecurityError


class TestRateLimiter:
    """Test cases for rate limiting."""
    
    def test_fixed_delay_strategy(self):
        """Test fixed delay strategy."""
        config = RateLimitConfig(strategy=RateLimitStrategy.FIXED, base_delay=1.0)
        limiter = RateLimiter(config)
        
        # Test basic delay
        delay = asyncio.run(limiter.acquire())
        assert delay == 1.0
        
        # Test consecutive failures
        limiter.record_failure("test_error")
        delay = asyncio.run(limiter.acquire())
        assert delay == 2.0  # Exponential backoff
        
        # Test success recovery
        limiter.record_success()
        delay = asyncio.run(limiter.acquire())
        assert delay == 1.0  # Back to base delay
        
        stats = limiter.get_stats()
        assert stats["consecutive_failures"] == 0
        assert stats["current_delay"] == 1.0
    
    def test_adaptive_strategy(self):
        """Test adaptive strategy."""
        config = RateLimitConfig(
            strategy=RateLimitStrategy.ADAPTIVE,
            base_delay=1.0,
            adaptive_factor=1.5,
            backoff_factor=2.0
        )
        limiter = RateLimiter(config)
        
        # Simulate successful requests to reduce delay
        for _ in range(5):
            limiter.record_success()
        
        delay = asyncio.run(limiter.acquire())
        assert delay < 1.0  # Reduced due to success
        
        stats = limiter.get_stats()
        assert stats["average_delay"] < 1.0
    
    def test_token_bucket_strategy(self):
        """Test token bucket strategy."""
        config = RateLimitConfig(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_window=10,
            window_size=60,
            burst_capacity=5
        )
        limiter = RateLimiter(config)
        
        # Test token consumption
        for _ in range(7):  # Should exhaust after 5 requests
            delay = asyncio.run(limiter.acquire())
            if _ < 5:
                assert delay == 0.0  # No delay when tokens available
            else:
                assert delay > 0.0  # Delay when tokens exhausted
        
        stats = limiter.get_stats()
        assert stats["total_requests"] == 7
    
    def test_multi_domain_limiter(self):
        """Test multi-domain rate limiting."""
        config = RateLimitConfig(strategy=RateLimitStrategy.ADAPTIVE)
        limiter = get_rate_limiter(config)
        
        # Test different domains
        domains = ["example.com", "test.org", "api.example.net"]
        
        for domain in domains:
            delay = asyncio.run(limiter.acquire(domain))
            assert isinstance(delay, float)
            limiter.record_success(domain)
        
        # Test domain-specific stats
        all_stats = limiter.get_stats()
        assert len(all_stats) == 3
        
        for domain in domains:
            domain_stats = limiter.get_stats(domain)
            assert domain_stats["total_requests"] == 1


class TestBrowserRenderer:
    """Test cases for browser rendering."""
    
    @pytest.mark.asyncio
    async def test_basic_rendering(self):
        """Test basic page rendering."""
        config = BrowserConfig(
            browser_type=BrowserType.CHROME,
            headless=True,
            timeout=30.0
        )
        
        async with BrowserRenderer(config) as renderer:
            # Test successful rendering
            result = await renderer.render_page("https://example.com")
            
            assert result["success"] is True
            assert "content" in result
            assert "metadata" in result
            assert result["render_time"] > 0
            
            # Test metadata extraction
            metadata = result["metadata"]
            assert "title" in metadata
            assert "url" in metadata
    
    @pytest.mark.asyncio
    async def test_javascript_execution(self):
        """Test JavaScript execution."""
        config = BrowserConfig(browser_type=BrowserType.CHROME, headless=True)
        
        async with BrowserRenderer(config) as renderer:
            # Test JavaScript execution
            result = await renderer.execute_javascript("return document.title;")
            
            assert result is not None
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in browser rendering."""
        config = BrowserConfig(browser_type=BrowserType.CHROME, headless=True)
        
        async with BrowserRenderer(config) as renderer:
            # Test invalid URL
            with pytest.raises(CrawlError):
                await renderer.render_page("invalid-url")
            
            # Test timeout
            config.timeout = 0.001  # Very short timeout
            renderer_with_timeout = BrowserRenderer(config)
            
            with pytest.raises(CrawlError):
                await renderer_with_timeout.render_page("https://example.com")


class TestSearchFallbacks:
    """Test cases for search fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_provider_fallback(self):
        """Test provider fallback functionality."""
        # Test all providers
        results = await test_all_providers("test query", max_results=5)
        
        assert len(results) > 0
        assert any(results[provider]["success"] for provider in results.keys())
        
        # Test specific provider
        manager = FallbackSearchManager(FallbackStrategy.PRIORITY_BASED)
        duckduckgo_results = await manager.search("test query", max_results=3)
        
        assert len(duckduckgo_results) >= 0
    
    def test_error_handling(self):
        """Test error handling in search fallbacks."""
        import asyncio
        manager = FallbackSearchManager()
        
        # Test with all providers failing
        with patch('DataOrchestra.web.search_fallbacks.DuckDuckGoAPI.search') as mock_search:
            mock_search.side_effect = Exception("Network error")
            
            with pytest.raises(NetworkError):
                asyncio.run(manager.search("test query"))


class TestInputSanitizer:
    """Test cases for input sanitization."""
    
    def test_url_sanitization(self):
        """Test URL sanitization."""
        config = SanitizationConfig(validation_level=ValidationLevel.STRICT)
        sanitizer = InputSanitizer(config)
        
        # Valid URLs
        valid_urls = [
            "https://example.com",
            "http://test.org/path",
            "https://sub.example.com"
        ]
        
        for url in valid_urls:
            result = sanitizer.sanitize_url(url, InputType.URL)
            assert result == url  # Should not change valid URLs
        
        # Invalid URLs
        invalid_urls = [
            "javascript:alert('XSS')",
            "file:///etc/passwd",
            "http://192.168.1.1",  # Private IP
            "https://malicious.com",
            "ftp://example.com",  # Invalid scheme
        ]
        
        for url in invalid_urls:
            with pytest.raises((ValidationError, SecurityError)):
                sanitizer.sanitize_url(url, InputType.URL)
    
    def test_query_sanitization(self):
        """Test query sanitization."""
        config = SanitizationConfig(validation_level=ValidationLevel.MODERATE)
        sanitizer = InputSanitizer(config)
        
        # Valid queries
        valid_queries = [
            "test search",
            "hello world",
            "data analysis"
        ]
        
        for query in valid_queries:
            result = sanitizer.sanitize_query(query, InputType.QUERY)
            assert result == query  # Should not change valid queries
        
        # Invalid queries
        invalid_queries = [
            "<script>alert('XSS')</script>",
            "'; DROP TABLE users; --",
            "${jndi:ldap://example.com}",
            "data:text/html,<script>alert(1)</script>"
        ]
        
        for query in invalid_queries:
            result = sanitizer.sanitize_query(query, InputType.QUERY)
            # Should remove dangerous content
            assert "<script>" not in result
            assert "alert" not in result.lower()
    
    def test_html_sanitization(self):
        """Test HTML content sanitization."""
        config = SanitizationConfig(validation_level=ValidationLevel.STRICT)
        sanitizer = InputSanitizer(config)
        
        # Dangerous HTML
        dangerous_html = """
        <script>alert('XSS')</script>
        <div onclick="malicious()">Click me</div>
        <iframe src="javascript:alert('XSS')"></iframe>
        """
        
        result = sanitizer.sanitize_html_content(dangerous_html)
        
        # Should remove dangerous elements
        assert "<script>" not in result
        assert "onclick" not in result
        assert "<iframe>" not in result
        assert "malicious" not in result
    
    def test_file_validation(self):
        """Test file upload validation."""
        config = SanitizationConfig(max_file_size=1024)
        sanitizer = InputSanitizer(config)
        
        # Valid file
        valid_content = b"Valid file content"
        result = sanitizer.validate_file_upload("test.txt", valid_content, "text/plain")
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Oversized file
        oversized_content = b"x" * (1024 * 1024 + 1)  # Over 1GB
        result = sanitizer.validate_file_upload("large.txt", oversized_content, "text/plain")
        
        assert result["valid"] is False
        assert any("too large" in error for error in result["errors"])


class TestRetryManager:
    """Test cases for retry manager."""
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff retry."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay=1.0,
            backoff_multiplier=2.0
        )
        
        call_count = 0
        successful_calls = []
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("Simulated failure")
            return "success"
        
        # Test retry with eventual success
        result = await execute_with_retry(
            failing_operation,
            "test_operation",
            config=config
        )
        
        assert result == "success"
        assert call_count == 3  # 2 failures + 1 success
        
        # Check stats
        retry_manager = RetryManager(config)
        stats = retry_manager.get_operation_stats("test_operation")
        assert stats["total_attempts"] == 3
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,
            circuit_breaker_threshold=3
        )
        
        retry_manager = RetryManager(config)
        
        async def always_failing_operation():
            raise NetworkError("Always fails")
        
        # Should open circuit after threshold
        for _ in range(4):
            try:
                await execute_with_retry(
                    always_failing_operation,
                    "circuit_test",
                    config=config
                )
            except NetworkError:
                pass  # Expected
        
        # Should be in circuit breaker state
        stats = retry_manager.get_operation_stats("circuit_test")
        assert stats["total_attempts"] >= 5


# Integration tests
class TestWebScrapingIntegration:
    """Integration tests for web scraping components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete web scraping workflow."""
        # Initialize components
        rate_limiter = get_rate_limiter()
        browser_config = BrowserConfig(headless=True)
        search_manager = FallbackSearchManager()
        sanitizer = InputSanitizer()
        
        # Test URL sanitization
        safe_url = sanitizer.sanitize_url("https://example.com/test", InputType.URL)
        assert safe_url == "https://example.com/test"
        
        # Test search with fallbacks
        search_results = await search_manager.search("test query", max_results=5)
        assert len(search_results) > 0
        
        # Test browser rendering
        async with BrowserRenderer(browser_config) as renderer:
            if search_results:
                result = await renderer.render_page(search_results[0])
                assert result["success"] is True
                
                # Test retry with rendered content
                retry_result = await execute_with_retry(
                    lambda: renderer.render_page("https://example.com/fallback"),
                    "render_with_retry",
                    config=RetryConfig(max_attempts=2)
                )
                
                assert retry_result is not None


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for web scraping components."""
    
    def test_rate_limiter_performance(self):
        """Benchmark rate limiter performance."""
        import timeit
        
        config = RateLimitConfig(strategy=RateLimitStrategy.ADAPTIVE)
        limiter = RateLimiter(config)
        
        # Benchmark acquire performance
        def test_acquire():
            return asyncio.run(limiter.acquire())
        
        # Measure performance
        time_taken = timeit.timeit(test_acquire, number=1000)
        avg_time = time_taken / 1000
        
        # Should be very fast (< 1ms)
        assert avg_time < 0.001
        
        print(f"Rate limiter average acquire time: {avg_time:.6f}s")
    
    def test_browser_renderer_performance(self):
        """Benchmark browser renderer performance."""
        config = BrowserConfig(headless=True)
        renderer = BrowserRenderer(config)
        
        # Simple performance test
        start_time = time.time()
        
        async def render_test():
            return await renderer.render_page("https://example.com")
        
        # Run multiple iterations
        for _ in range(10):
            asyncio.run(render_test())
        
        total_time = time.time() - start_time
        avg_time = total_time / 10
        
        # Should complete in reasonable time (< 5 seconds per page)
        assert avg_time < 5.0
        
        print(f"Browser renderer average render time: {avg_time:.2f}s")


if __name__ == "__main__":
    # Run performance benchmarks
    benchmark = TestPerformanceBenchmarks()
    benchmark.test_rate_limiter_performance()
    benchmark.test_browser_renderer_performance()
    
    # Run tests with pytest
    pytest.main([__file__])