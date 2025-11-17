"""
Custom exceptions for DataOrchestra.
"""
from typing import Any, Optional


class DataOrchestraError(Exception):
    """Base exception for all DataOrchestra errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class FileProcessingError(DataOrchestraError):
    """Raised when file processing fails."""
    pass


class DownloadError(DataOrchestraError):
    """Raised when downloading files fails."""
    pass


class CrawlError(DataOrchestraError):
    """Raised when web crawling fails."""
    pass


class ConfigurationError(DataOrchestraError):
    """Raised when there's a configuration error."""
    pass


class ValidationError(DataOrchestraError):
    """Raised when input validation fails."""
    pass


class NetworkError(DataOrchestraError):
    """Raised when network operations fail."""
    pass


class SecurityError(DataOrchestraError):
    """Raised when security validation fails."""
    pass


class TimeoutError(DataOrchestraError):
    """Raised when operations timeout."""
    pass