"""
Exception classes for DataOrchestra.
"""


class DataOrchestraError(Exception):
    """Base exception for all DataOrchestra errors."""
    pass


class FileProcessingError(DataOrchestraError):
    """Exception raised when file processing fails."""
    pass


class DownloadError(DataOrchestraError):
    """Exception raised when download operations fail."""
    pass


class CrawlError(DataOrchestraError):
    """Exception raised when web crawling fails."""
    pass


class ConfigurationError(DataOrchestraError):
    """Exception raised when configuration is invalid."""
    pass


class ValidationError(DataOrchestraError):
    """Exception raised when data validation fails."""
    pass


class NetworkError(DataOrchestraError):
    """Exception raised when network operations fail."""
    pass


class SecurityError(DataOrchestraError):
    """Exception raised when security checks fail."""
    pass


class TimeoutError(DataOrchestraError):
    """Exception raised when operations timeout."""
    pass
