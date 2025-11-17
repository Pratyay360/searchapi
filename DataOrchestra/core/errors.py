"""
Enhanced error handling framework with context and recovery.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum

from .logging_utils import get_logger


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for better organization."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PLUGIN = "plugin"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"


@dataclass
class ErrorContext:
    """Context information for errors."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    stack_trace: Optional[str] = None


class DataOrchestraError(Exception):
    """
    Enhanced base exception for all DataOrchestra errors.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PROCESSING,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
        user_action: Optional[str] = None,
        technical_details: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.user_action = user_action
        self.technical_details = technical_details
        self.logger = get_logger(self.__class__.__name__)
        
        # Log the error
        self._log_error()
    
    def _log_error(self) -> None:
        """Log the error with context."""
        log_data = {
            "error_code": self.error_code,
            "category": self.category,
            "severity": self.severity,
            "message": str(self),
            "recoverable": self.recoverable,
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "file_path": self.context.file_path,
                "url": self.context.url,
                "operation": self.context.operation,
                "component": self.context.component,
                "retry_count": self.context.retry_count,
                "additional_data": self.context.additional_data
            }
        }
        
        if self.details:
            log_data["details"] = self.details
        
        if self.technical_details:
            log_data["technical_details"] = self.technical_details
        
        if self.user_action:
            log_data["user_action"] = self.user_action
        
        if self.retry_after:
            log_data["retry_after"] = self.retry_after
        
        # Choose log level based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {self}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {self}", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {self}", extra=log_data)
        else:
            self.logger.info(f"Low severity error: {self}", extra=log_data)
    
    def with_context(self, **kwargs) -> 'DataOrchestraError':
        """Add context to the error."""
        if self.context is None:
            self.context = ErrorContext()
        
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        
        return self
    
    def with_retry(self, retry_after: float, retry_count: int = 1) -> 'DataOrchestraError':
        """Add retry information to the error."""
        self.retry_after = retry_after
        if self.context is None:
            self.context = ErrorContext()
        self.context.retry_count = retry_count
        return self
    
    def with_user_action(self, action: str) -> 'DataOrchestraError':
        """Add user action recommendation to the error."""
        self.user_action = action
        return self
    
    def with_details(self, **kwargs) -> 'DataOrchestraError':
        """Add detailed information to the error."""
        if self.details is None:
            self.details = {}
        
        self.details.update(kwargs)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error": str(self),
            "error_code": self.error_code,
            "category": self.category,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "user_action": self.user_action,
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "file_path": self.context.file_path,
                "url": self.context.url,
                "operation": self.context.operation,
                "component": self.context.component,
                "retry_count": self.context.retry_count,
                "additional_data": self.context.additional_data
            },
            "details": self.details,
            "technical_details": self.technical_details,
            "timestamp": self.context.timestamp
        }


class ValidationError(DataOrchestraError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        allowed_values: Optional[list] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.field = field
        self.value = value
        self.expected_type = expected_type
        self.allowed_values = allowed_values


class ProcessingError(DataOrchestraError):
    """Raised when text processing fails."""
    
    def __init__(
        self,
        message: str,
        processor: Optional[str] = None,
        stage: Optional[str] = None,
        input_text: Optional[str] = None,
        processing_time: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )
        self.processor = processor
        self.stage = stage
        self.input_text = input_text
        self.processing_time = processing_time


class NetworkError(DataOrchestraError):
    """Raised when network operations fail."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        self.url = url
        self.status_code = status_code
        self.response_time = response_time
        self.timeout = timeout


class FileSystemError(DataOrchestraError):
    """Raised when file system operations fail."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        permission: Optional[str] = None,
        disk_space: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.FILE_SYSTEM,
            **kwargs
        )
        self.file_path = file_path
        self.operation = operation
        self.permission = permission
        self.disk_space = disk_space


class ConfigurationError(DataOrchestraError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        config_file: Optional[str] = None,
        validation_errors: Optional[list] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        self.config_key = config_key
        self.config_value = config_value
        self.config_file = config_file
        self.validation_errors = validation_errors


class SecurityError(DataOrchestraError):
    """Raised when security-related issues occur."""
    
    def __init__(
        self,
        message: str,
        security_issue: Optional[str] = None,
        threat_level: Optional[str] = None,
        blocked_content: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.security_issue = security_issue
        self.threat_level = threat_level
        self.blocked_content = blocked_content


class PluginError(DataOrchestraError):
    """Raised when plugin operations fail."""
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        plugin_version: Optional[str] = None,
        plugin_error: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PLUGIN,
            **kwargs
        )
        self.plugin_name = plugin_name
        self.plugin_version = plugin_version
        self.plugin_error = plugin_error


class TimeoutError(DataOrchestraError):
    """Raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.timeout_duration = timeout_duration
        self.operation = operation


class RateLimitError(DataOrchestraError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        limit_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        limit: Optional[int] = None,
        reset_time: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.limit_type = limit_type
        self.current_usage = current_usage
        self.limit = limit
        self.reset_time = reset_time


class AuthenticationError(DataOrchestraError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        auth_provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.auth_method = auth_method
        self.auth_provider = auth_provider


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(self.__class__.__name__)
    
    def can_recover(self, error: DataOrchestraError) -> bool:
        """
        Check if this strategy can recover from the given error.
        
        Args:
            error: The error that occurred
            
        Returns:
            True if recovery is possible, False otherwise
        """
        raise NotImplementedError
    
    def recover(self, error: DataOrchestraError) -> bool:
        """
        Attempt to recover from the error.
        
        Args:
            error: The error that occurred
            
        Returns:
            True if recovery was successful, False otherwise
        """
        raise NotImplementedError


class RetryRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries the operation."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        super().__init__("RetryRecovery")
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def can_recover(self, error: DataOrchestraError) -> bool:
        """Check if error is recoverable with retry."""
        return (
            error.recoverable and
            error.context.retry_count < self.max_retries and
            error.category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]
        )
    
    def recover(self, error: DataOrchestraError) -> bool:
        """Attempt recovery by retrying with exponential backoff."""
        import time
        
        if not self.can_recover(error):
            return False
        
        delay = self.base_delay * (2 ** error.context.retry_count)
        self.logger.info(f"Retrying operation after {delay:.2f}s (attempt {error.context.retry_count + 1})")
        
        time.sleep(delay)
        
        # Update retry count
        error.context.retry_count += 1
        error.retry_after = time.time() + delay
        
        return True  # Recovery initiated


class FallbackRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that falls back to alternative methods."""
    
    def __init__(self, fallback_methods: list):
        super().__init__("FallbackRecovery")
        self.fallback_methods = fallback_methods
        self.current_method_index = 0
    
    def can_recover(self, error: DataOrchestraError) -> bool:
        """Check if fallback recovery is possible."""
        return (
            error.recoverable and
            self.current_method_index < len(self.fallback_methods)
        )
    
    def recover(self, error: DataOrchestraError) -> bool:
        """Attempt recovery using fallback method."""
        if not self.can_recover(error):
            return False
        
        if self.current_method_index >= len(self.fallback_methods):
            self.logger.error("All fallback methods exhausted")
            return False
        
        fallback_method = self.fallback_methods[self.current_method_index]
        self.logger.info(f"Attempting fallback method: {fallback_method}")
        
        try:
            result = fallback_method(error)
            if result:
                self.logger.info(f"Fallback recovery successful")
                return True
            else:
                self.current_method_index += 1
                return self.recover(error)  # Try next fallback
        except Exception as e:
            self.logger.error(f"Fallback method failed: {e}")
            self.current_method_index += 1
            return self.recover(error)  # Try next fallback


class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.strategies: Dict[ErrorCategory, list[ErrorRecoveryStrategy]] = {
            ErrorCategory.NETWORK: [RetryRecoveryStrategy()],
            ErrorCategory.TIMEOUT: [RetryRecoveryStrategy()],
            ErrorCategory.PROCESSING: [FallbackRecoveryStrategy([
                lambda error: self._retry_with_different_config(error),
                lambda error: self._retry_with_gentler_settings(error),
                lambda error: self._retry_with_minimal_processing(error)
            ])],
            ErrorCategory.FILE_SYSTEM: [RetryRecoveryStrategy(max_retries=2)],
        }
    
    def _retry_with_different_config(self, error: ProcessingError) -> bool:
        """Retry with different configuration."""
        self.logger.info("Retrying with different configuration")
        # Implementation would depend on specific error type
        return False
    
    def _retry_with_gentler_settings(self, error: ProcessingError) -> bool:
        """Retry with gentler processing settings."""
        self.logger.info("Retrying with gentler processing settings")
        # Implementation would adjust processing parameters
        return False
    
    def _retry_with_minimal_processing(self, error: ProcessingError) -> bool:
        """Retry with minimal processing."""
        self.logger.info("Retrying with minimal processing")
        # Implementation would reduce processing complexity
        return False
    
    def add_strategy(self, category: ErrorCategory, strategy: ErrorRecoveryStrategy) -> None:
        """Add a recovery strategy for a specific error category."""
        if category not in self.strategies:
            self.strategies[category] = []
        self.strategies[category].append(strategy)
        self.logger.info(f"Added recovery strategy for {category}: {strategy.name}")
    
    def recover(self, error: DataOrchestraError) -> bool:
        """Attempt to recover from error using appropriate strategies."""
        strategies = self.strategies.get(error.category, [])
        
        for strategy in strategies:
            if strategy.can_recover(error):
                try:
                    return strategy.recover(error)
                except Exception as e:
                    self.logger.error(f"Recovery strategy {strategy.name} failed: {e}")
                    continue
        
        self.logger.error(f"No recovery strategy could handle {error.category}: {error}")
        return False


class ErrorReporter:
    """Reports errors to external systems."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = get_logger(self.__class__.__name__)
        self.error_buffer: list[DataOrchestraError] = []
        self.max_buffer_size = 100
    
    def report_error(self, error: DataOrchestraError) -> None:
        """Report an error to external monitoring."""
        if not self.enabled:
            return
        
        self.error_buffer.append(error)
        
        # Flush buffer if it's full
        if len(self.error_buffer) >= self.max_buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush error buffer to external system."""
        if not self.error_buffer:
            return
        
        # Here you would integrate with external monitoring systems
        # For now, just log the errors
        for error in self.error_buffer:
            self.logger.error(f"Reported error: {error.to_dict()}")
        
        self.error_buffer.clear()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of reported errors."""
        if not self.error_buffer:
            return {"total_errors": 0}
        
        error_counts = {}
        for error in self.error_buffer:
            category = error.category
            error_counts[category] = error_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(self.error_buffer),
            "by_category": error_counts,
            "by_severity": {
                severity: len([e for e in self.error_buffer if e.severity == severity])
                for severity in ErrorSeverity
            }
        }