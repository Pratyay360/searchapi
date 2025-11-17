"""
Test suite for core functionality of DataOrchestra.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from DataOrchestra.core.base import BaseProcessor, FileProcessor, ProcessResult, FileProcessingResult
from DataOrchestra.core.config import Config, get_config, set_config, reset_config
from DataOrchestra.core.exceptions import (
    DataOrchestraError, FileProcessingError, DownloadError, 
    CrawlError, ConfigurationError, ValidationError
)
from DataOrchestra.core.logging_utils import get_logger


class ConcreteBaseProcessor(BaseProcessor):
    """Concrete implementation of BaseProcessor for testing."""
    
    def process(self, data):
        return f"Processed: {data}"


class ConcreteFileProcessor(FileProcessor):
    """Concrete implementation of FileProcessor for testing."""
    
    def __init__(self):
        super().__init__("TestFileProcessor", {".txt", ".md"})
    
    def process_file(self, file_path: Path) -> FileProcessingResult:
        return FileProcessingResult(
            success=True,
            input_path=file_path,
            data=f"File processed: {file_path}"
        )


class TestBaseProcessor:
    """Test cases for base processor functionality."""
    
    def test_base_processor_initialization(self):
        """Test base processor initialization."""
        processor = ConcreteBaseProcessor("TestProcessor")
        assert processor is not None
        assert processor.name == "TestProcessor"
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'validate_input')
    
    def test_base_processor_process_method(self):
        """Test base processor process method."""
        processor = ConcreteBaseProcessor("TestProcessor")
        
        result = processor.process("test data")
        assert result == "Processed: test data"
    
    def test_base_processor_validate_input(self):
        """Test base processor validate_input method."""
        processor = ConcreteBaseProcessor("TestProcessor")
        
        # Valid input should not raise an exception
        processor.validate_input("valid data")
        
        # Invalid input should raise ValidationError
        with pytest.raises(ValidationError):
            processor.validate_input(None)
    
    def test_base_processor_handle_error(self):
        """Test base processor handle_error method."""
        processor = ConcreteBaseProcessor("TestProcessor")
        
        # Should raise DataOrchestraError when handling an error
        with pytest.raises(DataOrchestraError):
            try:
                raise ValueError("Test error")
            except ValueError as e:
                processor.handle_error(e, "Test context")


class TestFileProcessor:
    """Test cases for file processor functionality."""
    
    def test_file_processor_initialization(self):
        """Test file processor initialization."""
        processor = ConcreteFileProcessor()
        assert processor is not None
        assert processor.name == "TestFileProcessor"
        assert processor.supported_extensions == {".txt", ".md"}
        assert hasattr(processor, 'process_file')
        assert hasattr(processor, 'process')
    
    def test_file_processor_process_file_method(self):
        """Test file processor process_file method."""
        processor = ConcreteFileProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            result = processor.process_file(tmp_path)
            
            assert result.success is True
            assert result.input_path == tmp_path
            assert f"File processed: {tmp_path}" in result.data
            
            # Clean up
            os.unlink(tmp_path)
    
    def test_file_processor_process_method(self):
        """Test file processor process method."""
        processor = ConcreteFileProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            result = processor.process(tmp_path)
            
            assert result.success is True
            assert result.input_path == tmp_path
            
            # Clean up
            os.unlink(tmp_path)
    
    def test_file_processor_unsupported_extension(self):
        """Test file processor with unsupported extension."""
        processor = ConcreteFileProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
            with pytest.raises(FileProcessingError):
                processor.process(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)


class TestProcessResult:
    """Test cases for ProcessResult."""
    
    def test_process_result_initialization(self):
        """Test ProcessResult initialization."""
        result = ProcessResult(
            success=True,
            data="test data",
            metadata={"source": "test"},
            duration=1.5
        )
        
        assert result.success is True
        assert result.data == "test data"
        assert result.metadata == {"source": "test"}
        assert result.duration == 1.5
        assert result.error is None
    
    def test_process_result_with_error(self):
        """Test ProcessResult with error."""
        result = ProcessResult(
            success=False,
            data=None,
            error="Error occurred",
            metadata={}
        )
        
        assert result.success is False
        assert result.data is None
        assert result.error == "Error occurred"
    
    def test_process_result_repr(self):
        """Test ProcessResult string representation."""
        result = ProcessResult(
            success=True,
            data="test data",
            metadata={"source": "test"},
            duration=0.5
        )
        
        repr_str = repr(result)
        assert "ProcessResult" in repr_str
        assert "success=True" in repr_str


class TestFileProcessingResult:
    """Test cases for FileProcessingResult."""
    
    def test_file_processing_result_initialization(self):
        """Test FileProcessingResult initialization."""
        temp_path = Path("/tmp/test.txt")
        result = FileProcessingResult(
            success=True,
            data="test data",
            input_path=temp_path,
            output_path=Path("/tmp/output.txt"),
            file_size_before=100,
            file_size_after=200,
            metadata={"source": "test"},
            duration=1.2
        )
        
        assert result.success is True
        assert result.data == "test data"
        assert result.input_path == temp_path
        assert result.output_path == Path("/tmp/output.txt")
        assert result.file_size_before == 100
        assert result.file_size_after == 200
        assert result.metadata == {"source": "test"}
        assert result.duration == 1.2
    
    def test_file_processing_result_with_error(self):
        """Test FileProcessingResult with error."""
        temp_path = Path("/tmp/test.txt")
        result = FileProcessingResult(
            success=False,
            data=None,
            input_path=temp_path,
            error="File not found"
        )
        
        assert result.success is False
        assert result.input_path == temp_path
        assert result.error == "File not found"
    
    def test_file_processing_result_repr(self):
        """Test FileProcessingResult string representation."""
        temp_path = Path("/tmp/test.txt")
        result = FileProcessingResult(
            success=True,
            data="test data",
            input_path=temp_path,
            metadata={"source": "test"},
            duration=0.8
        )
        
        repr_str = repr(result)
        assert "FileProcessingResult" in repr_str
        assert "success=True" in repr_str
        assert "test.txt" in repr_str


class TestConfig:
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test Config initialization with default values."""
        config = Config()
        
        # Test that all expected sub-configs exist
        assert hasattr(config, 'logging')
        assert hasattr(config, 'web')
        assert hasattr(config, 'processing')
        assert hasattr(config, 'download')
        assert hasattr(config, 'output_dir')
        assert hasattr(config, 'verbose')
    
    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            verbose=True,
            output_dir=Path("/tmp/test_output")
        )
        
        assert config.verbose is True
        assert config.output_dir == Path("/tmp/test_output")
    
    def test_config_singleton_pattern(self):
        """Test configuration singleton pattern."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
        assert isinstance(config1, Config)
        assert isinstance(config2, Config)
    
    def test_config_modification(self):
        """Test modifying configuration."""
        original_verbose = get_config().verbose
        
        # Modify configuration
        new_config = Config(verbose=True)
        set_config(new_config)
        
        # Verify change
        new_config_from_get = get_config()
        assert new_config_from_get.verbose is True
        
        # Reset to original
        reset_config()
        assert get_config().verbose == original_verbose
    
    def test_config_reset(self):
        """Test configuration reset functionality."""
        # Change config
        original_verbose = get_config().verbose
        set_config(Config(verbose=True))
        
        # Verify change
        assert get_config().verbose is True
        
        # Reset and verify original values
        reset_config()
        assert get_config().verbose == original_verbose
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        valid_config = Config()
        valid_config.validate()
        
        # Invalid config should fail
        invalid_config = Config()
        invalid_config.web.workers = 0  # Should be at least 1
        with pytest.raises(ValueError):
            invalid_config.validate()


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_data_orchestra_error(self):
        """Test DataOrchestraError base exception."""
        with pytest.raises(DataOrchestraError):
            raise DataOrchestraError("Test error")
        
        try:
            raise DataOrchestraError("Test error")
        except DataOrchestraError as e:
            assert str(e) == "Test error"
            assert "Test error" in repr(e)
    
    def test_data_orchestra_error_with_details(self):
        """Test DataOrchestraError with details."""
        error = DataOrchestraError("Test error", details={"code": 404, "reason": "Not found"})
        assert error.details == {"code": 404, "reason": "Not found"}
    
    def test_file_processing_error(self):
        """Test FileProcessingError."""
        with pytest.raises(FileProcessingError):
            raise FileProcessingError("File error")
        
        try:
            raise FileProcessingError("File error")
        except FileProcessingError as e:
            assert str(e) == "File error"
    
    def test_download_error(self):
        """Test DownloadError."""
        with pytest.raises(DownloadError):
            raise DownloadError("Download failed")
        
        try:
            raise DownloadError("Download failed")
        except DownloadError as e:
            assert str(e) == "Download failed"
    
    def test_crawl_error(self):
        """Test CrawlError."""
        with pytest.raises(CrawlError):
            raise CrawlError("Crawl failed")
        
        try:
            raise CrawlError("Crawl failed")
        except CrawlError as e:
            assert str(e) == "Crawl failed"
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config error")
        
        try:
            raise ConfigurationError("Config error")
        except ConfigurationError as e:
            assert str(e) == "Config error"
    
    def test_validation_error(self):
        """Test ValidationError."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")
        
        try:
            raise ValidationError("Validation failed")
        except ValidationError as e:
            assert str(e) == "Validation failed"


class TestLoggingUtils:
    """Test cases for logging utilities."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "DataOrchestra.test_module"
        
        # Test with different name
        logger2 = get_logger("another_module")
        assert logger2.name == "DataOrchestra.another_module"


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])
