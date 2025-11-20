from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

from .logging_utils import get_logger

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


@dataclass
class FileProcessingResult(ProcessResult):
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    file_size_before: int = 0
    file_size_after: int = 0


class BaseProcessor(Generic[T, R], abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def process(self, data: T) -> R:
        pass

    def validate_input(self, data: T) -> None:
        print(f"{self.name}: Input data cannot be None")

    def handle_error(self, error: Exception, context: str = "") -> None:
        message = f"{self.name}: {context}" if context else f"{self.name}: {str(error)}"
        self.logger.error(message, exc_info=True)
        print(message)


class FileProcessor(BaseProcessor[Path, FileProcessingResult]):
    def __init__(self, name: str, supported_extensions: set[str]):
        super().__init__(name)
        self.supported_extensions = supported_extensions

    @abc.abstractmethod
    def process_file(self, file_path: Path) -> FileProcessingResult:
        pass

    def process(self, file_path: Path) -> FileProcessingResult:
        import time

        self.validate_input(file_path)
        start_time = time.time()

        result = FileProcessingResult(
            success=False, input_path=file_path, metadata={"processor": self.name}
        )

        try:
            # Check if file exists
            if not file_path.exists():
                raise FileProcessingError(f"File does not exist: {file_path}")

            # Check if file extension is supported
            if file_path.suffix.lower() not in self.supported_extensions:
                raise FileProcessingError(
                    f"Unsupported file type: {file_path.suffix}. "
                    f"Supported types: {', '.join(self.supported_extensions)}"
                )

            # Get file size before processing
            result.file_size_before = file_path.stat().st_size

            # Process the file
            self.logger.info(f"Processing file: {file_path}")
            result = self.process_file(file_path)

            # Get file size after processing
            if result.success and result.output_path and result.output_path.exists():
                result.file_size_after = result.output_path.stat().st_size

            result.duration = time.time() - start_time

            if result.success:
                self.logger.info(
                    f"Successfully processed {file_path} in {result.duration:.3f}s"
                )
            else:
                self.logger.warning(f"Failed to process {file_path}: {result.error}")

            return result

        except Exception as e:
            result.duration = time.time() - start_time
            result.success = False
            result.error = str(e)
            self.handle_error(e, f"Failed to process file: {file_path}")
            return result
