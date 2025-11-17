"""
Unified text processing pipeline with configurable stages and strategy pattern.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Protocol
from dataclasses import dataclass, field

from ..core.base import BaseProcessor, ProcessResult
from ..core.config import get_config, ProcessingConfig
from ..core.logging_utils import get_logger, LogContext
from ..core.exceptions import DataOrchestraError, ValidationError
from .stages import ProcessingStage
from .strategies import CleaningStrategy
from .quality import TextQualityAssessor
from .cache import CacheManager


class TextProcessingPipeline(BaseProcessor[str, str]):
    """
    Advanced text processing pipeline with configurable stages and strategies.
    """
    
    def __init__(
        self,
        stages: Optional[List[ProcessingStage]] = None,
        strategy: Optional[CleaningStrategy] = None,
        quality_assessor: Optional[TextQualityAssessor] = None,
        cache_manager: Optional[CacheManager] = None,
        config: Optional[ProcessingConfig] = None
    ):
        super().__init__("TextProcessingPipeline")
        self.config = config or get_config().processing
        self.stages = stages or self._default_stages()
        self.strategy = strategy
        self.quality_assessor = quality_assessor or TextQualityAssessor()
        self.cache_manager = cache_manager or CacheManager()
        self.logger = get_logger(self.__class__.__name__)
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
    
    def _default_stages(self) -> List[ProcessingStage]:
        """Create default processing stages."""
        from .stages import (
            UnicodeNormalizationStage,
            NoiseRemovalStage,
            ContentFilteringStage,
            StructureAnalysisStage,
            FormatStandardizationStage
        )
        return [
            UnicodeNormalizationStage(),
            NoiseRemovalStage(),
            ContentFilteringStage(),
            StructureAnalysisStage(),
            FormatStandardizationStage()
        ]
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for better performance."""
        import re
        self._url_pattern = re.compile(r"https?://\S+")
        self._email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self._ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        self._control_chars_pattern = re.compile(r"[\x00-\x1F\x7F-\x9F]")
        self._repeated_punct_pattern = re.compile(r"([!?.,])\1+")
        self._whitespace_pattern = re.compile(r"\s+")
    
    def process(self, text: str) -> str:
        """
        Process text through the complete pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        with LogContext(self.logger, "process", text_length=len(text)):
            # Check cache first
            cache_key = self._get_cache_key(text)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for text hash: {cache_key[:8]}...")
                return cached_result
            
            try:
                # Apply processing stages
                result = text
                stage_results = []
                
                for stage in self.stages:
                    stage_start = time.time()
                    result = stage.process(result, self.config)
                    stage_duration = time.time() - stage_start
                    
                    stage_results.append({
                        'stage': stage.__class__.__name__,
                        'duration': stage_duration,
                        'input_length': len(text),
                        'output_length': len(result)
                    })
                    
                    self.logger.debug(f"Stage {stage.__class__.__name__} completed in {stage_duration:.3f}s")
                
                # Apply strategy if provided
                if self.strategy:
                    result = self.strategy.apply(result, self.config)
                
                # Assess quality
                quality_score = self.quality_assessor.assess(result)
                
                # Cache result
                self.cache_manager.set(cache_key, result)
                
                # Log metrics
                self._log_metrics(text, result, stage_results, quality_score)
                
                return result
                
            except Exception as e:
                self.handle_error(e, f"Failed to process text")
                raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _log_metrics(
        self, 
        original_text: str, 
        processed_text: str, 
        stage_results: List[Dict[str, Any]], 
        quality_score: float
    ) -> None:
        """Log processing metrics."""
        total_duration = sum(r['duration'] for r in stage_results)
        reduction_ratio = (len(original_text) - len(processed_text)) / len(original_text) if original_text else 0
        
        self.logger.info(
            f"Processing completed: "
            f"original={len(original_text)} chars, "
            f"processed={len(processed_text)} chars, "
            f"reduction={reduction_ratio:.1%}, "
            f"quality={quality_score:.2f}, "
            f"duration={total_duration:.3f}s"
        )
    
    async def process_async(self, text: str) -> str:
        """
        Asynchronously process text through the pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        # For now, just run the sync version in an executor
        # In a full implementation, stages would be async-aware
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, text)
    
    def process_file(self, file_path: Path) -> ProcessResult:
        """
        Process a text file with the pipeline.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            ProcessResult with processing details
        """
        try:
            # Try different encodings
            text = None
            for encoding in self.config.encoding_fallbacks:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise ValidationError(f"Cannot decode {file_path.name} with any supported encoding")
            
            # Process the text
            processed_text = self.process(text)
            
            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            
            return ProcessResult(
                success=True,
                data=processed_text,
                metadata={
                    "processor": self.name,
                    "original_length": len(text),
                    "processed_length": len(processed_text),
                    "stages_applied": len(self.stages),
                    "strategy": self.strategy.__class__.__name__ if self.strategy else None
                }
            )
            
        except Exception as e:
            return ProcessResult(
                success=False,
                error=str(e),
                metadata={"processor": self.name}
            )
    
    def add_stage(self, stage: ProcessingStage, position: Optional[int] = None) -> None:
        """
        Add a processing stage to the pipeline.
        
        Args:
            stage: Processing stage to add
            position: Position to insert at (None for end)
        """
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)
        self.logger.info(f"Added stage: {stage.__class__.__name__}")
    
    def remove_stage(self, stage_class: type) -> bool:
        """
        Remove a processing stage from the pipeline.
        
        Args:
            stage_class: Class of stage to remove
            
        Returns:
            True if stage was removed, False if not found
        """
        for i, stage in enumerate(self.stages):
            if isinstance(stage, stage_class):
                removed = self.stages.pop(i)
                self.logger.info(f"Removed stage: {removed.__class__.__name__}")
                return True
        return False
    
    def set_strategy(self, strategy: CleaningStrategy) -> None:
        """
        Set the cleaning strategy for the pipeline.
        
        Args:
            strategy: Cleaning strategy to use
        """
        self.strategy = strategy
        self.logger.info(f"Set strategy: {strategy.__class__.__name__}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline configuration.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "stages": [stage.__class__.__name__ for stage in self.stages],
            "strategy": self.strategy.__class__.__name__ if self.strategy else None,
            "quality_assessor": self.quality_assessor.__class__.__name__,
            "cache_enabled": self.cache_manager.enabled,
            "config": {
                "min_word_length": self.config.min_word_length,
                "min_sentence_length": self.config.min_sentence_length,
                "max_token_limit": self.config.max_token_limit,
                "noise_words": list(self.config.noise_words)
            }
        }