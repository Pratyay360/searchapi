"""
Unified text processor that merges all existing implementations.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseProcessor, ProcessResult
from ..core.config import get_config, ProcessingConfig
from ..core.logging_utils import get_logger, LogContext
from ..core.errors import ProcessingError, ValidationError
from .pipeline import TextProcessingPipeline
from .strategies import (
    AggressiveCleaningStrategy,
    GentleCleaningStrategy,
    DomainSpecificStrategy,
    LanguageAwareStrategy,
    AdaptiveStrategy
)
from .quality import TextQualityAssessor
from .cache import CacheManager


class UnifiedTextProcessor(BaseProcessor[Union[str, Path], str]):
    """
    Unified text processor that combines all existing implementations.
    """
    
    def __init__(
        self,
        strategy: Optional[str] = None,
        quality_threshold: Optional[float] = None,
        enable_cache: bool = True,
        enable_parallel: bool = False,
        max_workers: Optional[int] = None,
        config: Optional[ProcessingConfig] = None
    ):
        super().__init__("UnifiedTextProcessor")
        self.config = config or get_config().processing
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.cache_manager = CacheManager() if enable_cache else None
        self.quality_assessor = TextQualityAssessor()
        
        # Create pipeline with default stages
        self.pipeline = TextProcessingPipeline(
            config=self.config,
            cache_manager=self.cache_manager,
            quality_assessor=self.quality_assessor
        )
        
        # Set strategy based on parameter
        self.strategy = self._create_strategy(strategy)
        if self.strategy:
            self.pipeline.set_strategy(self.strategy)
        
        self.quality_threshold = quality_threshold or getattr(self.config, 'quality_threshold', 0.5)
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or getattr(self.config, 'max_workers', 4)
        
        # Performance tracking
        self._processing_stats = {
            'files_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _create_strategy(self, strategy_name: Optional[str]):
        """Create cleaning strategy based on name."""
        if not strategy_name:
            return AdaptiveStrategy()
        
        strategy_map = {
            'aggressive': AggressiveCleaningStrategy(),
            'gentle': GentleCleaningStrategy(),
            'domain': DomainSpecificStrategy(),
            'language': LanguageAwareStrategy(),
            'adaptive': AdaptiveStrategy()
        }
        
        strategy = strategy_map.get(strategy_name.lower())
        if strategy:
            self.logger.info(f"Using strategy: {strategy_name}")
        else:
            self.logger.warning(f"Unknown strategy: {strategy_name}, using adaptive")
            strategy = AdaptiveStrategy()
        
        return strategy
    
    def process(self, input_data: Union[str, Path]) -> ProcessResult:
        """
        Process text or file using the unified pipeline.
        
        Args:
            input_data: Text string or file path to process
            
        Returns:
            ProcessResult with processing details
        """
        start_time = time.time()
        
        try:
            if isinstance(input_data, Path):
                return self._process_file(input_data)
            else:
                return self._process_text(input_data)
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, cache_hit=False)
            
            error = ProcessingError(
                message=f"Failed to process input: {str(e)}",
                processing_time=processing_time,
                input_text=input_data if isinstance(input_data, str) else str(input_data),
                processor=self.name
            )
            
            self.logger.error(f"Processing failed: {e}")
            return ProcessResult(
                success=False,
                error=str(error),
                metadata=error.to_dict()
            )
    
    def _process_file(self, file_path: Path) -> ProcessResult:
        """Process a single file using the pipeline."""
        with LogContext(self.logger, "process_file", file_path=str(file_path)):
            # Check if file exists
            if not file_path.exists():
                raise ValidationError(f"File does not exist: {file_path}")
            
            # Check if file is supported
            supported_extensions = {'.txt', '.md', '.json', '.csv', '.html', '.htm'}
            if file_path.suffix.lower() not in supported_extensions:
                raise ValidationError(
                    f"Unsupported file type: {file_path.suffix}. "
                    f"Supported types: {', '.join(supported_extensions)}"
                )
            
            # Use pipeline to process the file
            result = self.pipeline.process_file(file_path)
            
            # Update statistics
            processing_time = result.metadata.get('processing_time', 0.0)
            self._update_stats(processing_time, cache_hit=False)
            
            # Add quality assessment
            if result.success and result.data:
                quality_score = self.quality_assessor.assess(result.data)
                result.metadata['quality_score'] = quality_score
                result.metadata['quality_level'] = self._get_quality_level(quality_score)
                
                # Log quality assessment
                if quality_score < self.quality_threshold:
                    self.logger.warning(
                        f"Low quality text: {file_path.name} "
                        f"(score: {quality_score:.3f})"
                    )
            
            return result
    
    def _process_text(self, text: str) -> ProcessResult:
        """Process text using the pipeline."""
        with LogContext(self.logger, "process_text", text_length=len(text)):
            # Check cache first
            cache_key = self._get_cache_key(text)
            cached_result = self.cache_manager.get_text_processing_result(text) if self.cache_manager else None
            
            if cached_result is not None:
                self._update_stats(0.0, cache_hit=True)
                self.logger.debug(f"Cache hit for text: {cache_key[:8]}")
                
                return ProcessResult(
                    success=True,
                    data=cached_result,
                    metadata={
                        'processor': self.name,
                        'cached': True,
                        'processing_time': 0.0
                    }
                )
            
            # Process through pipeline
            result_text = self.pipeline.process(text)
            processing_time = 0.001  # Minimal time for cached text
            
            # Cache the result
            if self.cache_manager:
                self.cache_manager.set_text_processing_result(text, result_text)
            
            # Assess quality
            quality_score = self.quality_assessor.assess(result_text)
            quality_level = self._get_quality_level(quality_score)
            
            self._update_stats(processing_time, cache_hit=False)
            
            self.logger.info(
                f"Processed text: quality={quality_score:.3f} "
                f"(level: {quality_level}), length={len(result_text)}"
            )
            
            return ProcessResult(
                success=True,
                data=result_text,
                metadata={
                    'processor': self.name,
                    'strategy': self.strategy.name if self.strategy else None,
                    'quality_score': quality_score,
                    'quality_level': quality_level,
                    'original_length': len(text),
                    'processed_length': len(result_text),
                    'processing_time': processing_time,
                    'cached': False
                }
            )
    
    async def process_async(self, input_data: Union[str, Path]) -> ProcessResult:
        """Asynchronously process text or file."""
        if isinstance(input_data, Path):
            # For file processing, we can run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._process_file, input_data
            )
        else:
            # For text processing, run the async pipeline
            result = await self.pipeline.process_async(input_data)
            return ProcessResult(
                success=True,
                data=result,
                metadata={
                    'processor': self.name,
                    'processing_time': 0.0,
                    'async': True
                }
            )
    
    def process_batch(
        self,
        inputs: List[Union[str, Path]],
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None
    ) -> List[ProcessResult]:
        """
        Process multiple inputs in batch.
        
        Args:
            inputs: List of text strings or file paths
            parallel: Whether to process in parallel (overrides instance setting)
            max_workers: Maximum number of workers (overrides instance setting)
            
        Returns:
            List of ProcessResult objects
        """
        parallel = parallel if parallel is not None else self.enable_parallel
        workers = max_workers or self.max_workers
        
        if not parallel or workers <= 1:
            # Sequential processing
            results = []
            for input_data in inputs:
                result = self.process(input_data)
                results.append(result)
            return results
        
        # Parallel processing
        if parallel:
            return self._process_batch_parallel(inputs, workers)
        else:
            return self._process_batch_sequential(inputs)
    
    def _process_batch_parallel(self, inputs: List[Union[str, Path]], workers: int) -> List[ProcessResult]:
        """Process batch in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results: List[ProcessResult] = [None] * len(inputs)  # type: ignore
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_input = {
                executor.submit(self.process, input_data): input_data 
                for input_data in inputs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_input.keys()):
                input_data = future_to_input[future]
                results[inputs.index(input_data)] = future.result()
        
        return results
    
    def _process_batch_sequential(self, inputs: List[Union[str, Path]]) -> List[ProcessResult]:
        """Process batch sequentially."""
        results = []
        for input_data in inputs:
            result = self.process(input_data)
            results.append(result)
        return results
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description from score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _update_stats(self, processing_time: float, cache_hit: bool) -> None:
        """Update processing statistics."""
        self._processing_stats['files_processed'] += 1
        self._processing_stats['total_processing_time'] += processing_time
        
        if cache_hit:
            self._processing_stats['cache_hits'] += 1
        else:
            self._processing_stats['cache_misses'] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self._processing_stats.copy()
        
        # Calculate derived stats
        if stats['files_processed'] > 0:
            stats['avg_processing_time'] = (
                stats['total_processing_time'] / stats['files_processed']
            )
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
                if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
            )
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._processing_stats = {
            'files_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.logger.info("Processing statistics reset")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration."""
        return {
            'processor': self.name,
            'strategy': self.strategy.name if self.strategy else None,
            'quality_threshold': self.quality_threshold,
            'cache_enabled': self.cache_manager is not None,
            'parallel_enabled': self.enable_parallel,
            'max_workers': self.max_workers,
            'config': {
                'min_word_length': self.config.min_word_length,
                'min_sentence_length': self.config.min_sentence_length,
                'max_token_limit': self.config.max_token_limit,
                'noise_words': list(self.config.noise_words)
            },
            'stats': self.get_processing_stats()
        }
    
    def set_strategy(self, strategy_name: str) -> None:
        """Change the cleaning strategy."""
        self.strategy = self._create_strategy(strategy_name)
        self.pipeline.set_strategy(self.strategy)
        self.logger.info(f"Strategy changed to: {strategy_name}")
    
    def set_quality_threshold(self, threshold: float) -> None:
        """Set the quality threshold."""
        self.quality_threshold = threshold
        self.logger.info(f"Quality threshold set to: {threshold}")
    
    def enable_caching(self, enabled: bool) -> None:
        """Enable or disable caching."""
        if enabled and not self.cache_manager:
            self.cache_manager = CacheManager()
        elif not enabled and self.cache_manager:
            self.cache_manager = None
        
        self.logger.info(f"Caching {'enabled' if enabled else 'disabled'}")
    
    def enable_parallel_processing(self, enabled: bool, max_workers: int = 4) -> None:
        """Enable or disable parallel processing."""
        self.enable_parallel = enabled
        self.max_workers = max_workers
        self.logger.info(f"Parallel processing {'enabled' if enabled else 'disabled'} (workers: {max_workers})")
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear()
        self.logger.info("Cache cleared")
    
    def add_stage(self, stage, position: Optional[int] = None) -> None:
        """Add a processing stage to the pipeline."""
        self.pipeline.add_stage(stage, position)
        self.logger.info(f"Added stage: {stage.name}")
    
    def remove_stage(self, stage_class: type) -> None:
        """Remove a processing stage from the pipeline."""
        self.pipeline.remove_stage(stage_class)
        self.logger.info(f"Removed stage: {stage_class.__name__}")


# Legacy compatibility functions
def clean_text(text: str, strategy: str = "adaptive") -> str:
    """Legacy function for text cleaning."""
    processor = UnifiedTextProcessor(strategy=strategy)
    result = processor.process(text)
    return result.data if result.success else text


def process_file(file_path: Path, strategy: str = "adaptive") -> ProcessResult:
    """Legacy function for file processing."""
    processor = UnifiedTextProcessor(strategy=strategy)
    return processor.process(file_path)


def process_batch(
    inputs: List[Union[str, Path]], 
    strategy: str = "adaptive",
    parallel: bool = False,
    max_workers: int = 4
) -> List[ProcessResult]:
    """Legacy function for batch processing."""
    processor = UnifiedTextProcessor(
        strategy=strategy,
        enable_parallel=parallel,
        max_workers=max_workers
    )
    return processor.process_batch(inputs)


def get_processing_stats() -> Dict[str, Any]:
    """Legacy function for getting processing stats."""
    # This would need to be implemented as a global stats manager
    return {}