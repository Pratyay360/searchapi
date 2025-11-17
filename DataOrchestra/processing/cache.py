"""
Caching system for text processing operations.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

from cachetools import TTLCache, cachedmethod
from ..core.logging_utils import get_logger


@dataclass
class CacheConfig:
    """Configuration for cache system."""
    enabled: bool = True
    memory_max_size: int = 1000
    disk_cache_dir: Optional[Path] = None
    ttl_seconds: int = 3600  # 1 hour
    compression_enabled: bool = True


class CacheManager:
    """
    Advanced caching system with memory and disk storage.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Memory cache for frequently accessed items
        self._memory_cache = TTLCache(
            maxsize=self.config.memory_max_size,
            ttl=self.config.ttl_seconds
        )

        # Disk cache for persistent storage
        self._disk_cache_dir = self.config.disk_cache_dir or Path.home() / \
            ".dataorchestra" / "cache"
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Cache initialized: memory={self.config.memory_max_size}, "
            f"disk={self._disk_cache_dir}, ttl={self.config.ttl_seconds}s"
        )

    @property
    def enabled(self) -> bool:
        """Check if cache is enabled."""
        return self.config.enabled

    @cachedmethod(lambda self: self._memory_cache, key=lambda self, text: f"text_hash_{self._hash_text(text)}")
    def get_text_processing_result(self, text: str) -> Optional[str]:
        """
        Get cached text processing result.

        Args:
            text: Original text that was processed

        Returns:
            Cached processed text or None if not found
        """
        return self._memory_cache.get(text)

    def set_text_processing_result(self, text: str, result: str) -> None:
        """
        Cache text processing result.

        Args:
            text: Original text that was processed
            result: Processed text to cache
        """
        self._memory_cache[text] = result
        self.logger.debug(
            f"Cached text processing result: {self._hash_text(text)[:8]}")

    @cachedmethod(lambda self: self._memory_cache, key=lambda self, file_path: f"file_content_{file_path}")
    def get_file_content(self, file_path: Path) -> Optional[str]:
        """
        Get cached file content.

        Args:
            file_path: Path to file

        Returns:
            Cached file content or None if not found
        """
        # Try memory cache first
        cache_key = str(file_path)
        cached_content = self._memory_cache.get(cache_key)
        if cached_content is not None:
            return cached_content

        # Try disk cache
        disk_content = self._get_from_disk(file_path)
        if disk_content is not None:
            # Store in memory cache for faster access
            self._memory_cache[cache_key] = disk_content
            return disk_content

        return None

    def set_file_content(self, file_path: Path, content: str) -> None:
        """
        Cache file content.

        Args:
            file_path: Path to file
            content: File content to cache
        """
        if not self.config.enabled:
            return

        # Store in memory cache
        cache_key = str(file_path)
        self._memory_cache[cache_key] = content

        # Store on disk for persistence
        self._save_to_disk(file_path, content)
        self.logger.debug(f"Cached file content: {file_path}")

    def get_language_detection(self, text: str) -> Optional[str]:
        """
        Get cached language detection result.

        Args:
            text: Text to detect language for

        Returns:
            Cached language code or None if not found
        """
        cache_key = f"lang_detect_{self._hash_text(text)}"
        return self._memory_cache.get(cache_key)

    def set_language_detection(self, text: str, language: str) -> None:
        """
        Cache language detection result.

        Args:
            text: Text that was analyzed
            language: Detected language code
        """
        cache_key = f"lang_detect_{self._hash_text(text)}"
        self._memory_cache[cache_key] = language
        self.logger.debug(f"Cached language detection: {language}")

    def get_quality_assessment(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached quality assessment result.

        Args:
            text: Text that was assessed

        Returns:
            Cached quality metrics or None if not found
        """
        cache_key = f"quality_{self._hash_text(text)}"
        return self._memory_cache.get(cache_key)

    def set_quality_assessment(self, text: str, metrics: Dict[str, Any]) -> None:
        """
        Cache quality assessment result.

        Args:
            text: Text that was assessed
            metrics: Quality metrics dictionary
        """
        cache_key = f"quality_{self._hash_text(text)}"
        self._memory_cache[cache_key] = metrics
        self.logger.debug(
            f"Cached quality assessment: {metrics.get('overall_score', 0):.3f}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self._memory_cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache by key.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._memory_cache[key] = value
        self.logger.debug(f"Cached value: {key[:16]}")

    def clear(self, pattern: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            pattern: Pattern to match for clearing (None for all)
        """
        if pattern is None:
            self._memory_cache.clear()
            self._clear_disk_cache()
            self.logger.info("Cleared all cache entries")
        else:
            # Clear matching entries from memory cache
            keys_to_remove = [
                k for k in self._memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._memory_cache[key]

            # Clear matching files from disk cache
            self._clear_disk_cache_pattern(pattern)
            self.logger.info(f"Cleared cache entries matching: {pattern}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        memory_stats = {
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_max": self.config.memory_max_size,
            "memory_cache_usage": len(self._memory_cache) / self.config.memory_max_size,
            "disk_cache_dir": str(self._disk_cache_dir),
            "disk_cache_enabled": self.config.disk_cache_dir is not None
        }

        # Count disk cache files
        if self._disk_cache_dir.exists():
            disk_files = list(self._disk_cache_dir.glob("*.cache"))
            memory_stats["disk_cache_files"] = len(disk_files)

            # Calculate disk cache size
            total_size = sum(
                f.stat().st_size for f in disk_files if f.is_file())
            memory_stats["disk_cache_size_bytes"] = total_size
            memory_stats["disk_cache_size_mb"] = total_size / (1024 * 1024)

        return memory_stats

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_disk_cache_path(self, key: str) -> Path:
        """Get disk cache file path for key."""
        return self._disk_cache_dir / f"{key}.cache"

    def _get_from_disk(self, key: Union[str, Path]) -> Optional[str]:
        """Get value from disk cache."""
        if not self.config.disk_cache_dir:
            return None

        cache_file = self._get_disk_cache_path(str(key))
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                if self.config.compression_enabled:
                    import gzip
                    with gzip.open(f, 'rb') as gz_file:
                        return pickle.load(gz_file)
                else:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def _save_to_disk(self, key: Union[str, Path], value: Any) -> None:
        """Save value to disk cache."""
        if not self.config.disk_cache_dir:
            return

        cache_file = self._get_disk_cache_path(str(key))

        try:
            # Ensure directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_file, 'wb') as f:
                if self.config.compression_enabled:
                    import gzip
                    with gzip.open(f, 'wb') as gz_file:
                        pickle.dump(value, gz_file)
                else:
                    pickle.dump(value, f)

            self.logger.debug(f"Saved to disk cache: {cache_file.name}")

        except Exception as e:
            self.logger.error(f"Failed to save to disk cache: {e}")

    def _clear_disk_cache(self) -> None:
        """Clear all disk cache files."""
        if not self._disk_cache_dir.exists():
            return

        try:
            import shutil
            shutil.rmtree(self._disk_cache_dir)
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Cleared disk cache")
        except Exception as e:
            self.logger.error(f"Failed to clear disk cache: {e}")

    def _clear_disk_cache_pattern(self, pattern: str) -> None:
        """Clear disk cache files matching pattern."""
        if not self._disk_cache_dir.exists():
            return

        try:
            import glob
            cache_files = glob.glob(
                str(self._disk_cache_dir / f"*{pattern}*.cache"))
            for cache_file in cache_files:
                try:
                    Path(cache_file).unlink()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to delete cache file {cache_file}: {e}")

            self.logger.info(f"Cleared disk cache files matching: {pattern}")

        except Exception as e:
            self.logger.error(f"Failed to clear disk cache pattern: {e}")

    def cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        # TTL cache automatically handles expiration
        # For disk cache, we'd need to implement timestamp checking
        # This is a simplified version
        self.logger.debug(
            "Cache cleanup completed (TTL cache handles expiration)")
