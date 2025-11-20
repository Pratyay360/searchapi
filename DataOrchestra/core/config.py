from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import DirectoryPath


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def _validate(cls, v):
        if isinstance(v, LogLevel):
            return v
        if isinstance(v, str):
            try:
                return LogLevel(v.upper())
            except ValueError:
                raise ValueError(f"Invalid log level: {v}")
        raise ValueError(f"Invalid log level type: {type(v)}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(cls._validate)


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"

    @classmethod
    def _validate(cls, v):
        if isinstance(v, OutputFormat):
            return v
        if isinstance(v, str):
            try:
                return OutputFormat(v.lower())
            except ValueError:
                raise ValueError(f"Invalid output format: {v}")
        raise ValueError(f"Invalid output format type: {type(v)}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(cls._validate)


class CacheConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable caching system")
    memory_max_size: int = Field(
        default=1000, ge=1, le=10000, description="Maximum memory cache size"
    )
    disk_cache_dir: Optional[DirectoryPath] = Field(
        default=None, description="Disk cache directory"
    )
    ttl_seconds: int = Field(
        default=3600, ge=60, le=86400, description="Cache TTL in seconds"
    )
    compression_enabled: bool = Field(
        default=True, description="Enable cache compression"
    )


class WebConfig(BaseModel):
    timeout: float = Field(
        default=12.0, ge=1.0, le=300.0, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=4, ge=0, le=10, description="Maximum retry attempts"
    )
    backoff_factor: float = Field(
        default=0.6, ge=0.1, le=2.0, description="Backoff multiplier"
    )
    delay_range: tuple[float, float] = Field(
        default=(1.0, 3.0), description="Delay range in seconds"
    )
    workers: int = Field(
        default=5, ge=1, le=20, description="Number of concurrent workers"
    )
    max_pages: int = Field(
        default=4000, ge=1, le=10000, description="Maximum pages to crawl"
    )
    max_depth: int = Field(default=2, ge=1, le=10, description="Maximum crawl depth")
    user_agent: Optional[str] = Field(
        default=None, description="Custom user agent string"
    )
    respect_robots: bool = Field(default=True, description="Respect robots.txt files")
    rate_limit_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Rate limit delay in seconds"
    )


class ProcessingConfig(BaseModel):
    min_word_length: int = Field(
        default=2, ge=1, le=10, description="Minimum word length to keep"
    )
    min_sentence_length: int = Field(
        default=10, ge=5, le=100, description="Minimum sentence length to keep"
    )
    max_token_limit: int = Field(
        default=5000, ge=100, le=50000, description="Maximum token limit for splitting"
    )
    noise_words: set[str] = Field(
        default_factory=lambda: {
            "http",
            "https",
            "www",
            "com",
            "org",
            "contact",
            "follow",
        },
        description="Words to filter out as noise",
    )
    encoding_fallbacks: list[str] = Field(
        default_factory=lambda: ["utf-8", "utf-32", "latin1"],
        description="Encoding fallbacks to try",
    )
    language_detection_enabled: bool = Field(
        default=True, description="Enable automatic language detection"
    )
    quality_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum quality threshold"
    )
    preserve_case: bool = Field(default=False, description="Preserve original case")
    remove_duplicates: bool = Field(
        default=True, description="Remove duplicate sentences/phrases"
    )

    @field_validator("noise_words")
    def validate_noise_words(cls, v):
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        return set(v) if v else set()


class DownloadConfig(BaseModel):
    timeout: float = Field(
        default=15.0, ge=1.0, le=300.0, description="Download timeout in seconds"
    )
    chunk_size: int = Field(
        default=8192, ge=1024, le=1048576, description="Download chunk size in bytes"
    )
    max_workers: int = Field(
        default=3, ge=1, le=10, description="Maximum concurrent downloads"
    )
    retry_attempts: int = Field(
        default=3, ge=1, le=10, description="Retry attempts for downloads"
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    max_file_size: Optional[int] = Field(
        default=None, ge=1, description="Maximum file size in bytes"
    )


class SecurityConfig(BaseModel):
    enable_path_validation: bool = Field(
        default=True, description="Enable path traversal protection"
    )
    enable_url_validation: bool = Field(
        default=True, description="Enable URL validation"
    )
    allowed_domains: Optional[set[str]] = Field(
        default=None, description="Allowed domains for web scraping"
    )
    blocked_domains: Optional[set[str]] = Field(
        default=None, description="Blocked domains for web scraping"
    )
    max_url_length: int = Field(
        default=2048, ge=100, le=8192, description="Maximum URL length"
    )
    enable_audit_logging: bool = Field(
        default=True, description="Enable security audit logging"
    )
    sanitize_user_input: bool = Field(
        default=True, description="Sanitize all user input"
    )


class LoggingConfig(BaseModel):
    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(
        default="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        description="Log format string",
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S", description="Date format string"
    )
    file_output: Optional[DirectoryPath] = Field(
        default=None, description="Log file output directory"
    )
    console_output: bool = Field(default=True, description="Enable console output")
    max_file_size: Optional[int] = Field(
        default=None, ge=1, description="Maximum log file size in bytes"
    )
    enable_rotation: bool = Field(default=False, description="Enable log file rotation")


class MonitoringConfig(BaseModel):
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_retention_days: int = Field(
        default=30, ge=1, le=365, description="Days to retain metrics"
    )
    enable_performance_profiling: bool = Field(
        default=False, description="Enable performance profiling"
    )
    health_check_interval: int = Field(
        default=300, ge=60, le=3600, description="Health check interval in seconds"
    )
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "error_rate": 0.1,
            "processing_time": 30.0,
            "memory_usage": 0.8,
            "disk_usage": 0.9,
        },
        description="Alert thresholds for various metrics",
    )


class Config(BaseModel):
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    output_dir: DirectoryPath = Field(
        default=Path("output"), description="Output directory for processed files"
    )
    verbose: bool = Field(default=False, description="Enable verbose output")

    @field_validator("output_dir")
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="before")
    def validate_config_consistency(cls, values):
        """Validate cross-configuration consistency."""
        # Check that output directory exists
        if not values.get("output_dir"):
            values["output_dir"] = Path("output")

        # Validate cache directory
        cache_config = values.get("cache", {})
        if cache_config.get("disk_cache_dir") is None:
            cache_config["disk_cache_dir"] = Path.home() / ".dataorchestra" / "cache"

        # Validate logging configuration
        logging_config = values.get("logging", {})
        if (
            logging_config.get("file_output")
            and not logging_config.get("file_output").parent.exists()
        ):
            logging_config["file_output"].parent.mkdir(parents=True, exist_ok=True)

        return values

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {yaml_path}: {e}")

    @classmethod
    def from_json(cls, json_path: Path) -> "Config":
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        try:
            import json

            with open(json_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading {json_path}: {e}")

    @classmethod
    def from_env(cls) -> "Config":
        config_dict = {}

        # Logging configuration
        if log_level := os.getenv("DATAORCHESTRA_LOG_LEVEL"):
            config_dict.setdefault("logging", {})["level"] = LogLevel(log_level.lower())

        if log_format := os.getenv("DATAORCHESTRA_LOG_FORMAT"):
            config_dict.setdefault("logging", {})["format"] = log_format

        if log_file := os.getenv("DATAORCHESTRA_LOG_FILE"):
            config_dict.setdefault("logging", {})["file_output"] = Path(log_file)

        # Web configuration
        if timeout := os.getenv("DATAORCHESTRA_TIMEOUT"):
            config_dict.setdefault("web", {})["timeout"] = float(timeout)

        if workers := os.getenv("DATAORCHESTRA_WORKERS"):
            config_dict.setdefault("web", {})["workers"] = int(workers)

        if max_pages := os.getenv("DATAORCHESTRA_MAX_PAGES"):
            config_dict.setdefault("web", {})["max_pages"] = int(max_pages)

        # Processing configuration
        if min_word_length := os.getenv("DATAORCHESTRA_MIN_WORD_LENGTH"):
            config_dict.setdefault("processing", {})["min_word_length"] = int(
                min_word_length
            )

        if max_token_limit := os.getenv("DATAORCHESTRA_MAX_TOKEN_LIMIT"):
            config_dict.setdefault("processing", {})["max_token_limit"] = int(
                max_token_limit
            )

        # Output directory
        if output_dir := os.getenv("DATAORCHESTRA_OUTPUT_DIR"):
            config_dict["output_dir"] = Path(output_dir)

        # Verbose mode
        if verbose := os.getenv("DATAORCHESTRA_VERBOSE"):
            config_dict["verbose"] = verbose.lower() in ("true", "1", "yes")

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    def to_yaml(self, yaml_path: Path) -> None:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving to {yaml_path}: {e}")

    def to_json(self, json_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import json

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving to {json_path}: {e}")

    def validate(self):
        pass

    def get_effective_config(self) -> Dict[str, Any]:
        return self.to_dict()

    def merge_with(self, other_config: "Config") -> "Config":
        current_dict = self.to_dict()
        other_dict = other_config.to_dict()

        # Deep merge dictionaries
        merged = current_dict.copy()
        for key, value in other_dict.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key].update(value)
            else:
                merged[key] = value

        return Config.from_dict(merged)

    def update_from_env(self) -> None:
        env_config = Config.from_env()
        self = self.merge_with(env_config)

    def get_section(self, section_name: str) -> Dict[str, Any]:
        return getattr(self, section_name, {})

    def set_section(self, section_name: str, section_config: Dict[str, Any]) -> None:
        setattr(self, section_name, section_config)

    def get_value(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        value = self

        try:
            for key in keys:
                value = getattr(value, key)
            return value if value is not None else default
        except AttributeError:
            return default

    def set_value(self, path: str, value: Any) -> None:
        keys = path.split(".")
        config_obj = self

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            config_obj = getattr(config_obj, key)

        # Set the final value
        setattr(config_obj, keys[-1], value)

    def __str__(self) -> str:
        return f"Config(output_dir={self.output_dir}, verbose={self.verbose})"


# Global configuration instance with lazy loading
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        # Try to load from environment first
        try:
            _config = Config.from_env()
        except Exception as e:
            # Fall back to default configuration
            _config = Config()
            import logging

            logging.warning(f"Could not load config from environment: {e}")
            logging.info("Using default configuration")

    return _config


def set_config(config: Config) -> None:
    global _config
    config.validate()
    _config = config


def reset_config() -> None:
    global _config
    _config = None


def load_config_from_file(config_path: Path) -> Config:
    config = (
        Config.from_yaml(config_path)
        if config_path.suffix.lower() in [".yml", ".yaml"]
        else Config.from_json(config_path)
    )
    set_config(config)
    return config


def save_config_to_file(config: Config, config_path: Path) -> None:
    if config_path.suffix.lower() in [".yml", ".yaml"]:
        config.to_yaml(config_path)
    else:
        config.to_json(config_path)


def create_sample_config(output_path: Path) -> None:
    sample_config = Config()

    # Add some example values
    sample_config.web.max_pages = 100
    sample_config.processing.min_word_length = 3
    sample_config.cache.enabled = True
    sample_config.security.enable_path_validation = True

    save_config_to_file(sample_config, output_path)
