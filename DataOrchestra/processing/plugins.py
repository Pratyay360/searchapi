"""
Plugin system for extensible text processing.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Type, Callable, Optional
from dataclasses import dataclass

from ..core.logging_utils import get_logger
from ..core.exceptions import ValidationError


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    version: str
    description: str
    author: str
    processor_class: Type
    config_schema: Optional[Dict[str, Any]] = None


class PluginManager:
    """
    Manages loading and registration of text processing plugins.
    """
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.plugin_dir = plugin_dir or Path.home() / ".dataorchestra" / "plugins"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        self._processors: Dict[str, PluginInfo] = {}
        self._strategies: Dict[str, PluginInfo] = {}
        self._assessors: Dict[str, PluginInfo] = {}
        
        self.logger.info(f"Plugin manager initialized: {self.plugin_dir}")
    
    def discover_plugins(self) -> None:
        """Discover and load plugins from the plugin directory."""
        if not self.plugin_dir.exists():
            self.logger.info(f"Plugin directory not found: {self.plugin_dir}")
            return
        
        # Scan for plugin files
        for plugin_file in self.plugin_dir.glob("*_plugin.py"):
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file.name}: {e}")
    
    def _load_plugin(self, plugin_file: Path) -> None:
        """Load a single plugin file."""
        module_name = plugin_file.stem
        
        try:
            # Import the plugin module
            import importlib.util
            spec = importlib.util.spec_from_file_location(str(plugin_file))
            if spec is None:
                self.logger.warning(f"Could not create spec for plugin: {plugin_file}")
                return
            module = importlib.util.module_from_spec(spec)
            
            # Look for plugin info
            if hasattr(module, 'PLUGIN_INFO'):
                plugin_info = module.PLUGIN_INFO
            else:
                # Try to auto-detect plugin components
                plugin_info = self._auto_detect_plugin_info(module, module_name)
            
            # Register plugin components
            self._register_plugin_components(module, plugin_info)
            
            self.logger.info(f"Loaded plugin: {plugin_info.name} v{plugin_info.version}")
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_file}: {e}")
            raise
    
    def _auto_detect_plugin_info(self, module: Any, module_name: str) -> PluginInfo:
        """Auto-detect plugin information from module."""
        # Look for processor classes
        processors = []
        strategies = []
        assessors = []
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not name.startswith('_'):
                # Check if it's a processor
                if hasattr(obj, 'process') and hasattr(obj, '__init__'):
                    processors.append(obj)
                
                # Check if it's a strategy
                if hasattr(obj, 'apply') and hasattr(obj, '__init__'):
                    strategies.append(obj)
                
                # Check if it's an assessor
                if hasattr(obj, 'assess') and hasattr(obj, '__init__'):
                    assessors.append(obj)
        
        # Create plugin info based on what we found
        if processors:
            processor_class = processors[0]  # Use the first one found
            return PluginInfo(
                name=module_name.replace('_plugin', ''),
                version=getattr(processor_class, 'VERSION', '1.0.0'),
                description=getattr(processor_class, 'DESCRIPTION', f'Custom processor from {module_name}'),
                author=getattr(processor_class, 'AUTHOR', 'Unknown'),
                processor_class=processor_class,
                config_schema=getattr(processor_class, 'CONFIG_SCHEMA', None)
            )
        elif strategies:
            strategy_class = strategies[0]  # Use the first one found
            return PluginInfo(
                name=module_name.replace('_plugin', ''),
                version=getattr(strategy_class, 'VERSION', '1.0.0'),
                description=getattr(strategy_class, 'DESCRIPTION', f'Custom strategy from {module_name}'),
                author=getattr(strategy_class, 'AUTHOR', 'Unknown'),
                processor_class=strategy_class,
                config_schema=getattr(strategy_class, 'CONFIG_SCHEMA', None)
            )
        elif assessors:
            assessor_class = assessors[0]  # Use the first one found
            return PluginInfo(
                name=module_name.replace('_plugin', ''),
                version=getattr(assessor_class, 'VERSION', '1.0.0'),
                description=getattr(assessor_class, 'DESCRIPTION', f'Custom assessor from {module_name}'),
                author=getattr(assessor_class, 'AUTHOR', 'Unknown'),
                processor_class=assessor_class,
                config_schema=getattr(assessor_class, 'CONFIG_SCHEMA', None)
            )
        else:
            raise ValidationError(f"No valid plugin components found in {module_name}")
    
    def _register_plugin_components(self, module: Any, plugin_info: PluginInfo) -> None:
        """Register plugin components in appropriate categories."""
        processor_class = plugin_info.processor_class
        
        # Determine the type of plugin
        if hasattr(processor_class, 'process'):
            self._processors[plugin_info.name] = plugin_info
        elif hasattr(processor_class, 'apply'):
            self._strategies[plugin_info.name] = plugin_info
        elif hasattr(processor_class, 'assess'):
            self._assessors[plugin_info.name] = plugin_info
    
    def get_processor(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a registered processor by name.
        
        Args:
            name: Name of the processor
            config: Optional configuration for the processor
            
        Returns:
            Processor instance or None if not found
        """
        if name not in self._processors:
            self.logger.warning(f"Processor not found: {name}")
            return None
        
        plugin_info = self._processors[name]
        
        try:
            # Validate config if schema is available
            if plugin_info.config_schema and config:
                self._validate_config(config, plugin_info.config_schema)
            
            # Create processor instance
            if config:
                return plugin_info.processor_class(**config)
            else:
                return plugin_info.processor_class()
                
        except Exception as e:
            self.logger.error(f"Failed to create processor {name}: {e}")
            return None
    
    def get_strategy(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a registered strategy by name.
        
        Args:
            name: Name of the strategy
            config: Optional configuration for the strategy
            
        Returns:
            Strategy instance or None if not found
        """
        if name not in self._strategies:
            self.logger.warning(f"Strategy not found: {name}")
            return None
        
        plugin_info = self._strategies[name]
        
        try:
            # Validate config if schema is available
            if plugin_info.config_schema and config:
                self._validate_config(config, plugin_info.config_schema)
            
            # Create strategy instance
            if config:
                return plugin_info.processor_class(**config)
            else:
                return plugin_info.processor_class()
                
        except Exception as e:
            self.logger.error(f"Failed to create strategy {name}: {e}")
            return None
    
    def get_assessor(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a registered assessor by name.
        
        Args:
            name: Name of the assessor
            config: Optional configuration for the assessor
            
        Returns:
            Assessor instance or None if not found
        """
        if name not in self._assessors:
            self.logger.warning(f"Assessor not found: {name}")
            return None
        
        plugin_info = self._assessors[name]
        
        try:
            # Validate config if schema is available
            if plugin_info.config_schema and config:
                self._validate_config(config, plugin_info.config_schema)
            
            # Create assessor instance
            if config:
                return plugin_info.processor_class(**config)
            else:
                return plugin_info.processor_class()
                
        except Exception as e:
            self.logger.error(f"Failed to create assessor {name}: {e}")
            return None
    
    def _validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        # Simple validation - can be enhanced with jsonschema
        for key, value_type in schema.items():
            if key in config:
                if not isinstance(config[key], value_type):
                    raise ValidationError(f"Config key '{key}' should be {value_type.__name__}, got {type(config[key]).__name__}")
            elif key not in config and schema.get('required', False):
                raise ValidationError(f"Required config key '{key}' is missing")
    
    def list_processors(self) -> List[str]:
        """List all available processors."""
        return list(self._processors.keys())
    
    def list_strategies(self) -> List[str]:
        """List all available strategies."""
        return list(self._strategies.keys())
    
    def list_assessors(self) -> List[str]:
        """List all available assessors."""
        return list(self._assessors.keys())
    
    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """Get information about a specific plugin."""
        if name in self._processors:
            return self._processors[name]
        elif name in self._strategies:
            return self._strategies[name]
        elif name in self._assessors:
            return self._assessors[name]
        return None
    
    def reload_plugins(self) -> None:
        """Reload all plugins from the plugin directory."""
        self._processors.clear()
        self._strategies.clear()
        self._assessors.clear()
        self.discover_plugins()
        self.logger.info("Plugins reloaded")
    
    def create_plugin_template(self, name: str, plugin_type: str = "processor") -> str:
        """
        Create a template for a new plugin.
        
        Args:
            name: Name of the plugin
            plugin_type: Type of plugin (processor, strategy, assessor)
            
        Returns:
            Template code as string
        """
        if plugin_type == "processor":
            template = f'''"""
{name} plugin for DataOrchestra.

This plugin provides a custom text processor.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from ..core.base import BaseProcessor
from ..core.logging_utils import get_logger


class {name}Processor(BaseProcessor[str, str]):
    """
    Custom text processor: {name}.
    """
    
    VERSION = "1.0.0"
    DESCRIPTION = "Custom {name} processor"
    AUTHOR = "Your Name"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("{name}Processor")
        self.config = config or {{}}
        self.logger = get_logger(self.__class__.__name__)
    
    def process(self, text: str) -> str:
        """
        Process text using custom {name} logic.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        # Implement your custom processing logic here
        self.logger.info(f"Processing text with {name} processor")
        
        # Example: simple text transformation
        processed_text = text.upper()  # Replace with your logic
        
        return processed_text


# Register the plugin
PLUGIN_INFO = {{
    "name": "{name}",
    "version": VERSION,
    "description": DESCRIPTION,
    "author": AUTHOR,
    "processor_class": {name}Processor,
    "config_schema": {{
        "required": False,
        "type": "object",
        "properties": {{
            "custom_param": {{
                "type": "string",
                "description": "Custom parameter for {name} processing"
            }}
        }}
    }}
}}
'''
        elif plugin_type == "strategy":
            template = f'''"""
{name} strategy for DataOrchestra.

This plugin provides a custom text cleaning strategy.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from ..core.config import ProcessingConfig
from ..core.logging_utils import get_logger


class {name}Strategy:
    """
    Custom {name} cleaning strategy.
    """
    
    VERSION = "1.0.0"
    DESCRIPTION = "Custom {name} cleaning strategy"
    AUTHOR = "Your Name"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.logger = get_logger(self.__class__.__name__)
    
    def apply(self, text: str, config: ProcessingConfig) -> str:
        """
        Apply {name} cleaning strategy to text.
        
        Args:
            text: Input text to clean
            config: Processing configuration
            
        Returns:
            Cleaned text
        """
        # Implement your custom strategy logic here
        self.logger.info(f"Applying {name} strategy")
        
        # Example: simple text transformation
        cleaned_text = text.replace("{name}", "[CLEANED]")  # Replace with your logic
        
        return cleaned_text


# Register the strategy
PLUGIN_INFO = {{
    "name": "{name}",
    "version": VERSION,
    "description": DESCRIPTION,
    "author": AUTHOR,
    "processor_class": {name}Strategy,
    "config_schema": {{
        "required": False,
        "type": "object",
        "properties": {{
            "custom_param": {{
                "type": "string",
                "description": "Custom parameter for {name} strategy"
            }}
        }}
    }}
}}
'''
        elif plugin_type == "assessor":
            template = f'''"""
{name} assessor for DataOrchestra.

This plugin provides a custom text quality assessor.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from ..core.logging_utils import get_logger


class {name}Assessor:
    """
    Custom {name} text quality assessor.
    """
    
    VERSION = "1.0.0"
    DESCRIPTION = "Custom {name} quality assessor"
    AUTHOR = "Your Name"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.logger = get_logger(self.__class__.__name__)
    
    def assess(self, text: str) -> float:
        """
        Assess text quality using custom {name} logic.
        
        Args:
            text: Text to assess
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Implement your custom assessment logic here
        self.logger.info(f"Assessing text with {name} assessor")
        
        # Example: simple scoring based on text characteristics
        word_count = len(text.split())
        score = min(1.0, word_count / 100.0)  # Simple scoring
        
        return score


# Register the assessor
PLUGIN_INFO = {{
    "name": "{name}",
    "version": VERSION,
    "description": DESCRIPTION,
    "author": AUTHOR,
    "processor_class": {name}Assessor,
    "config_schema": {{
        "required": False,
        "type": "object",
        "properties": {{
            "custom_param": {{
                "type": "number",
                "description": "Custom parameter for {name} assessment"
            }}
        }}
    }}
}}
'''
        
        return template
    
    def install_plugin(self, plugin_path: Path) -> None:
        """
        Install a plugin from a file.
        
        Args:
            plugin_path: Path to plugin file
        """
        try:
            import shutil
            target_path = self.plugin_dir / plugin_path.name
            
            # Copy plugin to plugin directory
            shutil.copy2(plugin_path, target_path)
            
            self.logger.info(f"Installed plugin: {plugin_path.name}")
            
            # Reload plugins to include the new one
            self.reload_plugins()
            
        except Exception as e:
            self.logger.error(f"Failed to install plugin {plugin_path.name}: {e}")
            raise
    
    def uninstall_plugin(self, name: str) -> None:
        """
        Uninstall a plugin by name.
        
        Args:
            name: Name of the plugin to uninstall
        """
        try:
            # Remove from registry
            if name in self._processors:
                del self._processors[name]
            elif name in self._strategies:
                del self._strategies[name]
            elif name in self._assessors:
                del self._assessors[name]
            
            # Remove plugin file if it exists
            plugin_file = self.plugin_dir / f"{name}_plugin.py"
            if plugin_file.exists():
                import os
                os.remove(plugin_file)
            
            self.logger.info(f"Uninstalled plugin: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to uninstall plugin {name}: {e}")
            raise