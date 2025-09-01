"""
Centralized configuration management for the GIS Document Processing Application.
Provides a unified interface for all application settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config_data = {}
        self._load_default_config()
        self._load_config_file()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        app_dir = Path(__file__).parent.parent
        return str(app_dir / "config" / "app_config.json")
    
    def _load_default_config(self):
        """Load default configuration settings"""
        self.config_data = {
            # Application settings
            'app': {
                'name': 'GIS Document Processing Application',
                'version': '2.0.0',
                'window_title': 'GIS Document Processing - Address Extraction',
                'window_size': {'width': 1200, 'height': 800},
                'max_file_size_mb': 100,
                'supported_formats': ['.pdf', '.docx', '.xlsx', '.txt']
            },
            
            # LLM settings
            'llm': {
                'default_type': 'Ollama',
                'default_model': 'llama3.1:8b',
                'default_server_url': 'http://localhost:11434',
                'timeout': 30,
                'max_retries': 3,
                'use_contextual': True,
                'chunk_size': 1000
            },
            
            # Processing settings
            'processing': {
                'enable_ocr': False,
                'enable_quality_assessment': True,
                'min_confidence_threshold': 0.5,
                'target_crs': 'EPSG:4326',
                'max_concurrent_chunks': 5
            },
            
            # Export settings
            'export': {
                'default_format': 'csv',
                'default_output_dir': './output',
                'include_quality_scores': True,
                'create_backup': False
            },
            
            # GUI settings
            'gui': {
                'theme': 'default',
                'show_progress_details': True,
                'auto_save_settings': True,
                'recent_files_count': 10
            },
            
            # Logging settings
            'logging': {
                'level': 'INFO',
                'file': './logs/app.log',
                'max_file_size_mb': 10,
                'backup_count': 5
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file if it exists"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge file config with defaults
                self._merge_config(self.config_data, file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.info("No configuration file found, using defaults")
                
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {str(e)}, using defaults")
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure config directory exists
            config_dir = Path(self.config_file).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., 'llm.default_model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config_data
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., 'llm.default_model')
            value: Value to set
        """
        try:
            keys = key_path.split('.')
            config = self.config_data
            
            # Navigate to the parent dictionary
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            
        except Exception as e:
            logger.error(f"Failed to set configuration value {key_path}: {str(e)}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'llm', 'processing')
            
        Returns:
            Dictionary with section configuration
        """
        return self.config_data.get(section, {}).copy()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.get_section('llm')
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.get_section('processing')
    
    def get_export_config(self) -> Dict[str, Any]:
        """Get export configuration"""
        return self.get_section('export')
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration"""
        return self.get_section('gui')
    
    def update_llm_config(self, config: Dict[str, Any]):
        """Update LLM configuration section"""
        self.config_data['llm'].update(config)
    
    def update_processing_config(self, config: Dict[str, Any]):
        """Update processing configuration section"""
        self.config_data['processing'].update(config)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate LLM configuration
        llm_config = self.get_section('llm')
        if not llm_config.get('default_model'):
            result['errors'].append('LLM default model not specified')
            result['valid'] = False
        
        if not llm_config.get('default_server_url'):
            result['warnings'].append('LLM server URL not specified')
        
        # Validate processing configuration
        processing_config = self.get_section('processing')
        threshold = processing_config.get('min_confidence_threshold', 0)
        if not (0 <= threshold <= 1):
            result['errors'].append('Confidence threshold must be between 0 and 1')
            result['valid'] = False
        
        # Validate export configuration
        export_config = self.get_section('export')
        output_dir = export_config.get('default_output_dir')
        if output_dir:
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                result['warnings'].append(f'Cannot create output directory: {output_dir}')
        
        return result
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self._load_default_config()
        logger.info("Configuration reset to defaults")
    
    def export_config(self, file_path: str):
        """
        Export configuration to a file
        
        Args:
            file_path: Path to export file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            logger.info(f"Configuration exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export configuration: {str(e)}")
    
    def import_config(self, file_path: str):
        """
        Import configuration from a file
        
        Args:
            file_path: Path to import file
        """
        try:
            with open(file_path, 'r') as f:
                imported_config = json.load(f)
            
            self._merge_config(self.config_data, imported_config)
            logger.info(f"Configuration imported from {file_path}")
        except Exception as e:
            logger.error(f"Failed to import configuration: {str(e)}")


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_config(key_path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return get_config_manager().get(key_path, default)


def set_config(key_path: str, value: Any):
    """Convenience function to set configuration value"""
    get_config_manager().set(key_path, value)


def save_config():
    """Convenience function to save configuration"""
    get_config_manager().save_config()