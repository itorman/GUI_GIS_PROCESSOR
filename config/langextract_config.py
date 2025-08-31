"""
Configuration file for langextract optimization and settings.
Contains parameters for efficient address extraction and batch processing.
"""

import os
from typing import Dict, Any

# Langextract configuration
LANGEXTRACT_CONFIG = {
    # Extraction settings
    'max_retries': 3,
    'timeout': 60,
    'fallback_strategy': 'partial',  # Extract what's possible even if schema validation fails
    
    # Batch processing optimization
    'batch_size': 5,  # Number of chunks to process together
    'parallel_processing': False,  # Enable parallel processing for large documents
    
    # Schema validation settings
    'strict_validation': False,  # Allow partial validation for better extraction
    'auto_salvage': True,  # Automatically try to fix invalid results
    
    # Language detection and support
    'multilingual_support': True,  # Enable support for multiple languages
    'default_language': 'en',  # Default language for extraction
    
    # Performance tuning
    'cache_schemas': True,  # Cache schemas for better performance
    'optimize_prompts': True,  # Use optimized prompts for better extraction
}

# Language-specific extraction hints
LANGUAGE_EXTRACTION_HINTS = {
    'en': {
        'address_patterns': [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)',
            r'[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\s+\d+'
        ],
        'postal_code_pattern': r'\b\d{5}(?:-\d{4})?\b',  # US ZIP codes
        'city_patterns': [r'\b[A-Z][a-z]+\s*(?:City|Town|Village|Borough)\b']
    },
    'es': {
        'address_patterns': [
            r'\d+\s+[A-Za-z\s]+(?:Calle|Avenida|Ave|Carretera|Ctra|Paseo|Plaza|Pl)',
            r'[A-Za-z\s]+(?:Calle|Avenida|Ave|Carretera|Ctra|Paseo|Plaza|Pl)\s+\d+'
        ],
        'postal_code_pattern': r'\b\d{5}\b',  # Spanish postal codes
        'city_patterns': [r'\b[A-Z][a-z]+\s*(?:Ciudad|Pueblo|Villa|Municipio)\b']
    },
    'fr': {
        'address_patterns': [
            r'\d+\s+[A-Za-z\s]+(?:Rue|Avenue|Ave|Boulevard|Bd|Place|Pl|Chemin|Ch)',
            r'[A-Za-z\s]+(?:Rue|Avenue|Ave|Boulevard|Bd|Place|Pl|Chemin|Ch)\s+\d+'
        ],
        'postal_code_pattern': r'\b\d{5}\b',  # French postal codes
        'city_patterns': [r'\b[A-Z][a-z]+\s*(?:Ville|Commune|Canton)\b']
    },
    'de': {
        'address_patterns': [
            r'\d+\s+[A-Za-z\s]+(?:Straße|Str|Gasse|Weg|Platz|Pl|Allee|Al)',
            r'[A-Za-z\s]+(?:Straße|Str|Gasse|Weg|Platz|Pl|Allee|Al)\s+\d+'
        ],
        'postal_code_pattern': r'\b\d{5}\b',  # German postal codes
        'city_patterns': [r'\b[A-Z][a-z]+\s*(?:Stadt|Dorf|Gemeinde)\b']
    }
}

# Coordinate extraction patterns
COORDINATE_PATTERNS = {
    'decimal_degrees': [
        r'(-?\d+\.\d+)\s*[°º]\s*([NS])\s*,?\s*(-?\d+\.\d+)\s*[°º]\s*([EW])',
        r'([NS])\s*(-?\d+\.\d+)\s*[°º]\s*,?\s*([EW])\s*(-?\d+\.\d+)\s*[°º]',
        r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)',  # Simple lat,lon format
    ],
    'degrees_minutes_seconds': [
        r'(\d+)°\s*(\d+)\'?\s*(\d+\.?\d*)"?\s*([NS])\s*,?\s*(\d+)°\s*(\d+)\'?\s*(\d+\.?\d*)"?\s*([EW])',
        r'([NS])\s*(\d+)°\s*(\d+)\'?\s*(\d+\.?\d*)"?\s*,?\s*([EW])\s*(\d+)°\s*(\d+)\'?\s*(\d+\.?\d*)"?'
    ],
    'utm': [
        r'(\d{1,2})\s*([NS])\s*(\d{6,7}\.?\d*)\s*(\d{6,7}\.?\d*)',
        r'UTM\s*(\d{1,2})\s*([NS])\s*(\d{6,7}\.?\d*)\s*(\d{6,7}\.?\d*)'
    ]
}

# Performance optimization settings
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_size': 1000,  # Maximum number of cached extractions
    'chunk_optimization': True,  # Optimize chunk sizes for better extraction
    'memory_management': True,  # Enable memory optimization for large documents
    'parallel_threshold': 10,  # Minimum chunks for parallel processing
}

# Error handling and recovery
ERROR_HANDLING_CONFIG = {
    'max_consecutive_failures': 3,
    'retry_delay': 1.0,  # Seconds between retries
    'graceful_degradation': True,  # Continue processing even if some chunks fail
    'log_failures': True,  # Log all extraction failures for analysis
}

def get_langextract_config() -> Dict[str, Any]:
    """Get the complete langextract configuration"""
    config = LANGEXTRACT_CONFIG.copy()
    
    # Override with environment variables if available
    for key in config:
        env_key = f'LANGEXTRACT_{key.upper()}'
        if env_key in os.environ:
            try:
                # Try to convert to appropriate type
                value = os.environ[env_key]
                if isinstance(config[key], bool):
                    config[key] = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(config[key], int):
                    config[key] = int(value)
                elif isinstance(config[key], float):
                    config[key] = float(value)
                else:
                    config[key] = value
            except (ValueError, TypeError):
                pass  # Keep default value if conversion fails
    
    return config

def get_language_hints(language_code: str = 'en') -> Dict[str, Any]:
    """Get language-specific extraction hints"""
    return LANGUAGE_EXTRACTION_HINTS.get(language_code.lower(), LANGUAGE_EXTRACTION_HINTS['en'])

def get_coordinate_patterns() -> Dict[str, Any]:
    """Get coordinate extraction patterns"""
    return COORDINATE_PATTERNS

def get_performance_config() -> Dict[str, Any]:
    """Get performance optimization settings"""
    return PERFORMANCE_CONFIG

def get_error_handling_config() -> Dict[str, Any]:
    """Get error handling configuration"""
    return ERROR_HANDLING_CONFIG
