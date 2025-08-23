"""
Configuration file for GIS Document Processing Application
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "GIS Document Processing"
APP_VERSION = "1.0.0"
APP_AUTHOR = "GIS Development Team"

# Default paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# LLM Configuration
DEFAULT_LLM_CONFIG = {
    'type': 'Ollama',
    'server_url': 'http://localhost:11434',
    'model': 'llama3:8b',
    'api_key': None,
    'max_retries': 3,
    'timeout': 60
}

# Document processing settings
DEFAULT_PROCESSING_CONFIG = {
    'chunk_size': 1000,
    'enable_ocr': False,
    'max_file_size_mb': 100,
    'supported_formats': ['.pdf', '.docx', '.xlsx', '.xls', '.txt']
}

# Coordinate system settings
DEFAULT_CRS = 'EPSG:4326'  # WGS84
SUPPORTED_CRS = [
    'EPSG:4326',  # WGS84
    'EPSG:3857',  # Web Mercator
    'EPSG:25830', # ETRS89 / UTM zone 30N
    'EPSG:32630', # WGS84 / UTM zone 30N
]

# Export settings
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'excel_engine': 'xlsxwriter',
    'shapefile_crs': 'EPSG:4326',
    'arcgis_crs': 'EPSG:4326'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOG_DIR / 'app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# GUI settings
GUI_CONFIG = {
    'window_size': (1200, 800),
    'table_max_rows': 10000,
    'progress_update_interval': 100,  # milliseconds
    'theme': 'Fusion'
}

# Validation rules
VALIDATION_RULES = {
    'min_address_length': 5,
    'max_address_length': 500,
    'min_text_length': 10,
    'coordinate_precision': 6,
    'max_coordinate_value': 1000000
}

# Error messages
ERROR_MESSAGES = {
    'file_not_found': 'File not found: {file_path}',
    'unsupported_format': 'Unsupported file format: {format}',
    'llm_connection_failed': 'Failed to connect to LLM service: {error}',
    'processing_failed': 'Document processing failed: {error}',
    'export_failed': 'Export failed: {error}',
    'no_data': 'No data to process or export',
    'invalid_coordinates': 'Invalid coordinate values found',
    'missing_required_fields': 'Missing required fields: {fields}'
}

# Success messages
SUCCESS_MESSAGES = {
    'document_loaded': 'Document loaded successfully: {filename}',
    'processing_complete': 'Document processing completed successfully',
    'export_complete': 'Data exported successfully to {format}',
    'llm_connected': 'Successfully connected to LLM service'
}

# File size limits (in bytes)
FILE_SIZE_LIMITS = {
    'pdf': 100 * 1024 * 1024,      # 100 MB
    'docx': 50 * 1024 * 1024,      # 50 MB
    'xlsx': 50 * 1024 * 1024,      # 50 MB
    'txt': 10 * 1024 * 1024        # 10 MB
}

# OCR settings
OCR_CONFIG = {
    'dpi': 300,
    'language': 'eng',
    'tesseract_config': '--psm 6',  # Assume uniform block of text
    'image_format': 'PNG'
}

# Geocoding settings (if using external geocoding service)
GEOCODING_CONFIG = {
    'service': 'nominatim',  # OpenStreetMap Nominatim
    'base_url': 'https://nominatim.openstreetmap.org/search',
    'user_agent': 'GIS_Document_Processor/1.0',
    'rate_limit': 1.0,  # requests per second
    'timeout': 10
} 