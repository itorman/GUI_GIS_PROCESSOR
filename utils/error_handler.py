"""
Centralized error handling and logging utilities.
Provides consistent error handling patterns across the application.
"""

import logging
import traceback
from enum import Enum
from typing import Optional, Any, Dict
from pathlib import Path


class ErrorLevel(Enum):
    """Error severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ApplicationError(Exception):
    """Base application error with enhanced context"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/reporting"""
        return {
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'type': self.__class__.__name__
        }


class DocumentProcessingError(ApplicationError):
    """Errors related to document processing"""
    pass


class LLMError(ApplicationError):
    """Errors related to LLM operations"""
    pass


class ExportError(ApplicationError):
    """Errors related to data export"""
    pass


class ConfigurationError(ApplicationError):
    """Errors related to configuration"""
    pass


class ErrorHandler:
    """Centralized error handler with logging and user-friendly messages"""
    
    def __init__(self, logger_name: str = __name__):
        """
        Initialize error handler
        
        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler(log_dir / "error.log")
            file_handler.setLevel(logging.ERROR)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.DEBUG)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    user_message: str = None) -> Dict[str, Any]:
        """
        Handle an error with logging and user-friendly message generation
        
        Args:
            error: The exception that occurred
            context: Additional context information
            user_message: Custom user-friendly message
            
        Returns:
            Dictionary with error information for UI display
        """
        error_info = {
            'success': False,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'user_message': user_message or self._get_user_friendly_message(error),
            'context': context or {},
            'traceback': None
        }
        
        # Log the error
        if isinstance(error, ApplicationError):
            self.logger.error(f"Application Error: {error.to_dict()}")
        else:
            self.logger.error(f"Unexpected Error: {str(error)}", exc_info=True)
            error_info['traceback'] = traceback.format_exc()
        
        return error_info
    
    def _get_user_friendly_message(self, error: Exception) -> str:
        """
        Generate user-friendly error messages
        
        Args:
            error: The exception
            
        Returns:
            User-friendly error message
        """
        if isinstance(error, DocumentProcessingError):
            return "There was a problem processing the document. Please check the file format and try again."
        
        elif isinstance(error, LLMError):
            return "There was an issue connecting to the language model. Please check your LLM settings and try again."
        
        elif isinstance(error, ExportError):
            return "Failed to export the data. Please check the output path and permissions."
        
        elif isinstance(error, ConfigurationError):
            return "There's an issue with the application configuration. Please check your settings."
        
        elif "No module named" in str(error):
            module_name = str(error).split("'")[1] if "'" in str(error) else "unknown"
            return f"Missing required component: {module_name}. Please install the necessary dependencies."
        
        elif "Permission denied" in str(error):
            return "Permission denied. Please check file/folder permissions and try again."
        
        elif "No such file or directory" in str(error):
            return "File not found. Please check the file path and try again."
        
        else:
            return "An unexpected error occurred. Please check the logs for more details."
    
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """Log informational message"""
        if context:
            message = f"{message} - Context: {context}"
        self.logger.info(message)
    
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning message"""
        if context:
            message = f"{message} - Context: {context}"
        self.logger.warning(message)
    
    def log_error(self, message: str, context: Dict[str, Any] = None):
        """Log error message"""
        if context:
            message = f"{message} - Context: {context}"
        self.logger.error(message)


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(error: Exception, context: Dict[str, Any] = None, 
                user_message: str = None) -> Dict[str, Any]:
    """Convenience function for error handling"""
    return get_error_handler().handle_error(error, context, user_message)


def log_info(message: str, context: Dict[str, Any] = None):
    """Convenience function for info logging"""
    get_error_handler().log_info(message, context)


def log_warning(message: str, context: Dict[str, Any] = None):
    """Convenience function for warning logging"""
    get_error_handler().log_warning(message, context)


def log_error(message: str, context: Dict[str, Any] = None):
    """Convenience function for error logging"""
    get_error_handler().log_error(message, context)


def safe_execute(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with execution results
    """
    try:
        result = func(*args, **kwargs)
        return {
            'success': True,
            'result': result,
            'error': None
        }
    except Exception as e:
        error_info = handle_error(e)
        return {
            'success': False,
            'result': None,
            'error': error_info
        }