"""
File utility functions for GIS Document Processing Application
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def validate_file(file_path: str, max_size_mb: int = 100) -> Dict[str, Any]:
        """
        Validate a file for processing
        
        Args:
            file_path: Path to the file
            max_size_mb: Maximum file size in MB
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                result['errors'].append(f"File not found: {file_path}")
                return result
            
            # Check file size
            file_size = file_path.stat().st_size
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                result['errors'].append(f"File too large: {file_size / (1024*1024):.1f} MB > {max_size_mb} MB")
            
            # Get file info
            result['file_info'] = {
                'name': file_path.name,
                'size': file_size,
                'size_mb': file_size / (1024*1024),
                'extension': file_path.suffix.lower(),
                'mime_type': FileUtils.get_mime_type(file_path),
                'last_modified': file_path.stat().st_mtime
            }
            
            # Check file extension
            supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.txt']
            if file_path.suffix.lower() not in supported_extensions:
                result['errors'].append(f"Unsupported file format: {file_path.suffix}")
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                result['errors'].append("File is not readable")
            
            # Check if file is empty
            if file_size == 0:
                result['errors'].append("File is empty")
            
            # Set valid flag
            result['valid'] = len(result['errors']) == 0
            
            # Add warnings for large files
            if file_size > 50 * 1024 * 1024:  # 50 MB
                result['warnings'].append("Large file - processing may take longer")
            
        except Exception as e:
            result['errors'].append(f"File validation error: {str(e)}")
            logger.error(f"File validation failed: {e}")
        
        return result
    
    @staticmethod
    def get_mime_type(file_path: Path) -> str:
        """Get MIME type of a file"""
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type or 'application/octet-stream'
        except Exception:
            return 'application/octet-stream'
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
        """
        Calculate file hash for integrity checking
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            File hash string or None if failed
        """
        try:
            hash_func = getattr(hashlib, algorithm.lower())
            
            with open(file_path, 'rb') as f:
                file_hash = hash_func()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
            
            return file_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            return None
    
    @staticmethod
    def create_backup(file_path: str, backup_dir: str = None) -> Optional[str]:
        """
        Create a backup copy of a file
        
        Args:
            file_path: Path to the original file
            backup_dir: Backup directory (optional)
            
        Returns:
            Path to backup file or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if backup_dir is None:
                backup_dir = file_path.parent / 'backup'
            
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create backup filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_file = backup_path / backup_filename
            
            # Copy file
            import shutil
            shutil.copy2(file_path, backup_file)
            
            logger.info(f"Backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age
        
        Args:
            temp_dir: Directory containing temp files
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        try:
            temp_path = Path(temp_dir)
            if not temp_path.exists():
                return 0
            
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            cleaned_count = 0
            
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.debug(f"Cleaned up temp file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temp file {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary files")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0
    
    @staticmethod
    def get_file_encoding(file_path: str) -> str:
        """
        Detect file encoding
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Default to utf-8 if detection fails
        return 'utf-8'
    
    @staticmethod
    def split_large_file(file_path: str, max_chunk_size_mb: int = 10) -> List[str]:
        """
        Split a large file into smaller chunks
        
        Args:
            file_path: Path to the large file
            max_chunk_size_mb: Maximum chunk size in MB
            
        Returns:
            List of chunk file paths
        """
        try:
            file_path = Path(file_path)
            max_chunk_size = max_chunk_size_mb * 1024 * 1024
            
            if file_path.stat().st_size <= max_chunk_size:
                return [str(file_path)]
            
            chunk_files = []
            chunk_num = 1
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk_data = f.read(max_chunk_size)
                    if not chunk_data:
                        break
                    
                    chunk_filename = f"{file_path.stem}_chunk_{chunk_num:03d}{file_path.suffix}"
                    chunk_path = file_path.parent / chunk_filename
                    
                    with open(chunk_path, 'wb') as chunk_file:
                        chunk_file.write(chunk_data)
                    
                    chunk_files.append(str(chunk_path))
                    chunk_num += 1
            
            logger.info(f"Split {file_path.name} into {len(chunk_files)} chunks")
            return chunk_files
            
        except Exception as e:
            logger.error(f"Failed to split file: {e}")
            return []
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """
        Calculate total size of a directory
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        try:
            total_size = 0
            dir_path = Path(directory)
            
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
            return 0 