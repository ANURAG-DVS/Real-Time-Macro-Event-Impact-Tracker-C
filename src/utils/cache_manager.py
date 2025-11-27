"""
Cache Manager - Universal Caching Infrastructure

This module provides a centralized, universal caching system for the Real-Time Macro Event Impact Tracker.
The CacheManager provides efficient data caching with support for multiple formats and comprehensive
cache management capabilities.

The CacheManager supports:
- Universal cache key generation from any function arguments
- Multiple data formats (pickle for complex objects, JSON for simple data)
- Configurable expiry times and cache enabling/disabling
- Comprehensive cache statistics and management
- Context manager support for clean resource handling
"""

import pickle
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict
from config.settings import config


class CacheManager:
    """
    Universal cache manager for efficient data caching across the application.

    This class provides a centralized caching infrastructure that can be used by any
    component in the system. It supports multiple data formats, configurable expiry,
    and comprehensive cache management operations.

    Features:
    - Universal cache key generation from function arguments
    - Support for pickle and JSON data formats
    - Configurable cache expiry and enable/disable functionality
    - Comprehensive cache statistics and management
    - Context manager support for clean resource handling

    Attributes:
        cache_dir: Directory where cache files are stored
        expiry_hours: Hours after which cache entries expire
        logger: Logger instance for operation tracking
        enabled: Whether caching is enabled (can be toggled at runtime)
    """

    def __init__(self,
                 cache_dir: Optional[Path] = None,
                 expiry_hours: Optional[int] = None,
                 enabled: Optional[bool] = None) -> None:
        """
        Initialize the cache manager with configuration.

        Args:
            cache_dir: Directory for cache storage (defaults to config.CACHE_DIR)
            expiry_hours: Hours before cache expires (defaults to config.CACHE_EXPIRY_HOURS)
            enabled: Whether caching is enabled (defaults to config.ENABLE_CACHE)
        """
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.expiry_hours = expiry_hours or config.CACHE_EXPIRY_HOURS
        self.enabled = enabled if enabled is not None else config.ENABLE_CACHE
        self.logger = logging.getLogger('cache_manager')

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"CacheManager initialized: dir={self.cache_dir}, "
            f"expiry={self.expiry_hours}h, enabled={self.enabled}"
        )

    def generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from function arguments.

        This method creates a deterministic hash from any combination of positional
        and keyword arguments, allowing any function to cache its results universally.

        Args:
            *args: Positional arguments to include in key generation
            **kwargs: Keyword arguments to include in key generation

        Returns:
            str: Unique cache key as MD5 hash string

        Example:
            >>> cache = CacheManager()
            >>> key = cache.generate_cache_key('function_name', arg1='value1', arg2=42)
            >>> print(key)  # MD5 hash string
        """
        # Convert all arguments to strings for consistent hashing
        arg_strings = []

        # Add positional arguments
        for arg in args:
            arg_strings.append(str(arg))

        # Add keyword arguments (sorted for consistency)
        for key in sorted(kwargs.keys()):
            arg_strings.append(f"{key}={kwargs[key]}")

        # Create combined string and hash
        combined = "|".join(arg_strings)
        cache_key = hashlib.md5(combined.encode('utf-8')).hexdigest()

        self.logger.debug(f"Generated cache key: {cache_key} from {len(args)} args, {len(kwargs)} kwargs")
        return cache_key

    def is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if a cache file exists and is not expired.

        Args:
            cache_path: Path to the cache file

        Returns:
            bool: True if cache is valid (exists and not expired), False otherwise

        Note:
            This method only checks file existence and modification time.
            It does not validate file contents or format.
        """
        if not cache_path.exists():
            self.logger.debug(f"Cache file does not exist: {cache_path}")
            return False

        # Check file modification time
        try:
            file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_hours = (datetime.now() - file_mtime).total_seconds() / 3600

            if age_hours > self.expiry_hours:
                self.logger.debug(
                    f"Cache file expired: {cache_path.name} "
                    f"(age: {age_hours:.1f}h > {self.expiry_hours}h)"
                )
                return False

            self.logger.debug(f"Cache file valid: {cache_path.name} (age: {age_hours:.1f}h)")
            return True

        except (OSError, ValueError) as e:
            self.logger.warning(f"Error checking cache file {cache_path}: {str(e)}")
            return False

    def load(self, cache_key: str, file_format: str = 'pickle') -> Optional[Any]:
        """
        Load data from cache if available and valid.

        Args:
            cache_key: Unique cache key (typically from generate_cache_key)
            file_format: Format to load ('pickle' or 'json')

        Returns:
            Optional[Any]: Cached data if available and valid, None otherwise

        Raises:
            ValueError: If file_format is not supported

        Example:
            >>> cache = CacheManager()
            >>> key = cache.generate_cache_key('my_function', param='value')
            >>> data = cache.load(key, 'json')
            >>> if data is not None:
            ...     print("Cache hit!")
            ... else:
            ...     print("Cache miss")
        """
        if not self.enabled:
            self.logger.debug("Caching disabled, skipping cache load")
            return None

        if file_format not in ['pickle', 'json']:
            raise ValueError(f"Unsupported file format: {file_format}. Use 'pickle' or 'json'.")

        cache_path = self.cache_dir / f"{cache_key}.{file_format}"

        if not self.is_cache_valid(cache_path):
            self.logger.debug(f"Cache miss: {cache_key}")
            return None

        try:
            if file_format == 'pickle':
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:  # json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            self.logger.debug(f"Cache hit: {cache_key} ({file_format})")
            return data

        except Exception as e:
            self.logger.warning(f"Error loading cache {cache_key}: {str(e)}")
            return None

    def save(self, data: Any, cache_key: str, file_format: str = 'pickle') -> None:
        """
        Save data to cache file.

        Args:
            data: Data to cache (must be serializable in chosen format)
            cache_key: Unique cache key (typically from generate_cache_key)
            file_format: Format to save ('pickle' or 'json')

        Raises:
            ValueError: If file_format is not supported or data is not serializable

        Example:
            >>> cache = CacheManager()
            >>> data = {'result': 42, 'timestamp': '2024-01-01'}
            >>> key = cache.generate_cache_key('compute_result', input=10)
            >>> cache.save(data, key, 'json')
        """
        if not self.enabled:
            self.logger.debug("Caching disabled, skipping cache save")
            return

        if file_format not in ['pickle', 'json']:
            raise ValueError(f"Unsupported file format: {file_format}. Use 'pickle' or 'json'.")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self.cache_dir / f"{cache_key}.{file_format}"

        try:
            if file_format == 'pickle':
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:  # json
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

            self.logger.debug(f"Cached data saved: {cache_key} ({file_format})")

        except Exception as e:
            error_msg = f"Error saving cache {cache_key}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cache files based on age criteria.

        Args:
            older_than_hours: If specified, only delete files older than this many hours.
                            If None, delete all cache files.

        Returns:
            int: Number of files deleted

        Example:
            >>> cache = CacheManager()
            >>> # Delete files older than 48 hours
            >>> deleted = cache.clear_cache(older_than_hours=48)
            >>> print(f"Deleted {deleted} old cache files")
            >>>
            >>> # Delete all cache files
            >>> deleted = cache.clear_cache()
            >>> print(f"Deleted {deleted} cache files")
        """
        deleted_count = 0

        if older_than_hours is None:
            # Delete all cache files
            for cache_file in self.cache_dir.glob('*'):
                if cache_file.is_file():
                    try:
                        cache_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error deleting {cache_file}: {str(e)}")

            self.logger.info(f"Cleared all cache files: {deleted_count} deleted")
        else:
            # Delete files older than specified hours
            cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)

            for cache_file in self.cache_dir.glob('*'):
                if cache_file.is_file():
                    try:
                        if cache_file.stat().st_mtime < cutoff_time:
                            cache_file.unlink()
                            deleted_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error checking/deleting {cache_file}: {str(e)}")

            self.logger.info(f"Cleared cache files older than {older_than_hours}h: {deleted_count} deleted")

        return deleted_count

    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get cache directory size and file count statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary with:
                - 'total_size_mb': Total size in megabytes (float)
                - 'file_count': Number of cache files (int)

        Example:
            >>> cache = CacheManager()
            >>> stats = cache.get_cache_size()
            >>> print(f"Cache size: {stats['total_size_mb']:.2f} MB, {stats['file_count']} files")
        """
        total_size = 0
        file_count = 0

        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                try:
                    total_size += cache_file.stat().st_size
                    file_count += 1
                except Exception as e:
                    self.logger.warning(f"Error getting size for {cache_file}: {str(e)}")

        total_size_mb = total_size / (1024 * 1024)  # Convert to MB

        return {
            'total_size_mb': round(total_size_mb, 2),
            'file_count': file_count
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics and information.

        Returns:
            Dict[str, Any]: Comprehensive statistics dictionary with:
                - 'total_files': Total number of cache files
                - 'total_size_mb': Total cache size in MB
                - 'avg_file_age_hours': Average age of cache files in hours
                - 'oldest_file_hours': Age of oldest file in hours
                - 'newest_file_hours': Age of newest file in hours
                - 'expired_files': Number of expired files
                - 'valid_files': Number of valid (non-expired) files

        Example:
            >>> cache = CacheManager()
            >>> stats = cache.get_cache_stats()
            >>> print(f"Cache has {stats['valid_files']} valid files, "
            ...       f"averaging {stats['avg_file_age_hours']:.1f}h old")
        """
        files_info = []

        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                try:
                    mtime = cache_file.stat().st_mtime
                    size = cache_file.stat().st_size
                    age_hours = (datetime.now().timestamp() - mtime) / 3600
                    is_expired = age_hours > self.expiry_hours

                    files_info.append({
                        'path': cache_file,
                        'age_hours': age_hours,
                        'size_bytes': size,
                        'is_expired': is_expired
                    })
                except Exception as e:
                    self.logger.warning(f"Error getting stats for {cache_file}: {str(e)}")

        if not files_info:
            return {
                'total_files': 0,
                'total_size_mb': 0.0,
                'avg_file_age_hours': 0.0,
                'oldest_file_hours': 0.0,
                'newest_file_hours': 0.0,
                'expired_files': 0,
                'valid_files': 0
            }

        # Calculate statistics
        total_files = len(files_info)
        total_size = sum(f['size_bytes'] for f in files_info)
        ages = [f['age_hours'] for f in files_info]
        expired_count = sum(1 for f in files_info if f['is_expired'])
        valid_count = total_files - expired_count

        return {
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'avg_file_age_hours': round(sum(ages) / len(ages), 2),
            'oldest_file_hours': round(max(ages), 2),
            'newest_file_hours': round(min(ages), 2),
            'expired_files': expired_count,
            'valid_files': valid_count
        }

    def __enter__(self):
        """
        Context manager entry.

        Returns:
            CacheManager: Self for use in with statement

        Example:
            >>> with CacheManager() as cache:
            ...     data = cache.load('my_key')
            ...     if data is None:
            ...         data = expensive_computation()
            ...         cache.save(data, 'my_key')
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.

        Currently performs no cleanup operations but can be extended
        for future resource management needs.
        """
        pass
