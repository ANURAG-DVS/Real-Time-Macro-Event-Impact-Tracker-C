"""
Utilities Package - Helper Functions and Common Operations

This package contains utility functions and helper classes used throughout
the Macro Event Impact Tracker. These utilities provide common functionality
for caching, data manipulation, formatting, and system operations.

Key Components:

1. Cache Manager (cache_manager.py)
   - Intelligent caching system for API responses
   - Automatic cache expiry and cleanup
   - File-based persistence with pickle serialization
   - Cache hit/miss statistics and monitoring

2. Helpers (helpers.py)
   - Data formatting and validation functions
   - Date/time manipulation utilities
   - Percentage calculations and formatting
   - Ticker symbol validation
   - DataFrame cleaning and normalization

Features:
- Efficient caching to minimize API calls and improve performance
- Consistent data formatting across the application
- Robust error handling and validation
- Type-safe operations with proper error messages
- Memory-efficient data structures and processing

Classes:
    CacheManager: File-based caching with automatic expiry

Functions:
    format_datetime: Convert datetime objects to readable strings
    parse_datetime: Parse string dates into datetime objects
    calculate_percentage_change: Calculate percent change between values
    format_percentage: Format decimal as percentage string
    validate_ticker: Check if ticker symbol is valid
    validate_date_range: Ensure date ranges are logical

Usage:
    from src.utils.cache_manager import CacheManager
    from src.utils.helpers import format_percentage, calculate_percentage_change

    # Caching
    cache = CacheManager()
    cache.save(data, 'cache_key')

    # Formatting
    pct_change = calculate_percentage_change(100, 110)  # Returns 10.0
    formatted = format_percentage(pct_change)  # Returns "+10.00%"
"""
