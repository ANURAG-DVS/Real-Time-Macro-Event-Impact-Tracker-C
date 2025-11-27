"""
Configuration settings for Real-Time Macro Event Impact Tracker.

This module provides centralized configuration for FREE data sources including:
- FRED (Federal Reserve Economic Data) API for economic indicators
- Yahoo Finance API for market data and asset prices

All data sources are completely FREE and require no paid subscriptions.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class for the Real-Time Macro Event Impact Tracker.

    This class centralizes all configuration settings for FREE data sources,
    directory management, caching, and logging. All APIs used are free tier.

    Attributes:
        API Configuration:
            FRED_API_KEY: API key for Federal Reserve Economic Data (FREE)

        Directory Configuration:
            BASE_DIR: Root directory of the project
            DATA_DIR: Directory for data storage
            CACHE_DIR: Directory for cached data
            EXPORT_DIR: Directory for exported results
            LOGS_DIR: Directory for log files

        Cache & Performance Settings:
            CACHE_EXPIRY_HOURS: Hours before cache expires
            ENABLE_CACHE: Whether to use caching
            DEFAULT_TIMEZONE: Default timezone for data

        Rate Limiting (respectful API usage):
            YFINANCE_DELAY_SECONDS: Delay between Yahoo Finance requests
            FRED_DELAY_SECONDS: Delay between FRED API requests
            MAX_RETRIES: Maximum retry attempts for failed requests
    """

    # API Configuration (FREE)
    FRED_API_KEY: str = os.getenv('FRED_API_KEY', '')

    # Directory Configuration
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    EXPORT_DIR: Path = DATA_DIR / "exports"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Timezone
    DEFAULT_TIMEZONE: str = "America/New_York"

    # Cache Settings
    CACHE_EXPIRY_HOURS: int = 24
    ENABLE_CACHE: bool = True

    # Rate Limiting (to be respectful to free APIs)
    YFINANCE_DELAY_SECONDS: float = 1.0
    FRED_DELAY_SECONDS: float = 0.5
    MAX_RETRIES: int = 3

    # Macro Economic Indicators (FRED API - FREE)
    MACRO_INDICATORS: Dict[str, Dict[str, Any]] = {
        'CPI': {
            'fred_code': 'CPIAUCSL',
            'name': 'Consumer Price Index',
            'frequency': 'monthly',
            'typical_release_time': '08:30',
            'description': 'Measures change in prices paid by consumers'
        },
        'CORE_CPI': {
            'fred_code': 'CPILFESL',
            'name': 'Core CPI (ex Food & Energy)',
            'frequency': 'monthly',
            'typical_release_time': '08:30',
            'description': 'CPI excluding volatile food and energy'
        },
        'NFP': {
            'fred_code': 'PAYEMS',
            'name': 'Non-Farm Payrolls',
            'frequency': 'monthly',
            'typical_release_time': '08:30',
            'description': 'Change in number of employed people'
        },
        'UNEMPLOYMENT': {
            'fred_code': 'UNRATE',
            'name': 'Unemployment Rate',
            'frequency': 'monthly',
            'typical_release_time': '08:30',
            'description': 'Percentage of labor force unemployed'
        },
        'GDP': {
            'fred_code': 'GDP',
            'name': 'Gross Domestic Product',
            'frequency': 'quarterly',
            'typical_release_time': '08:30',
            'description': 'Total value of goods and services produced'
        },
        'PCE': {
            'fred_code': 'PCEPI',
            'name': 'Personal Consumption Expenditures',
            'frequency': 'monthly',
            'typical_release_time': '08:30',
            'description': 'Fed\'s preferred inflation measure'
        },
        'RETAIL_SALES': {
            'fred_code': 'RSXFS',
            'name': 'Retail Sales',
            'frequency': 'monthly',
            'typical_release_time': '08:30',
            'description': 'Total receipts of retail stores'
        }
    }

    # Market Assets (Yahoo Finance - FREE)
    MARKET_ASSETS: Dict[str, Dict[str, str]] = {
        'SPY': {
            'name': 'S&P 500 ETF',
            'type': 'equity',
            'sector': 'broad_market',
            'description': 'Tracks S&P 500 index'
        },
        'QQQ': {
            'name': 'Nasdaq 100 ETF',
            'type': 'equity',
            'sector': 'technology',
            'description': 'Tracks tech-heavy Nasdaq 100'
        },
        'DIA': {
            'name': 'Dow Jones ETF',
            'type': 'equity',
            'sector': 'broad_market',
            'description': 'Tracks Dow Jones Industrial Average'
        },
        'IWM': {
            'name': 'Russell 2000 ETF',
            'type': 'equity',
            'sector': 'small_cap',
            'description': 'Tracks small-cap stocks'
        },
        'TLT': {
            'name': '20+ Year Treasury Bond ETF',
            'type': 'rates',
            'sector': 'bonds',
            'description': 'Long-term government bonds'
        },
        'IEF': {
            'name': '7-10 Year Treasury ETF',
            'type': 'rates',
            'sector': 'bonds',
            'description': 'Intermediate-term government bonds'
        },
        'SHY': {
            'name': '1-3 Year Treasury ETF',
            'type': 'rates',
            'sector': 'bonds',
            'description': 'Short-term government bonds'
        },
        'GLD': {
            'name': 'Gold ETF',
            'type': 'commodity',
            'sector': 'precious_metals',
            'description': 'Tracks gold prices'
        },
        'UUP': {
            'name': 'US Dollar Index ETF',
            'type': 'fx',
            'sector': 'currency',
            'description': 'Tracks US dollar vs basket of currencies'
        },
        'VXX': {
            'name': 'VIX Short-Term Futures ETF',
            'type': 'volatility',
            'sector': 'vol',
            'description': 'Tracks expected market volatility'
        }
    }

    # Analysis Time Windows
    ANALYSIS_WINDOWS: List[Dict[str, Any]] = [
        {'label': '15 Minutes', 'minutes': 15},
        {'label': '30 Minutes', 'minutes': 30},
        {'label': '1 Hour', 'minutes': 60},
        {'label': '2 Hours', 'minutes': 120},
        {'label': '4 Hours', 'minutes': 240},
        {'label': '1 Day', 'minutes': 390}
    ]

    def validate_config(self) -> bool:
        """
        Validate configuration settings and set up required directories and logging.

        This method performs the following validations:
        1. Checks if FRED_API_KEY is set and not empty
        2. Creates all required directories if they don't exist
        3. Sets up logging configuration

        Returns:
            bool: True if all validation passes

        Raises:
            ValueError: If FRED_API_KEY is missing or empty
        """
        # Validate FRED API Key
        if not self.FRED_API_KEY or self.FRED_API_KEY.strip() == '':
            raise ValueError(
                "FRED_API_KEY not found. Get your FREE API key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        # Create required directories
        directories = [self.DATA_DIR, self.CACHE_DIR, self.EXPORT_DIR, self.LOGS_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.setup_logging()

        return True

    def get_cache_path(self, filename: str) -> Path:
        """
        Get the full path for a cache file.

        Args:
            filename: Name of the cache file

        Returns:
            Path: Full path to the cache file
        """
        return self.CACHE_DIR / filename

    def get_export_path(self, filename: str) -> Path:
        """
        Get the full path for an export file.

        Args:
            filename: Name of the export file

        Returns:
            Path: Full path to the export file
        """
        return self.EXPORT_DIR / filename

    def get_log_path(self, filename: str) -> Path:
        """
        Get the full path for a log file.

        Args:
            filename: Name of the log file

        Returns:
            Path: Full path to the log file
        """
        return self.LOGS_DIR / filename

    def setup_logging(self) -> None:
        """
        Set up logging configuration for both console and file output.

        Configures logging to:
        - Console: INFO level and above
        - File: DEBUG level and above with rotation
        - Format: Timestamp - Logger Name - Level - Message
        """
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler (INFO level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation (DEBUG level)
        log_file_path = self.get_log_path('app.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


# Create global configuration instance
config = Config()
config.validate_config()
