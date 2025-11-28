"""
Basic integration tests for Macro Event Impact Tracker

These tests verify that core functionality works without making excessive API calls.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from src.utils.helpers import *
from src.utils.cache_manager import CacheManager


class TestConfiguration(unittest.TestCase):
    """Test configuration and setup"""

    def test_config_loads(self):
        """Test that configuration loads successfully"""
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.MACRO_INDICATORS)
        self.assertIsNotNone(config.MARKET_ASSETS)

    def test_directories_exist(self):
        """Test that required directories are created"""
        self.assertTrue(config.CACHE_DIR.exists())
        self.assertTrue(config.EXPORT_DIR.exists())

    def test_indicators_valid(self):
        """Test that all indicators have required fields"""
        for code, details in config.MACRO_INDICATORS.items():
            self.assertIn('fred_code', details)
            self.assertIn('name', details)
            self.assertIn('frequency', details)


class TestHelperFunctions(unittest.TestCase):
    """Test utility helper functions"""

    def test_format_datetime(self):
        """Test datetime formatting"""
        dt = datetime(2024, 1, 15, 10, 30)
        formatted = format_datetime(dt)
        self.assertIsInstance(formatted, str)
        self.assertIn('2024', formatted)

    def test_parse_datetime(self):
        """Test datetime parsing"""
        date_str = "2024-01-15"
        parsed = parse_datetime(date_str)
        self.assertIsInstance(parsed, datetime)
        self.assertEqual(parsed.year, 2024)

    def test_calculate_percentage_change(self):
        """Test percentage change calculation"""
        old_val = 100
        new_val = 110
        change = calculate_percentage_change(old_val, new_val)
        self.assertAlmostEqual(change, 10.0)

    def test_format_percentage(self):
        """Test percentage formatting"""
        val = 0.0234
        formatted = format_percentage(val)
        self.assertEqual(formatted, "+2.34%")


class TestCacheManager(unittest.TestCase):
    """Test caching functionality"""

    def setUp(self):
        """Set up test cache manager"""
        self.cache = CacheManager()

    def test_cache_save_and_load(self):
        """Test saving and loading from cache"""
        test_data = {"test": "data", "value": 123}
        cache_key = "test_key"

        # Save to cache
        self.cache.save(test_data, cache_key, file_format='pickle')

        # Load from cache
        loaded_data = self.cache.load(cache_key, file_format='pickle')

        self.assertEqual(test_data, loaded_data)

    def test_cache_expiry(self):
        """Test that cache respects expiry time"""
        # This is a placeholder - implement with time manipulation
        pass


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""

    def test_validate_ticker(self):
        """Test ticker validation"""
        self.assertTrue(validate_ticker("SPY"))
        self.assertTrue(validate_ticker("AAPL"))
        self.assertFalse(validate_ticker("spy"))  # Should be uppercase
        self.assertFalse(validate_ticker("123"))  # Should start with letter

    def test_validate_date_range(self):
        """Test date range validation"""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        self.assertTrue(validate_date_range(start, end))

        # Invalid: end before start
        with self.assertRaises(ValueError):
            validate_date_range(end, start)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
