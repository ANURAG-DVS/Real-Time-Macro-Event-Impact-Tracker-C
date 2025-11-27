"""
FRED Data Fetcher - Optimized for FREE API Tier

This module provides a comprehensive data fetcher for Federal Reserve Economic Data (FRED)
API, specifically optimized for the FREE tier with intelligent caching, rate limiting,
and robust error handling.

The FREDDataFetcher class handles:
- Economic indicator data fetching
- Intelligent caching to minimize API calls
- Rate limiting to respect FREE tier limits
- Comprehensive error handling and retry logic
- Data enrichment with metadata

All data sources are completely FREE - no paid subscriptions required.
"""

import time
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple

import pandas as pd
import requests
import certifi
from fredapi import Fred

from config.settings import config


class FREDDataFetcher:
    """
    FRED (Federal Reserve Economic Data) API data fetcher optimized for FREE tier.

    This class provides comprehensive economic data fetching with:
    - Intelligent caching to minimize API calls
    - Rate limiting to respect FREE tier limits
    - Robust error handling and retry logic
    - Data enrichment with metadata

    Attributes:
        logger: Logger instance for operation tracking
        fred_client: FRED API client instance
        cache_dir: Directory for cached data storage
        delay: Rate limiting delay between API calls (seconds)
    """

    def __init__(self) -> None:
        """
        Initialize FRED data fetcher with validation and connection testing.

        Raises:
            ValueError: If FRED API key is invalid or connection fails
        """
        self.logger = logging.getLogger('fred_fetcher')

        # Validate API key
        if not config.FRED_API_KEY:
            raise ValueError(
                "FRED API key not configured. Get your FREE API key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        try:
            # Set SSL certificate file to use certifi's bundle
            import os
            os.environ['SSL_CERT_FILE'] = certifi.where()

            # Initialize FRED client
            self.fred_client = Fred(api_key=config.FRED_API_KEY)
            self.cache_dir = config.CACHE_DIR
            self.delay = config.FRED_DELAY_SECONDS

            # Test connection with a simple API call
            self.logger.info("Testing FRED API connection...")
            test_series = self.fred_client.get_series('GDP', start='2023-01-01', end='2023-01-02')
            if test_series is None or len(test_series) == 0:
                raise ValueError("FRED API connection test failed")

            self.logger.info("✅ FRED API connection validated successfully")

        except Exception as e:
            error_msg = f"FRED API initialization failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _rate_limit(self) -> None:
        """
        Apply rate limiting to respect FREE tier limits.

        Sleeps for the configured delay period before API calls.
        """
        self.logger.debug(f"Rate limiting: waiting {self.delay}s to respect free tier")
        time.sleep(self.delay)

    def _get_cache_key(self, series_id: str, start_date: str, end_date: str) -> str:
        """
        Generate unique cache key for data series.

        Args:
            series_id: FRED series identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            str: Unique cache key with .pkl extension
        """
        return f"{series_id}_{start_date}_{end_date}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and not expired.

        Args:
            cache_key: Cache file key

        Returns:
            Optional[pd.DataFrame]: Cached data if available and valid, None otherwise
        """
        cache_path = self.cache_dir / cache_key

        if not cache_path.exists():
            self.logger.debug(f"Cache miss: {cache_key} not found")
            return None

        # Check cache expiry
        cache_age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
        if cache_age_hours > config.CACHE_EXPIRY_HOURS:
            self.logger.debug(f"Cache expired: {cache_key} ({cache_age_hours:.1f}h old)")
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.debug(f"Cache hit: {cache_key}")
            return data
        except Exception as e:
            self.logger.warning(f"Cache load error for {cache_key}: {str(e)}")
            return None

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """
        Save data to cache file.

        Args:
            data: DataFrame to cache
            cache_key: Cache file key
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / cache_key

            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

            self.logger.debug(f"Cached data saved: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Cache save error for {cache_key}: {str(e)}")

    def get_series_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch time series data from FRED API with caching.

        Args:
            series_id: FRED series identifier (e.g., 'GDP', 'CPIAUCSL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'value', 'series_id']

        Raises:
            ValueError: If series_id is invalid or API call fails
            ConnectionError: If network/API issues occur

        Example:
            >>> fetcher = FREDDataFetcher()
            >>> data = fetcher.get_series_data('GDP', '2020-01-01', '2023-12-31')
            >>> print(data.head())
                     date        value series_id
            0  2020-01-01  21538.032      GDP
            1  2020-04-01  19636.731      GDP
        """
        cache_key = self._get_cache_key(series_id, start_date, end_date)

        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        # Cache miss - fetch from API
        self.logger.info(f"Fetching FRED data: {series_id} ({start_date} to {end_date})")

        max_retries = config.MAX_RETRIES
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Fetch data from FRED
                series_data = self.fred_client.get_series(series_id, start=start_date, end=end_date)

                if series_data is None or len(series_data) == 0:
                    raise ValueError(f"No data available for series {series_id}")

                # Convert to DataFrame
                df = pd.DataFrame({
                    'date': series_data.index,
                    'value': series_data.values,
                    'series_id': series_id
                })

                # Remove any NaN values
                df = df.dropna()

                # Cache the result
                self._save_to_cache(df, cache_key)

                self.logger.info(f"✅ Fetched {len(df)} data points for {series_id}")
                return df

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"FRED API attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"FRED API failed after {max_retries + 1} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    raise ConnectionError(error_msg)

    def get_indicator_data(self, indicator_code: str, lookback_months: int = 12) -> pd.DataFrame:
        """
        Fetch economic indicator data with metadata enrichment.

        Args:
            indicator_code: Indicator code from config.MACRO_INDICATORS (e.g., 'CPI', 'GDP')
            lookback_months: Number of months of historical data to fetch

        Returns:
            pd.DataFrame: Enriched DataFrame with indicator metadata

        Raises:
            ValueError: If indicator_code is not found in configuration

        Example:
            >>> fetcher = FREDDataFetcher()
            >>> cpi_data = fetcher.get_indicator_data('CPI', lookback_months=24)
            >>> print(cpi_data.head())
                     date     value series_id indicator_code                  name
            0  2022-01-01  281.148  CPIAUCSL            CPI  Consumer Price Index
        """
        if indicator_code not in config.MACRO_INDICATORS:
            available_indicators = list(config.MACRO_INDICATORS.keys())
            raise ValueError(
                f"Unknown indicator '{indicator_code}'. Available indicators: {available_indicators}"
            )

        indicator_info = config.MACRO_INDICATORS[indicator_code]
        series_id = indicator_info['fred_code']

        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_months * 30)).strftime('%Y-%m-%d')

        # Fetch data
        df = self.get_series_data(series_id, start_date, end_date)

        # Add indicator metadata
        df['indicator_code'] = indicator_code
        df['name'] = indicator_info['name']
        df['frequency'] = indicator_info['frequency']
        df['description'] = indicator_info['description']

        self.logger.info(f"✅ Fetched {len(df)} data points for {indicator_code} ({indicator_info['name']})")
        return df

    def get_latest_release(self, indicator_code: str) -> Dict:
        """
        Get the most recent data release for an economic indicator.

        Args:
            indicator_code: Indicator code from config.MACRO_INDICATORS

        Returns:
            Dict: Latest release information with keys:
                - 'indicator': indicator code
                - 'date': release datetime
                - 'value': indicator value
                - 'series_id': FRED series ID
                - 'name': indicator name

        Raises:
            ValueError: If no data available for indicator

        Example:
            >>> fetcher = FREDDataFetcher()
            >>> latest = fetcher.get_latest_release('CPI')
            >>> print(latest)
            {
                'indicator': 'CPI',
                'date': Timestamp('2024-01-01'),
                'value': 308.417,
                'series_id': 'CPIAUCSL',
                'name': 'Consumer Price Index'
            }
        """
        # Get recent data (last 3 months to ensure we get latest)
        df = self.get_indicator_data(indicator_code, lookback_months=3)

        if df.empty:
            raise ValueError(f"No data available for indicator {indicator_code}")

        # Get the most recent data point
        latest_row = df.iloc[-1]

        return {
            'indicator': indicator_code,
            'date': latest_row['date'],
            'value': latest_row['value'],
            'series_id': latest_row['series_id'],
            'name': latest_row['name']
        }

    def get_release_dates(self, indicator_code: str, lookback_months: int = 12) -> List[datetime]:
        """
        Get all release dates for an indicator within the lookback period.

        Args:
            indicator_code: Indicator code from config.MACRO_INDICATORS
            lookback_months: Number of months to look back

        Returns:
            List[datetime]: Sorted list of release dates

        Example:
            >>> fetcher = FREDDataFetcher()
            >>> dates = fetcher.get_release_dates('CPI', lookback_months=6)
            >>> print(dates[:3])
            [Timestamp('2023-10-01'), Timestamp('2023-11-01'), Timestamp('2023-12-01')]
        """
        df = self.get_indicator_data(indicator_code, lookback_months=lookback_months)

        # Extract unique dates and sort
        release_dates = sorted(df['date'].unique())

        self.logger.info(f"Found {len(release_dates)} release dates for {indicator_code}")
        return release_dates

    def calculate_change(self, indicator_code: str) -> Dict:
        """
        Calculate the change between the two most recent data points.

        Args:
            indicator_code: Indicator code from config.MACRO_INDICATORS

        Returns:
            Dict: Change calculations with keys:
                - 'absolute_change': absolute difference (new - old)
                - 'percent_change': percentage change
                - 'previous_value': older value
                - 'current_value': newer value

        Raises:
            ValueError: If insufficient data points available

        Example:
            >>> fetcher = FREDDataFetcher()
            >>> change = fetcher.calculate_change('CPI')
            >>> print(change)
            {
                'absolute_change': 0.3,
                'percent_change': 0.097,
                'previous_value': 307.671,
                'current_value': 307.971
            }
        """
        # Get recent data (enough for 2+ data points)
        df = self.get_indicator_data(indicator_code, lookback_months=6)

        if len(df) < 2:
            raise ValueError(f"Insufficient data points for {indicator_code} to calculate change")

        # Get the two most recent values
        recent_data = df.tail(2)
        previous_value = recent_data.iloc[0]['value']
        current_value = recent_data.iloc[1]['value']

        # Calculate changes
        absolute_change = current_value - previous_value
        percent_change = (absolute_change / previous_value) * 100 if previous_value != 0 else 0

        return {
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            'previous_value': previous_value,
            'current_value': current_value
        }

    def estimate_release_time(self, release_date: datetime, indicator_code: str) -> datetime:
        """
        Estimate the release time for an economic indicator.

        Args:
            release_date: Date of the data release (cannot be None)
            indicator_code: Indicator code from config.MACRO_INDICATORS

        Returns:
            datetime: Estimated release datetime with time component

        Raises:
            ValueError: If release_date is None or invalid

        Example:
            >>> from datetime import datetime
            >>> fetcher = FREDDataFetcher()
            >>> release_dt = fetcher.estimate_release_time(datetime(2024, 1, 15), 'CPI')
            >>> print(release_dt)
            2024-01-15 08:30:00-05:00
        """
        if release_date is None:
            raise ValueError(f"release_date cannot be None for {indicator_code}")

        if not isinstance(release_date, datetime):
            raise ValueError(f"release_date must be a datetime object for {indicator_code}, got {type(release_date)}")

        indicator_info = config.MACRO_INDICATORS[indicator_code]
        typical_time = indicator_info.get('typical_release_time', '08:30')

        # Parse time
        time_parts = typical_time.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1]) if len(time_parts) > 1 else 0

        # Combine date with time
        release_datetime = release_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        self.logger.debug(f"Estimated release time for {indicator_code}: {release_datetime}")
        return release_datetime
