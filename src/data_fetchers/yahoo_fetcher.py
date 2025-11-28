"""
Yahoo Finance Data Fetcher - 100% FREE Market Data

This module provides comprehensive market data fetching from Yahoo Finance,
specifically optimized for the FREE tier with no API key required.

The YahooDataFetcher class handles:
- Intraday and daily market data fetching
- Intelligent caching to minimize API calls
- Rate limiting to respect FREE tier limits
- Market hours validation and adjustment
- Return calculations and volatility analysis
- Robust error handling and retry logic

FREE tier limitations (acceptable trade-offs for zero cost):
- Intraday data: Only last 7 days available
- Rate limiting: 1-2 requests per second recommended
- Occasional data gaps: Handled with forward-fill
- No pre/post market data on free tier

All market data is completely FREE - no paid subscriptions required.
"""

import time
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import pytz

from config.settings import config


class YahooDataFetcher:
    """
    Yahoo Finance API data fetcher optimized for FREE tier - 100% FREE, no API key required!

    This class provides comprehensive market data fetching with:
    - Intelligent caching to minimize API calls
    - Rate limiting to respect FREE tier limits
    - Robust error handling and retry logic
    - Market hours validation and time adjustment
    - Return calculations and volatility analysis

    FREE tier limitations (acceptable trade-offs for zero cost):
    - Intraday data: Only last 7 days available
    - Rate limiting: 1-2 requests per second recommended
    - Occasional data gaps: Handled with forward-fill
    - No pre/post market data on free tier

    Attributes:
        logger: Logger instance for operation tracking
        cache_dir: Directory for cached data storage
        timezone: Eastern Time timezone for market hours
        delay: Rate limiting delay between API calls (seconds)
    """

    def __init__(self) -> None:
        """
        Initialize Yahoo Finance data fetcher with configuration.

        Sets up logging, caching directory, timezone, and rate limiting
        parameters from the centralized configuration.
        """
        self.logger = logging.getLogger('yahoo_fetcher')
        self.cache_dir = config.CACHE_DIR
        self.timezone = pytz.timezone(config.DEFAULT_TIMEZONE)
        self.delay = config.YFINANCE_DELAY_SECONDS

        self.logger.info("✅ Yahoo Finance data fetcher initialized (100% FREE, no API key required!)")

    def _rate_limit(self) -> None:
        """
        Apply rate limiting to respect FREE tier limits.

        Sleeps for the configured delay period before API calls to be
        respectful to Yahoo Finance's free service.
        """
        self.logger.debug(f"Rate limiting: waiting {self.delay}s (being respectful to free API)")
        time.sleep(self.delay)

    def _get_cache_key(self, ticker: str, start: str, end: str, interval: str) -> str:
        """
        Generate unique cache key for market data.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            interval: Data interval ('1m', '1d', etc.)

        Returns:
            str: Unique cache key with .pkl extension
        """
        return f"{ticker}_{interval}_{start}_{end}.pkl"

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

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize DataFrame.

        Performs the following cleaning operations:
        - Remove rows with all NaN values
        - Forward fill NaN in price columns
        - Ensure datetime index is timezone-aware
        - Sort by timestamp

        Args:
            df: Raw DataFrame from Yahoo Finance

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df

        # Remove rows with all NaN values
        df = df.dropna(how='all')

        # Forward fill NaN values in price columns
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        existing_price_cols = [col for col in price_columns if col in df.columns]
        if existing_price_cols:
            df[existing_price_cols] = df[existing_price_cols].fillna(method='ffill')

        # Ensure datetime index is timezone-aware
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone)
            elif df.index.tz != self.timezone:
                df.index = df.index.tz_convert(self.timezone)

        # Sort by timestamp
        df = df.sort_index()

        return df

    def is_market_hours(self, check_time: datetime) -> bool:
        """
        Check if the given time falls within regular market hours.

        Market hours are Monday-Friday, 9:30 AM - 4:00 PM Eastern Time.

        Args:
            check_time: Datetime to check

        Returns:
            bool: True if during market hours, False otherwise

        Example:
            >>> from datetime import datetime
            >>> fetcher = YahooDataFetcher()
            >>> fetcher.is_market_hours(datetime(2024, 1, 15, 10, 0))
            True
        """
        # Convert to Eastern Time if not already
        if check_time.tzinfo is None:
            check_time = self.timezone.localize(check_time)
        else:
            check_time = check_time.astimezone(self.timezone)

        # Check if weekday (Monday=0 to Friday=4)
        if check_time.weekday() > 4:
            return False

        # Check time range: 9:30 AM to 4:00 PM
        market_open = check_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = check_time.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= check_time <= market_close

    def get_next_market_open(self, from_time: datetime) -> datetime:
        """
        Find the next market opening time from the given time.

        Args:
            from_time: Starting datetime

        Returns:
            datetime: Next market open datetime (9:30 AM ET)

        Example:
            >>> from datetime import datetime
            >>> fetcher = YahooDataFetcher()
            >>> next_open = fetcher.get_next_market_open(datetime(2024, 1, 15, 16, 30))
            >>> print(next_open)
            2024-01-16 09:30:00-05:00
        """
        # Convert to Eastern Time
        if from_time.tzinfo is None:
            current_time = self.timezone.localize(from_time)
        else:
            current_time = from_time.astimezone(self.timezone)

        # If currently during market hours, return current time
        if self.is_market_hours(current_time):
            return current_time

        # If before 9:30 AM on a weekday, return today at 9:30 AM
        if current_time.weekday() <= 4:  # Monday-Friday
            market_open_today = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            if current_time < market_open_today:
                return market_open_today

        # Find next weekday
        days_ahead = 1
        if current_time.weekday() == 4:  # Friday
            days_ahead = 3  # Skip weekend
        elif current_time.weekday() == 5:  # Saturday
            days_ahead = 2  # Next Monday
        elif current_time.weekday() == 6:  # Sunday
            days_ahead = 1  # Next Monday

        next_market_day = current_time + timedelta(days=days_ahead)
        return next_market_day.replace(hour=9, minute=30, second=0, microsecond=0)

    def adjust_to_market_hours(self, event_time: datetime) -> Tuple[datetime, str]:
        """
        Adjust event time to nearest valid market hours.

        Args:
            event_time: Original event datetime

        Returns:
            Tuple[datetime, str]: (adjusted_datetime, reason_string)

        Reasons:
            - "adjusted_to_market_open": Moved to next market open
            - "weekend_adjusted": Moved from weekend to Monday
            - "no_adjustment": Already within market hours

        Example:
            >>> from datetime import datetime
            >>> fetcher = YahooDataFetcher()
            >>> adjusted, reason = fetcher.adjust_to_market_hours(datetime(2024, 1, 13, 18, 0))
            >>> print(f"Adjusted to: {adjusted}, Reason: {reason}")
            Adjusted to: 2024-01-15 09:30:00-05:00, Reason: weekend_adjusted
        """
        if self.is_market_hours(event_time):
            return event_time, "no_adjustment"

        adjusted_time = self.get_next_market_open(event_time)

        if event_time.weekday() > 4:  # Weekend
            reason = "weekend_adjusted"
        else:
            reason = "adjusted_to_market_open"

        self.logger.info(f"Event time adjusted: {event_time} -> {adjusted_time} ({reason})")
        return adjusted_time, reason

    def get_intraday_data(self, ticker: str, event_datetime: datetime,
                         hours_before: int = 1, hours_after: int = 4) -> pd.DataFrame:
        """
        Fetch intraday market data around an event time.

        IMPORTANT: Yahoo Finance FREE tier only provides last 7 days of intraday data.
        If event_datetime is older than 7 days, will log warning and return empty DataFrame.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'SPY')
            event_datetime: Event datetime (will be adjusted to market hours if needed)
            hours_before: Hours of data before event (default: 1)
            hours_after: Hours of data after event (default: 4)

        Returns:
            pd.DataFrame: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']

        Raises:
            ValueError: If ticker is invalid or no data available
            ConnectionError: If network/API issues occur

        Example:
            >>> from datetime import datetime
            >>> fetcher = YahooDataFetcher()
            >>> data = fetcher.get_intraday_data('AAPL', datetime(2024, 1, 15, 10, 0), 2, 6)
            >>> print(data.head())
                           timestamp    open    high     low   close  volume ticker
            0 2024-01-15 08:00:00-05:00  185.92  186.27  185.92  186.15   12345  AAPL
        """
        # Check if event is within last 7 days (FREE tier limitation)
        days_since_event = (datetime.now() - event_datetime.replace(tzinfo=None)).days
        if days_since_event > 7:
            self.logger.warning(
                f"Event datetime {event_datetime} is {days_since_event} days old. "
                "Yahoo Finance FREE tier only provides 7 days of intraday data. "
                "Use get_daily_data() for historical analysis."
            )
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker'])

        # Adjust event time to market hours
        adjusted_event, adjustment_reason = self.adjust_to_market_hours(event_datetime)
        if adjustment_reason != "no_adjustment":
            self.logger.info(f"Event time adjusted for {ticker}: {adjustment_reason}")

        # Calculate time window
        start_time = adjusted_event - timedelta(hours=hours_before)
        end_time = adjusted_event + timedelta(hours=hours_after)

        # Convert to string format for cache key
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

        cache_key = self._get_cache_key(ticker, start_str, end_str, '1m')

        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        # Cache miss - fetch from Yahoo Finance
        self.logger.info(f"Fetching intraday data: {ticker} around {adjusted_event}")

        max_retries = config.MAX_RETRIES
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Fetch data from Yahoo Finance
                df = yf.download(
                    ticker,
                    start=start_time.strftime('%Y-%m-%d'),
                    end=(end_time + timedelta(days=1)).strftime('%Y-%m-%d'),  # Add buffer
                    interval='1m',
                    progress=False
                )

                if df.empty:
                    if attempt == max_retries:
                        raise ValueError(f"No intraday data available for {ticker} around {adjusted_event}")
                    continue

                # Clean and standardize data
                df = self._clean_dataframe(df)

                # Filter to exact time window
                df = df[(df.index >= start_time) & (df.index <= end_time)]

                if df.empty:
                    if attempt == max_retries:
                        raise ValueError(f"No data found in time window for {ticker}")
                    continue

                # Reset index and add timestamp column
                df = df.reset_index()
                df.rename(columns={'Datetime': 'timestamp', 'index': 'timestamp'}, inplace=True)

                # Rename columns to lowercase
                df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                }, inplace=True)

                # Add ticker column
                df['ticker'] = ticker

                # Ensure consistent column order
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']
                df = df[columns]

                # Cache the result
                self._save_to_cache(df, cache_key)

                self.logger.info(f"✅ Fetched {len(df)} intraday data points for {ticker}")
                return df

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Yahoo Finance attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Yahoo Finance failed after {max_retries + 1} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    if "No data found" in str(e):
                        raise ValueError(f"Invalid ticker symbol '{ticker}' or no data available")
                    else:
                        raise ConnectionError(error_msg)

    def get_daily_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch daily market data for longer time periods.

        This is the fallback method when intraday data is not available (events > 7 days old).

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data

        Returns:
            pd.DataFrame: DataFrame with same structure as intraday data

        Example:
            >>> from datetime import datetime
            >>> fetcher = YahooDataFetcher()
            >>> data = fetcher.get_daily_data('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))
            >>> print(data.head())
                           timestamp    open    high     low   close    volume ticker
            0 2023-01-03 09:30:00-05:00  130.28  130.90  124.17  125.07  112117500  AAPL
        """
        # Convert dates to string format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        cache_key = self._get_cache_key(ticker, start_str, end_str, '1d')

        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        self.logger.info(f"Fetching daily data: {ticker} from {start_str} to {end_str}")

        max_retries = config.MAX_RETRIES
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Fetch daily data
                df = yf.download(ticker, start=start_str, end=end_str, interval='1d', progress=False)

                if df.empty:
                    if attempt == max_retries:
                        raise ValueError(f"No daily data available for {ticker} in date range")
                    continue

                # Clean and standardize data
                df = self._clean_dataframe(df)

                # Reset index and add timestamp column
                df = df.reset_index()
                df.rename(columns={'Date': 'timestamp', 'index': 'timestamp'}, inplace=True)

                # Rename columns to lowercase
                df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                }, inplace=True)

                # Add ticker column
                df['ticker'] = ticker

                # Ensure consistent column order
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']
                df = df[columns]

                # Cache the result
                self._save_to_cache(df, cache_key)

                self.logger.info(f"✅ Fetched {len(df)} daily data points for {ticker}")
                return df

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Daily data attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Daily data fetch failed after {max_retries + 1} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    if "No data found" in str(e):
                        raise ValueError(f"Invalid ticker symbol '{ticker}' or no data available")
                    else:
                        raise ConnectionError(error_msg)

    def get_multi_asset_data(self, tickers: List[str], event_datetime: datetime,
                           hours_before: int = 1, hours_after: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for multiple tickers around an event time.

        Args:
            tickers: List of ticker symbols
            event_datetime: Event datetime
            hours_before: Hours before event
            hours_after: Hours after event

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping ticker to DataFrame

        Raises:
            Exception: If all tickers fail to fetch

        Example:
            >>> fetcher = YahooDataFetcher()
            >>> data = fetcher.get_multi_asset_data(['AAPL', 'MSFT', 'SPY'], datetime(2024, 1, 15, 10, 0))
            >>> print(f"Fetched data for {len(data)} tickers")
            Fetched data for 3 tickers
        """
        results = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers):
            try:
                self.logger.info(f"Fetching {i+1}/{len(tickers)}: {ticker}")
                df = self.get_intraday_data(ticker, event_datetime, hours_before, hours_after)
                if not df.empty:
                    results[ticker] = df
                else:
                    failed_tickers.append(ticker)
                    self.logger.warning(f"No data available for {ticker}")
            except Exception as e:
                failed_tickers.append(ticker)
                self.logger.error(f"Failed to fetch {ticker}: {str(e)}")
                continue

        if not results:
            raise Exception(
                f"Failed to fetch data for any of the requested tickers: {tickers}. "
                "Check ticker symbols and ensure event_datetime is within last 7 days for intraday data."
            )

        if failed_tickers:
            self.logger.warning(f"Failed to fetch data for tickers: {failed_tickers}")

        self.logger.info(f"✅ Successfully fetched data for {len(results)}/{len(tickers)} tickers")
        return results

    def calculate_returns(self, price_data: pd.DataFrame, event_time: datetime) -> Dict[str, float]:
        """
        Calculate price returns over different time windows from event time.

        Args:
            price_data: DataFrame with price data (from get_intraday_data or get_daily_data)
            event_time: Event datetime to calculate returns from

        Returns:
            Dict[str, float]: Returns for each time window in config.ANALYSIS_WINDOWS

        Example:
            >>> fetcher = YahooDataFetcher()
            >>> data = fetcher.get_intraday_data('AAPL', datetime(2024, 1, 15, 10, 0))
            >>> returns = fetcher.calculate_returns(data, datetime(2024, 1, 15, 10, 0))
            >>> print(returns)
            {'15 Minutes': 1.23, '30 Minutes': 2.45, '1 Hour': 3.67, ...}
        """
        if price_data.empty:
            self.logger.warning("Empty price data provided to calculate_returns")
            return {window['label']: 0.0 for window in config.ANALYSIS_WINDOWS}

        # Ensure event_time is timezone-aware
        if event_time.tzinfo is None:
            event_time = self.timezone.localize(event_time)
        elif event_time.tzinfo != self.timezone:
            event_time = event_time.astimezone(self.timezone)

        # Find closest price timestamp to event time
        price_data = price_data.copy()
        price_data['time_diff'] = abs(price_data['timestamp'] - event_time)
        event_row = price_data.loc[price_data['time_diff'].idxmin()]

        # Extract the event price - use .iloc[0] to get scalar value instead of Series
        event_price = event_row['close'].iloc[0] if hasattr(event_row['close'], 'iloc') else event_row['close']

        # Handle case where event price is missing or invalid
        if pd.isna(event_price):
            self.logger.warning(f"No valid price found at event time {event_time}")
            return {window['label']: 0.0 for window in config.ANALYSIS_WINDOWS}

        returns = {}

        # Calculate returns for each configured time window (15min, 30min, 1hr, etc.)
        for window in config.ANALYSIS_WINDOWS:
            window_minutes = window['minutes']
            # Calculate the future timestamp for this time window
            window_time = event_time + timedelta(minutes=window_minutes)

            # Find the price data point closest to the target window time
            # This handles irregular trading intervals and market gaps
            price_data['window_diff'] = abs(price_data['timestamp'] - window_time)
            window_row = price_data.loc[price_data['window_diff'].idxmin()]

            # Extract window price safely
            window_price = window_row['close'].iloc[0] if hasattr(window_row['close'], 'iloc') else window_row['close']

            # Calculate percentage return: ((future_price - event_price) / event_price) * 100
            if pd.isna(window_price):
                self.logger.debug(f"No valid price found at {window_minutes} minutes window")
                returns[window['label']] = 0.0
            else:
                pct_return = ((window_price - event_price) / event_price) * 100
                returns[window['label']] = round(pct_return, 2)

        self.logger.debug(f"Calculated returns for {len(returns)} time windows")
        return returns

    def get_volatility(self, price_data: pd.DataFrame, window_minutes: int = 30) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            price_data: DataFrame with price data
            window_minutes: Rolling window size in minutes

        Returns:
            pd.Series: Volatility values (annualized)

        Raises:
            ValueError: If insufficient data for volatility calculation

        Example:
            >>> fetcher = YahooDataFetcher()
            >>> data = fetcher.get_intraday_data('SPY', datetime(2024, 1, 15, 10, 0))
            >>> vol = fetcher.get_volatility(data, window_minutes=60)
            >>> print(f"Average volatility: {vol.mean():.2f}%")
            Average volatility: 12.34%
        """
        if price_data.empty or len(price_data) < window_minutes:
            raise ValueError(f"Insufficient data for volatility calculation. Need at least {window_minutes} data points.")

        # Calculate minute-by-minute returns
        returns = price_data['close'].pct_change().dropna()

        if len(returns) < window_minutes:
            raise ValueError(f"Insufficient return data for {window_minutes}-minute volatility window")

        # Calculate rolling standard deviation
        rolling_std = returns.rolling(window=window_minutes).std()

        # Annualize volatility (252 trading days * 390 minutes per day)
        annualized_vol = rolling_std * np.sqrt(252 * 390) * 100  # Convert to percentage

        self.logger.debug(f"Calculated volatility with {window_minutes}-minute window")
        return annualized_vol.dropna()

    def get_current_price(self, ticker: str) -> Dict:
        """
        Get the most recent price information for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict: Current price information with keys:
                - 'ticker': ticker symbol
                - 'current_price': current/last price
                - 'previous_close': previous day's close
                - 'open': today's open price
                - 'day_high': today's high
                - 'day_low': today's low
                - 'volume': today's volume
                - 'timestamp': data timestamp

        Raises:
            ValueError: If ticker is invalid
            ConnectionError: If API issues occur

        Example:
            >>> fetcher = YahooDataFetcher()
            >>> price = fetcher.get_current_price('AAPL')
            >>> print(f"AAPL: ${price['current_price']:.2f}")
            AAPL: $185.42
        """
        self.logger.info(f"Fetching current price for {ticker}")

        max_retries = config.MAX_RETRIES
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Get ticker info
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info

                if not info:
                    if attempt == max_retries:
                        raise ValueError(f"Unable to fetch data for {ticker}. Invalid ticker symbol?")
                    continue

                # Extract price data - Yahoo Finance uses different keys during/after market hours
                # During market hours: 'regularMarketPrice'
                # After market hours: 'currentPrice' or 'previousClose'
                current_price = (info.get('regularMarketPrice') or
                               info.get('currentPrice') or
                               info.get('previousClose'))

                if current_price is None:
                    if attempt == max_retries:
                        raise ValueError(f"No price data available for {ticker}")
                    continue

                # Extract price data
                price_data = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'previous_close': info.get('previousClose'),
                    'open': info.get('regularMarketOpen') or info.get('open'),
                    'day_high': info.get('regularMarketDayHigh') or info.get('dayHigh'),
                    'day_low': info.get('regularMarketDayLow') or info.get('dayLow'),
                    'volume': info.get('volume'),
                    'timestamp': datetime.now(self.timezone)
                }

                self.logger.info(f"✅ Fetched current price for {ticker}: ${price_data['current_price']:.2f}")
                return price_data

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Current price attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Current price fetch failed after {max_retries + 1} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    if "No data found" in str(e):
                        raise ValueError(f"Invalid ticker symbol '{ticker}'")
                    else:
                        raise ConnectionError(error_msg)
