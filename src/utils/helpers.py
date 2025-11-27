"""
Helper Functions - Utility Functions for the Real-Time Macro Event Impact Tracker

This module provides a comprehensive collection of utility functions used throughout
the application. These functions handle common operations like datetime formatting,
mathematical calculations, file operations, data processing, and market-specific utilities.

The helper functions are designed to:
- Reduce code duplication across modules
- Provide consistent error handling and logging
- Follow the project's coding standards and patterns
- Integrate seamlessly with the configuration system
"""

import pandas as pd
import numpy as np
import datetime
import pytz
import logging
from typing import Union, List, Optional, Tuple, Dict, Any
from pathlib import Path
from config.settings import config


# Set up logger for this module
logger = logging.getLogger('helpers')


def format_datetime(dt: Union[datetime.datetime, str], format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format a datetime object or string to a standardized string format.

    Args:
        dt: Datetime object or string to format
        format_str: Format string (default: '%Y-%m-%d %H:%M:%S')

    Returns:
        str: Formatted datetime string, or empty string if input is None

    Examples:
        >>> from datetime import datetime
        >>> format_datetime(datetime(2024, 1, 15, 10, 30))
        '2024-01-15 10:30:00'
        >>> format_datetime("2024-01-15")
        '2024-01-15'
        >>> format_datetime(None)
        ''
    """
    if dt is None:
        return ""

    if isinstance(dt, str):
        # If it's already a string, try to validate format
        try:
            # Attempt to parse and reformat to ensure consistency
            parsed = datetime.datetime.strptime(dt, format_str)
            return parsed.strftime(format_str)
        except ValueError:
            # If parsing fails, return as-is
            return dt

    if isinstance(dt, datetime.datetime):
        return dt.strftime(format_str)

    # For other types, convert to string
    return str(dt)


def parse_datetime(date_str: str) -> Optional[datetime.datetime]:
    """
    Parse a date string using multiple common formats.

    Attempts to parse the date string using various common formats in order
    of preference. Returns None if parsing fails.

    Args:
        date_str: Date string to parse

    Returns:
        Optional[datetime.datetime]: Parsed datetime object, or None if parsing fails

    Examples:
        >>> parse_datetime("2024-01-15")
        datetime.datetime(2024, 1, 15, 0, 0)
        >>> parse_datetime("01/15/2024")
        datetime.datetime(2024, 1, 15, 0, 0)
        >>> parse_datetime("invalid")
        None
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Common date formats to try, in order of preference
    formats = [
        '%Y-%m-%d',           # 2024-01-15
        '%Y-%m-%d %H:%M:%S',  # 2024-01-15 10:30:00
        '%m/%d/%Y',           # 01/15/2024
        '%d-%m-%Y',           # 15-01-2024
        '%Y/%m/%d',           # 2024/01/15
        '%d/%m/%Y',           # 15/01/2024
        '%b %d, %Y',          # Jan 15, 2024
        '%B %d, %Y',          # January 15, 2024
    ]

    for fmt in formats:
        try:
            parsed = datetime.datetime.strptime(date_str.strip(), fmt)
            logger.debug(f"Successfully parsed '{date_str}' with format '{fmt}'")
            return parsed
        except ValueError:
            continue

    logger.warning(f"Failed to parse date string: '{date_str}'")
    return None


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate the percentage change between two values.

    Formula: ((new_value - old_value) / old_value) * 100

    Args:
        old_value: The original/baseline value
        new_value: The new/current value

    Returns:
        float: Percentage change, or 0.0 if old_value is zero or None

    Examples:
        >>> calculate_percentage_change(100, 110)
        10.0
        >>> calculate_percentage_change(100, 90)
        -10.0
        >>> calculate_percentage_change(0, 100)
        0.0
    """
    if old_value is None or new_value is None:
        return 0.0

    try:
        old_value = float(old_value)
        new_value = float(new_value)

        if old_value == 0:
            # Avoid division by zero
            return 0.0

        return ((new_value - old_value) / old_value) * 100

    except (ValueError, TypeError, ZeroDivisionError):
        logger.warning(f"Error calculating percentage change: old={old_value}, new={new_value}")
        return 0.0


def ensure_directory_exists(path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it and its parents if necessary.

    Args:
        path: Directory path to create

    Raises:
        PermissionError: If directory cannot be created due to permissions

    Examples:
        >>> ensure_directory_exists("data/exports/my_analysis")
        >>> # Creates the directory structure if it doesn't exist
    """
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        if path_obj.exists():
            logger.debug(f"Directory ensured: {path}")
        else:
            logger.error(f"Failed to create directory: {path}")

    except PermissionError as e:
        logger.error(f"Permission denied creating directory {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise


def get_trading_days(start_date: datetime.datetime, end_date: datetime.datetime) -> List[datetime.datetime]:
    """
    Get a list of trading days (business days) between two dates.

    Uses pandas.bdate_range to generate business days. Note: This provides
    a simple weekday-based approach and doesn't account for holidays.

    Args:
        start_date: Start date for the range
        end_date: End date for the range

    Returns:
        List[datetime.datetime]: List of trading day datetime objects

    Examples:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 5)
        >>> get_trading_days(start, end)
        [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2),
         datetime.datetime(2024, 1, 3), datetime.datetime(2024, 1, 4)]
    """
    try:
        # Use pandas bdate_range for business days
        bdates = pd.bdate_range(start=start_date, end=end_date, freq='B')

        # Convert to list of datetime objects
        trading_days = [dt.to_pydatetime() for dt in bdates]

        logger.debug(f"Generated {len(trading_days)} trading days from {start_date} to {end_date}")
        return trading_days

    except Exception as e:
        logger.error(f"Error generating trading days: {e}")
        return []


def calculate_annualized_volatility(returns: Union[List, np.ndarray, pd.Series],
                                   periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from a series of returns.

    Args:
        returns: Series of periodic returns (e.g., daily returns)
        periods_per_year: Number of periods in a year (default: 252 for daily)

    Returns:
        float: Annualized volatility as a percentage, or 0.0 if calculation fails

    Examples:
        >>> import numpy as np
        >>> daily_returns = np.random.normal(0, 0.02, 100)  # 100 days of returns
        >>> calculate_annualized_volatility(daily_returns)
        31.42  # Approximately 31.42% annualized volatility
    """
    if returns is None or len(returns) == 0:
        logger.warning("Empty or None returns provided to calculate_annualized_volatility")
        return 0.0

    try:
        # Convert to numpy array for consistent handling
        if isinstance(returns, list):
            returns = np.array(returns)
        elif isinstance(returns, pd.Series):
            returns = returns.values

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        # Calculate standard deviation
        std_dev = np.std(returns, ddof=1)  # Use sample standard deviation

        # Annualize
        annualized_vol = std_dev * np.sqrt(periods_per_year)

        # Convert to percentage
        annualized_vol_pct = annualized_vol * 100

        logger.debug(f"Calculated annualized volatility: {annualized_vol_pct:.2f}%")
        return annualized_vol_pct

    except Exception as e:
        logger.error(f"Error calculating annualized volatility: {e}")
        return 0.0


def create_summary_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Create summary statistics for specified or all numeric columns in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of column names to analyze (default: all numeric columns)

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with stats for each column

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        >>> create_summary_stats(df)
        {
            'A': {'mean': 3.0, 'median': 3.0, 'std': 1.58, ...},
            'B': {'mean': 30.0, 'median': 30.0, 'std': 15.81, ...}
        }
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to create_summary_stats")
        return {}

    # Determine which columns to analyze
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        logger.warning("No numeric columns found in DataFrame")
        return {}

    summary_stats = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue

        col_data = df[col].dropna()

        if col_data.empty:
            logger.warning(f"Column '{col}' has no valid data")
            continue

        try:
            # Calculate comprehensive statistics
            stats = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                '25th_percentile': col_data.quantile(0.25),
                '75th_percentile': col_data.quantile(0.75),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }

            # Clean up NaN values
            for key, value in stats.items():
                if pd.isna(value):
                    stats[key] = 0.0

            summary_stats[col] = stats

        except Exception as e:
            logger.error(f"Error calculating stats for column '{col}': {e}")
            continue

    logger.debug(f"Generated summary stats for {len(summary_stats)} columns")
    return summary_stats


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string with sign.

    Args:
        value: Decimal value to format (e.g., 0.0234 for 2.34%)
        decimals: Number of decimal places to show

    Returns:
        str: Formatted percentage string

    Examples:
        >>> format_percentage(0.0234)
        '+2.34%'
        >>> format_percentage(-0.0156)
        '-1.56%'
        >>> format_percentage(0.0)
        '+0.00%'
    """
    if value is None:
        return "0.00%"

    try:
        value = float(value)

        # Format with specified decimals
        formatted_value = ".2f"

        # Add sign
        if value >= 0:
            return f"+{formatted_value}%"
        else:
            return f"{formatted_value}%"

    except (ValueError, TypeError):
        logger.warning(f"Invalid value for percentage formatting: {value}")
        return "0.00%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero occurs.

    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if denominator is zero or None

    Returns:
        float: Division result or default value

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=999)
        999
    """
    if numerator is None or denominator is None:
        return default

    try:
        numerator = float(numerator)
        denominator = float(denominator)

        if denominator == 0:
            return default

        return numerator / denominator

    except (ValueError, TypeError, ZeroDivisionError):
        logger.warning(f"Safe divide error: {numerator}/{denominator}")
        return default


def merge_dataframes_on_time(dfs: Dict[str, pd.DataFrame], time_col: str = 'timestamp') -> pd.DataFrame:
    """
    Merge multiple DataFrames on a timestamp column using outer join.

    Args:
        dfs: Dictionary mapping names to DataFrames
        time_col: Name of the timestamp column to merge on

    Returns:
        pd.DataFrame: Merged DataFrame with forward-filled missing values

    Examples:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'timestamp': ['2024-01-01', '2024-01-02'], 'A': [1, 2]})
        >>> df2 = pd.DataFrame({'timestamp': ['2024-01-01', '2024-01-03'], 'B': [10, 30]})
        >>> merged = merge_dataframes_on_time({'df1': df1, 'df2': df2})
        >>> print(merged)
             timestamp    A     B
        0   2024-01-01  1.0  10.0
        1   2024-01-02  2.0   NaN
        2   2024-01-03  NaN  30.0
    """
    if not dfs or not isinstance(dfs, dict):
        logger.warning("Invalid DataFrames provided to merge_dataframes_on_time")
        return pd.DataFrame()

    try:
        # Start with the first DataFrame
        merged_df = None

        for name, df in dfs.items():
            if df is None or df.empty:
                logger.warning(f"Skipping empty DataFrame: {name}")
                continue

            if time_col not in df.columns:
                logger.warning(f"Timestamp column '{time_col}' not found in DataFrame: {name}")
                continue

            if merged_df is None:
                merged_df = df.copy()
            else:
                # Merge with outer join and add suffix for duplicate columns
                merged_df = pd.merge(merged_df, df, on=time_col, how='outer', suffixes=('', f'_{name}'))

        if merged_df is None:
            return pd.DataFrame()

        # Sort by timestamp
        if time_col in merged_df.columns:
            merged_df = merged_df.sort_values(time_col)

        # Forward fill missing values
        merged_df = merged_df.fillna(method='ffill')

        logger.debug(f"Merged {len(dfs)} DataFrames into {len(merged_df)} rows")
        return merged_df

    except Exception as e:
        logger.error(f"Error merging DataFrames: {e}")
        return pd.DataFrame()


def export_to_csv(df: pd.DataFrame, filename: str, subdir: str = '') -> None:
    """
    Export a DataFrame to CSV in the exports directory.

    Args:
        df: DataFrame to export
        filename: Base filename (timestamp will be added)
        subdir: Optional subdirectory within exports folder

    Raises:
        Exception: If export fails

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> export_to_csv(df, 'my_data.csv')
        >>> # Saves to: data/exports/my_data_20240115_103000.csv
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to export_to_csv")
        return

    try:
        # Build export path
        export_dir = config.EXPORT_DIR
        if subdir:
            export_dir = export_dir / subdir
            ensure_directory_exists(export_dir)

        # Add timestamp to filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = Path(filename).stem  # Remove extension if present
        full_filename = f"{base_name}_{timestamp}.csv"
        export_path = export_dir / full_filename

        # Export to CSV
        df.to_csv(export_path, index=False)

        logger.info(f"Exported DataFrame to: {export_path}")

    except Exception as e:
        error_msg = f"Error exporting DataFrame to CSV: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)


def load_from_csv(filename: str, subdir: str = '') -> pd.DataFrame:
    """
    Load a DataFrame from CSV in the exports directory.

    Args:
        filename: CSV filename to load
        subdir: Optional subdirectory within exports folder

    Returns:
        pd.DataFrame: Loaded DataFrame, or empty DataFrame if loading fails

    Examples:
        >>> df = load_from_csv('my_data_20240115_103000.csv')
        >>> print(df.shape)
        (100, 5)
    """
    try:
        # Build file path
        load_dir = config.EXPORT_DIR
        if subdir:
            load_dir = load_dir / subdir

        file_path = load_dir / filename

        if not file_path.exists():
            logger.warning(f"CSV file not found: {file_path}")
            return pd.DataFrame()

        # Load CSV with automatic datetime parsing
        df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)

        logger.info(f"Loaded DataFrame from: {file_path} ({len(df)} rows)")
        return df

    except Exception as e:
        logger.error(f"Error loading CSV {filename}: {e}")
        return pd.DataFrame()


def get_market_status() -> Dict[str, Any]:
    """
    Get current market status information.

    Returns:
        Dict[str, Any]: Market status information with keys:
            - 'is_open': Boolean indicating if market is currently open
            - 'time_to_open': Minutes to next market open (None if open)
            - 'time_to_close': Minutes to next market close (None if closed)
            - 'current_time': Current time in ET
            - 'next_open_time': Next market open time
            - 'next_close_time': Next market close time

    Examples:
        >>> status = get_market_status()
        >>> if status['is_open']:
        ...     print(f"Market is open, closes in {status['time_to_close']} minutes")
        ... else:
        ...     print(f"Market is closed, opens in {status['time_to_open']} minutes")
    """
    try:
        # Get current time in Eastern Time
        et_tz = pytz.timezone(config.DEFAULT_TIMEZONE)
        current_time = datetime.datetime.now(et_tz)

        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

        # Check if it's a weekday
        is_weekday = current_time.weekday() < 5  # Monday=0 to Friday=4

        if not is_weekday:
            # Weekend - find next Monday 9:30 AM
            days_to_monday = (7 - current_time.weekday()) % 7
            if days_to_monday == 0:  # If it's Monday, go to next Monday
                days_to_monday = 7

            next_open = (current_time + datetime.timedelta(days=days_to_monday)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            next_close = next_open.replace(hour=16, minute=0)

            return {
                'is_open': False,
                'time_to_open': int((next_open - current_time).total_seconds() / 60),
                'time_to_close': None,
                'current_time': current_time,
                'next_open_time': next_open,
                'next_close_time': next_close
            }

        # Weekday logic
        if market_open <= current_time <= market_close:
            # Market is open
            time_to_close = int((market_close - current_time).total_seconds() / 60)

            return {
                'is_open': True,
                'time_to_open': None,
                'time_to_close': time_to_close,
                'current_time': current_time,
                'next_open_time': None,
                'next_close_time': market_close
            }
        elif current_time < market_open:
            # Before market open today
            time_to_open = int((market_open - current_time).total_seconds() / 60)

            return {
                'is_open': False,
                'time_to_open': time_to_open,
                'time_to_close': None,
                'current_time': current_time,
                'next_open_time': market_open,
                'next_close_time': market_close
            }
        else:
            # After market close today - find next trading day
            next_trading_day = get_trading_days(current_time.date(), current_time.date() + datetime.timedelta(days=7))
            next_day = None
            for day in next_trading_day:
                if day.date() > current_time.date():
                    next_day = day
                    break

            if next_day:
                next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
                next_close = next_day.replace(hour=16, minute=0, second=0, microsecond=0)
                time_to_open = int((next_open - current_time).total_seconds() / 60)
            else:
                # Fallback
                next_open = (current_time + datetime.timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
                next_close = next_open.replace(hour=16, minute=0)
                time_to_open = 1440  # 24 hours in minutes

            return {
                'is_open': False,
                'time_to_open': time_to_open,
                'time_to_close': None,
                'current_time': current_time,
                'next_open_time': next_open,
                'next_close_time': next_close
            }

    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return {
            'is_open': False,
            'time_to_open': None,
            'time_to_close': None,
            'current_time': datetime.datetime.now(),
            'next_open_time': None,
            'next_close_time': None,
            'error': str(e)
        }


def validate_ticker(ticker: str) -> bool:
    """
    Validate a stock ticker symbol format.

    Basic validation checks:
    - Not empty or None
    - Uppercase letters
    - Alphanumeric characters only
    - Reasonable length (1-10 characters)

    Args:
        ticker: Ticker symbol to validate

    Returns:
        bool: True if ticker appears valid, False otherwise

    Examples:
        >>> validate_ticker('AAPL')
        True
        >>> validate_ticker('aapl')
        False
        >>> validate_ticker('INVALID!')
        False
    """
    if not ticker or not isinstance(ticker, str):
        return False

    ticker = ticker.strip()

    # Check length
    if len(ticker) < 1 or len(ticker) > 10:
        return False

    # Check if uppercase
    if ticker != ticker.upper():
        logger.debug(f"Ticker not uppercase: {ticker}")
        return False

    # Check if alphanumeric only (allowing some special characters like dots)
    allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
    if not all(c in allowed_chars for c in ticker):
        logger.debug(f"Ticker contains invalid characters: {ticker}")
        return False

    return True


def validate_date_range(start_date: datetime.datetime, end_date: datetime.datetime) -> bool:
    """
    Validate a date range for reasonableness.

    Checks:
    - start_date is before end_date
    - Dates are not in the future
    - Date range is not unreasonably long (max 10 years)
    - Dates are not too far in the past (max 50 years ago)

    Args:
        start_date: Start date of the range
        end_date: End date of the range

    Returns:
        bool: True if date range is valid

    Raises:
        ValueError: If date range is invalid

    Examples:
        >>> from datetime import datetime
        >>> validate_date_range(datetime(2020, 1, 1), datetime(2024, 1, 1))
        True
        >>> validate_date_range(datetime(2024, 1, 1), datetime(2020, 1, 1))
        ValueError: Start date must be before end date
    """
    current_date = datetime.datetime.now()

    # Check date order
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")

    # Check future dates
    if start_date > current_date or end_date > current_date:
        raise ValueError("Date range cannot include future dates")

    # Check maximum range (10 years)
    date_diff_years = (end_date - start_date).days / 365.25
    if date_diff_years > 10:
        raise ValueError("Date range cannot exceed 10 years")

    # Check minimum date (50 years ago)
    min_allowed_date = current_date - datetime.timedelta(days=50*365.25)
    if start_date < min_allowed_date:
        raise ValueError("Start date cannot be more than 50 years in the past")

    logger.debug(f"Validated date range: {start_date} to {end_date}")
    return True
