"""
Data Fetchers Package - API Integrations for Financial Data

This package provides unified interfaces for fetching financial and economic data
from multiple FREE API sources. All data fetchers implement consistent interfaces
and include intelligent caching, rate limiting, and robust error handling.

Available Data Sources:

1. FRED (Federal Reserve Economic Data)
   - Provider: Federal Reserve Bank of St. Louis
   - Cost: 100% FREE (requires API key - 2 minute signup)
   - Data: Economic indicators, inflation, employment, GDP, etc.
   - Rate Limit: 120 requests/minute
   - Historical Data: 100+ years for most indicators

2. Yahoo Finance (yfinance)
   - Provider: Yahoo Finance API
   - Cost: 100% FREE (no API key required)
   - Data: Real-time and historical market prices
   - Rate Limit: 1-2 requests/second recommended
   - Intraday Data: Last 7 days only (FREE tier limitation)

Key Features:
- Unified API interfaces across all data sources
- Automatic timezone handling and data normalization
- Intelligent caching to minimize API calls
- Rate limiting to respect FREE tier limits
- Comprehensive error handling with helpful messages
- Market hours awareness for trading data

Classes:
    FREDDataFetcher: Economic indicator data from FRED API
    YahooDataFetcher: Market price data from Yahoo Finance

Usage:
    from src.data_fetchers.fred_fetcher import FREDDataFetcher
    from src.data_fetchers.yahoo_fetcher import YahooDataFetcher

    # Economic data
    fred = FREDDataFetcher()
    cpi_data = fred.get_series_data('CPIAUCSL', '2023-01-01', '2024-01-01')

    # Market data
    yahoo = YahooDataFetcher()
    spy_prices = yahoo.get_intraday_data('SPY', event_datetime, 1, 4)
"""
