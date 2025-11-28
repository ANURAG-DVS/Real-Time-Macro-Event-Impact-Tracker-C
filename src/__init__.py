"""
Macro Event Impact Tracker - Core Source Package

This package contains the core implementation of the Real-Time Macro Event Impact Tracker,
a comprehensive financial analysis tool for studying how major economic releases affect
financial markets in real-time.

The package is organized into the following submodules:

- analyzers: Core analysis logic for processing economic events and market reactions
- data_fetchers: API integrations for FRED economic data and Yahoo Finance market data
- utils: Utility functions for caching, helpers, and common operations
- visualizers: Chart generation and plotting functionality

All data sources are 100% FREE:
- FRED API: Economic indicators (requires free API key)
- Yahoo Finance: Market data (no API key needed)

Key Features:
- Real-time event impact analysis across multiple asset classes
- Multi-timeframe market reaction tracking (15min to 1hr windows)
- Statistical significance testing and historical comparisons
- Interactive visualizations and professional dashboards
- Intelligent caching and rate limiting for API efficiency

Usage:
    from src.analyzers.impact_analyzer import ImpactAnalyzer
    from src.data_fetchers.fred_fetcher import FREDDataFetcher
    from src.data_fetchers.yahoo_fetcher import YahooDataFetcher

    # Initialize components
    analyzer = ImpactAnalyzer()
    results = analyzer.analyze_single_event('CPI', datetime(2024, 1, 15))
"""
