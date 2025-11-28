"""
Impact Analyzer - Core Analysis Engine for Macro Event Impact Tracking

This module provides the core analysis engine for the Real-Time Macro Event Impact Tracker.
The ImpactAnalyzer orchestrates data fetching, market analysis, and statistical testing to
provide comprehensive insights into how macroeconomic events affect financial markets.

The ImpactAnalyzer supports:
- Single event analysis with multi-timeframe market reactions
- Multi-asset impact assessment across different market sectors
- Statistical significance testing and historical comparisons
- Volatility analysis before and after events
- Cross-asset correlation analysis during event windows
- Batch processing for historical event analysis
- Human-readable impact summaries and reports
"""

import pandas as pd
import numpy as np
import logging
import datetime
from typing import Dict, List, Optional, Tuple
from scipy import stats

from src.data_fetchers.fred_fetcher import FREDDataFetcher
from src.data_fetchers.yahoo_fetcher import YahooDataFetcher
from src.utils.helpers import *
from config.settings import config


class ImpactAnalyzer:
    """
    Core analysis engine for macroeconomic event impact assessment.

    This class orchestrates the analysis of how macroeconomic data releases
    affect financial markets across multiple timeframes and asset classes.
    It integrates FRED economic data with Yahoo Finance market data to provide
    comprehensive impact analysis.

    Features:
    - Single event impact analysis with multi-timeframe reactions
    - Multi-asset analysis across different market sectors
    - Statistical significance testing
    - Volatility change analysis
    - Cross-asset correlation during events
    - Historical comparison and batch processing
    - Human-readable impact summaries

    Limitations:
    - Yahoo Finance FREE tier: Intraday data limited to last 7 days
    - Older events use daily data with adjusted time windows
    - Market hours and holiday awareness built-in

    Attributes:
        fred_fetcher: FRED data fetcher instance for economic indicators
        yahoo_fetcher: Yahoo Finance data fetcher instance for market data
        logger: Logger instance for operation tracking
    """

    def __init__(self, fred_fetcher: Optional[FREDDataFetcher] = None,
                 yahoo_fetcher: Optional[YahooDataFetcher] = None) -> None:
        """
        Initialize the impact analyzer with data fetchers.

        Args:
            fred_fetcher: FRED data fetcher instance (created if None)
            yahoo_fetcher: Yahoo Finance data fetcher instance (created if None)
        """
        self.fred_fetcher = fred_fetcher or FREDDataFetcher()
        self.yahoo_fetcher = yahoo_fetcher or YahooDataFetcher()
        self.logger = logging.getLogger('impact_analyzer')

        self.logger.info("✅ ImpactAnalyzer initialized with data fetchers")

    def analyze_single_event(self, indicator_code: str, release_date: datetime.datetime,
                           hours_after: int = 4) -> Dict:
        """
        Analyze the market impact of a single macroeconomic event.

        This is the main orchestration method that coordinates data fetching,
        market analysis, and statistical testing for a single event.

        Args:
            indicator_code: FRED indicator code (e.g., 'CPI', 'NFP')
            release_date: Date of the data release (None = use latest available)
            hours_after: Hours of market data to analyze after release

        Returns:
            Dict: Comprehensive analysis results with the following keys:
                - 'event_info': Event details (indicator, date, value, change, release_time)
                - 'market_reactions': DataFrame of returns by asset and time window
                - 'volatility_analysis': DataFrame of volatility changes by asset
                - 'top_movers': Dict of top gainers/losers by time window
                - 'sector_summary': DataFrame of sector-level aggregated reactions
                - 'correlation_matrix': DataFrame of cross-asset correlations
                - 'statistical_tests': Dict of significance test results

        Raises:
            ValueError: If indicator_code is invalid or data unavailable
            ConnectionError: If API failures prevent analysis

        Example:
            >>> analyzer = ImpactAnalyzer()
            >>> results = analyzer.analyze_single_event('CPI', datetime(2024, 1, 15, 8, 30))
            >>> print(f"Event: {results['event_info']['indicator']}")
            >>> print(f"Top gainer: {results['top_movers']['15min']['gainers'][0]}")
        """
        # Log the start of analysis with key parameters
        self.logger.info(f"🚀 Starting analysis for {indicator_code} on {release_date}")

        # Step 1: Get macro release data
        self.logger.info("📊 Fetching macroeconomic release data...")
        try:
            release_data = self.fred_fetcher.get_latest_release(indicator_code)
            # Use the date from latest release if no specific date provided
            actual_release_date = release_date if release_date is not None else release_data['date']
            release_time = self.fred_fetcher.estimate_release_time(actual_release_date, indicator_code)
        except Exception as e:
            raise ValueError(f"Failed to get release data for {indicator_code}: {e}")

        # Step 2: Check if we can use intraday data (Yahoo Finance 7-day limit)
        days_since_release = (datetime.datetime.now() - actual_release_date.replace(tzinfo=None)).days
        use_intraday = days_since_release <= 7

        if not use_intraday:
            self.logger.warning(
                f"Event is {days_since_release} days old. Using daily data instead of intraday "
                "(Yahoo Finance FREE tier limitation: intraday data only available for last 7 days)"
            )

        # Step 3: Get list of market assets to analyze
        market_assets = list(config.MARKET_ASSETS.keys())
        self.logger.info(f"📈 Analyzing {len(market_assets)} market assets")

        # Step 4: Fetch market data for all assets
        self.logger.info("📡 Fetching market data for all assets...")
        try:
            if use_intraday:
                market_data = self.yahoo_fetcher.get_multi_asset_data(
                    market_assets, release_time, hours_before=1, hours_after=hours_after
                )
            else:
                # Use daily data for older events
                start_date = actual_release_date - datetime.timedelta(days=1)
                end_date = actual_release_date + datetime.timedelta(days=hours_after // 24 + 1)

                market_data = {}
                for ticker in market_assets:
                    try:
                        df = self.yahoo_fetcher.get_daily_data(ticker, start_date, end_date)
                        if not df.empty:
                            market_data[ticker] = df
                    except Exception as e:
                        self.logger.warning(f"Failed to get daily data for {ticker}: {e}")
                        continue

        except Exception as e:
            raise ConnectionError(f"Failed to fetch market data: {e}")

        if not market_data:
            # Provide helpful error message with troubleshooting tips
            error_msg = (
                "No market data available for analysis. This could be due to:\n"
                f"1. Event date too old: {actual_release_date} (intraday data only available for last 7 days on Yahoo Finance FREE tier)\n"
                f"2. Market closed: Current time may be outside market hours (9:30 AM - 4:00 PM ET)\n"
                f"3. Invalid tickers: {market_assets}\n"
                f"4. Network/API issues: Check internet connection\n"
                "Try selecting a more recent event date or use daily data analysis."
            )
            raise ValueError(error_msg)

        self.logger.info(f"✅ Retrieved data for {len(market_data)} assets")

        # Step 5: Calculate returns for each asset at all time windows
        self.logger.info("📊 Calculating market reactions...")
        market_reactions = []
        volatility_analysis = []

        for ticker, price_data in market_data.items():
            if price_data.empty:
                continue

            # Calculate returns
            returns = self._calculate_asset_returns(price_data, release_time)
            if returns:
                asset_type = config.MARKET_ASSETS[ticker]['type']
                reaction_row = {
                    'ticker': ticker,
                    'asset_type': asset_type,
                    'sector': config.MARKET_ASSETS[ticker]['sector']
                }
                reaction_row.update(returns)
                market_reactions.append(reaction_row)

            # Calculate volatility changes
            vol_change = self._calculate_volatility_change(price_data, release_time)
            if vol_change:
                vol_row = {
                    'ticker': ticker,
                    'asset_type': config.MARKET_ASSETS[ticker]['type'],
                    'pre_event_vol': vol_change['pre_event_vol'],
                    'post_event_vol': vol_change['post_event_vol'],
                    'vol_change': vol_change['vol_change'],
                    'vol_change_pct': vol_change['vol_change_pct']
                }
                volatility_analysis.append(vol_row)

        # Convert to DataFrames
        market_reactions_df = pd.DataFrame(market_reactions)
        volatility_df = pd.DataFrame(volatility_analysis)

        # Debug logging for DataFrame structure
        self.logger.info(f"Market reactions DataFrame shape: {market_reactions_df.shape}")
        if not market_reactions_df.empty:
            self.logger.info(f"Market reactions columns: {list(market_reactions_df.columns)}")
            if 'ticker' in market_reactions_df.columns:
                self.logger.info(f"Sample tickers: {market_reactions_df['ticker'].head(3).tolist()}")
            else:
                self.logger.error("ERROR: 'ticker' column missing from market_reactions DataFrame!")
        else:
            self.logger.warning("Market reactions DataFrame is empty!")

        # Step 6: Identify top movers
        self.logger.info("🏆 Identifying top market movers...")
        top_movers = self.identify_top_movers(market_reactions_df)

        # Step 7: Analyze by sector
        self.logger.info("📊 Analyzing sector-level impacts...")
        sector_summary = self.analyze_by_sector(market_reactions_df)

        # Step 8: Calculate cross-asset correlations
        self.logger.info("🔗 Calculating cross-asset correlations...")
        correlation_matrix = self.calculate_cross_asset_correlation(release_time, hours_after)

        # Step 9: Statistical significance testing
        self.logger.info("📈 Performing statistical significance tests...")
        statistical_tests = {}
        if not market_reactions_df.empty:
            for window in config.ANALYSIS_WINDOWS:
                window_label = window['label']
                if window_label in market_reactions_df.columns:
                    returns = market_reactions_df[window_label].dropna()
                    if len(returns) > 0:
                        statistical_tests[window_label] = self.calculate_statistical_significance(returns)

        # Compile final results
        results = {
            'event_info': {
                'indicator': indicator_code,
                'date': release_date,
                'value': release_data.get('value'),
                'change': self.fred_fetcher.calculate_change(indicator_code).get('percent_change', 0),
                'release_time': release_time,
                'data_type': 'intraday' if use_intraday else 'daily'
            },
            'market_reactions': market_reactions_df,
            'volatility_analysis': volatility_df,
            'top_movers': top_movers,
            'sector_summary': sector_summary,
            'correlation_matrix': correlation_matrix,
            'statistical_tests': statistical_tests,
            'analysis_timestamp': datetime.datetime.now(),
            'assets_analyzed': len(market_data),
            'time_windows': [w['label'] for w in config.ANALYSIS_WINDOWS]
        }

        self.logger.info(f"✅ Analysis complete for {indicator_code} event")
        return results

    def _calculate_asset_returns(self, price_data: pd.DataFrame,
                               event_time: datetime.datetime) -> Optional[Dict[str, float]]:
        """
        Calculate returns for a single asset across all time windows.

        Args:
            price_data: DataFrame with price data
            event_time: Event timestamp

        Returns:
            Optional[Dict[str, float]]: Returns for each time window, or None if calculation fails
        """
        try:
            returns = self.yahoo_fetcher.calculate_returns(price_data, event_time)
            return returns
        except Exception as e:
            self.logger.warning(f"Failed to calculate returns: {e}")
            return None

    def _calculate_volatility_change(self, price_data: pd.DataFrame,
                                   event_time: datetime.datetime) -> Optional[Dict]:
        """
        Calculate volatility changes before and after an event.

        Args:
            price_data: DataFrame with price data
            event_time: Event timestamp

        Returns:
            Optional[Dict]: Volatility analysis results, or None if calculation fails
        """
        try:
            # Ensure event_time is timezone-aware and in the same timezone as price_data
            if event_time.tzinfo is None:
                # If event_time is naive, assume it's in the same timezone as the data
                if not price_data.empty and isinstance(price_data['timestamp'].iloc[0], pd.Timestamp):
                    event_time = price_data['timestamp'].iloc[0].tz_localize(event_time.replace(tzinfo=None))
                else:
                    # Default to UTC if we can't determine timezone
                    import pytz
                    event_time = pytz.UTC.localize(event_time)

            # Split data into pre and post event periods
            pre_event_data = price_data[price_data['timestamp'] < event_time]
            post_event_data = price_data[price_data['timestamp'] >= event_time]

            # Calculate volatility for each period
            pre_vol = 0.0
            if not pre_event_data.empty and len(pre_event_data) >= 30:  # Need minimum data points
                pre_returns = pre_event_data['close'].pct_change().dropna()
                if len(pre_returns) >= 30:
                    pre_vol = self.yahoo_fetcher.get_volatility(pre_event_data, window_minutes=30)

            post_vol = 0.0
            if not post_event_data.empty and len(post_event_data) >= 30:
                post_returns = post_event_data['close'].pct_change().dropna()
                if len(post_returns) >= 30:
                    post_vol = self.yahoo_fetcher.get_volatility(post_event_data, window_minutes=30)

            # Calculate change
            vol_change = post_vol - pre_vol
            vol_change_pct = (vol_change / pre_vol * 100) if pre_vol > 0 else 0.0

            return {
                'pre_event_vol': pre_vol,
                'post_event_vol': post_vol,
                'vol_change': vol_change,
                'vol_change_pct': vol_change_pct
            }

        except Exception as e:
            self.logger.warning(f"Failed to calculate volatility change: {e}")
            return None

    def identify_top_movers(self, market_reactions: pd.DataFrame, n: int = 3) -> Dict:
        """
        Identify top gainers and losers for each time window.

        Args:
            market_reactions: DataFrame with market reactions
            n: Number of top movers to identify

        Returns:
            Dict: Top movers by time window with gainers and losers
        """
        if market_reactions.empty:
            return {}

        top_movers = {}

        for window in config.ANALYSIS_WINDOWS:
            window_label = window['label']

            if window_label not in market_reactions.columns:
                continue

            # Sort by return magnitude
            sorted_data = market_reactions.dropna(subset=[window_label]).copy()
            sorted_data = sorted_data.sort_values(window_label, ascending=False)

            # Top gainers (positive returns)
            gainers_data = sorted_data[sorted_data[window_label] > 0].head(n)
            gainers = []
            for _, row in gainers_data.iterrows():
                gainers.append({
                    'ticker': row['ticker'],
                    'return': row[window_label],
                    'asset_type': row['asset_type']
                })

            # Top losers (negative returns)
            losers_data = sorted_data[sorted_data[window_label] < 0].tail(n)
            losers = []
            for _, row in losers_data.iterrows():
                losers.append({
                    'ticker': row['ticker'],
                    'return': row[window_label],
                    'asset_type': row['asset_type']
                })

            top_movers[window_label] = {
                'gainers': gainers,
                'losers': losers
            }

        return top_movers

    def analyze_by_sector(self, market_reactions: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze market reactions aggregated by sector/asset type.

        Args:
            market_reactions: DataFrame with market reactions

        Returns:
            pd.DataFrame: Sector-level summary with mean returns and volatility
        """
        if market_reactions.empty:
            return pd.DataFrame()

        sector_analysis = []

        # Group by asset type
        for asset_type, group in market_reactions.groupby('asset_type'):
            sector_row = {'sector': asset_type}

            # Calculate mean returns for each time window
            for window in config.ANALYSIS_WINDOWS:
                window_label = window['label']
                if window_label in group.columns:
                    mean_return = group[window_label].mean()
                    std_return = group[window_label].std()
                    sector_row[f'{window_label}_mean'] = mean_return
                    sector_row[f'{window_label}_std'] = std_return

            # Count assets in sector
            sector_row['asset_count'] = len(group)

            sector_analysis.append(sector_row)

        return pd.DataFrame(sector_analysis)

    def calculate_cross_asset_correlation(self, event_datetime: datetime.datetime,
                                        hours_window: int = 4) -> pd.DataFrame:
        """
        Calculate cross-asset correlations during the event window.

        Args:
            event_datetime: Event timestamp
            hours_window: Hours around event to analyze

        Returns:
            pd.DataFrame: Correlation matrix of asset returns
        """
        try:
            # Get all asset data for the event window
            market_assets = list(config.MARKET_ASSETS.keys())

            asset_data = self.yahoo_fetcher.get_multi_asset_data(
                market_assets, event_datetime, hours_before=1, hours_after=hours_window
            )

            if not asset_data:
                return pd.DataFrame()

            # Calculate returns for each asset
            returns_data = {}
            for ticker, price_df in asset_data.items():
                if not price_df.empty:
                    returns = price_df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[ticker] = returns

            if len(returns_data) < 2:
                return pd.DataFrame()

            # Create returns DataFrame and calculate correlation
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()

            return correlation_matrix

        except Exception as e:
            self.logger.warning(f"Failed to calculate cross-asset correlations: {e}")
            return pd.DataFrame()

    def calculate_statistical_significance(self, returns: pd.Series) -> Dict:
        """
        Test if returns are statistically significantly different from zero.

        Args:
            returns: Series of returns to test

        Returns:
            Dict: Statistical test results
        """
        try:
            returns = returns.dropna()
            if len(returns) < 3:  # Need minimum sample size
                return {
                    't_statistic': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'confidence_95': (0.0, 0.0),
                    'sample_size': len(returns)
                }

            # Perform one-sample t-test against zero
            t_stat, p_value = stats.ttest_1samp(returns, 0)

            # Calculate 95% confidence interval
            mean_return = returns.mean()
            std_error = returns.std() / np.sqrt(len(returns))
            confidence_interval = stats.t.interval(0.95, len(returns)-1, mean_return, std_error)

            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'confidence_95': confidence_interval,
                'sample_size': len(returns),
                'mean_return': mean_return
            }

        except Exception as e:
            self.logger.warning(f"Failed to calculate statistical significance: {e}")
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'confidence_95': (0.0, 0.0),
                'error': str(e)
            }

    def generate_impact_summary(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable summary of the impact analysis.

        Args:
            analysis_results: Results from analyze_single_event()

        Returns:
            str: Formatted summary text
        """
        if not analysis_results:
            return "No analysis results available"

        event_info = analysis_results.get('event_info', {})
        top_movers = analysis_results.get('top_movers', {})
        sector_summary = analysis_results.get('sector_summary', pd.DataFrame())

        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("MACRO EVENT IMPACT ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)

        # Event information
        summary_lines.append("📊 EVENT DETAILS:")
        summary_lines.append(f"  Indicator: {event_info.get('indicator', 'N/A')}")
        summary_lines.append(f"  Release Date: {event_info.get('date', 'N/A')}")
        summary_lines.append(f"  Value: {event_info.get('value', 'N/A')}")
        summary_lines.append(f"  Change: {format_percentage(event_info.get('change', 0))}")
        summary_lines.append(f"  Data Type: {event_info.get('data_type', 'N/A')}")

        # Market reactions
        market_reactions = analysis_results.get('market_reactions', pd.DataFrame())
        if not market_reactions.empty:
            summary_lines.append("📈 MARKET REACTIONS:")
            summary_lines.append(f"  Assets Analyzed: {len(market_reactions)}")

            # Show 15-minute reactions (most immediate)
            if '15 Minutes' in market_reactions.columns:
                returns_15min = market_reactions['15 Minutes'].dropna()
                if len(returns_15min) > 0:
                    positive_pct = (returns_15min > 0).mean() * 100
                    avg_return = returns_15min.mean()
                    summary_lines.append(f"  15-Min Reactions: {format_percentage(avg_return)} average")
                    summary_lines.append(f"  Positive Reactions: {positive_pct:.1f}% of assets")

        # Top movers
        if top_movers and '15 Minutes' in top_movers:
            summary_lines.append("🏆 TOP MOVERS (15 Minutes):")
            gainers = top_movers['15 Minutes'].get('gainers', [])
            losers = top_movers['15 Minutes'].get('losers', [])

            if gainers:
                top_gainer = gainers[0]
                summary_lines.append(f"  Top Gainer: {top_gainer['ticker']} ({format_percentage(top_gainer['return'])})")

            if losers:
                top_loser = losers[0]
                summary_lines.append(f"  Top Loser: {top_loser['ticker']} ({format_percentage(top_loser['return'])})")

        # Sector analysis
        if not sector_summary.empty:
            summary_lines.append("📊 SECTOR IMPACT:")
            for _, row in sector_summary.iterrows():
                sector = row['sector']
                if '15 Minutes_mean' in row:
                    avg_return = row['15 Minutes_mean']
                    asset_count = row['asset_count']
                    summary_lines.append(f"  {sector}: {format_percentage(avg_return)} ({asset_count} assets)")

        # Statistical significance
        statistical_tests = analysis_results.get('statistical_tests', {})
        if statistical_tests and '15 Minutes' in statistical_tests:
            test_results = statistical_tests['15 Minutes']
            summary_lines.append("\n📈 STATISTICAL SIGNIFICANCE:")
            if test_results.get('significant', False):
                summary_lines.append("  15-Min reactions are statistically significant (p < 0.05)")
            else:
                summary_lines.append("  15-Min reactions are not statistically significant (p >= 0.05)")
        # Volatility analysis
        volatility_df = analysis_results.get('volatility_analysis', pd.DataFrame())
        if not volatility_df.empty:
            avg_vol_change = volatility_df['vol_change_pct'].mean()
            summary_lines.append("📊 VOLATILITY ANALYSIS:")
            summary_lines.append(f"  Average Volatility Change: {format_percentage(avg_vol_change)}")

        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("Analysis completed at: " + str(analysis_results.get('analysis_timestamp', 'N/A')))

        return "\n".join(summary_lines)

    def batch_analyze_historical(self, indicator_code: str, start_date: datetime.datetime,
                               end_date: datetime.datetime) -> pd.DataFrame:
        """
        Analyze multiple historical events for an indicator.

        Args:
            indicator_code: FRED indicator code
            start_date: Start date for historical analysis
            end_date: End date for historical analysis

        Returns:
            pd.DataFrame: Historical analysis results
        """
        self.logger.info(f"🔄 Starting batch analysis for {indicator_code} from {start_date} to {end_date}")

        try:
            # Get all release dates in the range
            release_dates = self.fred_fetcher.get_release_dates(indicator_code, lookback_months=12)

            # Filter to date range
            release_dates = [d for d in release_dates
                           if start_date <= d <= end_date]

            if not release_dates:
                self.logger.warning(f"No release dates found for {indicator_code} in the specified range")
                return pd.DataFrame()

            self.logger.info(f"📅 Analyzing {len(release_dates)} historical events")

            # Analyze each event
            historical_results = []

            for i, release_date in enumerate(release_dates):
                self.logger.info(f"📊 Analyzing event {i+1}/{len(release_dates)}: {release_date}")

                try:
                    event_analysis = self.analyze_single_event(indicator_code, release_date)

                    # Extract key metrics
                    event_summary = {
                        'release_date': release_date,
                        'indicator': indicator_code,
                        'value': event_analysis['event_info'].get('value'),
                        'change_pct': event_analysis['event_info'].get('change'),
                        'assets_analyzed': event_analysis.get('assets_analyzed', 0)
                    }

                    # Add market reaction metrics
                    market_reactions = event_analysis.get('market_reactions', pd.DataFrame())
                    if not market_reactions.empty:
                        for window in config.ANALYSIS_WINDOWS:
                            window_label = window['label']
                            if window_label in market_reactions.columns:
                                avg_return = market_reactions[window_label].mean()
                                pos_pct = (market_reactions[window_label] > 0).mean() * 100
                                event_summary[f'{window_label.lower().replace(" ", "_")}_avg_return'] = avg_return
                                event_summary[f'{window_label.lower().replace(" ", "_")}_positive_pct'] = pos_pct

                    historical_results.append(event_summary)

                except Exception as e:
                    self.logger.error(f"Failed to analyze event on {release_date}: {e}")
                    continue

            # Convert to DataFrame
            results_df = pd.DataFrame(historical_results)

            if not results_df.empty:
                # Calculate aggregate statistics
                summary_stats = {
                    'total_events': len(results_df),
                    'avg_value': results_df['value'].mean() if 'value' in results_df.columns else None,
                    'avg_change_pct': results_df['change_pct'].mean() if 'change_pct' in results_df.columns else None,
                }

                # Add average returns for each time window
                for window in config.ANALYSIS_WINDOWS:
                    window_key = window['label'].lower().replace(" ", "_") + "_avg_return"
                    if window_key in results_df.columns:
                        summary_stats[f'avg_{window_key}'] = results_df[window_key].mean()

                self.logger.info(f"✅ Batch analysis complete: {len(results_df)} events analyzed")
                results_df.attrs['summary_stats'] = summary_stats

            return results_df

        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            return pd.DataFrame()

    def compare_to_historical_average(self, current_analysis: Dict,
                                    historical_data: pd.DataFrame) -> Dict:
        """
        Compare current event's market reaction to historical average.

        Args:
            current_analysis: Results from analyze_single_event()
            historical_data: DataFrame from batch_analyze_historical()

        Returns:
            Dict: Comparison metrics with z-scores and significance
        """
        if historical_data.empty or not current_analysis:
            return {'error': 'Insufficient data for comparison'}

        try:
            comparison_results = {}
            market_reactions = current_analysis.get('market_reactions', pd.DataFrame())

            if market_reactions.empty:
                return {'error': 'No market reaction data in current analysis'}

            for window in config.ANALYSIS_WINDOWS:
                window_label = window['label']
                window_key = window_label.lower().replace(" ", "_") + "_avg_return"

                if window_key not in historical_data.columns:
                    continue

                # Current event's average return
                if window_label in market_reactions.columns:
                    current_avg = market_reactions[window_label].mean()
                else:
                    continue

                # Historical averages
                historical_avgs = historical_data[window_key].dropna()
                if len(historical_avgs) < 3:  # Need minimum sample size
                    continue

                hist_mean = historical_avgs.mean()
                hist_std = historical_avgs.std()

                # Calculate z-score
                if hist_std > 0:
                    z_score = (current_avg - hist_mean) / hist_std
                else:
                    z_score = 0.0

                # Determine significance
                is_significant = abs(z_score) > 1.96  # 95% confidence level
                direction = "above" if z_score > 0 else "below"

                comparison_results[window_label] = {
                    'current_avg_return': current_avg,
                    'historical_avg_return': hist_mean,
                    'historical_std': hist_std,
                    'z_score': z_score,
                    'is_significant': is_significant,
                    'direction': direction,
                    'sample_size': len(historical_avgs)
                }

            return comparison_results

        except Exception as e:
            self.logger.error(f"Failed to compare to historical average: {e}")
            return {'error': str(e)}
