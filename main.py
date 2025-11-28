#!/usr/bin/env python3
"""
Macro Event Impact Tracker - Main CLI Application

A 100% FREE tool for analyzing how major economic releases (CPI, NFP, GDP, etc.)
impact financial markets in real-time.

Data Sources (All Free):
- FRED API: Economic indicator data (requires free API key)
- Yahoo Finance: Market price data (no API key needed)

Example Usage:
# Analyze latest CPI release
python main.py --indicator CPI --plot --export

# Analyze specific NFP release
python main.py --indicator NFP --date 2024-01-05 --hours-after 8

# Historical analysis
python main.py --indicator GDP --mode historical --start-date 2023-01-01 --end-date 2024-01-01

# List all available indicators
python main.py --list-indicators
"""

import argparse
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional
from src.data_fetchers.fred_fetcher import FREDDataFetcher
from src.data_fetchers.yahoo_fetcher import YahooDataFetcher
from src.analyzers.impact_analyzer import ImpactAnalyzer
from src.visualizers.plotter import ImpactPlotter
from src.utils.helpers import *
from config.settings import config


def setup_logging() -> None:
    """
    Configure logging for both console and file output.

    Console: INFO level with simple format
    File: DEBUG level with detailed format in logs/app.log
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove any existing handlers
    logger.handlers.clear()

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG and above)
    log_file = logs_dir / "app.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logging.info("✅ Logging configured (console: INFO, file: DEBUG)")


def setup_directories() -> None:
    """
    Ensure all required directories exist for the application.
    """
    directories = [
        Path("data/cache"),
        Path("data/exports"),
        Path("data/exports/plots"),
        Path("logs")
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f"✅ Directory ensured: {directory}")
        except PermissionError as e:
            logging.error(f"❌ Permission denied creating directory {directory}: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Error creating directory {directory}: {e}")
            raise


def validate_api_keys() -> None:
    """
    Validate that required API keys are set and working.
    """
    logging.info("🔑 Validating API keys...")

    # Check FRED API key
    if not config.FRED_API_KEY or config.FRED_API_KEY.strip() == '':
        print("""
❌ FRED API key not found or invalid.
Get your FREE API key at: https://fred.stlouisfed.org/docs/api/api_key.html
Then add it to your .env file: FRED_API_KEY=your_key_here
        """.strip())
        sys.exit(1)

    # Test FRED API connection
    try:
        fred_fetcher = FREDDataFetcher()
        # Try a simple API call
        test_data = fred_fetcher.get_latest_release('CPI')
        if test_data and 'value' in test_data:
            logging.info("✅ FRED API connection validated")
        else:
            raise ValueError("Test API call returned invalid data")
    except Exception as e:
        print(f"""
❌ FRED API validation failed: {e}
Please check your API key and internet connection.
Get your FREE API key at: https://fred.stlouisfed.org/docs/api/api_key.html
        """.strip())
        sys.exit(1)


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Macro Event Impact Tracker - Analyze how economic releases affect financial markets (100% FREE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest CPI release
  python main.py --indicator CPI --plot --export

  # Analyze specific NFP release
  python main.py --indicator NFP --date 2024-01-05 --hours-after 8

  # Historical analysis
  python main.py --indicator GDP --mode historical --start-date 2023-01-01 --end-date 2024-01-01

  # List all available indicators
  python main.py --list-indicators
        """
    )

    # Required arguments
    parser.add_argument(
        '--indicator',
        type=str,
        help=f"Economic indicator to analyze. Options: {', '.join(config.MACRO_INDICATORS.keys())}"
    )

    # Optional arguments
    parser.add_argument(
        '--date',
        type=str,
        help="Specific release date to analyze (YYYY-MM-DD). If not provided, uses latest release."
    )

    parser.add_argument(
        '--hours-after',
        type=int,
        default=4,
        help="Number of hours after release to analyze market impact (default: 4)"
    )

    parser.add_argument(
        '--mode',
        choices=['single', 'historical'],
        default='single',
        help="Analysis mode: 'single' for one event, 'historical' for multiple events (default: single)"
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help="Start date for historical analysis (YYYY-MM-DD, required if mode=historical)"
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help="End date for historical analysis (YYYY-MM-DD, required if mode=historical)"
    )

    parser.add_argument(
        '--export',
        action='store_true',
        help="Export analysis results to CSV files in data/exports/"
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help="Generate visualizations and save to data/exports/plots/"
    )

    parser.add_argument(
        '--list-indicators',
        action='store_true',
        help="Show all available economic indicators and exit"
    )

    parser.add_argument(
        '--cache',
        action='store_true',
        default=True,
        help="Use cached data when available (recommended for free tier)"
    )

    return parser.parse_args()


def list_available_indicators() -> None:
    """
    Display all available economic indicators in a formatted table.
    """
    print("\nAvailable Economic Indicators:")
    print("═" * 80)

    # Table header
    print(f"{'Code':<12} {'Name':<40} {'Frequency':<15}")
    print("─" * 80)

    # Table rows
    for code, info in config.MACRO_INDICATORS.items():
        name = info['name'][:37] + "..." if len(info['name']) > 37 else info['name']
        frequency = info.get('frequency', 'Unknown')
        print(f"{code:<12} {name:<40} {frequency:<15}")

    print("═" * 80)
    print()


def validate_inputs(args) -> bool:
    """
    Validate command-line arguments for consistency and correctness.
    """
    # Check if indicator is valid
    if not args.indicator or args.indicator not in config.MACRO_INDICATORS:
        available = list(config.MACRO_INDICATORS.keys())
        raise ValueError(f"Invalid indicator '{args.indicator}'. Available options: {available}")

    # Validate mode-specific requirements
    if args.mode == 'historical':
        if not args.start_date or not args.end_date:
            raise ValueError("Historical mode requires --start-date and --end-date")

        # Validate date formats
        try:
            start_date = parse_datetime(args.start_date)
            end_date = parse_datetime(args.end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format: {e}")

        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        # Check for future dates
        current_date = datetime.datetime.now()
        if start_date > current_date or end_date > current_date:
            raise ValueError("Date range cannot include future dates")

    elif args.mode == 'single':
        # Validate single date if provided
        if args.date:
            try:
                release_date = parse_datetime(args.date)
            except Exception as e:
                raise ValueError(f"Invalid date format. Use YYYY-MM-DD format: {e}")

            # Check if date is too old (Yahoo Finance limitation)
            days_since_release = (datetime.datetime.now() - release_date.replace(tzinfo=None)).days
            if days_since_release > 7:
                print(f"""
⚠️  Note: Event is {days_since_release} days old. Due to Yahoo Finance free tier limitations,
only daily data (not intraday) will be available for this analysis.
                """.strip())

    # Validate hours_after
    if args.hours_after <= 0 or args.hours_after > 24:
        raise ValueError("hours-after must be between 1 and 24")

    return True


def run_single_analysis(args) -> int:
    """
    Run single event analysis and return exit code.
    """
    try:
        # Initialize components
        logging.info("🔍 Initializing data fetchers...")
        fred_fetcher = FREDDataFetcher()
        yahoo_fetcher = YahooDataFetcher()
        analyzer = ImpactAnalyzer(fred_fetcher, yahoo_fetcher)
        plotter = ImpactPlotter() if args.plot else None

        # Get release date
        if args.date:
            release_date = parse_datetime(args.date)
        else:
            logging.info("📅 Finding latest release date...")
            # Get latest release to determine the date
            latest_release = fred_fetcher.get_latest_release(args.indicator)
            # For now, use today's date as approximation - in real usage would parse from release data
            release_date = datetime.datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)

        logging.info(f"🎯 Analyzing {args.indicator} release on {release_date.strftime('%Y-%m-%d')}")

        # Run analysis
        logging.info("📊 Running market impact analysis...")
        results = analyzer.analyze_single_event(args.indicator, release_date, args.hours_after)

        # Generate and display summary
        summary = analyzer.generate_impact_summary(results)
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        print(summary)
        print("="*80)

        # Export results if requested
        if args.export:
            logging.info("💾 Exporting results to CSV...")

            # Export market reactions
            market_df = results.get('market_reactions', pd.DataFrame())
            if not market_df.empty:
                export_to_csv(market_df, f"{args.indicator}_market_reactions")

            # Export volatility analysis
            vol_df = results.get('volatility_analysis', pd.DataFrame())
            if not vol_df.empty:
                export_to_csv(vol_df, f"{args.indicator}_volatility_analysis")

            print("✅ Results exported to data/exports/")

        # Generate plots if requested
        if args.plot and plotter:
            logging.info("📊 Generating visualizations...")
            plot_paths = plotter.generate_report_plots(results)
            if plot_paths:
                print(f"✅ {len(plot_paths)} plots saved to data/exports/plots/")
                for path in plot_paths:
                    print(f"   - {path.name}")

        logging.info("✅ Single event analysis completed successfully")
        return 0

    except ValueError as e:
        print(f"❌ Input validation error: {e}")
        return 1
    except ConnectionError as e:
        print(f"❌ Network/API error: {e}")
        print("   Please check your internet connection and API keys")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error in single analysis: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1


def run_historical_analysis(args) -> int:
    """
    Run historical analysis and return exit code.
    """
    try:
        # Initialize components
        logging.info("🔍 Initializing data fetchers...")
        fred_fetcher = FREDDataFetcher()
        yahoo_fetcher = YahooDataFetcher()
        analyzer = ImpactAnalyzer(fred_fetcher, yahoo_fetcher)
        plotter = ImpactPlotter() if args.plot else None

        # Parse dates
        start_date = parse_datetime(args.start_date)
        end_date = parse_datetime(args.end_date)

        logging.info(f"📅 Analyzing historical events from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Run historical analysis
        logging.info("📊 Running historical analysis...")
        historical_results = analyzer.batch_analyze_historical(args.indicator, start_date, end_date)

        if historical_results.empty:
            print(f"⚠️  No historical data found for {args.indicator} in the specified date range")
            return 1

        # Display summary statistics
        print(f"\n📊 Historical Analysis Summary for {args.indicator}")
        print("="*60)
        print(f"Total Events Analyzed: {len(historical_results)}")

        if hasattr(historical_results, 'attrs') and 'summary_stats' in historical_results.attrs:
            stats = historical_results.attrs['summary_stats']
            print(f"Average Value: {stats.get('avg_value', 'N/A')}")
            print(f"Average Change: {format_percentage(stats.get('avg_change_pct', 0))}")
            print(f"Average 1hr Return: {format_percentage(stats.get('1hr_avg_return', 0))}")

        # Export results if requested
        if args.export:
            logging.info("💾 Exporting historical results...")
            export_to_csv(historical_results, f"{args.indicator}_historical_analysis")
            print("✅ Historical results exported to data/exports/")

        # Generate historical trend plot if requested
        if args.plot and plotter and len(historical_results) > 2:
            logging.info("📊 Generating historical trend plot...")
            plot_path = f"data/exports/plots/{args.indicator}_historical_trends.png"
            plotter.plot_historical_trends(historical_results, args.indicator, plot_path)
            print(f"✅ Historical trend plot saved to {plot_path}")

        logging.info("✅ Historical analysis completed successfully")
        return 0

    except ValueError as e:
        print(f"❌ Input validation error: {e}")
        return 1
    except ConnectionError as e:
        print(f"❌ Network/API error: {e}")
        print("   Please check your internet connection and API keys")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error in historical analysis: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1


def main() -> int:
    """
    Main entry point for the Macro Event Impact Tracker CLI.
    """
    try:
        # Setup
        setup_logging()
        setup_directories()

        # Parse arguments
        args = parse_arguments()

        # Handle list indicators request
        if args.list_indicators:
            list_available_indicators()
            return 0

        # Welcome message
        print("""
╔═══════════════════════════════════════════════════════╗
║     MACRO EVENT IMPACT TRACKER (100% FREE!)          ║
║     Analyzing market reactions to economic releases   ║
╚═══════════════════════════════════════════════════════╝
        """.strip())

        # Validate API keys
        validate_api_keys()

        # Validate inputs
        validate_inputs(args)

        # Run analysis based on mode
        if args.mode == 'single':
            return run_single_analysis(args)
        elif args.mode == 'historical':
            return run_historical_analysis(args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        logging.info("Analysis interrupted by user")
        return 1

    except Exception as e:
        logging.error(f"Unexpected error in main: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        print("   Check the log file at logs/app.log for details")
        return 1

    finally:
        # Completion message
        if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print("\n✨ Analysis complete! Thank you for using the Macro Event Impact Tracker.")


if __name__ == "__main__":
    sys.exit(main())
