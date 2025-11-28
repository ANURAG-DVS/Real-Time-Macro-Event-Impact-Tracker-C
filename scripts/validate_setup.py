"""
Setup Validation Script

Run this before using the tracker to ensure everything is configured correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_result(check_name, passed, message=""):
    """Print check result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {check_name}")
    if message:
        print(f"       {message}")

def main():
    print("\n🔍 MACRO EVENT IMPACT TRACKER - Setup Validation")
    print("=" * 60)

    all_passed = True

    # Check 1: Python version
    print_header("Python Version")
    import sys
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 9
    print_result("Python 3.9+", passed, f"Found: Python {version.major}.{version.minor}.{version.micro}")
    all_passed = all_passed and passed

    # Check 2: Required packages
    print_header("Required Packages")
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'yfinance', 'fredapi', 'streamlit', 'scipy', 'requests'
    ]
    for package in required_packages:
        try:
            __import__(package)
            print_result(f"Package: {package}", True)
        except ImportError:
            print_result(f"Package: {package}", False, "Run: pip install -r requirements.txt")
            all_passed = False

    # Check 3: Configuration
    print_header("Configuration")
    try:
        from config.settings import config
        print_result("Config loads", True)

        # Check API keys
        if config.FRED_API_KEY and config.FRED_API_KEY != 'your_fred_api_key_here':
            print_result("FRED API Key", True)
        else:
            print_result("FRED API Key", False, "Set FRED_API_KEY in .env file")
            all_passed = False

    except Exception as e:
        print_result("Config loads", False, str(e))
        all_passed = False

    # Check 4: Directories
    print_header("Directory Structure")
    from config.settings import config
    directories = [
        ('Cache', config.CACHE_DIR),
        ('Export', config.EXPORT_DIR),
        ('Logs', config.LOGS_DIR)
    ]
    for name, path in directories:
        exists = path.exists()
        print_result(f"{name} directory", exists, str(path))
        if not exists:
            path.mkdir(parents=True, exist_ok=True)
            print(f"       Created: {path}")

    # Check 5: API Connectivity
    print_header("API Connectivity")

    # Test FRED API
    try:
        from src.data_fetchers.fred_fetcher import FREDDataFetcher
        fetcher = FREDDataFetcher()
        test_data = fetcher.get_series_data('CPIAUCSL', '2024-01-01', '2024-01-31')
        if test_data is not None and not test_data.empty:
            print_result("FRED API connection", True)
        else:
            print_result("FRED API connection", False, "No data returned")
            all_passed = False
    except Exception as e:
        print_result("FRED API connection", False, str(e))
        all_passed = False

    # Test Yahoo Finance
    try:
        from src.data_fetchers.yahoo_fetcher import YahooDataFetcher
        fetcher = YahooDataFetcher()

        # Check if markets are currently open
        from datetime import datetime
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)

        # Check if it's a weekday and within market hours
        is_weekday = now_et.weekday() < 5  # Monday-Friday
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = market_open <= now_et <= market_close

        if is_weekday and is_market_hours:
            # Markets are open - test should work
            current_price = fetcher.get_current_price('SPY')
            if current_price and 'current_price' in current_price and current_price['current_price'] is not None:
                print_result("Yahoo Finance connection", True)
            else:
                print_result("Yahoo Finance connection", False, "No data returned during market hours")
                all_passed = False
        else:
            # Markets are closed - this is expected
            market_status = "outside market hours" if is_weekday else "on weekend"
            print_result("Yahoo Finance connection", True,
                        f"Markets closed ({market_status}) - this is expected ✅")
            print("       Note: Yahoo Finance real-time data only available 9:30 AM - 4:00 PM ET weekdays")

    except Exception as e:
        print_result("Yahoo Finance connection", False, str(e))
        all_passed = False

    # Final Summary
    print_header("Validation Summary")
    if all_passed:
        print("\n🎉 SUCCESS! All checks passed. You're ready to use the tracker!")
        print("\nNext steps:")
        print("  1. Run CLI: python main.py --list-indicators")
        print("  2. Run dashboard: streamlit run streamlit_app.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see TROUBLESHOOTING.md")
        return 1

if __name__ == '__main__':
    sys.exit(main())
