# 📊 Macro Event Impact Tracker

> **Analyze how major economic releases (CPI, NFP, GDP, etc.) impact financial markets in real-time**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![100% Free](https://img.shields.io/badge/Cost-$0-brightgreen.svg)]()
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-success.svg)]()

## 🎯 Perfect For

- 📚 **Finance Students** building portfolio projects
- 💼 **Job Seekers** demonstrating market knowledge and coding skills
- 📈 **Analysts** exploring macro-market relationships
- 🎓 **Researchers** studying event impacts

## ✨ Features

- ✅ **Real-time Event Analysis** - Track market reactions minute-by-minute
- ✅ **Multi-Asset Tracking** - Equities, bonds, commodities, FX, volatility
- ✅ **Historical Analysis** - Analyze trends across multiple releases
- ✅ **Interactive Dashboard** - Beautiful Streamlit web interface
- ✅ **Professional Visualizations** - Publication-quality charts and plots
- ✅ **100% FREE** - No credit card, no hidden costs

## 💰 Cost Breakdown

| Component | Cost |
|-----------|------|
| FRED API | $0 (free forever) |
| Yahoo Finance | $0 (no signup needed) |
| Python & Libraries | $0 (open source) |
| Streamlit Deployment | $0 (free tier) |
| **TOTAL** | **$0** 🎉 |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- FRED API key (free, 2-minute signup)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/macro-event-tracker.git
cd macro-event-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your FRED API key
```

### Get FREE FRED API Key

1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill out simple form (takes 2 minutes)
4. Copy your key to `.env` file

No credit card required!

## 📖 Usage

### Command Line Interface
```bash
# Analyze latest CPI release
python main.py --indicator CPI --plot --export

# Analyze specific NFP release
python main.py --indicator NFP --date 2024-01-05 --hours-after 8

# Historical analysis
python main.py --indicator GDP --mode historical \
  --start-date 2023-01-01 --end-date 2024-01-01 --plot

# List all available indicators
python main.py --list-indicators
```

### Web Dashboard
```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser!

## 📊 Available Indicators

| Code | Name | Frequency |
|------|------|-----------|
| CPI | Consumer Price Index | Monthly |
| CORE_CPI | Core CPI (ex Food & Energy) | Monthly |
| NFP | Non-Farm Payrolls | Monthly |
| UNEMPLOYMENT | Unemployment Rate | Monthly |
| GDP | Gross Domestic Product | Quarterly |
| PCE | Personal Consumption Expenditures | Monthly |
| RETAIL_SALES | Retail Sales | Monthly |

## 🎨 Sample Output

### Event Analysis
```
════════════════════════════════════════════════════════════
MACRO EVENT IMPACT ANALYSIS
════════════════════════════════════════════════════════════
Indicator: CPI (Consumer Price Index)
Release Date: 2024-01-11 08:30:00 EST
Value: 3.4%
Change from Previous: +0.3%

MARKET REACTIONS (1 Hour After Release):
────────────────────────────────────────────────────────────
SPY (S&P 500 ETF):         -0.85%  ⬇️
QQQ (Nasdaq 100 ETF):      -1.23%  ⬇️
TLT (20Y Treasury ETF):    +0.67%  ⬆️
GLD (Gold ETF):            +1.12%  ⬆️
UUP (US Dollar ETF):       +0.45%  ⬆️
VXX (Volatility ETF):      +8.34%  ⬆️⬆️

SECTOR SUMMARY:
────────────────────────────────────────────────────────────
Equities:      -1.04% (Risk-off)
Bonds:         +0.67% (Flight to safety)
Commodities:   +1.12% (Inflation hedge)
Volatility:    +8.34% (Fear spike)
════════════════════════════════════════════════════════════
```

## 🏗️ Project Structure
```
macro-event-tracker/
├── config/
│   └── settings.py          # Configuration and API keys
├── src/
│   ├── data_fetchers/
│   │   ├── fred_fetcher.py  # FRED API integration
│   │   └── yahoo_fetcher.py # Yahoo Finance integration
│   ├── analyzers/
│   │   └── impact_analyzer.py # Core analysis logic
│   ├── visualizers/
│   │   └── plotter.py       # Chart generation
│   └── utils/
│       ├── helpers.py       # Utility functions
│       └── cache_manager.py # Data caching
├── data/
│   ├── cache/               # Cached API responses
│   └── exports/             # Analysis results
├── notebooks/
│   └── demo_analysis.ipynb  # Jupyter examples
├── main.py                  # CLI application
├── streamlit_app.py         # Web dashboard
└── requirements.txt         # Dependencies
```

## 🎓 Learning Outcomes

By building/using this project, you'll demonstrate:

✅ API integration skills (RESTful APIs)  
✅ Data manipulation with pandas  
✅ Financial market knowledge  
✅ Statistical analysis capabilities  
✅ Data visualization proficiency  
✅ Software engineering best practices  
✅ Real-world problem solving  

## 📈 Deployment (FREE)

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Visit https://streamlit.io/cloud
3. Connect your GitHub account
4. Select your repository
5. Add FRED_API_KEY in Secrets
6. Deploy!

Your app will be live at: `yourapp.streamlit.app`

## ⚠️ Limitations (Free Tier)

### Yahoo Finance (yfinance):
- Intraday data: Last 7 days only
- Rate limiting: 1-2 requests/second
- Occasional data gaps (handled automatically)

### FRED API:
- Rate limit: 120 requests/minute (more than enough!)

**These limitations don't affect the value of your project for recruiters!**

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## 📝 License

MIT License - Free to use for personal and commercial projects

## 🙏 Acknowledgments

- FRED (Federal Reserve Economic Data) for free economic data
- Yahoo Finance for free market data
- Streamlit for free hosting
- The open source community

## 📧 Contact

Questions? Open an issue on GitHub!

---

**⭐ If this project helped you, please star the repository!**
