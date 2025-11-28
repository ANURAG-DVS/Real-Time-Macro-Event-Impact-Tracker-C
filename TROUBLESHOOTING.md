# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. SSL Certificate Errors (macOS)

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution:**
```bash
pip install --upgrade certifi
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```

**Prevention:** Add to `~/.zshrc`:
```bash
echo "export SSL_CERT_FILE=\$(python3 -c \"import certifi; print(certifi.where())\")" >> ~/.zshrc
```

### 2. FRED API Connection Issues

**Error:** `FRED API initialization failed`

**Solutions:**
- ✅ Verify FRED API key is set in `.env` file
- ✅ Check internet connection
- ✅ Ensure API key is valid (no extra spaces)
- ✅ Try again (temporary API issues happen)

**Test API Key:**
```bash
python -c "from src.data_fetchers.fred_fetcher import FREDDataFetcher; FREDDataFetcher()"
```

### 3. Yahoo Finance Data Issues

**Error:** `No market data available for analysis`

**Common Causes & Solutions:**

**Market Closed:**
- US markets: 9:30 AM - 4:00 PM ET weekdays
- Solution: Wait for market hours or use historical data

**Data Age Limit:**
- Yahoo Finance FREE tier: Last 7 days only
- Solution: Use recent events or switch to daily data

**Rate Limiting:**
- Error: Too many requests
- Solution: Wait 1-2 minutes, reduce frequency

### 4. Import Errors (Apple Silicon)

**Error:** `ModuleNotFoundError` or architecture issues

**Solution:**
```bash
# Ensure you're using the right Python
python3 --version  # Should be 3.9+

# Reinstall packages
pip uninstall pandas numpy scipy yfinance fredapi
pip install --upgrade pip
pip install -r requirements.txt

# For Apple Silicon specifically
pip install --upgrade --force-reinstall --no-deps pandas
```

### 5. Virtual Environment Issues

**Error:** Packages not found after activation

**Solution:**
```bash
# Remove and recreate venv
rm -rf venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 6. Streamlit App Won't Start

**Error:** `ModuleNotFoundError` in Streamlit

**Solutions:**
- ✅ Ensure virtual environment is activated
- ✅ Check all dependencies are installed
- ✅ Try: `python -m streamlit run streamlit_app.py`
- ✅ Verify Python path is correct

### 7. DataFrame Display Issues

**Error:** `'ticker'` column not found

**Cause:** Market data fetching failed, resulting in empty DataFrame

**Solutions:**
- ✅ Check market hours (9:30 AM - 4:00 PM ET)
- ✅ Verify event date is within last 7 days
- ✅ Check internet connection
- ✅ Try different indicator or date

### 8. Memory/Performance Issues

**Error:** App runs slow or crashes

**Solutions:**
- ✅ Reduce analysis time window (hours_after parameter)
- ✅ Use fewer assets in analysis
- ✅ Clear cache: `rm -rf data/cache/*`
- ✅ Restart Streamlit app

### 9. Plotting Errors

**Error:** Charts not displaying

**Solutions:**
- ✅ Install matplotlib and plotly
- ✅ Check data is not empty before plotting
- ✅ Ensure proper timezone handling

### 10. Historical Analysis Issues

**Error:** No historical data found

**Solutions:**
- ✅ Check date range (not too far back)
- ✅ Verify indicator has data for that period
- ✅ Reduce date range if too large

## Diagnostic Commands

### Test FRED API
```bash
python -c "
from src.data_fetchers.fred_fetcher import FREDDataFetcher
fetcher = FREDDataFetcher()
print('FRED API: ✅ Working')
"
```

### Test Yahoo Finance
```bash
python -c "
from src.data_fetchers.yahoo_fetcher import YahooDataFetcher
fetcher = YahooDataFetcher()
print('Yahoo Finance: ✅ Working')
"
```

### Test Full Pipeline
```bash
python -c "
from src.analyzers.impact_analyzer import ImpactAnalyzer
analyzer = ImpactAnalyzer()
print('Full Pipeline: ✅ Working')
"
```

### Check Environment
```bash
python --version
pip list | grep -E "(pandas|numpy|scipy|yfinance|fredapi|streamlit)"
echo $SSL_CERT_FILE
```

## Getting Help

1. **Check Logs:** Look at console output for detailed error messages
2. **GitHub Issues:** Open an issue with full error message and steps to reproduce
3. **System Info:** Include your OS, Python version, and pip list output

## Prevention Tips

- ✅ Always activate virtual environment before running
- ✅ Keep dependencies updated: `pip install -r requirements.txt --upgrade`
- ✅ Test after system updates (especially macOS)
- ✅ Use recent event dates for best results
- ✅ Check market hours before analysis

## Quick Fixes

| Issue | Quick Command |
|-------|---------------|
| SSL Issues | `pip install --upgrade certifi` |
| Import Errors | `pip install -r requirements.txt --force-reinstall` |
| Cache Issues | `rm -rf data/cache/*` |
| Port Conflicts | `streamlit run streamlit_app.py --server.port 8502` |

---

**Still having issues?** Please include:
- Full error message
- Your operating system
- Python version (`python --version`)
- Output of `pip list | head -20`
