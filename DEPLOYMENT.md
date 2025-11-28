# 🚀 Deployment Guide

## Option 1: Streamlit Cloud (Recommended - FREE)

### Step-by-Step:

1. **Prepare Repository**
```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
```

2. **Sign up for Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub

3. **Deploy App**
   - Click "New app"
   - Select your repository
   - Main file: `streamlit_app.py`
   - Click "Deploy"

4. **Add Secrets**
   - In app settings, go to "Secrets"
   - Add:
```toml
     FRED_API_KEY = "your_key_here"
```

5. **Done!** Your app is live at `yourapp.streamlit.app`

### Troubleshooting:

- **App won't start**: Check requirements.txt is complete
- **API errors**: Verify FRED_API_KEY in secrets
- **Slow loading**: Enable caching in code (already done)

## Option 2: Local Deployment
```bash
streamlit run streamlit_app.py --server.port 8501
```

Access at: http://localhost:8501

## Option 3: Docker (Advanced)

Coming soon!

## 📊 Resource Usage (Streamlit Free Tier)

- Memory: ~200MB (well within 1GB limit)
- CPU: Minimal
- Storage: Cached data ~50MB
- **Verdict**: Perfect fit for free tier! ✅
