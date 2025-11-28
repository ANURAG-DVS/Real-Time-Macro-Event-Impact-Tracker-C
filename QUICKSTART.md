# ⚡ 5-Minute Quick Start

## Step 1: Get FRED API Key (2 minutes)

1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill form → Get key instantly

## Step 2: Setup Project (2 minutes)
```bash
git clone https://github.com/ANURAG-DVS/Real-Time-Macro-Event-Impact-Tracker-C.git
cd macro-event-tracker
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your FRED API key to .env file
```

## Step 3: Run First Analysis (1 minute)
```bash
python main.py --indicator CPI --plot --export
```

## Step 4: View Dashboard
```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501

## 🎉 Done!

You now have a professional market analysis tool running!

## Next Steps

- Try different indicators: `python main.py --list-indicators`
- Run historical analysis
- Customize the dashboard
- Deploy to Streamlit Cloud
