# Deployment Guide

## Deployment Choice

The project is a Streamlit-based Python application.

- Frontend: Streamlit components rendered from `app.py`
- Backend: the same Streamlit Python process
- Database: none
- Persistent data: CSV, Markdown, and generated research files
- External APIs: public market data through `yfinance`; no API key is required
- Entry point: `app.py`

The simplest production demo target is Streamlit Community Cloud because it can
run this repository directly from GitHub with `requirements.txt` and `app.py`.
Vercel is not appropriate for the current app because the project is not a
static frontend or serverless JavaScript application. Render and Railway can run
it, but they require more deployment plumbing than Streamlit Cloud.

## Files Prepared for Deployment

```text
app.py
requirements.txt
.streamlit/config.toml
.streamlit/secrets.toml.example
demo_data/
```

`demo_data/` is a lightweight public snapshot. The app first reads local
research outputs from `data/processed/` and `results/`. If those files are absent
in a cloud deployment, it falls back to `demo_data/` so the public demo is not
blank.

## Security

No private API keys are required for the public demo.

Do not commit:

```text
.env
*.env
.streamlit/secrets.toml
```

If future data vendors, broker APIs, or database credentials are added, store
them in Streamlit Cloud app secrets and access them through `st.secrets` or
environment variables.

## Local Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

To regenerate the full local research outputs:

```bash
python main.py
```

## Streamlit Community Cloud Deployment

1. Push the latest branch to GitHub.
2. Open Streamlit Community Cloud.
3. Click **Create app** or **New app**.
4. Choose the GitHub repository:

```text
wangyunfeiphilip/quant-trading-framework
```

5. Choose the branch that contains the dashboard changes.
6. Set the main file path:

```text
app.py
```

7. Select Python 3.11 in advanced settings if Streamlit asks for a Python version.
8. Keep app secrets empty for this public demo.
9. Deploy.

The public URL is created by Streamlit Cloud after deployment. You can set the
app slug to a readable name such as:

```text
quant-research-terminal
```

The final public URL will look like:

```text
https://quant-research-terminal.streamlit.app
```

## Expected Public Demo Behavior

Visitors can open the public link without logging in or installing code.

They can:

- view the Chinese dashboard
- inspect the demo portfolio performance and risk tables
- search project tickers and quant concepts
- use the stock explorer with built-in demo data
- query an external Yahoo Finance ticker when the provider is available
- run strategy backtests against the demo dataset
- use the derivatives pricing lab

External ticker lookup depends on Yahoo Finance availability and may be
rate-limited. The public demo remains usable through the bundled `demo_data/`
snapshot even if a live market-data request fails.
