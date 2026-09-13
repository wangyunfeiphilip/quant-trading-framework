# Public Demo

This directory is built by `.github/workflows/public-demo.yml` and published to
GitHub Pages. The generated page is a static, public-facing research snapshot;
it does not include private holdings, broker credentials, or API keys.

Code pushes build from `demo_data/` so the public page remains reproducible.
Weekday scheduled runs rebuild the research outputs from public market data and
publish a fresh static snapshot when the data-quality checks and pipeline succeed.

The interactive Streamlit app may hibernate after 12 hours without traffic on
the free Community Cloud tier. The GitHub Pages snapshot is the stable interview
link; the Streamlit app is the interactive research lab. Neither layer should be
described as an always-on trading or execution service.
