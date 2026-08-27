# Demo Data Snapshot

This directory contains a lightweight public demo snapshot used when the full
local research outputs are not present, such as on Streamlit Community Cloud.

The snapshot is derived from public market data and generated research outputs.
It is intentionally smaller than the full local dataset so the deployed demo can
load quickly. The equity feature snapshot covers 2023-01-03 through 2026-08-26;
the complete research pipeline can still be reproduced locally by running:

```bash
python main.py
```

The dashboard first looks for local files under `data/processed/` and `results/`.
If those files are absent, it falls back to this demo snapshot.
