"""Build a dependency-light static research demo for GitHub Pages."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
from datetime import datetime, timezone


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_series_csv(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    return {row.get("", ""): row.get("0", "") for row in rows}


def percent(value: str | float | None) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "NA"


def number(value: str | float | None, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def build_equity_svg(rows: list[dict[str, str]]) -> str:
    points = []
    for row in rows:
        try:
            points.append((row["date"], float(row["total_value"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(points) < 2:
        return '<div class="empty">No equity curve available.</div>'

    max_points = 360
    step = max(1, len(points) // max_points)
    points = points[::step]
    final_point = (rows[-1]["date"], float(rows[-1]["total_value"])) if rows else None
    if final_point is not None and points[-1] != final_point:
        points.append(final_point)
    values = [value for _, value in points]
    low, high = min(values), max(values)
    span = high - low or 1.0
    width, height = 960, 280
    coordinates = []
    for index, (_date, value) in enumerate(points):
        x = 18 + (width - 36) * index / max(1, len(points) - 1)
        y = 18 + (height - 36) * (1 - (value - low) / span)
        coordinates.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    area = f"18,{height - 18} {line} {width - 18},{height - 18}"
    return f"""
    <svg class="equity-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio equity curve">
      <defs>
        <linearGradient id="curveFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stop-color="#77f2d3" stop-opacity=".28" />
          <stop offset="1" stop-color="#77f2d3" stop-opacity="0" />
        </linearGradient>
      </defs>
      <line x1="18" y1="{height - 18}" x2="{width - 18}" y2="{height - 18}" stroke="#33434a" />
      <polygon points="{escape(area)}" fill="url(#curveFill)" />
      <polyline points="{escape(line)}" fill="none" stroke="#77f2d3" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    """


def build_html(data_root: Path, streamlit_url: str) -> str:
    results = data_root / "results"
    manifest_path = results / "backtest_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    data = manifest.get("data", {})
    backtest = manifest.get("backtest", {})
    validation = manifest.get("validation", {})
    performance = read_series_csv(results / "performance_summary.csv")
    portfolio_rows = read_csv_rows(results / "portfolio_value.csv")
    findings_path = results / "key_findings.md"
    findings = []
    if findings_path.exists():
        findings = [
            line[2:].strip()
            for line in findings_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
    cutoff = data.get("end_date") or (portfolio_rows[-1].get("date") if portfolio_rows else "NA")
    generated = manifest.get("generated_at_utc", "NA")
    warning_text = " | ".join(validation.get("warnings", []))
    finding_items = "".join(f"<li>{escape(item)}</li>" for item in findings)
    artifact_hash = data.get("configuration_sha256", "not available")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quant Research Terminal | Public Demo</title>
  <style>
    :root {{ --bg:#101617; --panel:#172124; --panel2:#1d2a2d; --ink:#f2f6f2; --muted:#a9b8b5; --teal:#77f2d3; --gold:#e7bf69; --line:rgba(225,245,239,.13); }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at 82% -10%,#24413d 0,transparent 31%),var(--bg); color:var(--ink); font:16px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:30px 22px 70px; }}
    nav {{ display:flex; justify-content:space-between; align-items:center; gap:18px; margin-bottom:74px; }} .brand {{ letter-spacing:.12em; font-size:.76rem; font-weight:800; color:var(--teal); }}
    nav a, .button {{ color:var(--ink); text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:9px 15px; background:rgba(255,255,255,.04); }} .button {{ background:var(--teal); color:#071412; border-color:var(--teal); font-weight:800; }}
    .hero {{ display:grid; grid-template-columns:1.15fr .85fr; gap:46px; align-items:end; margin-bottom:58px; }} .eyebrow {{ color:var(--teal); font-weight:800; letter-spacing:.14em; font-size:.78rem; text-transform:uppercase; }}
    h1 {{ font-size:clamp(3rem,8vw,7.2rem); line-height:.92; letter-spacing:-.06em; max-width:740px; margin:17px 0 25px; }} h1 span {{ color:var(--gold); }} .lede {{ color:var(--muted); font-size:1.15rem; max-width:660px; }}
    .terminal {{ border:1px solid var(--line); border-radius:18px; padding:24px; background:linear-gradient(145deg,rgba(255,255,255,.08),rgba(255,255,255,.025)); }} .terminal h2 {{ margin:0 0 20px; font-size:1rem; color:var(--teal); letter-spacing:.12em; }} .terminal-row {{ display:grid; grid-template-columns:100px 1fr; gap:18px; padding:16px 0; border-top:1px solid var(--line); color:var(--muted); }} .terminal-row strong {{ color:var(--ink); }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:13px; margin-bottom:46px; }} .metric, .panel {{ border:1px solid var(--line); border-radius:15px; background:rgba(255,255,255,.045); }} .metric {{ padding:18px; }} .metric b {{ display:block; font-size:1.65rem; color:var(--teal); }} .metric small {{ color:var(--muted); }}
    .section-title {{ margin:48px 0 16px; }} .section-title h2 {{ margin:0; font-size:1.7rem; }} .section-title p {{ margin:3px 0; color:var(--muted); }} .panel {{ padding:22px; }} .equity-chart {{ width:100%; height:auto; display:block; }}
    .findings {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; padding:0; list-style:none; }} .findings li {{ border-left:2px solid var(--teal); padding:12px 15px; color:var(--muted); background:rgba(255,255,255,.03); }}
    .audit {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }} .audit div {{ padding:14px; background:var(--panel2); border-radius:10px; }} .audit span {{ display:block; color:var(--muted); font-size:.8rem; }} .audit strong {{ display:block; margin-top:4px; }} .note {{ color:var(--muted); font-size:.9rem; }} footer {{ margin-top:58px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem; }}
    @media(max-width:800px) {{ .hero {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} .findings, .audit {{ grid-template-columns:1fr; }} nav {{ margin-bottom:45px; }} }}
  </style>
</head>
<body>
<main>
  <nav><div class="brand">QUANT RESEARCH TERMINAL</div><div><a href="{escape(streamlit_url)}" class="button">Open interactive lab ↗</a></div></nav>
  <section class="hero"><div><div class="eyebrow">Public research snapshot</div><h1>Markets through <span>models, risk, and motion.</span></h1><p class="lede">A transparent quantitative research demo for equity strategies, robust factor attribution, data diagnostics, and derivatives experiments.</p></div>
    <div class="terminal"><h2>RESEARCH PIPELINE</h2><div class="terminal-row"><strong>INPUT</strong><span>Adjusted OHLCV, technical features, factor data</span></div><div class="terminal-row"><strong>ENGINE</strong><span>Signal-lagged portfolio simulation with costs and slippage</span></div><div class="terminal-row"><strong>OUTPUT</strong><span>Returns, drawdowns, factors, diagnostics, and audit metadata</span></div></div></section>
  <section class="metrics"><div class="metric"><b>{percent(performance.get('cumulative_return'))}</b><small>Cumulative return</small></div><div class="metric"><b>{number(performance.get('sharpe_ratio'))}</b><small>Sharpe ratio</small></div><div class="metric"><b>{percent(performance.get('maximum_drawdown'))}</b><small>Maximum drawdown</small></div><div class="metric"><b>{escape(str(cutoff))}</b><small>Data cutoff</small></div></section>
  <section class="section-title"><h2>Portfolio equity curve</h2><p>Deterministic public snapshot, marked in adjusted prices.</p></section><div class="panel">{build_equity_svg(portfolio_rows)}</div>
  <section class="section-title"><h2>Key findings</h2><p>Generated by the research pipeline, not hard-coded into the page.</p></section><ul class="findings">{finding_items or '<li>No findings file available.</li>'}</ul>
  <section class="section-title"><h2>Reproducibility audit</h2><p>The page makes its assumptions visible before the chart makes its claims.</p></section><div class="panel"><div class="audit"><div><span>Data source</span><strong>{escape(str(data.get('source', 'NA')))}</strong></div><div><span>Universe</span><strong>{escape(str(data.get('ticker_count', 'NA')))} names</strong></div><div><span>Signal lag</span><strong>{escape(str(backtest.get('signal_lag_sessions', 'NA')))} session</strong></div><div><span>Transaction cost</span><strong>{escape(str(backtest.get('transaction_cost_bps', 'NA')))} bps</strong></div><div><span>Slippage</span><strong>{escape(str(backtest.get('slippage_bps', 'NA')))} bps</strong></div><div><span>Run timestamp</span><strong>{escape(str(generated))}</strong></div></div><p class="note">{escape(warning_text or 'The manifest did not report additional warnings.')}</p><p class="note">Configuration fingerprint: {escape(str(artifact_hash))}</p></div>
  <footer>Educational research software. Historical results are not investment advice or a guarantee of future performance.</footer>
</main>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("demo_data"))
    parser.add_argument("--output", type=Path, default=Path("public_demo/dist"))
    parser.add_argument("--streamlit-url", default="https://wyf-quant-research-terminal.streamlit.app")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text(build_html(args.data_root, args.streamlit_url), encoding="utf-8")
    (args.output / "health.json").write_text(
        json.dumps({"status": "ok", "built_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Static demo written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
