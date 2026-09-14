"""Refresh website-facing investment platform artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from investment_platform.pipeline import run_daily_research
from investment_platform.static_site import build_static_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh website summary data")
    parser.add_argument("--config", default="investment_platform.json", help="Path to platform config JSON or YAML")
    args = parser.parse_args()

    result = run_daily_research(Path(args.config))
    output_dir = PROJECT_ROOT / "results" / "investment_platform"
    latest_path = output_dir / "latest_snapshot.json"
    latest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    site_path = PROJECT_ROOT / "site" / "index.html"
    build_static_dashboard(result, site_path)
    print(site_path)


if __name__ == "__main__":
    main()
