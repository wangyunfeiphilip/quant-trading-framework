"""Run the personal AI investment research platform batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from investment_platform.pipeline import run_daily_research


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one investment research platform cycle")
    parser.add_argument("--config", default="investment_platform.json", help="Path to platform config JSON or YAML")
    args = parser.parse_args()

    result = run_daily_research(Path(args.config))
    print(result["report_path"])


if __name__ == "__main__":
    main()
