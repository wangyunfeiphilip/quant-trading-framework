"""Validate a research output directory before it is published."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.validation import validate_research_outputs, write_validation_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = validate_research_outputs(args.root)
    report_path = args.report or (args.root / "results" / "research_validation.json")
    write_validation_report(report, report_path)
    print(f"Research validation: {report['status']}")
    for message in report["errors"]:
        print(f"ERROR: {message}")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
