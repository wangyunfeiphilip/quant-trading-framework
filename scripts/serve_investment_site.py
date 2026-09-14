"""Local dashboard server with safe per-module run buttons."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = PROJECT_ROOT / "site"
MODULE_SCRIPT = PROJECT_ROOT / "scripts" / "run_investment_module.py"
ALLOWED_MODULES = {
    "all",
    "regime",
    "portfolio",
    "watchlist",
    "ranking",
    "stock_details",
    "thesis",
    "discovery",
    "catalysts",
    "earnings",
    "news",
    "smart_money",
    "sentiment",
    "valuation",
    "trade_plan",
    "sizing",
    "allocation",
    "behavior",
    "industry",
    "integrations",
    "data_sources",
    "research_stack",
    "search_logic",
}


class InvestmentSiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        module = parse_qs(parsed.query).get("module", ["all"])[0]
        if module not in ALLOWED_MODULES:
            self._send_json({"ok": False, "error": f"不允许运行模块：{module}"}, HTTPStatus.BAD_REQUEST)
            return
        command = [sys.executable, str(MODULE_SCRIPT), "--module", module, "--config", "investment_platform.json"]
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._send_json({"ok": False, "error": "模块运行超时"}, HTTPStatus.REQUEST_TIMEOUT)
            return
        if completed.returncode != 0:
            self._send_json({"ok": False, "error": completed.stderr or completed.stdout}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {"ok": True, "module": module}
        payload["message"] = f"{payload.get('module_label', module)}已运行完成"
        self._send_json(payload)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local investment dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8504)
    args = parser.parse_args()
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), InvestmentSiteHandler)
    print(f"Serving static AI Quant preview at http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
