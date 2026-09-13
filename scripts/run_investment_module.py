"""Run one investment platform module and refresh dashboard artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from investment_platform.pipeline import run_daily_research
from investment_platform.market_data_refresh import project_market_data_status, refresh_from_project_config
from investment_platform.static_site import build_static_dashboard


MODULES = {
    "all": "全量刷新",
    "regime": "市场环境与动态因子",
    "portfolio": "IBKR持仓",
    "watchlist": "Moomoo自选股",
    "ranking": "股票排名",
    "stock_details": "单股详情和日K",
    "thesis": "投资逻辑追踪",
    "discovery": "主题潜力股雷达",
    "catalysts": "事件催化",
    "earnings": "财报日历与解析",
    "news": "美联储和国际新闻",
    "smart_money": "聪明钱追踪",
    "sentiment": "市场情绪",
    "valuation": "情景估值",
    "trade_plan": "买入/卖出点位计划",
    "sizing": "仓位管理",
    "allocation": "现金约束仓位计划",
    "behavior": "个人行为分析",
    "industry": "行业竞争地图",
    "integrations": "开源引擎状态",
    "data_sources": "数据源状态",
    "research_stack": "开源框架接入蓝图",
    "search_logic": "搜索逻辑工作台",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an investment platform module")
    parser.add_argument("--module", default="all", choices=sorted(MODULES), help="Module to run")
    parser.add_argument("--config", default="investment_platform.json", help="Path to platform config")
    args = parser.parse_args()

    market_before = project_market_data_status(PROJECT_ROOT)
    market_refresh = {"refreshed": False, "reason": "already_current", **market_before}
    if market_before.get("stale"):
        try:
            market_refresh = refresh_from_project_config(PROJECT_ROOT)
        except Exception as exc:
            market_refresh = {**market_before, "refreshed": False, "error": str(exc)}

    snapshot = run_daily_research(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / "results" / "investment_platform"
    latest_path = output_dir / "latest_snapshot.json"
    latest_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    site_path = PROJECT_ROOT / "site" / "index.html"
    build_static_dashboard(snapshot, site_path)

    module_path = output_dir / f"module_{args.module}.json"
    payload = {
        "ok": True,
        "module": args.module,
        "module_label": MODULES[args.module],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "market_refresh": market_refresh,
        "site_path": str(site_path),
        "latest_snapshot_path": str(latest_path),
    }
    module_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
