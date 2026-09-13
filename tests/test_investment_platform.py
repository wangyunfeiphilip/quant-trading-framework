from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from investment_platform.allocation import build_allocation_plan
from investment_platform.earnings import build_earnings_reviews, enrich_earnings_reviews_with_news
from investment_platform.events import build_earnings_calendar
from investment_platform.integrations import integration_status
from investment_platform.ibkr_inputs import load_ibkr_positions
from investment_platform.market_regime import MarketRegimeDetector
from investment_platform.moomoo_inputs import build_watchlists_from_moomoo, load_moomoo_watchlist
from investment_platform.pipeline import _write_earnings_review_files
from investment_platform.position_sizing import PositionSizingEngine
from investment_platform.predictions import PredictionTracker, PredictionRecord
from investment_platform.static_site import _search_index
from investment_platform.thesis import ThesisTracker, build_thesis
from investment_platform.trade_plan import build_trade_plans
from investment_platform.valuation import build_scenarios, summarize_valuation


def test_market_regime_detects_ai_growth_environment() -> None:
    regime = MarketRegimeDetector().detect(
        {
            "vix": 16,
            "nasdaq_200dma_pct": 0.10,
            "market_breadth": 0.65,
            "credit_spread": 1.0,
            "yield_change_30d_bps": 5,
            "fed_policy_bias": "neutral",
        }
    )
    assert regime.name == "AI Growth Environment"
    assert regime.confidence >= 0.70


def test_thesis_tracker_flags_invalidation_condition() -> None:
    thesis = build_thesis(
        {
            "ticker": "NVDA",
            "thesis": "Data Center growth stays strong",
            "why_follow": [],
            "supporting_evidence": [],
            "key_risks": [],
            "invalidation_conditions": [
                {"metric": "data_center_growth", "operator": "<", "threshold": 0.20, "message": "Growth too low"}
            ],
        }
    )
    evaluated = ThesisTracker().evaluate([thesis], {"NVDA": {"data_center_growth": 0.12}})
    assert evaluated[0].status == "weakened"
    assert evaluated[0].triggered_conditions == ["Growth too low"]


def test_prediction_tracker_scores_due_prediction() -> None:
    prediction = PredictionRecord(
        prediction_id="p1",
        as_of=date(2026, 1, 1),
        ticker="AAPL",
        bullish_probability=0.70,
        horizon_days=30,
    )
    outcomes = PredictionTracker().evaluate(
        [prediction],
        date(2026, 1, 31),
        {"AAPL": {"2026-01-01": 100.0, "2026-01-31": 110.0}},
    )
    assert outcomes[0].actual_return == 0.10
    assert outcomes[0].correct


def test_scenario_valuation_probability_weighted_value() -> None:
    scenarios = build_scenarios(
        [
            {"name": "Bull", "fair_value": 150, "probability": 0.25},
            {"name": "Base", "fair_value": 100, "probability": 0.50},
            {"name": "Bear", "fair_value": 50, "probability": 0.25},
        ]
    )
    summary = summarize_valuation("TEST", scenarios, current_price=80)
    assert summary.weighted_fair_value == 100
    assert round(summary.upside_to_price or 0, 2) == 0.25


def test_position_sizing_caps_concentrated_high_risk_position() -> None:
    rec = PositionSizingEngine().recommend(
        ticker="NVDA",
        score=92,
        current_allocation=0.18,
        volatility=0.46,
        max_drawdown=-0.40,
        sector_exposure=0.50,
        average_correlation=0.80,
    )
    assert rec.max_allocation <= 0.08
    assert rec.suggested_action == "trim_or_hold"


def test_integration_status_reports_optional_engines() -> None:
    statuses = integration_status()
    names = {item.name for item in statuses}
    assert {"Qlib", "OpenBB", "Riskfolio-Lib", "TradingAgents"}.issubset(names)


def test_moomoo_watchlist_loader_normalizes_export() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "watchlist.json"
        path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "code": "US.NVDA",
                            "name": "NVIDIA",
                            "groups": ["All", "自选"],
                            "last_price": 216.85,
                            "technical": {
                                "supports": [{"low": 211.9, "high": 216.8, "center": 214.3, "strength": "强"}],
                                "resistances": [{"low": 220.2, "high": 221.9, "center": 221.0, "strength": "小"}],
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        rows = load_moomoo_watchlist([str(path)])
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["last_price"] == 216.85
    assert rows[0]["supports"][0]["strength"] == "强"
    assert build_watchlists_from_moomoo(rows)[0]["tickers"] == ["NVDA"]


def test_ibkr_position_loader_computes_allocations() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "ibkr_snapshot.py"
        path.write_text(
            "POSITIONS = ["
            "{'symbol': 'NVDA', 'qty': 1, 'avg': 100, 'ibkr_px': 120, 'mv': 120, 'upl': 20},"
            "{'symbol': 'SGOV', 'qty': 1, 'avg': 100, 'ibkr_px': 80, 'mv': 80, 'upl': -20}"
            "]",
            encoding="utf-8",
        )
        rows = load_ibkr_positions(str(path))
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["allocation"] == 0.6
    assert rows[1]["source"] == "ibkr_local_snapshot"


def test_allocation_plan_keeps_cash_buffer_before_buying() -> None:
    portfolio = [
        {"ticker": "SCHD", "allocation": 0.20, "sector": "Dividend"},
        {"ticker": "AVGO", "allocation": 0.02, "sector": "Semiconductors"},
    ]
    sizing = [
        PositionSizingEngine().recommend("SCHD", 55, 0.20, 0.15, -0.18, 0.20, 0.30),
        PositionSizingEngine().recommend("AVGO", 85, 0.02, 0.34, -0.29, 0.20, 0.50),
    ]
    policy = {
        "risk_profile": "中等风险",
        "minimum_cash_weight": 0.08,
        "target_daily_volatility": {"min": 0.02, "max": 0.03},
        "buckets": [
            {"name": "现金", "target": 0.08, "daily_volatility": 0.0, "tickers": {}},
            {"name": "红利/防守质量", "target": 0.12, "daily_volatility": 0.010, "tickers": {"SCHD": 0.12}},
            {"name": "AI硬件/半导体", "target": 0.10, "daily_volatility": 0.026, "tickers": {"AVGO": 0.10}},
        ],
    }
    plan = build_allocation_plan(
        portfolio,
        sizing,
        {"SCHD": 55.0, "AVGO": 85.0},
        {"net_liquidation": 10_000, "available_funds": 300},
        policy,
    )
    assert plan["cash_shortfall"] == 0
    assert plan["rebalance_buying_power"] == 300
    assert not any(item["side"] == "reserve_cash" for item in plan["action_plan"])
    schd = next(item for item in plan["rows"] if item["ticker"] == "SCHD")
    assert schd["action"] == "cash_reserve_hold"
    avgo = next(item for item in plan["rows"] if item["ticker"] == "AVGO")
    assert avgo["feasible_trade_value"] == 300


def test_trade_plan_uses_levels_and_refuses_missing_data() -> None:
    details = [
        {
            "ticker": "AVGO",
            "price": 364.0,
            "portfolio": {"allocation": 0.02},
            "supports": [{"low": 351.0, "high": 362.5, "center": 357.8, "labels": ["20日低"]}],
            "resistances": [{"low": 368.0, "high": 371.5, "center": 369.5, "labels": ["MA200"]}],
            "valuation": {"weighted_fair_value": 434.5, "upside_to_price": 0.19},
        },
        {"ticker": "MRNA", "price": None, "supports": [], "resistances": [], "valuation": {}},
    ]
    plan = build_trade_plans(
        details,
        {
            "rows": [
                {"ticker": "AVGO", "action": "add_if_cash_available", "current_allocation": 0.02, "target_allocation": 0.07}
            ]
        },
        [{"ticker": "MRNA", "theme": "癌症疫苗", "next_check": "临床数据"}],
    )
    avgo = next(item for item in plan if item["ticker"] == "AVGO")
    mrna = next(item for item in plan if item["ticker"] == "MRNA")
    assert avgo["buy_zone"].startswith("$351.00")
    assert avgo["stop_loss"].startswith("$")
    assert avgo["sell_zone"].startswith("$")
    assert mrna["decision"] == "no_trade_until_data"


def test_trade_plan_does_not_sell_good_stock_only_because_position_is_high() -> None:
    details = [
        {
            "ticker": "NVDA",
            "price": 100.0,
            "score": 86.0,
            "portfolio": {"allocation": 0.12},
            "supports": [{"low": 90.0, "high": 95.0, "center": 92.5, "labels": ["20日低"]}],
            "resistances": [{"low": 110.0, "high": 116.0, "center": 113.0, "labels": ["20日高"]}],
            "valuation": {"weighted_fair_value": 135.0, "upside_to_price": 0.35},
            "thesis": {"status": "active"},
            "sentiment": {"label": "Fundamental driven", "speculation_risk": "normal"},
            "trend": "短线中性",
        }
    ]
    plan = build_trade_plans(
        details,
        {
            "rows": [
                {
                    "ticker": "NVDA",
                    "action": "trim_to_target",
                    "current_allocation": 0.12,
                    "target_allocation": 0.08,
                    "max_allocation": 0.10,
                }
            ]
        },
        [],
    )
    assert plan[0]["decision"] == "hold_winner_no_forced_trim"
    assert plan[0]["problem_area"] == "仓位约束"
    assert "不机械减仓" in plan[0]["reason"]


def test_dashboard_search_index_preserves_a_share_code() -> None:
    index = _search_index([{"ticker": "301321", "code": "SZ.301321", "name": "测试A股", "groups": ["A股观察"]}])
    assert index[0]["ticker"] == "301321"
    assert "SZ.301321" in index[0]["aliases"]


def test_earnings_reviews_score_reported_watchlist_names() -> None:
    calendar = [
        {
            "ticker": "NVDA",
            "company": "NVIDIA",
            "date": "2026-08-26",
            "time": "盘后",
            "status": "reported_needs_analysis",
            "watch_items": ["Data Center增速", "毛利率", "指引"],
        },
        {"ticker": "AVGO", "company": "Broadcom", "date": "2026-09-04", "status": "upcoming"},
    ]
    reviews = build_earnings_reviews(
        calendar,
        [
            {
                "ticker": "NVDA",
                "evidence_grade": "A",
                "metrics": {
                    "revenue_growth": 0.21,
                    "eps_growth": 0.24,
                    "gross_margin_change_bps": 140,
                    "free_cash_flow_margin": 0.28,
                    "guidance_signal": "raise",
                    "management_tone": "strong",
                    "risk_language": "neutral",
                },
            }
        ],
        {"NVDA", "AVGO"},
        date(2026, 8, 27),
    )
    assert len(reviews) == 1
    assert reviews[0]["ticker"] == "NVDA"
    assert reviews[0]["status"] == "analyzed"
    assert reviews[0]["verdict"] == "好于预期"
    assert reviews[0]["quality_score"] >= 72


def test_earnings_reviews_queue_missing_original_data_without_fabricating_verdict() -> None:
    reviews = build_earnings_reviews(
        [{"ticker": "NVDA", "company": "NVIDIA", "date": "2026-08-26", "status": "reported_needs_analysis"}],
        [],
        {"NVDA"},
        date(2026, 8, 27),
    )
    assert reviews[0]["status"] == "queued"
    assert reviews[0]["verdict"] == "待原始数据"
    assert reviews[0]["investment_value"] == "不下投资结论"


def test_earnings_reviews_do_not_queue_same_day_events_without_source_data() -> None:
    reviews = build_earnings_reviews(
        [{"ticker": "MRVL", "company": "Marvell", "date": "2026-08-27", "status": "today"}],
        [],
        {"MRVL"},
        date(2026, 8, 27),
    )
    assert reviews == []


def test_earnings_calendar_dedupes_configured_and_auto_rows() -> None:
    rows = build_earnings_calendar(
        [
            {"ticker": "MRVL", "company": "Marvell", "date": "2026-08-27", "source": "configured"},
            {"ticker": "MRVL", "company": "Marvell Technology", "date": "2026-08-27", "source": "nasdaq"},
        ],
        date(2026, 8, 27),
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "configured"


def test_earnings_reviews_include_moomoo_news_links() -> None:
    reviews = build_earnings_reviews(
        [{"ticker": "NVDA", "company": "NVIDIA", "date": "2026-08-26", "status": "reported_needs_analysis"}],
        [],
        {"NVDA"},
        date(2026, 8, 27),
    )
    enriched = enrich_earnings_reviews_with_news(
        reviews,
        {"NVDA": [{"title": "NVIDIA earnings call highlights", "publish_time": "2026-08-27 10:00:00", "url": "https://example.com"}]},
    )
    assert enriched[0]["moomoo_news"][0]["title"] == "NVIDIA earnings call highlights"
    assert "Moomoo" in enriched[0]["source_note"]


def test_earnings_reviews_treat_reported_metrics_as_analyzed() -> None:
    reviews = build_earnings_reviews(
        [{"ticker": "NVDA", "company": "NVIDIA", "date": "2026-08-26", "status": "reported_needs_analysis"}],
        [
            {
                "ticker": "NVDA",
                "reported_metrics": [{"metric": "Revenue", "actual": "$46.7B", "estimate": "$46.1B", "change": "+6% QoQ"}],
                "call_highlights": ["管理层确认 Blackwell 需求仍强"],
                "source_documents": ["NVIDIA FY2Q27 earnings release"],
            }
        ],
        {"NVDA"},
        date(2026, 8, 27),
    )
    assert reviews[0]["status"] == "analyzed"
    assert reviews[0]["quality_score"] is not None


def test_existing_real_earnings_report_is_not_overwritten_by_queue_stub() -> None:
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        review_dir = output_dir / "earnings_reviews"
        review_dir.mkdir(parents=True)
        path = review_dir / "NVDA_earnings_review_2026-08-27.md"
        path.write_text(
            "\n".join(
                [
                    "# NVDA 财报解析 - 2026-08-27",
                    "- 财报好坏：好于预期",
                    "- 质量分：83",
                    "- 投资逻辑影响：强化",
                    "- 投资价值：投资价值提高",
                    "- 操作建议：持有，接近支撑再加",
                    "- 证据等级：A",
                    "- 来源说明：本地一手资料精读",
                    "",
                    "## 核心结论",
                    "- 数据中心仍是核心驱动",
                ]
            ),
            encoding="utf-8",
        )
        written = _write_earnings_review_files(
            output_dir,
            date(2026, 8, 27),
            [
                {
                    "ticker": "NVDA",
                    "company": "NVIDIA",
                    "date": "2026-08-26",
                    "status": "queued",
                    "verdict": "待原始数据",
                    "quality_score": None,
                    "thesis_impact": "待确认",
                    "investment_value": "不下投资结论",
                    "action": "等待财报原文",
                    "evidence_grade": "C",
                    "source_note": "等待输入",
                }
            ],
        )
    assert written[0]["status"] == "analyzed"
    assert written[0]["verdict"] == "好于预期"
    assert written[0]["quality_score"] == 83.0
