"""Earnings calendar, macro news, and recommendation history helpers."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


def build_earnings_calendar(rows: list[dict[str, Any]], as_of: date) -> list[dict[str, Any]]:
    """Normalize configured or fetched earnings events."""

    normalized = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        event_date = str(row.get("date", ""))
        key = (ticker, event_date)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "ticker": ticker,
                "company": row.get("company", ticker),
                "date": event_date,
                "time": row.get("time", "待确认"),
                "status": row.get("status", _event_status(event_date, as_of)),
                "watch_items": row.get("watch_items", []),
                "post_earnings_action": row.get("post_earnings_action", "财报后自动检查营收/EPS/指引/毛利率/盘后反应"),
                "source": row.get("source", "configured"),
                "eps_forecast": row.get("eps_forecast"),
                "fiscal_period": row.get("fiscal_period"),
            }
        )
    return sorted(normalized, key=lambda item: str(item.get("date", "")))


def fetch_nasdaq_earnings_calendar(
    tickers: set[str],
    as_of: date,
    lookahead_days: int = 21,
    timeout: int = 8,
) -> list[dict[str, Any]]:
    """Fetch Nasdaq calendar rows by date and filter to tracked tickers."""

    wanted = {_normalize_ticker(ticker) for ticker in tickers}
    wanted.discard("")
    if not wanted:
        return []
    rows: list[dict[str, Any]] = []
    for offset in range(max(0, lookahead_days) + 1):
        target_date = as_of + timedelta(days=offset)
        for row in _fetch_nasdaq_date(target_date, timeout):
            ticker = _normalize_ticker(row.get("symbol", ""))
            if ticker not in wanted:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "company": _clean_html(row.get("name", "")) or ticker,
                    "date": target_date.isoformat(),
                    "time": _nasdaq_time_cn(str(row.get("time", ""))),
                    "watch_items": _watch_items_for_auto_calendar(ticker),
                    "source": "nasdaq_earnings_calendar",
                    "eps_forecast": _clean_html(row.get("epsForecast", "")),
                    "fiscal_period": _clean_html(row.get("fiscalQuarterEnding", "")),
                    "post_earnings_action": "自动抓取财报原文、Moomoo电话会/报道、SEC filing，并生成好坏与投资价值判断",
                }
            )
    return rows


def fetch_moomoo_earnings_news(
    earnings_rows: list[dict[str, Any]],
    size: int = 6,
    timeout: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch moomoo public news likely relevant to earnings/calls."""

    output: dict[str, list[dict[str, Any]]] = {}
    for row in earnings_rows:
        ticker = str(row.get("ticker", "")).upper()
        company = str(row.get("company", ""))
        if not ticker:
            continue
        items: list[dict[str, Any]] = []
        for keyword in _moomoo_keywords(ticker, company):
            items.extend(_fetch_moomoo_news(keyword, size=size, timeout=timeout))
            if len(items) >= size:
                break
        filtered = _dedupe_news([item for item in items if _is_earnings_news(item, ticker, company)])
        if filtered:
            output[ticker] = filtered[:size]
    return output


def build_news_list(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """Normalize macro, Fed, and international news rows."""

    result = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        result.append(
            {
                "category": row.get("category", category),
                "title": title,
                "time": row.get("time", "待更新"),
                "source": row.get("source", "configured feed"),
                "impact": row.get("impact", "待评估"),
                "watch": row.get("watch", "观察是否影响利率、美元、风险偏好和行业估值"),
                "url": row.get("url", ""),
            }
        )
    return result


def fetch_rss_news(feeds: list[dict[str, Any]], fallback: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """Fetch public RSS headlines, falling back to configured rows when offline."""

    fetched: list[dict[str, Any]] = []
    for feed in feeds:
        url = str(feed.get("url", "")).strip()
        if not url:
            continue
        limit = int(feed.get("limit", 5) or 5)
        source = str(feed.get("source", category))
        impact = str(feed.get("impact", "待评估"))
        watch = str(feed.get("watch", "观察是否影响利率、美元、风险偏好和行业估值"))
        try:
            request = Request(url, headers={"User-Agent": "AI-Quant-Research-Platform/1.0"})
            with urlopen(request, timeout=8) as response:
                xml_text = response.read()
            root = ET.fromstring(xml_text)
        except Exception:
            continue
        for item in root.findall(".//item")[:limit]:
            title = _text(item, "title")
            if not title:
                continue
            fetched.append(
                {
                    "category": feed.get("category", category),
                    "title": title,
                    "time": _text(item, "pubDate") or "RSS",
                    "source": source,
                    "impact": impact,
                    "watch": watch,
                    "url": _text(item, "link"),
                }
            )
    return fetched or build_news_list(fallback, category)


def attach_recommendation_history(
    candidates: list[dict[str, Any]], previous: list[dict[str, Any]], as_of: date
) -> list[dict[str, Any]]:
    """Mark whether an idea is new today or has appeared before."""

    history = {str(item.get("ticker", "")).upper(): item for item in previous}
    enriched = []
    for candidate in candidates:
        ticker = str(candidate.get("ticker", "")).upper()
        prior = history.get(ticker)
        tag = "new_today"
        first_seen = as_of.isoformat()
        prior_score = None
        if prior:
            tag = "previously_recommended"
            first_seen = str(prior.get("first_seen", first_seen))
            prior_score = prior.get("last_score")
        enriched.append({**candidate, "history_tag": tag, "first_seen": first_seen, "prior_score": prior_score})
    return enriched


def _event_status(value: str, as_of: date) -> str:
    try:
        event_date = date.fromisoformat(value)
    except ValueError:
        return "date_unknown"
    if event_date < as_of:
        return "reported_needs_analysis"
    if event_date == as_of:
        return "today"
    return "upcoming"


def _text(item: ET.Element, tag: str) -> str:
    node = item.find(tag)
    return "" if node is None or node.text is None else node.text.strip()


def _fetch_nasdaq_date(target_date: date, timeout: int) -> list[dict[str, Any]]:
    params = urlencode({"date": target_date.isoformat()})
    request = Request(
        f"https://api.nasdaq.com/api/calendar/earnings?{params}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/earnings",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except Exception:
        return []
    rows = payload.get("data", {}).get("rows", [])
    return rows if isinstance(rows, list) else []


def _fetch_moomoo_news(keyword: str, size: int, timeout: int) -> list[dict[str, Any]]:
    params = urlencode({"keyword": keyword, "size": max(1, min(size, 20)), "news_type": 1, "lang": "en", "sort_type": 2})
    request = Request(
        f"https://ai-news-search.moomoo.com/news_search?{params}",
        headers={"User-Agent": "moomoo-news-search/0.0.2 (Skill)"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except Exception:
        return []
    if payload.get("code") != 0:
        return []
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def _moomoo_keywords(ticker: str, company: str) -> list[str]:
    clean_company = re.sub(r"\b(Inc\.?|Corp\.?|Corporation|Technology|Limited|Ltd\.?)\b", "", company, flags=re.I).strip()
    keywords = [f"{company} earnings call", f"{ticker} earnings", company, ticker]
    if clean_company and clean_company != company:
        keywords.insert(1, f"{clean_company} earnings")
    return [item for item in dict.fromkeys(keywords) if item]


def _is_earnings_news(item: dict[str, Any], ticker: str, company: str) -> bool:
    title = _clean_html(item.get("title", "")).lower()
    company_tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", company) if len(token) >= 4]
    mentions_name = ticker.lower() in title or any(token in title for token in company_tokens[:2])
    mentions_earnings = any(token in title for token in ["earnings", "results", "conference call", "guidance", "财报", "业绩", "电话会"])
    return mentions_name and mentions_earnings


def _dedupe_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for item in items:
        title = _clean_html(item.get("title", ""))
        if not title or title in seen:
            continue
        seen.add(title)
        deduped.append(
            {
                "title": title,
                "publish_time": _format_publish_time(item.get("publish_time")),
                "url": item.get("url", ""),
            }
        )
    return deduped


def _format_publish_time(value: Any) -> str:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return str(value or "")
    if timestamp > 10_000_000_000:
        timestamp //= 1000
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_ticker(value: Any) -> str:
    ticker = str(value or "").upper().replace("US.", "").strip()
    if ticker.startswith("$"):
        ticker = ticker[1:]
    return ticker


def _clean_html(value: Any) -> str:
    return re.sub(r"<[^>]+>", "", unescape(str(value or ""))).strip()


def _nasdaq_time_cn(value: str) -> str:
    return {
        "time-after-hours": "盘后",
        "time-pre-market": "盘前",
        "time-not-supplied": "待确认",
    }.get(value, value or "待确认")


def _watch_items_for_auto_calendar(ticker: str) -> list[str]:
    special = {
        "MRVL": ["AI互连收入", "Custom XPU/ASIC订单", "数据中心收入增速", "下一季指引和毛利率"],
        "IREN": ["AI Cloud收入", "数据中心电力与算力交付", "现金消耗与融资需求", "电话会管理层指引"],
        "NVDA": ["Data Center增速", "毛利率", "下一季指引", "AI资本开支评论"],
        "AVGO": ["AI网络收入", "ASIC订单", "VMware整合", "自由现金流"],
        "MU": ["HBM供需", "DRAM/NAND定价", "毛利率恢复", "库存"],
    }
    return special.get(ticker, ["营收/EPS是否超预期", "管理层指引", "毛利率/现金流", "电话会关键表述"])
