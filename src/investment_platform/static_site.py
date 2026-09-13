"""Generate a dependency-free Chinese HTML dashboard from platform snapshots."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def build_static_dashboard(snapshot: dict[str, Any], output_path: str | Path) -> Path:
    """Write a standalone Chinese investment dashboard."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_html(snapshot), encoding="utf-8")
    return path


def _render_html(snapshot: dict[str, Any]) -> str:
    regime = snapshot.get("market_regime", {})
    weights = snapshot.get("factor_weights", {})
    prediction_summary = snapshot.get("prediction_summary", {})
    scores = snapshot.get("scores", [])
    theses = snapshot.get("theses", [])
    valuations = snapshot.get("valuations", [])
    sizing = snapshot.get("sizing", [])
    sentiment = snapshot.get("sentiment", [])
    behaviors = snapshot.get("behaviors", [])
    integrations = snapshot.get("integrations", [])
    portfolio = snapshot.get("portfolio", [])
    watchlists = snapshot.get("watchlists", [])
    theme_candidates = snapshot.get("theme_candidates", [])
    catalysts = snapshot.get("catalysts", [])
    smart_money = snapshot.get("smart_money", [])
    earnings_calendar = snapshot.get("earnings_calendar", [])
    earnings_reviews = snapshot.get("earnings_reviews", [])
    fed_news = snapshot.get("fed_news", [])
    international_news = snapshot.get("international_news", [])
    account_summary = snapshot.get("account_summary", {})
    allocation_plan = snapshot.get("allocation_plan", {})
    portfolio_focus = snapshot.get("portfolio_focus", {})
    portfolio_daily_review = snapshot.get("portfolio_daily_review", {})
    industry_map = snapshot.get("industry_map", {})
    data_sources = snapshot.get("data_sources", [])
    research_stack = snapshot.get("research_stack", [])
    search_playbooks = snapshot.get("search_playbooks", [])
    stock_details = snapshot.get("stock_details", [])
    trade_plans = snapshot.get("trade_plans", [])
    top_recommendations = snapshot.get("top_recommendations", [])
    search_index_json = _script_json(
        _search_index(stock_details, research_stack, theme_candidates, earnings_calendar, data_sources, earnings_reviews)
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>个人 AI 投研终端</title>
  <style>
    :root {{
      --bg: #f4f6fa;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --head: #111827;
      --blue: #2457d6;
      --green: #13823b;
      --red: #b42318;
      --amber: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{ background: var(--head); color: white; padding: 28px 32px 22px; }}
    header h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cbd5e1; line-height: 1.55; }}
    nav {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
    nav a {{
      color: white;
      text-decoration: none;
      border: 1px solid rgba(255,255,255,.22);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 13px;
    }}
    .search-row {{ display: grid; grid-template-columns: minmax(220px, 420px) auto 1fr; gap: 10px; align-items: center; margin-top: 16px; }}
    .search-row input {{
      width: 100%;
      min-height: 38px;
      border: 1px solid rgba(255,255,255,.28);
      border-radius: 6px;
      background: rgba(255,255,255,.10);
      color: white;
      padding: 8px 11px;
      font-size: 14px;
    }}
    .search-row input::placeholder {{ color: #cbd5e1; }}
    .search-row button {{
      min-height: 38px;
      border: 1px solid white;
      border-radius: 6px;
      background: white;
      color: #111827;
      font-weight: 750;
      padding: 8px 14px;
      cursor: pointer;
    }}
    #search-status {{ color: #dbeafe; font-size: 13px; line-height: 1.45; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
      overflow-x: auto;
    }}
    .panel-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
    .panel-head h2 {{ margin: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 19px; }}
    h3 {{ margin: 14px 0 8px; font-size: 15px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; background: #f8fafc; }}
    a {{ color: var(--blue); text-decoration: none; font-weight: 700; }}
    button.run {{
      border: 1px solid #b9c6dc;
      background: #f8fafc;
      color: #172033;
      border-radius: 6px;
      min-height: 34px;
      padding: 7px 11px;
      cursor: pointer;
      white-space: nowrap;
    }}
    button.run:hover {{ background: #eef4ff; border-color: #8aa7e6; }}
    .header-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 14px; }}
    .header-actions button.run {{ background: #ffffff; border-color: #ffffff; font-weight: 750; }}
    #run-status {{ color: #ffffff; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); border-radius: 6px; padding: 8px 10px; min-height: 20px; font-size: 13px; }}
    .metric .label {{ color: var(--muted); font-size: 13px; }}
    .metric .value {{ font-size: 24px; font-weight: 760; margin-top: 6px; word-break: break-word; }}
    .ok {{ color: var(--green); font-weight: 750; }}
    .bad {{ color: var(--red); font-weight: 750; }}
    .warn {{ color: var(--amber); font-weight: 750; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; margin: 0 4px 4px 0; }}
    .barrow {{ display: grid; grid-template-columns: 88px 1fr 52px; align-items: center; gap: 10px; margin: 10px 0; }}
    .bar {{ height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: var(--blue); }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.6; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    li {{ margin-bottom: 5px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 13px; margin-bottom: 10px; background: #fcfdff; }}
    .detail-grid {{ display: grid; grid-template-columns: 1.08fr .92fr; gap: 14px; }}
    details.stock {{ border: 1px solid var(--line); border-radius: 8px; padding: 0; margin-bottom: 12px; background: #fff; overflow: hidden; }}
    details.stock summary {{ cursor: pointer; list-style: none; padding: 14px 16px; display: flex; justify-content: space-between; gap: 12px; align-items: center; background: #fbfcff; }}
    details.stock summary::-webkit-details-marker {{ display: none; }}
    .stock-body {{ padding: 14px 16px 16px; border-top: 1px solid var(--line); }}
    .summary-main {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
    .summary-main strong {{ font-size: 17px; }}
    .summary-meta {{ color: var(--muted); font-size: 13px; }}
    .levels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .level-list {{ font-size: 13px; line-height: 1.5; }}
    .chart-wrap {{ width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    svg.chart {{ min-width: 700px; width: 100%; height: 270px; display: block; }}
    .cmd {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f8fafc; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 12px; color: #334155; }}
    @media (max-width: 980px) {{
      .grid, .two, .detail-grid, .levels {{ grid-template-columns: 1fr; }}
      main {{ padding: 14px; }}
      header {{ padding: 22px 18px; }}
      details.stock summary {{ align-items: flex-start; flex-direction: column; }}
      .search-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>个人 AI 投研终端</h1>
    <p>自动汇总 IBKR 持仓、Moomoo 自选股、多因子评分、投资逻辑、估值、支撑压力、日K、仓位、主题机会和日报。数据日期：{escape(str(snapshot.get("as_of", "")))}。</p>
    <nav>
      <a href="#overview">今日总览</a>
      <a href="#portfolio">持仓与自选股</a>
      <a href="#ranking">股票排名</a>
      <a href="#stock-details">股票详情</a>
      <a href="#thesis">投资逻辑</a>
      <a href="#opportunity">潜力股雷达</a>
      <a href="#research-stack">开源框架</a>
      <a href="#search-logic">搜索逻辑</a>
      <a href="#calendar">财报与新闻</a>
      <a href="#earnings-reviews">财报解析</a>
      <a href="#risk">估值与仓位</a>
      <a href="#data">数据源状态</a>
    </nav>
    <form id="stock-search" class="search-row">
      <input id="stock-search-input" type="search" autocomplete="off" placeholder="搜索代码/名称，例如 NVDA、AVGO、301321 或 SZ.301321">
      <button type="submit">搜索</button>
      <div id="search-status">A股 6 位代码会自动识别深市/沪市格式。</div>
    </form>
    <div class="header-actions">
      <button class="run" data-module="all" title="运行所有模块并刷新网站">一键运行全部</button>
      <div id="run-status">本地服务已连接。点击“运行”会执行对应模块并刷新页面。</div>
    </div>
  </header>
  <main>
    <section id="overview" class="grid">
      {_metric("市场环境", _regime_cn(regime.get("name", "NA")))}
      {_metric("环境置信度", _pct(regime.get("confidence")))}
      {_metric("AI预测准确率", _pct(prediction_summary.get("accuracy")))}
      {_metric("现金/购买力", _money(account_summary.get("available_funds")))}
    </section>

    <section class="panel">
      {_panel_head("今日买卖建议", "trade_plan")}
      {_top_recommendations(top_recommendations)}
    </section>

    <section class="panel">
      {_panel_head("精简组合配置", "portfolio")}
      {_portfolio_focus(portfolio_focus, allocation_plan, portfolio)}
    </section>

    <section class="panel">
      {_panel_head("今日/近期财报提醒与解析", "earnings")}
      {_earnings_digest(earnings_calendar, earnings_reviews)}
    </section>

    <section id="research-stack" class="panel">
      {_panel_head("开源框架接入蓝图", "research_stack")}
      {_research_stack(research_stack)}
    </section>

    <section id="search-logic" class="panel">
      {_panel_head("搜索逻辑工作台", "search_logic")}
      {_search_playbooks(search_playbooks)}
    </section>

    <section class="two">
      <div class="panel">
        {_panel_head("动态因子权重", "regime")}
        {_factor_bars(weights)}
        <p class="note">{escape(_reason_cn(str(weights.get("reason", ""))))}</p>
      </div>
      <div class="panel">
        {_panel_head("市场状态解释", "regime")}
        {_market_regime_readout(regime)}
        <h3>支持因素</h3>
        {_list([_driver_cn(item) for item in regime.get("drivers", [])])}
        <h3>风险提示</h3>
        {_list([_driver_cn(item) for item in (regime.get("cautions", []) or ["No major caution triggered"])])}
      </div>
    </section>

    <section id="portfolio" class="two">
      {_table_panel("我的 IBKR 持仓：成本、现价、盈亏", "portfolio", ["股票", "仓位", "数量", "成本", "现价", "市值", "浮盈亏", "盈亏率", "昨日盈亏", "行业"], [_portfolio_row(item) for item in portfolio], wrap=False)}
      {_table_panel("Moomoo 自选股分组", "watchlist", ["分组", "股票", "来源"], [_watchlist_row(item) for item in watchlists], wrap=False)}
    </section>

    <section class="panel">
      {_panel_head("昨日持仓涨跌回顾", "portfolio")}
      {_portfolio_daily_review(portfolio_daily_review)}
    </section>

    <section id="ranking">
      {_table_panel("股票综合排名", "ranking", ["排名", "股票", "总分", "强项/弱项"], [_score_row(idx, item) for idx, item in enumerate(scores, 1)])}
    </section>

    <section id="stock-details" class="panel">
      {_panel_head("单股详情：估值、合理价位、支撑压力、日K", "stock_details")}
      {_stock_details(stock_details)}
    </section>

    <section id="thesis">
      {_table_panel("投资逻辑追踪", "thesis", ["股票", "状态", "核心逻辑", "触发条件"], [_thesis_row(item) for item in theses])}
    </section>

    <section id="opportunity" class="two">
      {_table_panel("今日潜力股雷达", "discovery", ["股票", "主题", "潜力分", "历史", "优先级", "卡点", "搜索路径", "证据门槛"], [_theme_row(item) for item in theme_candidates], wrap=False)}
      {_table_panel("事件催化日历", "catalysts", ["股票", "事件", "日期", "紧急度", "期望波动", "要盯什么"], [_catalyst_row(item) for item in catalysts], wrap=False)}
    </section>

    <section id="calendar" class="two">
      {_table_panel("财报日历与财报后自动解析", "earnings", ["股票", "公司", "日期", "时间", "状态", "财报重点"], [_earnings_row(item) for item in earnings_calendar], wrap=False)}
      {_table_panel("美联储新闻与国际新闻", "news", ["类别", "时间", "标题", "影响", "需要观察"], [_news_row(item) for item in fed_news + international_news], wrap=False)}
    </section>

    <section id="earnings-reviews" class="panel">
      {_panel_head("新出财报解析列表", "earnings")}
      {_earnings_reviews(earnings_reviews)}
    </section>

    <section class="two">
      {_table_panel("聪明钱追踪", "smart_money", ["股票", "信号", "13F变化", "内部人净买入", "备注"], [_smart_money_row(item) for item in smart_money], wrap=False)}
      {_table_panel("市场情绪判断", "sentiment", ["股票", "上涨来源", "投机风险", "原因"], [_sentiment_row(item) for item in sentiment], wrap=False)}
    </section>

    <section id="risk" class="two">
      {_table_panel("情景估值", "valuation", ["股票", "概率加权合理价值", "相对当前价格"], [_valuation_row(item) for item in valuations], wrap=False)}
      {_table_panel("单票风险约束诊断（不作为买卖信号）", "sizing", ["股票", "当前仓位", "风控上限", "约束说明"], [_sizing_row(item) for item in sizing], wrap=False)}
    </section>

    <section class="panel">
      {_panel_head("买入/卖出点位计划", "trade_plan")}
      {_trade_plan_table(trade_plans)}
    </section>

    <section class="panel">
      {_panel_head("中等风险板块配置方案", "allocation")}
      {_allocation_plan(allocation_plan)}
    </section>

    <section class="two">
      <div class="panel">
        {_panel_head("个人投资行为提醒", "behavior")}
        {_behavior_cards(behaviors)}
      </div>
      <div class="panel">
        {_panel_head("行业竞争地图", "industry")}
        {_industry_map(industry_map)}
      </div>
    </section>

    <section id="data" class="two">
      {_table_panel("外部开源引擎状态", "integrations", ["引擎", "用途", "状态"], [_integration_row(item) for item in integrations], wrap=False)}
      {_table_panel("数据源接入状态", "data_sources", ["数据源", "用途", "状态", "备注"], [_data_source_row(item) for item in data_sources], wrap=False)}
    </section>
  </main>
  <script>
    const statusBox = document.getElementById('run-status');
    const stockSearchIndex = {search_index_json};
    const searchForm = document.getElementById('stock-search');
    const searchInput = document.getElementById('stock-search-input');
    const searchStatus = document.getElementById('search-status');
    function inferAshareCode(value) {{
      if (!/^\\d{{6}}$/.test(value)) return [];
      const candidates = [];
      if (/^[03]/.test(value)) candidates.push('SZ.' + value);
      if (/^[689]/.test(value)) candidates.push('SH.' + value);
      return candidates;
    }}
    function searchCandidates(raw) {{
      const upper = raw.trim().toUpperCase().replace(/\\s+/g, '');
      const tail = upper.includes('.') ? upper.split('.').pop() : upper;
      return new Set([upper, tail, ...inferAshareCode(tail)]);
    }}
    function findStock(raw) {{
      const query = raw.trim();
      if (!query) return null;
      const candidates = searchCandidates(query);
      const normalized = query.toUpperCase();
      return stockSearchIndex.find((item) => {{
        const title = String(item.title || '').toUpperCase();
        const ticker = String(item.ticker || '').toUpperCase();
        const code = String(item.code || '').toUpperCase();
        const name = String(item.name || '').toUpperCase();
        const aliases = (item.aliases || []).map((value) => String(value).toUpperCase());
        return candidates.has(ticker) || candidates.has(code) || title.includes(normalized) || name.includes(normalized) || aliases.some((value) => candidates.has(value) || value.includes(normalized));
      }});
    }}
    function goToStock(item) {{
      const target = document.getElementById(item.target_id || ('stock-' + item.ticker));
      if (!target) return false;
      if (target.tagName && target.tagName.toLowerCase() === 'details') target.open = true;
      target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
      searchStatus.textContent = '已找到：' + item.label + '。';
      return true;
    }}
    searchForm.addEventListener('submit', (event) => {{
      event.preventDefault();
      const query = searchInput.value.trim();
      const item = findStock(query);
      if (item && goToStock(item)) return;
      const tail = query.toUpperCase().replace(/\\s+/g, '').split('.').pop();
      const inferred = inferAshareCode(tail);
      if (inferred.length) {{
        searchStatus.textContent = '当前快照里没有 ' + query + '。A股请用 ' + inferred.join(' 或 ') + '；如果要在本网站看详情，请先把它加入 Moomoo 自选股/导出文件，然后运行“自选股”和“股票详情”。';
      }} else {{
        searchStatus.textContent = '当前快照没有找到 ' + query + '。请确认代码是否在 IBKR 持仓或 Moomoo 自选股里。';
      }}
    }});
    async function runModule(moduleName) {{
      const activeButton = document.querySelector('button[data-module="' + moduleName + '"]');
      if (activeButton) {{
        activeButton.disabled = true;
        activeButton.textContent = '运行中';
      }}
      statusBox.textContent = '正在运行：' + moduleName + '。这会在本地执行脚本并刷新页面，请等几秒钟。';
      try {{
        const response = await fetch('/run?module=' + encodeURIComponent(moduleName), {{method: 'POST'}});
        const payload = await response.json();
        if (!response.ok || !payload.ok) throw new Error(payload.error || '运行失败');
        statusBox.textContent = payload.message || '运行完成，正在刷新页面';
        setTimeout(() => window.location.reload(), 700);
      }} catch (error) {{
        statusBox.textContent = '运行按钮需要通过本地服务打开：http://127.0.0.1:8502。错误：' + error.message;
        if (activeButton) {{
          activeButton.disabled = false;
          activeButton.textContent = '运行';
        }}
      }}
    }}
    document.querySelectorAll('button[data-module]').forEach((button) => {{
      button.addEventListener('click', () => runModule(button.dataset.module));
    }});
  </script>
</body>
</html>
"""


def _panel_head(title: str, module: str) -> str:
    return f'<div class="panel-head"><h2>{escape(title)}</h2>{_run_button(module)}</div>'


def _run_button(module: str) -> str:
    return f'<button class="run" data-module="{escape(module)}" title="单独运行这个模块">运行</button>'


def _search_index(
    stock_details: list[dict[str, Any]],
    research_stack: list[dict[str, Any]] | None = None,
    theme_candidates: list[dict[str, Any]] | None = None,
    earnings_calendar: list[dict[str, Any]] | None = None,
    data_sources: list[dict[str, Any]] | None = None,
    earnings_reviews: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for item in stock_details:
        ticker = str(item.get("ticker", ""))
        if not ticker:
            continue
        code = str(item.get("code", ""))
        name = str(item.get("name", ""))
        groups = item.get("groups", [])
        watchlists = item.get("watchlists", [])
        aliases = [ticker, code, name, *[str(value) for value in groups], *[str(value) for value in watchlists]]
        rows.append(
            {
                "ticker": ticker,
                "code": code,
                "name": name,
                "label": " / ".join(part for part in [ticker, code, name] if part),
                "kind": "stock",
                "target_id": f"stock-{ticker}",
                "aliases": [value for value in aliases if value],
            }
        )
    for item in research_stack or []:
        name = str(item.get("name", ""))
        target = "framework-" + _slug(name)
        rows.append(
            {
                "kind": "section",
                "title": name,
                "label": f"开源框架：{name}",
                "target_id": target,
                "aliases": [
                    name,
                    str(item.get("layer", "")),
                    str(item.get("role", "")),
                    str(item.get("repo", "")),
                    *[str(value) for value in item.get("keywords", [])],
                ],
            }
        )
    for item in theme_candidates or []:
        ticker = str(item.get("ticker", ""))
        rows.append(
            {
                "kind": "section",
                "ticker": ticker,
                "title": str(item.get("theme", "")),
                "label": f"潜力股：{ticker} / {item.get('theme', '')}",
                "target_id": "opportunity",
                "aliases": [
                    ticker,
                    str(item.get("company", "")),
                    str(item.get("theme", "")),
                    str(item.get("bottleneck", "")),
                    str(item.get("next_check", "")),
                ],
            }
        )
    for item in earnings_calendar or []:
        ticker = str(item.get("ticker", ""))
        rows.append(
            {
                "kind": "section",
                "ticker": ticker,
                "title": f"{ticker} 财报",
                "label": f"财报日历：{ticker} / {item.get('date', '')}",
                "target_id": "calendar",
                "aliases": [ticker, str(item.get("company", "")), "earnings", "财报", str(item.get("date", ""))],
            }
        )
    for item in earnings_reviews or []:
        ticker = str(item.get("ticker", ""))
        rows.append(
            {
                "kind": "section",
                "ticker": ticker,
                "title": f"{ticker} 财报解析",
                "label": f"财报解析：{ticker} / {item.get('verdict', '')}",
                "target_id": "earnings-reviews",
                "aliases": [
                    ticker,
                    str(item.get("company", "")),
                    "earnings review",
                    "财报解析",
                    str(item.get("verdict", "")),
                    str(item.get("thesis_impact", "")),
                    str(item.get("investment_value", "")),
                ],
            }
        )
    for item in data_sources or []:
        name = str(item.get("name", ""))
        rows.append(
            {
                "kind": "section",
                "title": name,
                "label": f"数据源：{name}",
                "target_id": "data",
                "aliases": [name, str(item.get("role", "")), str(item.get("status", "")), "数据源"],
            }
        )
    return rows


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _metric(label: str, value: object) -> str:
    return f'<div class="panel metric"><div class="label">{escape(label)}</div><div class="value">{escape(str(value))}</div></div>'


def _pct(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "NA"


def _money(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "NA"


def _safe_float(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _factor_bars(weights: dict[str, Any]) -> str:
    labels = {"growth": "成长", "quality": "质量", "momentum": "动量", "value": "估值", "risk": "风险"}
    rows = []
    for key in ["growth", "quality", "momentum", "value", "risk"]:
        value = float(weights.get(key, 0.0) or 0.0)
        rows.append(f'<div class="barrow"><div>{labels[key]}</div><div class="bar"><span style="width:{value * 100:.0f}%"></span></div><div>{value:.0%}</div></div>')
    return "\n".join(rows)


def _list(items: list[Any]) -> str:
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"


def _table_panel(title: str, module: str, headers: list[str], rows: list[list[str]], wrap: bool = True) -> str:
    html = f"{_panel_head(title, module)}<table><thead><tr>{''.join(f'<th>{escape(h)}</th>' for h in headers)}</tr></thead><tbody>"
    if rows:
        for row in rows:
            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    else:
        html += f'<tr><td colspan="{len(headers)}">暂无数据</td></tr>'
    html += "</tbody></table>"
    return f'<div class="panel">{html}</div>' if wrap else f'<div class="panel">{html}</div>'


def _ticker_link(ticker: object) -> str:
    value = str(ticker or "")
    return f'<a href="#stock-{escape(value)}">{escape(value)}</a>'


def _ticker_links(tickers: list[Any]) -> str:
    return ", ".join(_ticker_link(item) for item in tickers)


def _portfolio_row(item: dict[str, Any]) -> list[str]:
    qty = item.get("quantity", "")
    avg = item.get("avg_cost", "")
    pnl = float(item.get("unrealized_pnl", 0.0) or 0.0)
    daily_pnl = float(item.get("daily_pnl", 0.0) or 0.0)
    price = item.get("ibkr_price")
    pnl_pct = _position_pnl_pct(item)
    cls = "ok" if pnl >= 0 else "bad"
    daily_cls = "ok" if daily_pnl >= 0 else "bad"
    return [
        _ticker_link(item.get("ticker", "")),
        escape(_pct(item.get("allocation"))),
        escape(str(qty)),
        escape(_money(avg)),
        escape(_money(price)),
        escape(_money(item.get("market_value"))),
        f'<span class="{cls}">{escape(_money(pnl))}</span>',
        f'<span class="{cls}">{escape(_pct(pnl_pct))}</span>',
        f'<span class="{daily_cls}">{escape(_money(daily_pnl))}</span>',
        escape(str(item.get("sector", ""))),
    ]


def _position_pnl_pct(item: dict[str, Any]) -> float | None:
    pnl = _safe_float(item.get("unrealized_pnl"))
    qty = _safe_float(item.get("quantity"))
    avg_cost = _safe_float(item.get("avg_cost"))
    price = _safe_float(item.get("ibkr_price"))
    if pnl is not None and qty and avg_cost:
        cost_basis = qty * avg_cost
        if cost_basis:
            return pnl / cost_basis
    if price is not None and avg_cost:
        return price / avg_cost - 1.0
    return None


def _portfolio_daily_review(review: dict[str, Any]) -> str:
    rows = review.get("rows", []) if isinstance(review, dict) else []
    if not rows:
        return '<p class="note">暂无昨日持仓涨跌数据。</p>'
    total = float(review.get("total_daily_pnl", 0.0) or 0.0)
    total_cls = "ok" if total >= 0 else "bad"
    html = (
        '<p class="note">'
        f'昨日持仓合计影响：<span class="{total_cls}">{escape(_money(total))}</span>，'
        f'约占账户净值 {escape(_pct(review.get("total_daily_return")))}。'
        '排序按单票昨日盈亏绝对值，方便看主要贡献和拖累。'
        '</p>'
        '<table><thead><tr><th>股票</th><th>日期</th><th>前收</th><th>最新收盘</th><th>涨跌幅</th><th>昨日盈亏</th><th>成本</th><th>现价</th><th>总浮盈亏</th><th>盈亏率</th><th>来源</th></tr></thead><tbody>'
    )
    for row in rows:
        daily = float(row.get("daily_pnl", 0.0) or 0.0)
        unrealized = float(row.get("unrealized_pnl", 0.0) or 0.0)
        daily_cls = "ok" if daily >= 0 else "bad"
        pnl_cls = "ok" if unrealized >= 0 else "bad"
        dates = " / ".join(str(value) for value in [row.get("previous_date"), row.get("latest_date")] if value)
        html += (
            "<tr>"
            f"<td>{_ticker_link(row.get('ticker', ''))}</td>"
            f"<td>{escape(dates or 'NA')}</td>"
            f"<td>{escape(_money(row.get('previous_close')))}</td>"
            f"<td>{escape(_money(row.get('latest_close')))}</td>"
            f"<td>{escape(_pct(row.get('daily_return')))}</td>"
            f'<td><span class="{daily_cls}">{escape(_money(daily))}</span></td>'
            f"<td>{escape(_money(row.get('avg_cost')))}</td>"
            f"<td>{escape(_money(row.get('current_price')))}</td>"
            f'<td><span class="{pnl_cls}">{escape(_money(unrealized))}</span></td>'
            f'<td><span class="{pnl_cls}">{escape(_pct(row.get("unrealized_pnl_pct")))}</span></td>'
            f"<td>{escape(str(row.get('daily_pnl_source', '')))}</td>"
            "</tr>"
        )
    html += "</tbody></table>"
    return html


def _watchlist_row(item: dict[str, Any]) -> list[str]:
    return [escape(str(item.get("name", ""))), _ticker_links(item.get("tickers", [])), escape(_source_cn(str(item.get("source", ""))))]


def _score_row(rank: int, item: dict[str, Any]) -> list[str]:
    notes = "; ".join(_score_note_cn(str(note)) for note in item.get("explanation", []))
    return [str(rank), _ticker_link(item.get("ticker", "")), escape(f"{float(item.get('total_score', 0.0)):.1f}"), escape(notes)]


def _thesis_row(item: dict[str, Any]) -> list[str]:
    status = str(item.get("status", ""))
    status_cn = "有效" if status == "active" else "走弱"
    status_class = "ok" if status == "active" else "warn"
    triggered = item.get("triggered_conditions", []) or ["无"]
    return [_ticker_link(item.get("ticker", "")), f'<span class="{status_class}">{status_cn}</span>', escape(str(item.get("thesis", ""))), escape("; ".join(str(note) for note in triggered))]


def _theme_row(item: dict[str, Any]) -> list[str]:
    lanes = "；".join(str(value) for value in item.get("search_lanes", [])[:3])
    gates = "；".join(str(value) for value in item.get("evidence_gates", [])[:3])
    return [
        _ticker_link(item.get("ticker", "")),
        escape(str(item.get("theme", ""))),
        escape(f"{float(item.get('potential_score', 0.0)):.1f}"),
        escape(_history_cn(str(item.get("history_tag", "")), item.get("first_seen"), item.get("prior_score"))),
        escape(_priority_cn(str(item.get("priority", "")))),
        escape(str(item.get("bottleneck", ""))),
        escape(lanes or str(item.get("next_check", ""))),
        escape(gates or str(item.get("next_check", ""))),
    ]


def _catalyst_row(item: dict[str, Any]) -> list[str]:
    return [_ticker_link(item.get("ticker", "")), escape(str(item.get("event", ""))), escape(str(item.get("date", ""))), escape("高" if item.get("urgency") == "high" else "观察"), escape(_pct(item.get("expected_move"))), escape(str(item.get("watch_item", "")))]


def _smart_money_row(item: dict[str, Any]) -> list[str]:
    return [_ticker_link(item.get("ticker", "")), escape(_smart_money_cn(str(item.get("smart_money_signal", "")))), escape(_pct(item.get("institutional_change"))), escape(_pct(item.get("insider_net_buying"))), escape(str(item.get("notes", "")))]


def _earnings_row(item: dict[str, Any]) -> list[str]:
    watch_items = item.get("watch_items", [])
    if isinstance(watch_items, list):
        watch = "；".join(str(value) for value in watch_items)
    else:
        watch = str(watch_items)
    return [
        _ticker_link(item.get("ticker", "")),
        escape(str(item.get("company", ""))),
        escape(str(item.get("date", ""))),
        escape(str(item.get("time", ""))),
        escape(_earnings_status_cn(str(item.get("status", "")))),
        escape(watch or str(item.get("post_earnings_action", ""))),
    ]


def _earnings_digest(calendar: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> str:
    review_by_ticker = {str(item.get("ticker", "")).upper(): item for item in reviews}
    rows = []
    for item in calendar[:12]:
        ticker = str(item.get("ticker", "")).upper()
        review = review_by_ticker.get(ticker, {})
        watch_items = item.get("watch_items", [])
        watch = "；".join(str(value) for value in watch_items) if isinstance(watch_items, list) else str(watch_items)
        verdict = str(review.get("verdict", "待公布/待解析"))
        action = str(review.get("action", "公布后自动抓取财报原文、SEC filing 和电话会要点"))
        status = _earnings_status_cn(str(item.get("status", "")))
        if review:
            status = _earnings_review_status_cn(str(review.get("status", "")))
        rows.append(
            [
                _ticker_link(ticker),
                escape(str(item.get("company", ""))),
                escape(str(item.get("date", ""))),
                escape(str(item.get("time", ""))),
                escape(status),
                escape(verdict),
                escape(watch),
                escape(action),
            ]
        )
    if not rows:
        return '<p class="note">暂无近期财报提醒。</p>'
    html = (
        '<p class="note">这个列表优先展示持仓和 Moomoo 自选股的今日/近期财报。财报公布后，下一次刷新会把它转入解析结果，'
        '用 financial-research-agent + ai-berkshire 判断好坏和投资价值。</p>'
        '<table><thead><tr><th>股票</th><th>公司</th><th>日期</th><th>时间</th><th>状态</th><th>解析结论</th><th>重点</th><th>下一步</th></tr></thead><tbody>'
    )
    for row in rows:
        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html += "</tbody></table>"
    return html


def _earnings_reviews(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">暂无已出财报解析。每日刷新会扫描持仓和 Moomoo 自选股里的财报事件。</p>'
    html = (
        '<p class="note">这里专门放“已经出财报”的股票。只有拿到可核验的财报原文、SEC文件、电话会纪要或结构化指标后，'
        '才会标为已解析；系统不会把占位报告当成真实结论。</p>'
        '<table><thead><tr><th>股票</th><th>公司</th><th>日期</th><th>状态</th><th>财报好坏</th><th>质量分</th><th>投资逻辑影响</th><th>投资价值</th><th>操作建议</th><th>证据</th><th>下一步</th></tr></thead><tbody>'
    )
    for item in items:
        score = item.get("quality_score")
        score_text = "NA" if score in (None, "") else f"{float(score):.1f}"
        status = str(item.get("status", ""))
        status_cls = "ok" if status == "analyzed" else "warn"
        next_steps = "；".join(str(value) for value in item.get("next_steps", [])[:2])
        report_path = str(item.get("report_path", ""))
        evidence = f"{item.get('evidence_grade', '')} / {item.get('source_note', '')}"
        if report_path:
            evidence = f'{escape(evidence)}<br><span class="cmd">{escape(report_path)}</span>'
        else:
            evidence = escape(evidence)
        html += (
            "<tr>"
            f"<td>{_ticker_link(item.get('ticker', ''))}</td>"
            f"<td>{escape(str(item.get('company', '')))}</td>"
            f"<td>{escape(str(item.get('date', '')))}</td>"
            f'<td><span class="{status_cls}">{escape(_earnings_review_status_cn(status))}</span></td>'
            f"<td>{escape(str(item.get('verdict', '')))}</td>"
            f"<td>{escape(score_text)}</td>"
            f"<td>{escape(str(item.get('thesis_impact', '')))}</td>"
            f"<td>{escape(str(item.get('investment_value', '')))}</td>"
            f"<td>{escape(str(item.get('action', '')))}</td>"
            f"<td>{evidence}</td>"
            f"<td>{escape(next_steps)}</td>"
            "</tr>"
        )
    html += "</tbody></table>"
    html += _earnings_detail_cards(items)
    return html


def _earnings_detail_cards(items: list[dict[str, Any]]) -> str:
    html = '<h3>财报详情</h3>'
    for item in items:
        ticker = str(item.get("ticker", ""))
        score = item.get("quality_score")
        score_text = "NA" if score in (None, "") else f"{float(score):.1f}"
        html += (
            f'<details class="stock" open><summary><div class="summary-main"><strong>{escape(ticker)}</strong>'
            f'<span>{escape(str(item.get("company", "")))}</span>'
            f'<span class="pill">{escape(str(item.get("verdict", "")))}</span>'
            f'<span class="pill">质量分 {escape(score_text)}</span></div>'
            f'<div class="summary-meta">{escape(str(item.get("date", "")))} {escape(str(item.get("time", "")))}</div></summary>'
            '<div class="stock-body">'
            '<div class="detail-grid">'
            '<div>'
            '<h3>核心结论</h3>'
            f'{_list([str(value) for value in item.get("key_points", [])] or ["暂无"])}'
            '<h3>已披露/待核验财务数据</h3>'
            f'{_earnings_metrics(item.get("reported_metrics", []), item.get("missing_data", []))}'
            '<h3>电话会 / 管理层要点</h3>'
            f'{_list([str(value) for value in item.get("call_highlights", [])] or [str(item.get("management_tone", "")) or "等待电话会纪要或Moomoo电话会报道"])}'
            '</div>'
            '<div>'
            '<h3>Moomoo 财报/电话会相关报道</h3>'
            f'{_news_links(item.get("moomoo_news", []))}'
            '<h3>原始证据 / 官网链接</h3>'
            f'{_source_links(item.get("source_documents", []), item.get("source_note", ""))}'
            '<h3>红旗</h3>'
            f'{_list([str(value) for value in item.get("red_flags", [])] or ["暂无"])}'
            '<h3>下一步</h3>'
            f'{_list([str(value) for value in item.get("next_steps", [])] or ["暂无"])}'
            '</div>'
            '</div>'
            '</div></details>'
        )
    return html


def _earnings_metrics(metrics: Any, missing_data: Any) -> str:
    if isinstance(metrics, list) and metrics:
        rows = []
        for item in metrics:
            if not isinstance(item, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(str(item.get('metric', '')))}</td>"
                f"<td>{escape(str(item.get('actual', '')))}</td>"
                f"<td>{escape(str(item.get('estimate', '')))}</td>"
                f"<td>{escape(str(item.get('change', '')))}</td>"
                f"<td>{escape(str(item.get('note', '')))}</td>"
                "</tr>"
            )
        if rows:
            return '<table><thead><tr><th>指标</th><th>实际</th><th>预期</th><th>变化</th><th>备注</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
    missing = [str(value) for value in missing_data or []]
    return '<p class="note">尚未拿到可核验财报指标。缺失：' + escape("、".join(missing) or "原始财报数据") + "。这个状态表示系统已有待处理事项，但还没有足够证据下投资结论。</p>"


def _news_links(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return '<p class="note">暂未抓到高相关 Moomoo 财报/电话会报道；系统会在下一次刷新继续尝试。</p>'
    html = "<ul>"
    for item in items[:6]:
        title = escape(str(item.get("title", "")))
        url = escape(str(item.get("url", "")))
        published = escape(str(item.get("publish_time", "")))
        if url:
            title = f'<a href="{url}" target="_blank" rel="noreferrer">{title}</a>'
        html += f"<li>{title}<br><span class='note'>{published}</span></li>"
    html += "</ul>"
    return html


def _source_links(items: Any, source_note: Any) -> str:
    values = [str(value) for value in items or [] if value]
    if not values:
        note = str(source_note or "")
        return f'<p class="note">{escape(note or "暂无原始证据链接")}</p>'
    html = "<ul>"
    for value in values[:8]:
        escaped = escape(value)
        if value.startswith("http"):
            escaped = f'<a href="{escaped}" target="_blank" rel="noreferrer">{escaped}</a>'
        html += f"<li>{escaped}</li>"
    html += "</ul>"
    note = str(source_note or "")
    if note:
        html += f'<p class="note">{escape(note)}</p>'
    return html


def _news_row(item: dict[str, Any]) -> list[str]:
    title = escape(str(item.get("title", "")))
    url = str(item.get("url", ""))
    if url:
        title = f'<a href="{escape(url)}" target="_blank" rel="noreferrer">{title}</a>'
    return [
        escape(str(item.get("category", ""))),
        escape(str(item.get("time", ""))),
        title,
        escape(str(item.get("impact", ""))),
        escape(str(item.get("watch", ""))),
    ]


def _sentiment_row(item: dict[str, Any]) -> list[str]:
    reasons = "; ".join(_sentiment_reason_cn(str(note)) for note in item.get("reasons", []))
    return [_ticker_link(item.get("ticker", "")), escape(_sentiment_cn(str(item.get("label", "")))), escape("高" if item.get("speculation_risk") == "high" else "正常"), escape(reasons)]


def _valuation_row(item: dict[str, Any]) -> list[str]:
    return [_ticker_link(item.get("ticker", "")), escape(_money(item.get("weighted_fair_value"))), escape(_pct(item.get("upside_to_price")))]


def _sizing_row(item: dict[str, Any]) -> list[str]:
    reasons = "; ".join(_sizing_reason_cn(str(note)) for note in item.get("reasons", []))
    return [_ticker_link(item.get("ticker", "")), escape(_pct(item.get("current_allocation"))), escape(_pct(item.get("max_allocation"))), escape(reasons)]


def _portfolio_focus(focus: dict[str, Any], allocation_plan: dict[str, Any], portfolio: list[dict[str, Any]]) -> str:
    if not focus:
        return '<p class="note">暂无精简组合方案。</p>'
    allocation_by_ticker = {
        str(row.get("ticker", "")).upper(): row
        for row in allocation_plan.get("rows", [])
        if row.get("ticker")
    }
    portfolio_by_ticker = {str(row.get("ticker", "")).upper(): row for row in portfolio if row.get("ticker")}
    html = (
        '<p class="note">'
        f'目标持仓数量：{escape(str(focus.get("target_position_count", "")))}。'
        f'{escape(str(focus.get("principle", "")))} '
        f'{escape(str(focus.get("cash_policy", "")))}'
        '</p>'
    )
    for tier in focus.get("tiers", []):
        if not isinstance(tier, dict):
            continue
        html += (
            f'<h3>{escape(str(tier.get("name", "")))}</h3>'
            f'<p class="note">{escape(str(tier.get("objective", "")))}</p>'
            '<table><thead><tr><th>股票</th><th>角色</th><th>当前仓位</th><th>目标仓位</th><th>模型动作</th><th>结论</th><th>原因</th></tr></thead><tbody>'
        )
        for item in tier.get("tickers", []):
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).upper()
            allocation_row = allocation_by_ticker.get(ticker, {})
            portfolio_row = portfolio_by_ticker.get(ticker, {})
            current = allocation_row.get("current_allocation", portfolio_row.get("allocation"))
            target = allocation_row.get("target_allocation", item.get("target"))
            decision = str(item.get("decision", ""))
            cls = "bad" if "退出" in decision or "降权" in decision else "ok" if "保留" in decision or "加" in decision else "warn"
            html += (
                "<tr>"
                f"<td>{_ticker_link(ticker)}</td>"
                f"<td>{escape(str(item.get('role', '')))}</td>"
                f"<td>{escape(_pct(current))}</td>"
                f"<td>{escape(_pct(target))}</td>"
                f"<td>{escape(_allocation_action_cn(str(allocation_row.get('action', ''))))}</td>"
                f'<td><span class="{cls}">{escape(decision)}</span></td>'
                f"<td>{escape(str(item.get('reason', '')))}</td>"
                "</tr>"
            )
        html += "</tbody></table>"
    return html


def _trade_plan_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">暂无买卖点位计划。</p>'
    html = (
        '<p class="note">这张表才是买卖点位参考；上面的单票风控上限只说明风险约束，不代表目标仓位。'
        '所有点位来自 Moomoo 日K支撑/压力、情景估值和组合现金约束；不会自动下单。</p>'
        '<table><thead><tr><th>股票</th><th>类型</th><th>现价</th><th>判断</th><th>问题来源</th><th>支撑买入区</th>'
        '<th>突破买点</th><th>止损/失效</th><th>减仓区</th><th>目标卖出区</th><th>原因</th></tr></thead><tbody>'
    )
    for item in items:
        decision = str(item.get("decision", ""))
        cls = _decision_class(decision)
        html += (
            "<tr>"
            f"<td>{_ticker_link(item.get('ticker', ''))}</td>"
            f"<td>{escape(str(item.get('source', '')))}</td>"
            f"<td>{escape(_money(item.get('price')))}</td>"
            f'<td><span class="{cls}">{escape(_trade_decision_cn(decision))}</span></td>'
            f"<td>{escape(str(item.get('problem_area', '')))}</td>"
            f"<td>{escape(str(item.get('buy_zone', '')))}</td>"
            f"<td>{escape(str(item.get('breakout_buy', '')))}</td>"
            f"<td>{escape(str(item.get('stop_loss', '')))}</td>"
            f"<td>{escape(str(item.get('trim_zone', '')))}</td>"
            f"<td>{escape(str(item.get('sell_zone', '')))}</td>"
            f"<td>{escape(str(item.get('reason', '')))}</td>"
            "</tr>"
        )
    html += "</tbody></table>"
    return html


def _top_recommendations(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">暂无可执行建议。先运行“股票详情”和“买卖点位计划”。</p>'
    html = (
        '<p class="note">这是页面最上方的行动摘要：SCHD/SGOV 视为现金或防守储备；其余持仓结合潜力、基本面、波动、回撤、相关性和现金约束给出结论。不会自动下单。</p>'
        '<table><thead><tr><th>股票</th><th>结论</th><th>问题来源</th><th>当前/目标仓位</th><th>建议金额</th><th>买入区</th><th>减仓区</th><th>止损/失效</th><th>目标卖出区</th><th>原因</th></tr></thead><tbody>'
    )
    for item in items:
        decision = str(item.get("decision", ""))
        cls = _decision_class(decision)
        amount = float(item.get("suggested_amount", 0.0) or 0.0)
        amount_cls = "ok" if amount > 0 else "bad" if amount < 0 else ""
        html += (
            "<tr>"
            f"<td>{_ticker_link(item.get('ticker', ''))}</td>"
            f'<td><span class="{cls}">{escape(_trade_decision_cn(decision))}</span></td>'
            f"<td>{escape(str(item.get('problem_area', '')))}</td>"
            f"<td>{escape(_pct(item.get('current_allocation')))} / {escape(_pct(item.get('target_allocation')))}</td>"
            f'<td><span class="{amount_cls}">{escape(_money(amount))}</span></td>'
            f"<td>{escape(str(item.get('buy_zone', '')))}</td>"
            f"<td>{escape(str(item.get('trim_zone', '')))}</td>"
            f"<td>{escape(str(item.get('stop_loss', '')))}</td>"
            f"<td>{escape(str(item.get('sell_zone', '')))}</td>"
            f"<td>{escape(str(item.get('reason', '')))}</td>"
            "</tr>"
        )
    html += "</tbody></table>"
    return html


def _research_stack(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">暂无开源框架接入信息。</p>'
    html = (
        '<p class="note">这些是平台的底层研究框架分工：Qlib 做核心量化，OpenBB 做数据入口，'
        'financial-research-agent 做财报/SEC 深研，ai-berkshire 做价值投资检查，FinRL 只做研究沙盒。</p>'
        '<table><thead><tr><th>框架</th><th>定位</th><th>直接复用</th><th>本平台补充</th><th>决策边界</th><th>状态</th></tr></thead><tbody>'
    )
    for item in items:
        status_cls = "ok" if item.get("status") == "installed" else "warn"
        repo = str(item.get("repo", ""))
        name = escape(str(item.get("name", "")))
        name_html = f'<a href="{escape(repo)}" target="_blank" rel="noreferrer">{name}</a>' if repo else name
        html += (
            f'<tr id="framework-{escape(_slug(str(item.get("name", ""))))}">'
            f"<td>{name_html}<br><span class='pill'>{escape(str(item.get('layer', '')))}</span></td>"
            f"<td>{escape(str(item.get('role', '')))}</td>"
            f"<td>{escape(str(item.get('direct_use', '')))}</td>"
            f"<td>{escape(str(item.get('added_by_platform', '')))}</td>"
            f"<td>{escape(str(item.get('decision_policy', '')))}</td>"
            f'<td><span class="{status_cls}">{escape(str(item.get("status_label", "")))}</span></td>'
            "</tr>"
        )
    html += "</tbody></table>"
    return html


def _search_playbooks(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">暂无搜索逻辑配置。</p>'
    html = (
        '<p class="note">搜索不再只看已有持仓。每天会按不同研究路径扫描：量化因子、财报/SEC、价值质量、事件催化和强化学习实验。'
        '真正进入买入建议前，必须通过证据门槛和风险预算。</p>'
        '<table><thead><tr><th>搜索路径</th><th>使用引擎</th><th>目标</th><th>信号</th><th>淘汰条件</th></tr></thead><tbody>'
    )
    for item in items:
        signals = "；".join(str(value) for value in item.get("signals", []))
        html += (
            "<tr>"
            f"<td>{escape(str(item.get('name', '')))}</td>"
            f"<td>{escape(str(item.get('engine', '')))}</td>"
            f"<td>{escape(str(item.get('goal', '')))}</td>"
            f"<td>{escape(signals)}</td>"
            f"<td>{escape(str(item.get('reject_if', '')))}</td>"
            "</tr>"
        )
    html += "</tbody></table>"
    return html


def _allocation_plan(plan: dict[str, Any]) -> str:
    if not plan:
        return '<p class="note">暂无现金约束仓位计划。</p>'
    rows = plan.get("rows", [])
    html = (
        '<p class="note">'
        f'{escape(str(plan.get("risk_profile", "中等风险")))}；账户净值 {escape(_money(plan.get("net_liquidation")))}；'
        f'现金/购买力 {escape(_money(plan.get("cash_value")))}；最低现金目标 {escape(_pct(plan.get("minimum_cash_weight")))}；'
        f'现金目标金额 {escape(_money(plan.get("target_cash_value")))}；现金缺口 {escape(_money(plan.get("cash_shortfall")))}；'
        f'目标账户日波动 {escape(_pct(plan.get("target_daily_volatility_min")))}-{escape(_pct(plan.get("target_daily_volatility_max")))}；'
        f'估算目标组合日波动 {escape(_pct(plan.get("estimated_daily_volatility")))}（{escape(_risk_budget_status_cn(str(plan.get("risk_budget_status", ""))))}）。'
        f'{escape(str(plan.get("constraint_note", "")))}'
        '</p>'
    )
    action_plan = plan.get("action_plan", [])
    if action_plan:
        html += '<h3>再平衡执行顺序</h3><table><thead><tr><th>顺序</th><th>动作</th><th>股票/现金</th><th>板块</th><th>金额</th><th>原因</th></tr></thead><tbody>'
        for row in action_plan:
            side = str(row.get("side", ""))
            cls = "bad" if side == "sell_or_stop_adding" else "ok" if side == "conditional_buy" else "warn"
            html += (
                "<tr>"
                f"<td>{escape(str(row.get('step', '')))}</td>"
                f'<td><span class="{cls}">{escape(_allocation_side_cn(side))}</span></td>'
                f"<td>{_ticker_link(row.get('ticker', '')) if row.get('ticker') != 'CASH' else '现金缓冲'}</td>"
                f"<td>{escape(str(row.get('bucket', '')))}</td>"
                f"<td>{escape(_money(row.get('amount')))}</td>"
                f"<td>{escape(str(row.get('reason', '')))}</td>"
                "</tr>"
            )
        html += "</tbody></table>"
    bucket_rows = plan.get("bucket_rows", [])
    if bucket_rows:
        html += '<h3>板块目标</h3><table><thead><tr><th>板块</th><th>当前占比</th><th>目标占比</th><th>需要买/卖</th></tr></thead><tbody>'
        for row in bucket_rows:
            delta = float(row.get("delta_value", 0.0) or 0.0)
            cls = "ok" if delta > 0 else "bad" if delta < 0 else ""
            html += (
                "<tr>"
                f"<td>{escape(str(row.get('bucket', '')))}</td>"
                f"<td>{escape(_pct(row.get('current_allocation')))}</td>"
                f"<td>{escape(_pct(row.get('target_allocation')))}</td>"
                f'<td><span class="{cls}">{escape(_money(delta))}</span></td>'
                "</tr>"
            )
        html += "</tbody></table>"
    html += '<h3>个股调整金额</h3><table><thead><tr><th>股票</th><th>板块</th><th>当前占比</th><th>目标占比</th><th>需要买/卖</th><th>本次最多执行</th><th>动作</th><th>说明</th></tr></thead><tbody>'
    for row in rows:
        delta = float(row.get("delta_value", 0.0) or 0.0)
        cls = "ok" if delta > 0 else "bad" if delta < 0 else ""
        feasible = row.get("feasible_trade_value")
        executable = _money(feasible) if feasible is not None and float(feasible or 0.0) > 0 else "-"
        html += (
            "<tr>"
            f"<td>{_ticker_link(row.get('ticker', ''))}</td>"
            f"<td>{escape(str(row.get('bucket', '')))}</td>"
            f"<td>{escape(_pct(row.get('current_allocation')))}</td>"
            f"<td>{escape(_pct(row.get('target_allocation')))}</td>"
            f'<td><span class="{cls}">{escape(_money(delta))}</span></td>'
            f"<td>{escape(executable)}</td>"
            f"<td>{escape(_allocation_action_cn(str(row.get('action', ''))))}</td>"
            f"<td>{escape(str(row.get('note', '')))}</td>"
            "</tr>"
        )
    html += "</tbody></table>"
    return html


def _stock_details(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">暂无单股详情。</p>'
    blocks = []
    for idx, item in enumerate(items):
        ticker = str(item.get("ticker", ""))
        price = _money(item.get("price"))
        score = "NA" if item.get("score") is None else f"{float(item.get('score')):.1f}"
        valuation = item.get("valuation", {}) or {}
        upside = _pct(valuation.get("upside_to_price"))
        open_attr = " open" if idx < 5 else ""
        portfolio = item.get("portfolio") or {}
        source_pills = "".join(f'<span class="pill">{escape(str(source))}</span>' for source in item.get("data_sources", []))
        blocks.append(
            f'<details id="stock-{escape(ticker)}" class="stock"{open_attr}>'
            f'<summary><div class="summary-main"><strong>{escape(ticker)}</strong><span>{escape(str(item.get("name", "")))}</span><span class="pill">分数 {escape(score)}</span><span class="pill">现价 {escape(price)}</span><span class="pill">估值空间 {escape(upside)}</span></div><div class="summary-meta">{escape(str(item.get("trend", "")))}</div></summary>'
            f'<div class="stock-body"><div class="detail-grid"><div>{_stock_chart(item)}{source_pills}<p class="note">{escape(str(item.get("candle_source", "")))}</p></div>'
            f'<div>{_stock_key_stats(item, valuation, portfolio)}{_levels(item)}{_stock_notes(item)}</div></div></div></details>'
        )
    return "\n".join(blocks)


def _stock_key_stats(item: dict[str, Any], valuation: dict[str, Any], portfolio: dict[str, Any]) -> str:
    rows = [
        ("现价", _money(item.get("price"))),
        ("概率加权合理价", _money(valuation.get("weighted_fair_value"))),
        ("相对现价空间", _pct(valuation.get("upside_to_price"))),
        ("持仓占比", _pct(portfolio.get("allocation")) if portfolio else "非持仓"),
        ("IBKR成本", _money(portfolio.get("avg_cost")) if portfolio else "NA"),
        ("估值来源", _valuation_source_cn(str(valuation.get("source", "")))),
        ("Moomoo动作", str(item.get("action", "")) or "无"),
    ]
    html = '<table><tbody>'
    for key, value in rows:
        html += f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>"
    html += "</tbody></table>"
    scenarios = valuation.get("scenarios", [])
    if scenarios:
        html += "<h3>情景估值</h3><div class='level-list'>" + "<br>".join(
            f"{escape(str(s.get('name')))}：{escape(_money(s.get('fair_value')))} / 概率 {escape(_pct(s.get('probability')))} / {escape(str(s.get('driver', '')))}"
            for s in scenarios
        ) + "</div>"
    return html


def _levels(item: dict[str, Any]) -> str:
    supports = item.get("supports", [])[:6]
    resistances = item.get("resistances", [])[:6]
    return (
        '<h3>支撑位 / 压力位</h3><div class="levels">'
        f'<div class="level-list"><strong>支撑</strong><br>{_level_lines(supports)}</div>'
        f'<div class="level-list"><strong>压力</strong><br>{_level_lines(resistances)}</div>'
        "</div>"
    )


def _level_lines(levels: list[dict[str, Any]]) -> str:
    if not levels:
        return "暂无"
    rows = []
    for item in levels:
        label = ",".join(str(label) for label in item.get("labels", [])[:4])
        rows.append(f"{_money(item.get('low'))}-{_money(item.get('high'))}（{escape(str(item.get('strength', '')))}） {escape(label)}")
    return "<br>".join(rows)


def _stock_notes(item: dict[str, Any]) -> str:
    thesis = item.get("thesis") or {}
    sizing = item.get("sizing") or {}
    sentiment = item.get("sentiment") or {}
    catalysts = item.get("catalysts") or []
    smart = item.get("smart_money") or {}
    parts = ["<h3>研究备注</h3><ul>"]
    if thesis:
        parts.append(f"<li>投资逻辑：{escape(str(thesis.get('thesis', '')))}；状态：{escape(str(thesis.get('status', '')))}</li>")
    if sizing:
        parts.append(f"<li>仓位建议：{escape(_action_cn(str(sizing.get('suggested_action', ''))))}，建议上限 {escape(_pct(sizing.get('max_allocation')))}</li>")
    if sentiment:
        parts.append(f"<li>情绪：{escape(_sentiment_cn(str(sentiment.get('label', ''))))}，投机风险 {escape(str(sentiment.get('speculation_risk', '')))}</li>")
    if smart:
        parts.append(f"<li>聪明钱：{escape(_smart_money_cn(str(smart.get('smart_money_signal', ''))))}；{escape(str(smart.get('notes', '')))}</li>")
    for catalyst in catalysts[:2]:
        parts.append(f"<li>催化：{escape(str(catalyst.get('event', '')))} / {escape(str(catalyst.get('date', '')))} / {escape(str(catalyst.get('watch_item', '')))}</li>")
    if len(parts) == 1:
        parts.append("<li>暂无额外研究备注。</li>")
    parts.append("</ul>")
    return "".join(parts)


def _stock_chart(item: dict[str, Any]) -> str:
    candles = item.get("candles", [])[-60:]
    if not candles:
        return '<div class="chart-wrap"><p class="note">暂无日K数据。</p></div>'
    width, height = 760, 270
    left, right, top, bottom = 48, 86, 16, 30
    plot_w = width - left - right
    plot_h = height - top - bottom
    levels = (item.get("supports", [])[:5], item.get("resistances", [])[:5])
    values = []
    for candle in candles:
        values.extend([float(candle.get("low", 0.0)), float(candle.get("high", 0.0))])
    for group in levels:
        for level in group:
            values.append(float(level.get("center", level.get("low", 0.0)) or 0.0))
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        max_v += 1.0
        min_v -= 1.0
    pad = (max_v - min_v) * 0.08
    min_v -= pad
    max_v += pad

    def y(value: float) -> float:
        return top + (max_v - value) / (max_v - min_v) * plot_h

    step = plot_w / max(len(candles), 1)
    parts = [f'<div class="chart-wrap"><svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(str(item.get("ticker", "")))} 日K图">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    for i in range(5):
        gy = top + plot_h * i / 4
        val = max_v - (max_v - min_v) * i / 4
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width-right}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="8" y="{gy+4:.1f}" font-size="11" fill="#64748b">{val:.1f}</text>')
    for idx, candle in enumerate(candles):
        x = left + idx * step + step / 2
        open_v, high_v, low_v, close_v = (float(candle[key]) for key in ["open", "high", "low", "close"])
        color = "#13823b" if close_v >= open_v else "#b42318"
        body_y = min(y(open_v), y(close_v))
        body_h = max(abs(y(open_v) - y(close_v)), 2.0)
        body_w = max(min(step * 0.55, 7), 3)
        parts.append(f'<line x1="{x:.1f}" y1="{y(high_v):.1f}" x2="{x:.1f}" y2="{y(low_v):.1f}" stroke="{color}" stroke-width="1.2"/>')
        parts.append(f'<rect x="{x-body_w/2:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" fill="{color}" opacity=".9"/>')
    for name, group, color in [("支撑", levels[0], "#13823b"), ("压力", levels[1], "#b45309")]:
        for level in group:
            center = float(level.get("center", level.get("low", 0.0)) or 0.0)
            ly = y(center)
            label = f"{name} {center:.1f}"
            parts.append(f'<line x1="{left}" y1="{ly:.1f}" x2="{width-right}" y2="{ly:.1f}" stroke="{color}" stroke-width="1.2" stroke-dasharray="5 4"/>')
            parts.append(f'<text x="{width-right+6}" y="{ly+4:.1f}" font-size="11" fill="{color}">{escape(label)}</text>')
    first_date = str(candles[0].get("time", ""))
    last_date = str(candles[-1].get("time", ""))
    parts.append(f'<text x="{left}" y="{height-9}" font-size="11" fill="#64748b">{escape(first_date)}</text>')
    parts.append(f'<text x="{width-right-76}" y="{height-9}" font-size="11" fill="#64748b">{escape(last_date)}</text>')
    parts.append("</svg></div>")
    return "".join(parts)


def _integration_row(item: dict[str, Any]) -> list[str]:
    status = '<span class="ok">已可用</span>' if item.get("available") else '<span class="warn">未安装</span>'
    return [escape(str(item.get("name", ""))), escape(str(item.get("role", ""))), status]


def _data_source_row(item: dict[str, Any]) -> list[str]:
    return [escape(str(item.get("name", ""))), escape(str(item.get("role", ""))), escape(_status_cn(str(item.get("status", "")))), escape(str(item.get("notes", "")))]


def _behavior_cards(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">今天没有触发明显行为偏差。</p>'
    cards = []
    for item in items:
        cards.append(
            '<div class="card">'
            f'<strong>{escape(_behavior_label_cn(str(item.get("label", ""))))}</strong> '
            f'<span class="pill">{escape(_severity_cn(str(item.get("severity", ""))))}</span>'
            f'<p>{escape(_behavior_evidence_cn(str(item.get("evidence", ""))))}</p>'
            f'<p class="note">{escape(_behavior_reminder_cn(str(item.get("reminder", ""))))}</p>'
            "</div>"
        )
    return "\n".join(cards)


def _industry_map(industry_map: dict[str, Any]) -> str:
    if not industry_map:
        return '<p class="note">暂无行业地图。</p>'
    blocks = []
    for theme, layers in industry_map.items():
        blocks.append(f"<h3>{escape(str(theme))}</h3>")
        for layer, nodes in layers.items():
            names = ", ".join(f"{node.get('ticker')}({node.get('company')})" for node in nodes)
            blocks.append(f'<p><span class="pill">{escape(str(layer))}</span>{escape(names)}</p>')
    return "\n".join(blocks)


def _market_regime_readout(regime: dict[str, Any]) -> str:
    score = _pct(regime.get("risk_on_score"))
    name = _regime_cn(regime.get("name", ""))
    return (
        '<p class="note">'
        f"当前市场被归类为 <strong>{escape(name)}</strong>，风险偏好分数 {escape(score)}。"
        "这个判断会直接影响选股：风险偏好强时仍可看成长和动量，但现在模型已经把风险因子提高，"
        "所以高波动、深回撤、同主题过度集中的股票会被压低仓位。"
        "</p>"
    )


def _regime_cn(value: object) -> str:
    return {"AI Growth Environment": "AI成长环境", "Risk On": "风险偏好开启", "Risk Off": "风险规避", "Transitional": "切换期"}.get(str(value), str(value))


def _reason_cn(value: str) -> str:
    return (
        value.replace("Risk-aware AI growth tape: growth and momentum still lead, but risk gets a dedicated 10% weight", "AI 主线仍占优，但风险因子固定提高到 10%，成长/动量/质量/估值整体降权。")
        .replace("Risk-on but not blind risk-on: factor weights are reduced 10% and risk gets 10%", "风险偏好开启，但不是盲目进攻；动态因子整体降 10%，风险因子提高到 10%。")
        .replace("Risk-off regime: risk control becomes a primary factor, while growth and momentum are cut", "风险规避阶段，风险控制成为核心因子，成长和动量明显降权。")
        .replace("Mixed tape: dynamic factors are reduced and risk receives a larger explicit weight", "市场切换期，动态因子降权，风险因子明显提高。")
    )


def _driver_cn(value: str) -> str:
    mapping = {
        "VIX is calm": "VIX 较低，说明市场暂时没有强烈避险定价，但仍要观察是否快速上穿 20/25",
        "Nasdaq is above its long-term trend": "纳指仍在长期趋势上方，成长股环境尚未被趋势破坏",
        "Market breadth is supportive": "市场宽度较好，说明上涨不是只靠极少数龙头支撑",
        "Credit stress is contained": "信用利差可控，暂未出现明显流动性/信用风险扩散",
        "Rates are not pressuring duration assets": "利率变化暂未明显压制高久期成长股估值",
        "Fed policy bias is supportive": "美联储政策口径偏支持流动性",
        "Fed policy bias is restrictive": "美联储政策偏紧，会压制估值和风险偏好",
        "Treasury yields are rising quickly": "美债收益率快速上行，成长股估值压力上升",
        "VIX is elevated": "VIX 偏高，市场波动和止损风险上升",
        "Market breadth is weak": "市场宽度走弱，反弹质量下降",
        "Credit spreads are widening": "信用利差走阔，风险资产需要降杠杆/降仓位",
        "No major caution triggered": "暂无重大警报",
    }
    return mapping.get(value, value)


def _source_cn(value: str) -> str:
    return {
        "moomoo_placeholder": "Moomoo 占位，待真实接入",
        "moomoo_watchlist_export": "Moomoo 自选股导出",
        "manual_research": "手动研究池",
        "ibkr_local_snapshot": "IBKR 本地持仓快照",
    }.get(value, value)


def _priority_cn(value: str) -> str:
    return {"top_research": "最高优先级", "high_research": "高优先级", "watch": "观察", "early_or_low_priority": "早期线索"}.get(value, value)


def _history_cn(value: str, first_seen: object, prior_score: object) -> str:
    if value == "previously_recommended":
        suffix = "" if prior_score is None else f"，上次分数 {prior_score}"
        return f"之前推荐过，首次 {first_seen}{suffix}"
    if value == "new_today":
        return "今日新增"
    return value or "未记录"


def _smart_money_cn(value: str) -> str:
    return {"accumulation": "机构/内部人偏增持", "distribution": "资金流出", "insider_selling_watch": "内部人卖出需观察", "neutral": "中性"}.get(value, value)


def _sentiment_cn(value: str) -> str:
    return {"Fundamental driven": "基本面驱动", "Speculation driven": "投机情绪驱动", "Mixed": "混合驱动"}.get(value, value)


def _action_cn(value: str) -> str:
    return {"trim_or_hold": "减仓或持有观察", "hold_or_watch": "持有或观察", "eligible_to_add": "风险预算内可研究加仓"}.get(value, value)


def _allocation_action_cn(value: str) -> str:
    return {
        "trim_to_target": "降到目标/不加仓",
        "add_if_cash_available": "有条件分批加仓",
        "wait_for_cash_or_trim": "现金不足，等待",
        "hold_near_target": "接近目标，持有",
        "cash_reserve_hold": "现金储备，持有",
    }.get(value, value)


def _allocation_side_cn(value: str) -> str:
    return {
        "sell_or_stop_adding": "先卖出/停止加仓",
        "reserve_cash": "保留现金",
        "conditional_buy": "有条件买入",
    }.get(value, value)


def _trade_decision_cn(value: str) -> str:
    return {
        "reduce_fundamental_risk": "基本面走弱减仓",
        "trim_valuation_risk": "估值风险减仓",
        "trim_speculative_strength": "消息投机减仓",
        "reduce_low_conviction": "低确定性降权",
        "hold_winner_no_forced_trim": "强票持有/不强制卖",
        "hold_risk_limit_no_add": "仓位约束/不加",
        "buy_pullback_or_breakout": "强逻辑等买点",
        "wait_for_cash": "等现金",
        "staged_buy_near_support": "支撑区分批",
        "do_not_chase_wait_breakout": "不追高/等突破",
        "watch_only": "观察",
        "wait_for_pullback": "等回调",
        "no_trade_until_data": "数据不足不交易",
    }.get(value, value)


def _decision_class(value: str) -> str:
    if value in {"reduce_fundamental_risk", "trim_valuation_risk", "trim_speculative_strength", "reduce_low_conviction"}:
        return "bad"
    if value in {"staged_buy_near_support", "buy_pullback_or_breakout"}:
        return "ok"
    return "warn"


def _risk_budget_status_cn(value: str) -> str:
    return {
        "within_target": "在目标内",
        "above_target": "高于目标",
        "below_target": "低于目标",
    }.get(value, value)


def _earnings_status_cn(value: str) -> str:
    return {
        "reported_needs_analysis": "已出，需要解析",
        "today": "今天",
        "upcoming": "即将公布",
        "date_unknown": "日期待确认",
    }.get(value, value)


def _earnings_review_status_cn(value: str) -> str:
    return {
        "analyzed": "已解析",
        "queued": "待原始数据",
    }.get(value, value)


def _status_cn(value: str) -> str:
    return {
        "pending_connection": "待接入",
        "adapter_ready_not_installed": "适配层已好，依赖未安装",
        "installed_adapter_ready": "依赖已安装，适配层可用",
        "local_snapshot": "本地快照可用",
    }.get(value, value)


def _valuation_source_cn(value: str) -> str:
    return {
        "scenario_model": "2027E低/中/高三情景估值",
        "moomoo_forward_valuation": "Moomoo前瞻估值带",
        "not_available": "暂无估值",
    }.get(value, value)


def _severity_cn(value: str) -> str:
    return {"medium": "中", "high": "高", "low": "低"}.get(value, value)


def _score_note_cn(value: str) -> str:
    return value.replace("Strong", "强项").replace("Weak", "弱项").replace("growth", "成长").replace("quality", "质量").replace("momentum", "动量").replace("value", "估值").replace("risk", "风险")


def _sizing_reason_cn(value: str) -> str:
    mapping = {
        "Moderate volatility limits allocation": "波动率较高，限制仓位",
        "High volatility caps allocation": "波动率高，严格限制仓位",
        "Large historical drawdown requires tighter sizing": "历史回撤较大，需要更严格仓位",
        "Sector exposure is already high": "行业暴露已经偏高",
        "High correlation reduces diversification benefit": "相关性较高，分散效果有限",
        "Current allocation is above recommended maximum": "当前仓位高于建议上限",
        "Allocation is within the current risk budget": "仓位在当前风险预算内",
        "High score leaves room under the risk cap": "评分较高，仍在风险上限以内",
    }
    return mapping.get(value, value)


def _sentiment_reason_cn(value: str) -> str:
    return value.replace("Positive news is backed by stronger evidence", "正面新闻有较强证据支持").replace("Social heat is high while evidence quality is weak", "社交热度高，但证据质量偏弱")


def _behavior_label_cn(value: str) -> str:
    return value.replace("Chasing strength after large one-day moves", "大涨后追入倾向")


def _behavior_evidence_cn(value: str) -> str:
    return value.replace("2 buys occurred after prior-day moves above 8%; success rate 50%.", "有 2 次买入发生在前一日涨幅超过 8% 之后，成功率约 50%。")


def _behavior_reminder_cn(value: str) -> str:
    return value.replace("Slow down after large single-day moves and require thesis confirmation before adding.", "单日大涨后放慢节奏，先确认投资逻辑和证据，再考虑加仓。")
