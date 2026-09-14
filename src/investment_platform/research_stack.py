"""Research framework registry and search playbooks.

The platform treats large open-source projects as modular engines. They are not
hard dependencies for the daily dashboard; each record explains what should be
reused directly, what needs an adapter, and how its output should affect the
research workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchFramework:
    name: str
    repo: str
    layer: str
    role: str
    direct_use: str
    platform_adapter: str
    added_by_platform: str
    decision_policy: str
    priority: int
    keywords: tuple[str, ...]


FRAMEWORKS = [
    ResearchFramework(
        name="Microsoft Qlib",
        repo="https://github.com/microsoft/qlib",
        layer="核心量化选股框架",
        role="机构式因子研究、模型训练、回测、组合优化和研究实验管理。",
        direct_use="数据处理、因子建模、LightGBM/ML模型、回测流水线和因子绩效分析思想。",
        platform_adapter="接入到 Multi Factor Model / Backtesting / Factor Lab；先以 adapter_ready 状态登记，等本地 Qlib 数据目录准备好再运行。",
        added_by_platform="把 IBKR/Moomoo/估值/财报/新闻结果合并成个人股票池，并把 Qlib 分数转换成中文可解释排名。",
        decision_policy="可以影响选股排序和回测证据，但不能单独触发买卖。",
        priority=1,
        keywords=("qlib", "factor", "backtest", "alpha", "quant", "因子", "回测", "选股"),
    ),
    ResearchFramework(
        name="OpenBB",
        repo="https://github.com/OpenBB-finance/OpenBB",
        layer="股票研究终端与数据入口",
        role="统一读取股票、财务、宏观、新闻、期权、分析师和公司资料。",
        direct_use="OpenBB Platform 的 Python 数据接口和供应商路由能力。",
        platform_adapter="接入 Data Layer；优先用于财务、新闻、宏观、期权和公司资料刷新。",
        added_by_platform="做数据质量门、缓存、中文摘要、持仓/自选股优先级和缺失降级提示。",
        decision_policy="作为数据源，不直接产生买卖结论。",
        priority=1,
        keywords=("openbb", "terminal", "data", "news", "macro", "fundamentals", "数据", "新闻", "宏观"),
    ),
    ResearchFramework(
        name="financial-research-agent",
        repo="https://github.com/kherarudransh-oss/financial-research-agent",
        layer="财报与 SEC 研究代理",
        role="SEC filings RAG、FinBERT 情绪、风险因素变化和定性/定量背离检测。",
        direct_use="EDGAR 检索、filing chunking、ChromaDB/RAG、FinBERT 分段情绪和 divergence detector 思路。",
        platform_adapter="接入 Earnings Analysis / Thesis Tracker；财报前后自动生成重点、风险语言变化和管理层语气变化。",
        added_by_platform="把 filing 信号映射到投资逻辑失效条件、持仓动作、财报日历和日报首屏。",
        decision_policy="只作为财报证据和 thesis 状态，不直接给买卖金额。",
        priority=2,
        keywords=("financial research agent", "sec", "edgar", "filings", "finbert", "rag", "earnings", "财报", "10-k", "10-q"),
    ),
    ResearchFramework(
        name="ai-berkshire",
        repo="https://github.com/xbtlin/ai-berkshire",
        layer="价值投资与公司质量检查清单",
        role="用巴菲特/芒格/段永平/李录式框架做商业模式、护城河、管理层和长期价值研究。",
        direct_use="价值投资研究 skill、长期复利公司检查清单、多代理反方分析和研究报告结构。",
        platform_adapter="接入 Scenario Valuation / Investment Thesis；生成低中高三档 2027E 估值和价值投资红旗。",
        added_by_platform="把长期质量判断和短线支撑压力、现金约束、组合波动放在同一张股票详情卡里。",
        decision_policy="可以提高/降低质量与估值信心，但买点仍必须经过价格、风险和现金约束。",
        priority=2,
        keywords=("ai berkshire", "berkshire", "buffett", "munger", "value", "valuation", "价值", "护城河", "管理层"),
    ),
    ResearchFramework(
        name="FinRL",
        repo="https://github.com/AI4Finance-Foundation/FinRL",
        layer="强化学习交易实验沙盒",
        role="训练/测试/交易三段式 DRL 原型，用于仓位与再平衡实验。",
        direct_use="portfolio allocation、stock trading 环境、A2C/DDPG/PPO/TD3/SAC 基准和实验流程。",
        platform_adapter="接入 Research Lab，不接入主买卖建议；只比较策略稳定性、回撤和换手。",
        added_by_platform="加上用户风险预算、不可自动下单边界、IBKR 只读校验和中文实验摘要。",
        decision_policy="默认不影响真实持仓建议；只有通过回测、样本外、风控审查后才显示为辅助证据。",
        priority=4,
        keywords=("finrl", "reinforcement learning", "rl", "ppo", "a2c", "ddpg", "仓位实验", "强化学习"),
    ),
]


def build_research_stack(integration_rows: list[Any]) -> list[dict[str, Any]]:
    """Return framework records with current local availability labels."""

    availability = {str(getattr(row, "name", "")).lower(): bool(getattr(row, "available", False)) for row in integration_rows}
    aliases = {
        "Microsoft Qlib": "qlib",
        "OpenBB": "openbb",
        "financial-research-agent": "financial-research-agent",
        "ai-berkshire": "ai-berkshire",
        "FinRL": "finrl",
    }
    output = []
    for item in sorted(FRAMEWORKS, key=lambda row: (row.priority, row.name.lower())):
        key = aliases[item.name]
        installed = availability.get(key, False)
        status = "installed" if installed else "adapter_ready"
        output.append(
            {
                "name": item.name,
                "repo": item.repo,
                "layer": item.layer,
                "role": item.role,
                "direct_use": item.direct_use,
                "platform_adapter": item.platform_adapter,
                "added_by_platform": item.added_by_platform,
                "decision_policy": item.decision_policy,
                "priority": item.priority,
                "status": status,
                "status_label": "已安装可调用" if installed else "已登记，待安装/待配置",
                "keywords": list(item.keywords),
            }
        )
    return output


def build_search_playbooks() -> list[dict[str, Any]]:
    """Describe upgraded discovery routes for the website and daily email."""

    return [
        {
            "name": "机构量化选股",
            "engine": "Microsoft Qlib",
            "goal": "从美股股票池中找稳定 alpha，而不是只看热门股票。",
            "signals": ["多因子暴露", "横截面排名", "样本外回测", "因子IC/分层收益", "风险暴露"],
            "reject_if": "回测只靠单一时期、换手过高、样本外失效或因子解释不清。",
        },
        {
            "name": "研究终端数据刷新",
            "engine": "OpenBB",
            "goal": "统一更新财务、新闻、宏观、期权和公司资料。",
            "signals": ["价格/财务", "分析师预期", "宏观利率", "新闻", "期权/波动率"],
            "reject_if": "关键字段缺失或供应商数据时点落后，必须降级并提示。",
        },
        {
            "name": "财报/SEC 深研",
            "engine": "financial-research-agent",
            "goal": "捕捉财报、10-K/10-Q/8-K 里的语气变化和风险语言升级。",
            "signals": ["SEC RAG", "FinBERT分段情绪", "风险因素变化", "MD&A语气", "定量/定性背离"],
            "reject_if": "只有标题新闻、没有原始 filing 或没有同比/环比语气变化。",
        },
        {
            "name": "价值投资审查",
            "engine": "ai-berkshire",
            "goal": "判断公司是不是值得长期跟踪的好生意，以及贵不贵。",
            "signals": ["护城河", "管理层", "自由现金流", "ROIC", "安全边际", "反方观点"],
            "reject_if": "商业模式不稳定、现金流质量差、估值只靠故事支撑。",
        },
        {
            "name": "强化学习实验",
            "engine": "FinRL",
            "goal": "作为仓位和再平衡实验，不直接替代人类决策。",
            "signals": ["训练/测试/交易分割", "回撤", "换手", "策略稳定性", "基准对比"],
            "reject_if": "样本外差、交易成本后无优势、动作不可解释或风险超过账户预算。",
        },
    ]
