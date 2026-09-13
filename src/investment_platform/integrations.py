"""Lazy adapters for open-source institutional research building blocks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IntegrationStatus:
    name: str
    package: str
    available: bool
    role: str
    install_hint: str


INTEGRATIONS = [
    IntegrationStatus("Qlib", "qlib", False, "factor research, model training, backtesting", "pip install '.[institutional]'"),
    IntegrationStatus("OpenBB", "openbb", False, "market, fundamental, macro, news, options data", "pip install '.[institutional]'"),
    IntegrationStatus("FinRL", "finrl", False, "reinforcement-learning research sandbox for allocation experiments", "install FinRL in a research-only environment"),
    IntegrationStatus("Riskfolio-Lib", "riskfolio", False, "portfolio risk and allocation optimization", "pip install '.[institutional]'"),
    IntegrationStatus("Alphalens Reloaded", "alphalens", False, "factor IC and quantile performance analysis", "pip install '.[institutional]'"),
    IntegrationStatus("EdgarTools", "edgar", False, "SEC filings, ownership, 13F and Form 4 research", "pip install '.[institutional]'"),
    IntegrationStatus("QuantStats", "quantstats", False, "performance reporting", "pip install '.[institutional]'"),
    IntegrationStatus("vectorbt", "vectorbt", False, "fast vectorized backtesting", "pip install '.[institutional]'"),
    IntegrationStatus(
        "TradingAgents",
        "tradingagents",
        False,
        "multi-agent AI investment research",
        "install in .venv-tradingagents with pip install '.[ai-agents]'",
    ),
    IntegrationStatus(
        "financial-research-agent",
        "fin_research_agent",
        False,
        "SEC RAG, FinBERT filing sentiment, and cross-signal divergence research",
        "install from https://github.com/kherarudransh-oss/financial-research-agent",
    ),
    IntegrationStatus(
        "ai-berkshire",
        "ai_berkshire",
        False,
        "value-investing research skills and long-term quality checklist",
        "install Codex skills from https://github.com/xbtlin/ai-berkshire",
    ),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _external_python_has_module(python_path: Path, module: str) -> bool:
    if not python_path.exists():
        return False
    try:
        result = subprocess.run(
            [str(python_path), "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module}') else 1)"],
            cwd=_project_root(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _codex_skill_installed(skill_names: list[str]) -> bool:
    codex_home = Path.home() / ".codex"
    skills_dir = codex_home / "skills"
    return all((skills_dir / skill / "SKILL.md").exists() for skill in skill_names)


def _integration_available(package: str) -> tuple[bool, str | None]:
    if find_spec(package) is not None:
        return True, None
    if package == "tradingagents":
        python_path = _project_root() / ".venv-tradingagents" / "bin" / "python"
        if _external_python_has_module(python_path, package):
            return True, ".venv-tradingagents"
    if package == "finrl":
        python_path = _project_root() / ".venv-finrl" / "bin" / "python"
        if _external_python_has_module(python_path, package):
            return True, ".venv-finrl"
    if package == "fin_research_agent":
        python_path = _project_root() / ".venv-financial-research-agent" / "bin" / "python"
        if _external_python_has_module(python_path, package):
            return True, ".venv-financial-research-agent"
    if package == "ai_berkshire" and _codex_skill_installed(
        [
            "investment-research",
            "investment-checklist",
            "quality-screen",
            "portfolio-review",
            "thesis-tracker",
        ]
    ):
        return True, "~/.codex/skills"
    return False, None


def integration_status() -> list[IntegrationStatus]:
    """Return availability for each optional open-source integration."""

    statuses: list[IntegrationStatus] = []
    for integration in INTEGRATIONS:
        available, source = _integration_available(integration.package)
        role = integration.role
        if source:
            role = f"{role} ({source})"
        statuses.append(
            IntegrationStatus(
                name=integration.name,
                package=integration.package,
                available=available,
                role=role,
                install_hint=integration.install_hint,
            )
        )
    return statuses


class LazyIntegration:
    """Base class that imports optional dependencies only when used."""

    package: str
    display_name: str

    def _module(self) -> Any:
        if find_spec(self.package) is None:
            raise RuntimeError(f"{self.display_name} is not installed. Install optional institutional dependencies first.")
        return import_module(self.package)


class QlibAdapter(LazyIntegration):
    package = "qlib"
    display_name = "Qlib"

    def init(self, **kwargs: Any) -> Any:
        module = self._module()
        return module.init(**kwargs)


class OpenBBAdapter(LazyIntegration):
    package = "openbb"
    display_name = "OpenBB"

    def obb(self) -> Any:
        module = self._module()
        return getattr(module, "obb", module)


class RiskfolioAdapter(LazyIntegration):
    package = "riskfolio"
    display_name = "Riskfolio-Lib"

    def portfolio_class(self) -> Any:
        module = self._module()
        return getattr(module, "Portfolio")


class AlphalensAdapter(LazyIntegration):
    package = "alphalens"
    display_name = "Alphalens Reloaded"

    def module(self) -> Any:
        return self._module()


class EdgarToolsAdapter(LazyIntegration):
    package = "edgar"
    display_name = "EdgarTools"

    def module(self) -> Any:
        return self._module()


class TradingAgentsAdapter(LazyIntegration):
    package = "tradingagents"
    display_name = "TradingAgents"

    def graph_class(self) -> Any:
        self._module()
        graph_module = import_module("tradingagents.graph.trading_graph")
        return getattr(graph_module, "TradingAgentsGraph")
