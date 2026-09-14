"""Industry competitive map helpers."""

from __future__ import annotations

from collections import defaultdict

from investment_platform.schemas import IndustryNode


def build_industry_nodes(rows: list[dict[str, object]]) -> list[IndustryNode]:
    return [
        IndustryNode(
            theme=str(row["theme"]),
            layer=str(row["layer"]),
            ticker=str(row["ticker"]),
            company=str(row.get("company", row["ticker"])),
            moat=str(row.get("moat", "")),
            competitive_risk=str(row.get("competitive_risk", "")),
        )
        for row in rows
    ]


def group_by_theme(nodes: list[IndustryNode]) -> dict[str, dict[str, list[IndustryNode]]]:
    grouped: dict[str, dict[str, list[IndustryNode]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        grouped[node.theme][node.layer].append(node)
    return {theme: dict(layers) for theme, layers in grouped.items()}
