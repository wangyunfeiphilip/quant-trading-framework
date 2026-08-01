"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyYAML is required to read config.yaml. Install requirements.txt first.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("configuration file must contain a YAML mapping")
    return loaded
