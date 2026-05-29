from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default_thresholds.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_source_path"] = str(config_path)
    data.setdefault("metadata", {})
    data.setdefault("thresholds", {})
    return data


def thresholds(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("thresholds", {})


def threshold(config: dict[str, Any], key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    return thresholds(config).get(key, default or {})
