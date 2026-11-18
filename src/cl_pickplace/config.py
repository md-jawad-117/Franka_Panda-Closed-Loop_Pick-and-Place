"""Config loading. The config is a plain nested dict loaded from YAML."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def override(cfg: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Return a copy of cfg with cfg[a][b]... = value for dotted key 'a.b'."""
    out = copy.deepcopy(cfg)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return out
