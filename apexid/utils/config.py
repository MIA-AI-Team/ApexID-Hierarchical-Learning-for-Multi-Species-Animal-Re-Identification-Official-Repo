from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Return an already-loaded config dict or load one from YAML."""
    if config is not None:
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


__all__ = ["DEFAULT_CONFIG_PATH", "load_config"]
