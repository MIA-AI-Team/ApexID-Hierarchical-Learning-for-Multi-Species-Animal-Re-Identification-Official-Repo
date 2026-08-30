import copy
import os
import re
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
_ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_string(value, environment):
    """Expand shell-style variables without invoking a shell."""
    result = value
    for _ in range(100):
        unresolved = []

        def replace(match):
            name, fallback = match.group(1), match.group(2)
            configured = environment.get(name)
            if configured not in {None, ""}:
                return str(configured)
            if fallback is not None:
                return fallback
            unresolved.append(name)
            return match.group(0)

        expanded = _ENVIRONMENT_REFERENCE.sub(replace, result)
        if unresolved:
            names = ", ".join(sorted(set(unresolved)))
            raise ValueError(f"Unresolved configuration environment variable(s): {names}")
        if expanded == result or not _ENVIRONMENT_REFERENCE.search(expanded):
            return expanded
        result = expanded
    raise ValueError(f"Environment-variable expansion did not converge for {value!r}")


def expand_config_environment(config, environment=None):
    """Recursively expand ``${VAR}`` and ``${VAR:-fallback}`` references."""
    environment = os.environ if environment is None else environment
    if isinstance(config, str):
        return _expand_string(config, environment)
    if isinstance(config, dict):
        return {key: expand_config_environment(value, environment) for key, value in config.items()}
    if isinstance(config, list):
        return [expand_config_environment(value, environment) for value in config]
    if isinstance(config, tuple):
        return tuple(expand_config_environment(value, environment) for value in config)
    return config


def load_config(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Return a deep-copied config with recursive environment expansion."""
    if config is not None:
        return expand_config_environment(copy.deepcopy(config))

    with open(config_path, "r", encoding="utf-8") as f:
        return expand_config_environment(yaml.safe_load(f) or {})


__all__ = ["DEFAULT_CONFIG_PATH", "expand_config_environment", "load_config"]
