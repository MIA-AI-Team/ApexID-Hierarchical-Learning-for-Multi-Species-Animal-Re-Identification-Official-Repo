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


def _merge_config(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path, parents=()):
    path = Path(path).resolve()
    if path in parents:
        chain = " -> ".join(str(item) for item in (*parents, path))
        raise ValueError(f"Circular config inheritance: {chain}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    if "extends" not in config:
        return config

    base_path = config.pop("extends")
    if not isinstance(base_path, str) or not base_path.strip() or Path(base_path).is_absolute():
        raise ValueError(f"Expected extends to be a relative YAML path in {path}")
    base = _load_yaml(path.parent / base_path, (*parents, path))
    return _merge_config(base, config)


def load_config(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Resolve YAML inheritance and expand environment variables in the config."""
    if config is not None:
        return expand_config_environment(copy.deepcopy(config))

    return expand_config_environment(_load_yaml(config_path))


__all__ = ["DEFAULT_CONFIG_PATH", "expand_config_environment", "load_config"]
