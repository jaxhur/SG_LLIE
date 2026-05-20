"""YAML configuration loading and recursive override helpers."""

from pathlib import Path

import yaml


class ConfigLoader:
    """Load YAML files and expose them as plain nested dictionaries."""

    def __init__(self, config_path):
        """Store the YAML path that will be parsed when `load` is called."""
        self.config_path = Path(config_path)

    def load(self):
        """Read the YAML file and return a dictionary; empty files become `{}`."""
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return config


def deep_update(base, updates):
    """Recursively merge `updates` into `base` and return the mutated `base` dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def require_keys(mapping, keys, context):
    """Raise a clear error when any key from `keys` is missing in `mapping`."""
    missing = [key for key in keys if not mapping.get(key)]
    if missing:
        raise ValueError(f"Missing required {context}: {', '.join(missing)}")
