"""YAML 配置读取工具。"""

from pathlib import Path

import yaml


class ConfigLoader:
    """读取 YAML 文件，并返回普通 Python 字典。"""

    def __init__(self, config_path):
        """保存配置文件路径，真正读取发生在 load 方法中。"""
        self.config_path = Path(config_path)

    def load(self):
        """读取 YAML 配置文件并返回字典；空文件会返回空字典。"""
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return config


def deep_update(base, updates):
    """递归合并两个字典，常用于配置覆盖。"""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def require_keys(mapping, keys, context):
    """检查必填字段是否存在，缺失时抛出清晰错误。"""
    missing = [key for key in keys if not mapping.get(key)]
    if missing:
        raise ValueError(f"Missing required {context}: {', '.join(missing)}")
