"""Configuration loading utilities."""

import yaml
from typing import Dict, Any


def load_config(config_path: str = "data_configs.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
