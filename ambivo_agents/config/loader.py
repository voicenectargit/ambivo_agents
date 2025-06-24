
# ambivo_agents/config/loader.py
"""
Configuration loader for ambivo_agents.
All configurations must be sourced from agent_config.yaml - no defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigurationError(Exception):
    """Raised when configuration is missing or invalid."""
    pass


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from agent_config.yaml.

    Args:
        config_path: Optional path to config file. If None, searches for agent_config.yaml
                    in current directory and parent directories.

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If config file is not found or invalid
    """
    if config_path:
        config_file = Path(config_path)
    else:
        # Search for agent_config.yaml starting from current directory
        config_file = _find_config_file()

    if not config_file or not config_file.exists():
        raise ConfigurationError(
            "agent_config.yaml not found. This file is required for ambivo_agents to function. "
            "Please create agent_config.yaml in your project root or specify the path explicitly."
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not config:
            raise ConfigurationError("agent_config.yaml is empty or contains invalid YAML")

        # Validate required sections
        _validate_config(config)

        #print(f"====config/loader/load_config()Returning config Loaded configuration from {config}")

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in agent_config.yaml: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load agent_config.yaml: {e}")


def _find_config_file() -> Path:
    """Find agent_config.yaml in current directory or parent directories."""
    current_dir = Path.cwd()

    # Check current directory first
    config_file = current_dir / "agent_config.yaml"
    if config_file.exists():
        return config_file

    # Check parent directories
    for parent in current_dir.parents:
        config_file = parent / "agent_config.yaml"
        if config_file.exists():
            return config_file

    return None


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration sections exist.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigurationError: If required sections are missing
    """
    required_sections = ['redis', 'llm']
    missing_sections = []

    for section in required_sections:
        if section not in config:
            missing_sections.append(section)

    if missing_sections:
        raise ConfigurationError(
            f"Required configuration sections missing: {missing_sections}. "
            "Please check your agent_config.yaml file."
        )

    # Validate Redis config
    redis_config = config['redis']
    required_redis_fields = ['host', 'port']
    missing_redis_fields = [field for field in required_redis_fields if field not in redis_config]

    if missing_redis_fields:
        raise ConfigurationError(
            f"Required Redis configuration fields missing: {missing_redis_fields}"
        )

    # Validate LLM config
    llm_config = config['llm']
    has_api_key = any(key in llm_config for key in [
        'openai_api_key', 'anthropic_api_key', 'aws_access_key_id'
    ])

    if not has_api_key:
        raise ConfigurationError(
            "At least one LLM provider API key is required in llm configuration. "
            "Supported providers: openai_api_key, anthropic_api_key, aws_access_key_id"
        )


def get_config_section(section: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get a specific configuration section.

    Args:
        section: Section name (e.g., 'redis', 'llm', 'web_scraping')
        config: Optional config dict. If None, loads from file.

    Returns:
        Configuration section dictionary

    Raises:
        ConfigurationError: If section is not found
    """
    if config is None:
        config = load_config()

    if section not in config:
        raise ConfigurationError(f"Configuration section '{section}' not found in agent_config.yaml")

    return config[section]


def validate_agent_capabilities(config: Dict[str, Any] = None) -> Dict[str, bool]:
    """
    Validate and return available agent capabilities based on configuration.

    Args:
        config: Optional config dict. If None, loads from file.

    Returns:
        Dictionary of capability name -> enabled status
    """
    if config is None:
        config = load_config()

    capabilities = {
        'assistant': True,  # Always available
        'code_execution': True,  # Always available with Docker
        'proxy': True,  # Always available
    }

    # Optional capabilities based on configuration
    agent_caps = config.get('agent_capabilities', {})

    capabilities['web_scraping'] = (
            agent_caps.get('enable_web_scraping', False) and
            'web_scraping' in config
    )

    capabilities['knowledge_base'] = (
            agent_caps.get('enable_knowledge_base', False) and
            'knowledge_base' in config
    )

    capabilities['web_search'] = (
            agent_caps.get('enable_web_search', False) and
            'web_search' in config
    )

    capabilities['media_processing'] = (
            agent_caps.get('enable_media_processing', False) and
            'media_processing' in config
    )

    return capabilities