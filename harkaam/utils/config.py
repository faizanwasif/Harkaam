"""
Configuration utilities for the Harkaam framework.

This module provides functions for managing configuration
settings, including API keys for LLM providers.
"""

import os
from typing import Dict, Optional, Any
import json
from pathlib import Path

# Default configuration paths
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.harkaam")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "config.json")

# Global configuration store
_config: Dict[str, Any] = {}

def initialize_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize the configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        The loaded configuration
    """
    global _config
    
    config_path = config_path or DEFAULT_CONFIG_FILE
    
    # Create default config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Load configuration from file if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                _config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_path}: {e}")
            _config = {}
    else:
        _config = {}
    
    # Load environment variables
    _load_env_vars()
    
    return _config

def _load_env_vars() -> None:
    """
    Load configuration from environment variables.
    
    Looks for environment variables with the prefix HARKAAM_
    and adds them to the configuration.
    """
    for key, value in os.environ.items():
        if key.startswith("HARKAAM_"):
            config_key = key[8:].lower()  # Remove HARKAAM_ prefix and lowercase
            
            # Handle special cases for API keys
            if key == "HARKAAM_OPENAI_API_KEY":
                set_api_key("openai", value)
            elif key == "HARKAAM_ANTHROPIC_API_KEY":
                set_api_key("anthropic", value)
            else:
                _config[config_key] = value

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        The current configuration
    """
    global _config
    
    # Initialize if not already loaded
    if not _config:
        initialize_config()
    
    return _config

def set_config(key: str, value: Any) -> None:
    """
    Set a configuration value.
    
    Args:
        key: The configuration key
        value: The configuration value
    """
    global _config
    
    # Initialize if not already loaded
    if not _config:
        initialize_config()
    
    _config[key] = value

def save_config(config_path: Optional[str] = None) -> None:
    """
    Save the configuration to a file.
    
    Args:
        config_path: Path to the configuration file
    """
    global _config
    
    config_path = config_path or DEFAULT_CONFIG_FILE
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save configuration to file
    with open(config_path, 'w') as f:
        json.dump(_config, f, indent=2)

def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for a provider.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        
    Returns:
        The API key if found, None otherwise
    """
    global _config
    
    # Initialize if not already loaded
    if not _config:
        initialize_config()
    
    # Check for provider-specific API key
    api_keys = _config.get("api_keys", {})
    
    # Try environment variables first
    if provider.upper() == "OPENAI":
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key
    elif provider.upper() == "ANTHROPIC":
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            return env_key
    
    # Fall back to configuration file
    return api_keys.get(provider.lower())

def set_api_key(provider: str, api_key: str) -> None:
    """
    Set the API key for a provider.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        api_key: The API key
    """
    global _config
    
    # Initialize if not already loaded
    if not _config:
        initialize_config()
    
    # Ensure api_keys exists
    if "api_keys" not in _config:
        _config["api_keys"] = {}
    
    _config["api_keys"][provider.lower()] = api_key