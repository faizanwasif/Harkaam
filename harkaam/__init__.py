"""
Harkaam - A modular framework for building agent-based systems with multiple agent architectures.
"""

__version__ = "0.1.0"

from harkaam.agents.base import BaseAgent
from harkaam.utils import set_api_key, get_api_key, initialize_config, save_config

# Convenience alias
Agent = BaseAgent

def setup(openai_api_key=None, anthropic_api_key=None, save=True):
    """
    Set up the Harkaam framework with API keys.
    
    Args:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        save: Whether to save the API keys to the configuration file
    """
    initialize_config()
    
    if openai_api_key:
        set_api_key("openai", openai_api_key)
    
    if anthropic_api_key:
        set_api_key("anthropic", anthropic_api_key)
    
    if save and (openai_api_key or anthropic_api_key):
        save_config()

__all__ = [
    "Agent",
    "BaseAgent",
    "setup",
    "set_api_key",
    "get_api_key",
]