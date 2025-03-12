"""
LLM integration module for the Harkaam framework.

This module provides classes for integrating with different LLM providers
such as OpenAI and Anthropic.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from harkaam.utils.config import get_api_key

class BaseLLM(ABC):
    """
    Base class for LLM integrations.
    
    LLM integrations provide a common interface for different
    LLM providers such as OpenAI and Anthropic.
    """
    
    @abstractmethod
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> Tuple[str, str]:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            A tuple of (thinking, response)
        """
        pass

class OpenAILLM(BaseLLM):
    """
    OpenAI LLM integration.
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize a new OpenAI LLM integration.
        
        Args:
            model: The model to use
            api_key: The API key to use (overrides configuration)
        """
        try:
            import openai
        except ImportError:
            raise ImportError("The 'openai' package is required to use the OpenAI LLM")
        
        # Use provided API key or get from configuration
        if api_key is None:
            api_key = get_api_key("openai")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not found. Please provide an API key or set it in the configuration."
                )
        
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> Tuple[str, str]:
        """
        Generate a response from the OpenAI API.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            A tuple of (thinking, response)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # No explicit thinking output from OpenAI API
        thinking = ""
        response_text = response.choices[0].message.content
        
        return thinking, response_text

class AnthropicLLM(BaseLLM):
    """
    Anthropic LLM integration.
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize a new Anthropic LLM integration.
        
        Args:
            model: The model to use
            api_key: The API key to use (overrides configuration)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("The 'anthropic' package is required to use the Anthropic LLM")
        
        # Use provided API key or get from configuration
        if api_key is None:
            api_key = get_api_key("anthropic")
            if api_key is None:
                raise ValueError(
                    "Anthropic API key not found. Please provide an API key or set it in the configuration."
                )
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> Tuple[str, str]:
        """
        Generate a response from the Anthropic API.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            A tuple of (thinking, response)
        """
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # No explicit thinking output from Anthropic API
        thinking = ""
        response_text = response.content[0].text
        
        return thinking, response_text

# Factory function to create an LLM client
def create_llm(provider_model: str, api_key: Optional[str] = None) -> BaseLLM:
    """
    Create an LLM client for the given provider and model.
    
    Args:
        provider_model: The LLM provider and model in format "provider:model"
        api_key: The API key to use
        
    Returns:
        An LLM client
    """
    try:
        provider, model = provider_model.split(":", 1)
    except ValueError:
        raise ValueError(f"Invalid provider_model format: {provider_model}, expected 'provider:model'")
    
    if provider.lower() == "openai":
        return OpenAILLM(model, api_key)
    elif provider.lower() == "anthropic":
        return AnthropicLLM(model, api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")