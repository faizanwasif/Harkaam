"""
Utility script to set up API keys for the Harkaam framework.
"""

import os
import sys
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from harkaam.utils import set_api_key, initialize_config, save_config, get_api_key

def main():
    parser = argparse.ArgumentParser(description="Set up API keys for the Harkaam framework")
    parser.add_argument("--openai", help="OpenAI API key")
    parser.add_argument("--anthropic", help="Anthropic API key")
    parser.add_argument("--show", action="store_true", help="Show current API keys (masked)")
    args = parser.parse_args()
    
    # Initialize configuration
    initialize_config()
    
    # Show current API keys if requested
    if args.show:
        openai_key = get_api_key("openai")
        anthropic_key = get_api_key("anthropic")
        
        print("Current API keys:")
        print(f"OpenAI: {mask_api_key(openai_key)}")
        print(f"Anthropic: {mask_api_key(anthropic_key)}")
        print()
    
    # Set API keys if provided
    if args.openai:
        set_api_key("openai", args.openai)
        print("OpenAI API key set")
    
    if args.anthropic:
        set_api_key("anthropic", args.anthropic)
        print("Anthropic API key set")
    
    # Save configuration if any keys were set
    if args.openai or args.anthropic:
        save_config()
        print("Configuration saved")
    
    # If no arguments were provided, prompt for API keys
    if not args.show and not args.openai and not args.anthropic:
        print("Enter API keys (leave blank to keep current value):")
        
        # Get current keys
        current_openai = get_api_key("openai")
        current_anthropic = get_api_key("anthropic")
        
        # Prompt for OpenAI API key
        if current_openai:
            print(f"Current OpenAI API key: {mask_api_key(current_openai)}")
        openai_key = input("OpenAI API key: ").strip()
        if openai_key:
            set_api_key("openai", openai_key)
            print("OpenAI API key set")
        
        # Prompt for Anthropic API key
        if current_anthropic:
            print(f"Current Anthropic API key: {mask_api_key(current_anthropic)}")
        anthropic_key = input("Anthropic API key: ").strip()
        if anthropic_key:
            set_api_key("anthropic", anthropic_key)
            print("Anthropic API key set")
        
        # Save configuration if any keys were set
        if openai_key or anthropic_key:
            save_config()
            print("Configuration saved")

def mask_api_key(api_key):
    """Mask an API key for display."""
    if not api_key:
        return "Not set"
    
    # Show first 4 and last 4 characters
    if len(api_key) <= 8:
        return "****"
    
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]

if __name__ == "__main__":
    main()