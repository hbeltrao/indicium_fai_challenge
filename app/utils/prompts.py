"""
Prompt Loading Utility.

This module provides functions to load prompt templates from external .txt files,
improving modularity and making it easier to refine prompts without changing code.
"""
import os
from typing import Optional

def load_prompt(filename: str) -> str:
    """
    Load a prompt from the app/prompts directory.
    
    Args:
        filename: Name of the prompt file (e.g. 'column_mapping_system.txt')
        
    Returns:
        The content of the prompt file as a string
        
    Raises:
        FileNotFoundError: If the prompt file does not exist
    """
    # Get the base directory of the app
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "prompts", filename)
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_chat_prompt_content(base_name: str) -> tuple[str, str]:
    """
    Load both system and human prompts for a chat interaction.
    
    Expected filenames: {base_name}_system.txt and {base_name}_human.txt
    
    Args:
        base_name: Base name of the prompt (e.g. 'column_mapping')
        
    Returns:
        Tuple of (system_prompt, human_prompt)
    """
    system_prompt = load_prompt(f"{base_name}_system.txt")
    human_prompt = load_prompt(f"{base_name}_human.txt")
    return system_prompt, human_prompt
