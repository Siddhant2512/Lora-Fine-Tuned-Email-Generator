"""Configuration management for LoRA-Mail Assistant."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    """Get the appropriate device for inference (MPS for M3 Mac)."""
    import torch
    
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_lora_target_modules(base_model: str) -> list:
    """Get appropriate LoRA target modules based on model architecture."""
    base_model_lower = base_model.lower()
    
    # Llama models (Llama 3.2, Llama 2, etc.)
    if "llama" in base_model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # GPT-2 models
    elif "gpt2" in base_model_lower or "gpt-2" in base_model_lower:
        return ["c_attn", "c_proj"]
    
    # OPT models
    elif "opt" in base_model_lower:
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    # DialoGPT (GPT-2 based)
    elif "dialogpt" in base_model_lower:
        return ["c_attn", "c_proj"]
    
    # Default to GPT-2 style (most common)
    else:
        return ["c_attn", "c_proj"]

