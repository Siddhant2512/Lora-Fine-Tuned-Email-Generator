#!/usr/bin/env python3
"""Check if the setup is correct for training and inference."""
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    required = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "fastapi",
        "streamlit",
        "datasets",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True


def check_huggingface_auth():
    """Check if Hugging Face authentication is set up."""
    print("\nChecking Hugging Face authentication...")
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"  ✅ Authenticated as: {user.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"  ❌ Not authenticated: {e}")
        print("  Run: huggingface-cli login")
        return False


def check_mps():
    """Check if MPS is available."""
    print("\nChecking MPS (Metal Performance Shaders)...")
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  ✅ MPS is available")
            return True
        else:
            print("  ⚠️  MPS not available (will use CPU)")
            return False
    except Exception as e:
        print(f"  ❌ Error checking MPS: {e}")
        return False


def check_config():
    """Check if config file exists and is valid."""
    print("\nChecking configuration...")
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"  ❌ config.yaml not found at {config_path}")
        return False
    
    try:
        from src.config import load_config
        config = load_config()
        base_model = config['model']['base_model']
        print(f"  ✅ Config loaded")
        print(f"  📋 Base model: {base_model}")
        
        # Check if it's Llama
        if "llama" in base_model.lower():
            print("  ⚠️  Llama model detected - ensure Hugging Face auth is set up")
        
        return True
    except Exception as e:
        print(f"  ❌ Error loading config: {e}")
        return False


def check_data():
    """Check if training data exists."""
    print("\nChecking training data...")
    data_path = Path(__file__).parent.parent / "data" / "train_dataset.jsonl"
    if not data_path.exists():
        print(f"  ⚠️  Training data not found at {data_path}")
        print("  Create data/train_dataset.jsonl with your email examples")
        return False
    
    # Count examples
    try:
        count = sum(1 for _ in open(data_path))
        print(f"  ✅ Training data found ({count} examples)")
        if count < 5:
            print("  ⚠️  Recommended: at least 5-10 examples for good results")
        return True
    except Exception as e:
        print(f"  ❌ Error reading data: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 50)
    print("LoRA-Mail Assistant Setup Check")
    print("=" * 50)
    
    results = []
    results.append(("Dependencies", check_dependencies()))
    results.append(("Configuration", check_config()))
    results.append(("Training Data", check_data()))
    
    # Only check HF auth if using Llama
    try:
        from src.config import load_config
        config = load_config()
        if "llama" in config['model']['base_model'].lower():
            results.append(("Hugging Face Auth", check_huggingface_auth()))
    except:
        pass
    
    results.append(("MPS Support", check_mps()))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to train.")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

