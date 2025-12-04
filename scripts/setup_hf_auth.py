#!/usr/bin/env python3
"""Helper script to set up Hugging Face authentication."""
import os
import sys
from pathlib import Path

def main():
    """Guide user through HF authentication."""
    print("=" * 50)
    print("Hugging Face Authentication Setup")
    print("=" * 50)
    print()
    print("Llama models require Hugging Face authentication.")
    print()
    print("Steps:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is enough)")
    print("3. Copy the token")
    print()
    
    method = input("Choose method:\n1. Use huggingface-cli login (recommended)\n2. Set environment variable\nChoice [1/2]: ").strip()
    
    if method == "1":
        print("\nRunning: huggingface-cli login")
        print("Enter your token when prompted.")
        os.system("huggingface-cli login")
    elif method == "2":
        token = input("\nEnter your Hugging Face token: ").strip()
        if token:
            # Add to .env file or export
            env_file = Path(__file__).parent.parent / ".env"
            with open(env_file, "w") as f:
                f.write(f"HF_TOKEN={token}\n")
            print(f"\n✅ Token saved to {env_file}")
            print("Note: You may need to load this in your shell:")
            print(f"  export HF_TOKEN={token}")
        else:
            print("❌ No token provided")
            return 1
    else:
        print("Invalid choice")
        return 1
    
    # Verify
    print("\nVerifying authentication...")
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✅ Authenticated as: {user.get('name', 'Unknown')}")
        return 0
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

