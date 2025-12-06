"""
Main entry point for Hugging Face Spaces deployment.
This file allows HF Spaces to run the Streamlit app from the ui/ directory.
"""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main Streamlit app
from ui.streamlit_app import main

if __name__ == "__main__":
    main()

