#!/usr/bin/env python
"""CLI wrapper for model inference."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_env.inference import main

if __name__ == "__main__":
    main()
