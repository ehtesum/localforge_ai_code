#!/usr/bin/env python
"""CLI wrapper for the AI Coder Agent."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_env.agent import main

if __name__ == "__main__":
    main()
