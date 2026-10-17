"""Ultimo-MARL: Multi-Agent Reinforcement Learning Ecosystem for Ultimo."""

import sys

# Ensure UTF-8 console output streams on Windows to prevent UnicodeEncodeError with Unicode symbols/emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__version__ = "0.1.0"
__author__ = "Stanisław Golec"
__all__ = ["__version__", "__author__"]
