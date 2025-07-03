#!/usr/bin/env python3
"""
Legacy wrapper for inference_328.py - redirects to the new CLI.

This script maintains backward compatibility for existing scripts
that may be calling inference_328.py directly.
"""

import sys
import warnings

# Show deprecation warning
warnings.warn(
    "inference_328.py is deprecated. Please use 'python run_synctalk.py generate' or 'python inference_cli.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import and run the new CLI
from inference_cli import main

if __name__ == "__main__":
    # The new CLI handles 328x328 models by default
    main()