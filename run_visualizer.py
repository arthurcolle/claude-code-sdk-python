#!/usr/bin/env python3
"""Runner script for the claude self-image visualizer."""

import os
import subprocess
import sys
from pathlib import Path

# Set up the Python path
repo_root = Path(__file__).parent
experimental_src = repo_root / "experimental" / "src_claude_max"

# Run the visualizer with correct PYTHONPATH
env = {
    **os.environ,
    "PYTHONPATH": f"{experimental_src}:{repo_root / 'src'}:{os.environ.get('PYTHONPATH', '')}"
}
result = subprocess.run(
    [sys.executable, str(repo_root / "experimental" / "claude_self_image_visualizer.py")],
    env=env
)

sys.exit(result.returncode)