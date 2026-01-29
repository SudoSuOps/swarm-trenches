"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add package directories to path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "swarmcode-cli"))
sys.path.insert(0, str(root_dir / "swarmcode-agent"))
sys.path.insert(0, str(root_dir / "swarmcode-trainer"))
sys.path.insert(0, str(root_dir / "swarmcode-registry"))
