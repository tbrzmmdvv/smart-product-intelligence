"""Colab/local path bootstrap — run before importing src."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_project() -> Path:
    """Find repo root, chdir, and add to sys.path."""
    candidates = [
        Path("/content/smart-product-intelligence"),
        Path("/content/drive/MyDrive/smart-product-intelligence"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for path in candidates:
        if (path / "src" / "data.py").exists():
            os.chdir(path)
            root = str(path.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            print("Project root:", path)
            return path.resolve()
    raise ModuleNotFoundError(
        "Proje bulunamadi. Colab'da once calistirin:\n"
        "!git clone https://github.com/tbrzmmdvv/smart-product-intelligence.git"
    )
