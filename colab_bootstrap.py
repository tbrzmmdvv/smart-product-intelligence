"""Guaranteed Colab import setup — run before any `from src...` imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def setup_colab(project_name: str = "smart-product-intelligence") -> Path:
    root = Path(f"/content/{project_name}")
    if not (root / "src" / "data.py").exists():
        raise FileNotFoundError(
            f"Project not found at {root}. Run:\n"
            f"!git clone https://github.com/tbrzmmdvv/{project_name}.git"
        )

    init_py = root / "src" / "__init__.py"
    if not init_py.exists():
        init_py.write_text('"""Project source package."""\n', encoding="utf-8")

    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    importlib.invalidate_caches()

    import src.data  # noqa: F401

    print("Project root:", root_str)
    print("Import OK: src.data")
    return root
