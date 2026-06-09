"""Shared utilities for metrics, plotting, and notebook setup."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_notebook_path() -> Path:
    """Add project root to sys.path for notebook imports."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_learning_curves(history: Any, title: str, save_path: Path | None = None) -> None:
    hist = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in hist:
        axes[0].plot(hist["loss"], label="train")
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], label="val")
        axes[0].set_title(f"{title} — Loss")
        axes[0].legend()

    metric_keys = [k for k in hist if k not in {"loss", "val_loss"} and not k.startswith("val_")]
    if metric_keys:
        metric = metric_keys[0]
        val_metric = f"val_{metric}"
        axes[1].plot(hist[metric], label="train")
        if val_metric in hist:
            axes[1].plot(hist[val_metric], label="val")
        axes[1].set_title(f"{title} — {metric}")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def rating_band_labels() -> list[str]:
    return ["1", "2", "3", "4", "5"]


def top_k_similar(query_vec: np.ndarray, matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Return indices of top-k cosine similarities."""
    query = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    sims = (matrix / norms) @ query
    return np.argsort(-sims)[:k]


def format_pros_cons(text: str) -> dict[str, list[str]]:
    """Parse simple bullet-style pros/cons from LLM output."""
    pros: list[str] = []
    cons: list[str] = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("pros"):
            current = "pros"
            continue
        if lower.startswith("cons"):
            current = "cons"
            continue
        if stripped.startswith(("-", "*", "•")) and current:
            item = stripped.lstrip("-*• ").strip()
            if item:
                (pros if current == "pros" else cons).append(item)
    return {"pros": pros, "cons": cons}
