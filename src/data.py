"""Shared data loading, cleaning, and split utilities."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DEFAULT_CATEGORY = "All_Beauty"
MAX_PRODUCTS = 8000
MAX_REVIEWS = 40000
MAX_IMAGES = 5000
RATING_BANDS = [1, 2, 3, 4, 5]


def project_root() -> Path:
    """Return repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_dir() -> Path:
    path = data_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def images_dir() -> Path:
    path = data_dir() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def splits_dir() -> Path:
    path = data_dir() / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_raw_datasets(
    category: str = DEFAULT_CATEGORY,
    max_products: int = MAX_PRODUCTS,
    max_reviews: int = MAX_REVIEWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Amazon Reviews 2023 review + metadata tables for one category."""
    from datasets import load_dataset

    review_config = f"raw_review_{category}"
    meta_config = f"raw_meta_{category}"

    print(f"Loading reviews: {review_config}")
    reviews_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        review_config,
        trust_remote_code=True,
        split="full",
    )
    print(f"Loading metadata: {meta_config}")
    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        meta_config,
        trust_remote_code=True,
        split="full",
    )

    reviews = pd.DataFrame(reviews_ds)
    meta = pd.DataFrame(meta_ds)

    if max_products and "parent_asin" in meta.columns:
        meta = meta.drop_duplicates(subset=["parent_asin"]).head(max_products)
    elif max_products and "asin" in meta.columns:
        meta = meta.drop_duplicates(subset=["asin"]).head(max_products)

    product_ids = set(meta["parent_asin"] if "parent_asin" in meta.columns else meta["asin"])
    id_col = "parent_asin" if "parent_asin" in reviews.columns else "asin"
    reviews = reviews[reviews[id_col].isin(product_ids)].head(max_reviews)

    return reviews, meta


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_categories(raw: Any) -> str:
    if raw is None:
        return "unknown"
    if isinstance(raw, list):
        return raw[0] if raw else "unknown"
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (json.JSONDecodeError, ValueError):
            pass
        return raw.split("|")[0] if "|" in raw else raw
    return str(raw)


def _first_image_url(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if isinstance(raw, str):
        return raw
    return None


def build_product_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Create product-level table from metadata."""
    id_col = "parent_asin" if "parent_asin" in meta.columns else "asin"
    products = meta.copy()
    products = products.rename(columns={id_col: "product_id"})
    products["price"] = products.get("price", np.nan).map(_safe_float)
    products["average_rating"] = products.get("average_rating", np.nan).map(_safe_float)
    products["rating_number"] = products.get("rating_number", 0).map(
        lambda x: int(_safe_float(x, 0))
    )
    products["main_category"] = products.get("main_category", "unknown").fillna("unknown")
    if "categories" in products.columns:
        products["category_leaf"] = products["categories"].map(_parse_categories)
    else:
        products["category_leaf"] = products["main_category"]

    title_col = "title" if "title" in products.columns else None
    desc_col = "description" if "description" in products.columns else None
    feat_col = "features" if "features" in products.columns else None

    def _join_features(val: Any) -> str:
        if isinstance(val, list):
            return " ".join(str(v) for v in val)
        return str(val) if val is not None else ""

    products["title"] = products[title_col].fillna("") if title_col else ""
    if desc_col:
        products["description"] = products[desc_col].fillna("").astype(str)
    else:
        products["description"] = ""
    if feat_col:
        products["features_text"] = products[feat_col].map(_join_features)
    else:
        products["features_text"] = ""

    products["full_text"] = (
        products["title"] + " " + products["description"] + " " + products["features_text"]
    ).str.strip()

    image_col = None
    for candidate in ("images", "image", "imageURLHighRes"):
        if candidate in products.columns:
            image_col = candidate
            break
    if image_col:
        products["image_url"] = products[image_col].map(_first_image_url)
    else:
        products["image_url"] = None

    keep = [
        "product_id",
        "title",
        "description",
        "features_text",
        "full_text",
        "price",
        "average_rating",
        "rating_number",
        "main_category",
        "category_leaf",
        "image_url",
    ]
    return products[keep].drop_duplicates(subset=["product_id"]).reset_index(drop=True)


def build_review_table(reviews: pd.DataFrame) -> pd.DataFrame:
    """Create review-level table."""
    id_col = "parent_asin" if "parent_asin" in reviews.columns else "asin"
    df = reviews.copy()
    df = df.rename(columns={id_col: "product_id"})
    df["rating"] = df["rating"].astype(float)
    df["rating_band"] = df["rating"].round().astype(int).clip(1, 5)
    df["review_title"] = df.get("title", "").fillna("").astype(str)
    df["review_text"] = df.get("text", "").fillna("").astype(str)
    df["review_length"] = df["review_text"].str.len()
    df["helpful_vote"] = df.get("helpful_vote", 0).fillna(0).astype(int)
    df["verified_purchase"] = df.get("verified_purchase", False).fillna(False).astype(int)

    keep = [
        "product_id",
        "rating",
        "rating_band",
        "review_title",
        "review_text",
        "review_length",
        "helpful_vote",
        "verified_purchase",
    ]
    if "timestamp" in df.columns:
        keep.append("timestamp")
    return df[keep].reset_index(drop=True)


def merge_tables(products: pd.DataFrame, review_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach product metadata to reviews."""
    enriched_reviews = review_df.merge(products, on="product_id", how="inner")
    return products, enriched_reviews


def split_by_product(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split by product_id to prevent leakage."""
    from sklearn.model_selection import train_test_split

    product_ids = products["product_id"].unique()
    train_ids, test_ids = train_test_split(
        product_ids, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_ids, test_size=relative_val, random_state=random_state
    )

    def _subset(ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        id_set = set(ids)
        p = products[products["product_id"].isin(id_set)].copy()
        r = reviews[reviews["product_id"].isin(id_set)].copy()
        return p, r

    train_p, train_r = _subset(train_ids)
    val_p, val_r = _subset(val_ids)
    test_p, test_r = _subset(test_ids)

    return {
        "train_products": train_p,
        "val_products": val_p,
        "test_products": test_p,
        "train_reviews": train_r,
        "val_reviews": val_r,
        "test_reviews": test_r,
    }


def save_splits(splits: dict[str, pd.DataFrame], directory: Path | None = None) -> None:
    directory = directory or splits_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    manifest = {
        "files": list(splits.keys()),
        "counts": {k: len(v) for k, v in splits.items()},
    }
    with open(directory / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_splits(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    directory = directory or splits_dir()
    splits: dict[str, pd.DataFrame] = {}
    for path in directory.glob("*.parquet"):
        splits[path.stem] = pd.read_parquet(path)
    if not splits:
        raise FileNotFoundError(
            f"No split files found in {directory}. Run notebooks/00_eda.ipynb first."
        )
    return splits


def _safe_filename(product_id: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "_", product_id)
    return cleaned[:120]


def download_product_images(
    products: pd.DataFrame,
    max_images: int = MAX_IMAGES,
    timeout: int = 10,
) -> pd.DataFrame:
    """Download and cache product images; return manifest with local paths."""
    out = products.copy()
    out = out[out["image_url"].notna()].head(max_images)
    local_paths: list[str | None] = []
    ok_flags: list[bool] = []

    for _, row in tqdm(out.iterrows(), total=len(out), desc="Downloading images"):
        url = row["image_url"]
        filename = _safe_filename(str(row["product_id"])) + ".jpg"
        local_path = images_dir() / filename
        success = False
        if local_path.exists() and local_path.stat().st_size > 0:
            success = True
        elif url:
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 200 and resp.content:
                    local_path.write_bytes(resp.content)
                    success = True
            except requests.RequestException:
                success = False
        local_paths.append(str(local_path) if success else None)
        ok_flags.append(success)

    out["local_image_path"] = local_paths
    out["image_available"] = ok_flags
    manifest_path = artifacts_dir() / "image_manifest.parquet"
    out.to_parquet(manifest_path, index=False)
    return out


def load_image_manifest() -> pd.DataFrame:
    path = artifacts_dir() / "image_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError("Image manifest missing. Run image download in 00_eda.ipynb.")
    return pd.read_parquet(path)


def prepare_tabular_features(products: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric + categorical features for tabular models."""
    df = products.copy()
    df["price"] = df["price"].fillna(df["price"].median())
    df["rating_number"] = df["rating_number"].fillna(0)
    df["log_review_count"] = np.log1p(df["rating_number"])
    df["category_leaf"] = df["category_leaf"].fillna("unknown")
    df["main_category"] = df["main_category"].fillna("unknown")
    feature_cols = ["price", "log_review_count", "main_category", "category_leaf"]
    return df, feature_cols


def aggregate_reviews_for_products(reviews: pd.DataFrame) -> pd.DataFrame:
    """Product-level review aggregates for tabular features."""
    agg = (
        reviews.groupby("product_id")
        .agg(
            mean_review_rating=("rating", "mean"),
            review_count=("rating", "count"),
            mean_helpful=("helpful_vote", "mean"),
            mean_review_length=("review_length", "mean"),
        )
        .reset_index()
    )
    return agg
