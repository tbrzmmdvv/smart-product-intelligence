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
    from huggingface_hub import hf_hub_download

    repo_id = "McAuley-Lab/Amazon-Reviews-2023"

    meta_file = f"raw_meta_{category}/full-00000-of-00001.parquet"
    review_file = f"raw/review_categories/{category}.jsonl"

    print(f"Loading metadata parquet: {meta_file}")
    meta_path = hf_hub_download(repo_id=repo_id, filename=meta_file, repo_type="dataset")
    meta = pd.read_parquet(meta_path)

    if max_products:
        id_col = "parent_asin" if "parent_asin" in meta.columns else "asin"
        meta = meta.drop_duplicates(subset=[id_col]).head(max_products)

    product_ids = set(
        meta["parent_asin"] if "parent_asin" in meta.columns else meta["asin"]
    )

    print(f"Loading reviews jsonl: {review_file}")
    review_path = hf_hub_download(repo_id=repo_id, filename=review_file, repo_type="dataset")
    reviews = _load_reviews_jsonl(review_path, product_ids, max_reviews)

    return reviews, meta


def _load_reviews_jsonl(
    path: str | Path,
    product_ids: set[str],
    max_reviews: int,
) -> pd.DataFrame:
    """Stream JSONL reviews and keep rows for selected products."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record.get("parent_asin") or record.get("asin")
            if pid not in product_ids:
                continue
            rows.append(record)
            if max_reviews and len(rows) >= max_reviews:
                break
    return pd.DataFrame(rows)


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


def _scalar_value(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        if val.size == 1:
            return val.item()
        return val.tolist()
    return val


def _first_image_url(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None

    raw = _scalar_value(raw)

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped.lower() == "nan":
            return None
        if stripped.startswith("http"):
            return stripped
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                raw = json.loads(stripped.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

    if isinstance(raw, dict):
        for key in ("hi_res", "large", "thumb", "large_image_url", "medium_image_url"):
            val = _scalar_value(raw.get(key))
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    url = _first_image_url(item)
                    if url:
                        return url
                continue
            val_str = str(val).strip()
            if val_str.startswith("http"):
                return val_str
        return None

    if isinstance(raw, list) and raw:
        return _first_image_url(raw[0])

    val_str = str(raw).strip()
    if val_str.startswith("http"):
        return val_str
    return None


def attach_image_urls(
    products: pd.DataFrame,
    category: str = DEFAULT_CATEGORY,
) -> pd.DataFrame:
    """Ensure product rows have image_url by merging from metadata parquet."""
    df = products.copy()
    if "image_url" in df.columns and df["image_url"].notna().any():
        return df

    from huggingface_hub import hf_hub_download

    meta_file = f"raw_meta_{category}/full-00000-of-00001.parquet"
    meta_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=meta_file,
        repo_type="dataset",
    )
    meta_urls = build_product_table(pd.read_parquet(meta_path))[["product_id", "image_url"]]
    if "image_url" in df.columns:
        df = df.drop(columns=["image_url"])
    df = df.merge(meta_urls, on="product_id", how="left")
    return df


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
    if "helpful_votes" in df.columns:
        df["helpful_vote"] = df["helpful_votes"].fillna(0).astype(int)
    elif "helpful_vote" in df.columns:
        df["helpful_vote"] = df["helpful_vote"].fillna(0).astype(int)
    else:
        df["helpful_vote"] = 0
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
    category: str = DEFAULT_CATEGORY,
) -> pd.DataFrame:
    """Download and cache product images; return manifest with local paths."""
    out = attach_image_urls(products, category=category)
    if "image_url" not in out.columns:
        out["image_url"] = None
    out = out[out["image_url"].notna() & (out["image_url"].astype(str).str.len() > 0)].head(max_images)
    print(f"Products with image URLs: {len(out)}")
    if len(out) == 0:
        raise RuntimeError(
            "No image URLs found. Check metadata or re-run 00_eda.ipynb."
        )

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
