"""Build tabular feature vector for inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import aggregate_reviews_for_products, prepare_tabular_features


def build_tabular_vector(product_row: pd.Series, reviews: pd.DataFrame, prep_path: Path) -> np.ndarray:
    bundle = joblib.load(prep_path)
    numeric = bundle["numeric"]
    categorical = bundle["categorical"]
    enc = bundle["encoder"]
    mean = bundle["mean"]
    std = bundle["std"]

    row_df = product_row.to_frame().T.copy()
    agg = aggregate_reviews_for_products(reviews)
    row_df = row_df.merge(agg, on="product_id", how="left")
    row_df, _ = prepare_tabular_features(row_df)
    row = row_df.iloc[0]

    num = row[numeric].astype(float).fillna(0).values.astype(np.float32)
    cat = enc.transform(pd.DataFrame([row[categorical].fillna("unknown")]))
    vec = np.hstack([num, cat[0]]).astype(np.float32)
    return (vec - mean) / (std + 1e-8)
