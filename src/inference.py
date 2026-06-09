"""Inference helpers for the integrated Gradio demo."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.data import artifacts_dir, load_image_manifest, load_splits, project_root
from src.tabular_inference import build_tabular_vector
from src.utils import top_k_similar


CHECKPOINTS = project_root() / "data" / "checkpoints"


def _load_json(name: str) -> dict[str, Any]:
    path = artifacts_dir() / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_product_catalog() -> pd.DataFrame:
    splits = load_splits()
    products = pd.concat(
        [splits["train_products"], splits["val_products"], splits["test_products"]],
        ignore_index=True,
    ).drop_duplicates(subset=["product_id"])
    manifest = load_image_manifest()
    products = products.merge(
        manifest[["product_id", "local_image_path", "image_available"]],
        on="product_id",
        how="left",
    )
    return products


@lru_cache(maxsize=1)
def get_review_catalog() -> pd.DataFrame:
    splits = load_splits()
    return pd.concat(
        [splits["train_reviews"], splits["val_reviews"], splits["test_reviews"]],
        ignore_index=True,
    )


def predict_tabular_rating(product_row: pd.Series) -> dict[str, Any]:
    """Predict rating band using saved MLP + baseline metadata."""
    meta = _load_json("tabular_metrics.json")
    baseline_mae = meta.get("baseline", {}).get("mae")
    mlp_mae = meta.get("mlp", {}).get("mae")

    prep_path = CHECKPOINTS / "tabular_preprocessor.joblib"
    model_path = CHECKPOINTS / "tabular_mlp.keras"
    if not prep_path.exists() or not model_path.exists():
        rating = float(product_row.get("average_rating", 3.5))
        return {
            "predicted_rating": round(rating, 2),
            "predicted_band": int(round(rating)),
            "note": "Using average_rating fallback — train M1 first.",
            "baseline_mae": baseline_mae,
            "mlp_mae": mlp_mae,
        }

    reviews = get_review_catalog()
    x = build_tabular_vector(product_row, reviews, prep_path)
    model = tf.keras.models.load_model(model_path)
    pred = float(model.predict(x.reshape(1, -1), verbose=0)[0][0])
    pred = float(np.clip(pred, 1.0, 5.0))
    return {
        "predicted_rating": round(pred, 2),
        "predicted_band": int(round(pred)),
        "baseline_mae": baseline_mae,
        "mlp_mae": mlp_mae,
    }


def predict_image_category(image_path: str) -> dict[str, Any]:
    """Predict product category from image using transfer-learning checkpoint."""
    meta = _load_json("vision_metrics.json")
    model_path = CHECKPOINTS / "vision_transfer.keras"
    labels_path = CHECKPOINTS / "vision_label_map.json"

    if not model_path.exists() or not labels_path.exists():
        return {"category": "unknown", "confidence": 0.0, "note": "Train M2 first."}

    with open(labels_path, encoding="utf-8") as f:
        label_map = json.load(f)
    inv_map = {int(v): k for k, v in label_map.items()}

    img = Image.open(image_path).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(arr.reshape(1, 128, 128, 3), verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "category": inv_map.get(idx, "unknown"),
        "confidence": float(probs[idx]),
        "metrics": meta,
    }


def search_similar_products(query: str, k: int = 5) -> pd.DataFrame:
    """Semantic search over product descriptions."""
    emb_path = artifacts_dir() / "product_embeddings.npy"
    index_path = artifacts_dir() / "product_embedding_index.json"
    if not emb_path.exists() or not index_path.exists():
        products = get_product_catalog()
        return products.head(k)[["product_id", "title", "category_leaf", "average_rating"]]

    from sentence_transformers import SentenceTransformer

    embeddings = np.load(emb_path)
    with open(index_path, encoding="utf-8") as f:
        product_ids = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    idxs = top_k_similar(query_vec, embeddings, k=k)
    products = get_product_catalog().set_index("product_id")
    rows = []
    for rank, idx in enumerate(idxs, start=1):
        pid = product_ids[idx]
        if pid not in products.index:
            continue
        row = products.loc[pid]
        rows.append(
            {
                "rank": rank,
                "product_id": pid,
                "title": row["title"],
                "category": row["category_leaf"],
                "rating": row.get("average_rating", np.nan),
            }
        )
    return pd.DataFrame(rows)


def summarize_product_reviews(product_id: str, mode: str = "grounded") -> str:
    """Summarize reviews into pros/cons using a small LLM or heuristic fallback."""
    reviews = get_review_catalog()
    subset = reviews[reviews["product_id"] == product_id].head(30)
    if subset.empty:
        return "No reviews found for this product."

    texts = subset["review_text"].tolist()
    combined = "\n".join(f"- {t[:300]}" for t in texts[:15])

    try:
        from transformers import pipeline

        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,
        )
        summary = summarizer(combined[:3500], max_length=130, min_length=40, do_sample=False)[0][
            "summary_text"
        ]
    except Exception:
        pos = subset[subset["rating"] >= 4]["review_text"].head(3).tolist()
        neg = subset[subset["rating"] <= 2]["review_text"].head(3).tolist()
        pros = "\n".join(f"- {t[:120]}" for t in pos) or "- Generally well rated."
        cons = "\n".join(f"- {t[:120]}" for t in neg) or "- Few negative mentions."
        summary = f"Pros:\n{pros}\n\nCons:\n{cons}"

    if mode == "grounded":
        return summary + "\n\n[Grounded in retrieved review snippets above.]"
    return summary + "\n\n[Ungrounded summary — may omit rare issues.]"


def answer_product_question(product_id: str, question: str) -> str:
    """Simple retrieval-grounded QA over reviews + metadata."""
    products = get_product_catalog().set_index("product_id")
    reviews = get_review_catalog()
    subset = reviews[reviews["product_id"] == product_id].head(20)

    if product_id in products.index:
        p = products.loc[product_id]
        context = (
            f"Title: {p['title']}\n"
            f"Category: {p['category_leaf']}\n"
            f"Price: {p.get('price', 'N/A')}\n"
            f"Average rating: {p.get('average_rating', 'N/A')}\n"
        )
    else:
        context = ""

    review_snippets = subset["review_text"].head(8).tolist()
    retrieved = "\n".join(f"- {t[:250]}" for t in review_snippets)
    prompt_context = f"{context}\nReviews:\n{retrieved}\n\nQuestion: {question}"

    try:
        from transformers import pipeline

        qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa(question=question, context=prompt_context[:4000])
        return f"{result['answer']} (score={result['score']:.2f})"
    except Exception:
        if review_snippets:
            return f"Based on reviews: {review_snippets[0][:300]}"
        return "Not enough context to answer confidently."


def generate_product_image(title: str, description: str) -> Image.Image | None:
    """Generate lifestyle/product image with Stable Diffusion."""
    prompt = f"Professional product photo, e-commerce hero image, {title}. {description[:200]}"
    try:
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        return image
    except Exception as exc:
        print(f"Diffusion failed: {exc}")
        return None
