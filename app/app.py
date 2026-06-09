"""Milestone 7 — Smart Product Intelligence Gradio demo."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    answer_product_question,
    generate_product_image,
    get_product_catalog,
    predict_image_category,
    predict_tabular_rating,
    search_similar_products,
    summarize_product_reviews,
)


def _product_choices() -> list[tuple[str, str]]:
    try:
        catalog = get_product_catalog()
        catalog = catalog.dropna(subset=["title"]).head(200)
        return [
            (f"{row.title[:70]} ({row.product_id})", row.product_id)
            for row in catalog.itertuples()
        ]
    except Exception:
        return [("Run 00_eda.ipynb first", "demo")]


def analyze_product(product_id: str, query: str, qa_question: str, grounded: bool):
    catalog = get_product_catalog()
    row = catalog[catalog["product_id"] == product_id]
    if row.empty:
        return "Product not found.", "", "", None, None

    product = row.iloc[0]
    tabular = predict_tabular_rating(product)
    tabular_text = (
        f"Predicted rating: **{tabular['predicted_rating']}** "
        f"(band {tabular['predicted_band']})\n"
        f"MLP MAE: {tabular.get('mlp_mae', 'N/A')} | "
        f"Baseline MAE: {tabular.get('baseline_mae', 'N/A')}"
    )

    vision_text = "No cached image."
    image_path = product.get("local_image_path")
    if pd.notna(image_path) and Path(str(image_path)).exists():
        vision = predict_image_category(str(image_path))
        vision_text = (
            f"Predicted category: **{vision['category']}** "
            f"(confidence {vision.get('confidence', 0):.2f})"
        )
        display_image = str(image_path)
    else:
        display_image = None

    similar = search_similar_products(product.get("title", "") or query, k=5)
    similar_md = similar.to_markdown(index=False) if not similar.empty else "No matches."

    summary = summarize_product_reviews(
        product_id,
        mode="grounded" if grounded else "zero_shot",
    )
    answer = answer_product_question(product_id, qa_question or "Is this product good?")

    gen_image = generate_product_image(
        str(product.get("title", "")),
        str(product.get("description", "")),
    )

    return tabular_text, vision_text, similar_md, summary, answer, display_image, gen_image


def build_app() -> gr.Blocks:
    choices = _product_choices()
    with gr.Blocks(title="Smart Product Intelligence") as demo:
        gr.Markdown(
            """
            # Smart Product Intelligence
            Integrated capstone demo: tabular rating, vision category, semantic search,
            review summary, grounded QA, and diffusion hero image.
            """
        )

        with gr.Row():
            product_dd = gr.Dropdown(
                choices=choices,
                value=choices[0][1] if choices else None,
                label="Select product",
            )
            query = gr.Textbox(label="Semantic search query", placeholder="moisturizer for dry skin")
            qa_question = gr.Textbox(
                label="Buyer question",
                value="What do customers like most about this product?",
            )
            grounded = gr.Checkbox(value=True, label="Grounded summary (RAG)")

        run_btn = gr.Button("Analyze", variant="primary")

        with gr.Row():
            tabular_out = gr.Markdown(label="M1 — Tabular rating")
            vision_out = gr.Markdown(label="M2 — Vision category")

        similar_out = gr.Markdown(label="M3 — Similar products")
        summary_out = gr.Textbox(label="M5 — Pros / Cons summary", lines=8)
        qa_out = gr.Textbox(label="M5 — Grounded QA", lines=4)

        with gr.Row():
            product_img = gr.Image(label="Product image", type="filepath")
            generated_img = gr.Image(label="M6 — Generated hero image")

        run_btn.click(
            analyze_product,
            inputs=[product_dd, query, qa_question, grounded],
            outputs=[tabular_out, vision_out, similar_out, summary_out, qa_out, product_img, generated_img],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
