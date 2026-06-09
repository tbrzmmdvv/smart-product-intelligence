# Smart Product Intelligence

End-to-end capstone project for **Khazar University — Hands-on Deep Learning**.

Category: **All_Beauty** (Amazon Reviews 2023)

## Project structure

```
Project_Neural/
├── notebooks/          # Milestones 0–6
├── src/                # Shared Python modules
├── app/                # Milestone 7 Gradio demo
├── data/               # Cached data (gitignored)
├── report/             # Final report outline
├── requirements.txt
└── README.md
```

## Quick start (Google Colab — recommended)

1. Upload this folder to Google Drive **or** push to GitHub.
2. Open Colab with GPU: **Runtime → Change runtime type → T4 GPU**
3. Run notebooks **in order**:

| Order | Notebook | Time (approx.) |
|-------|----------|----------------|
| 0 | `00_eda.ipynb` | 30–60 min (download + images) |
| 1 | `01_tabular_mlp.ipynb` | 15 min |
| 2 | `02_vision_cnn_transfer.ipynb` | 45 min |
| 3 | `03_text_embeddings.ipynb` | 20 min |
| 4 | `04_transformers.ipynb` | 45 min |
| 5 | `05_llm_rag_finetune.ipynb` | 15 min |
| 6 | `06_diffusion.ipynb` | 10 min |
| 7 | `app/app.py` | 5 min |

### Colab first cell (example)

```python
!pip install -q -r requirements.txt
%cd /content/Project_Neural  # adjust path
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab notebooks/00_eda.ipynb
```

## Run the demo (Milestone 7)

After notebooks 0–6 finish and checkpoints exist:

```bash
python app/app.py
```

Gradio opens in the browser and combines:
- M1 tabular rating prediction
- M2 image category
- M3 semantic similar products
- M5 review summary + grounded QA
- M6 diffusion hero image

## Milestones checklist

- [x] M0 — EDA + product-level splits + image cache
- [x] M1 — Linear baseline + MLP (Keras)
- [x] M2 — CNN scratch vs MobileNetV2 transfer
- [x] M3 — TF-IDF vs embeddings + semantic search
- [x] M4 — DistilBERT fine-tune
- [x] M5 — Summarizer + RAG QA
- [x] M6 — Stable Diffusion demo
- [x] M7 — Gradio integration

## Important notes

1. **Split by product**, not by review (already handled in `src/data.py`).
2. **Download images in M0** — do not fetch live during training.
3. Report **macro-F1** for classification, not accuracy alone.
4. Always compare against a **baseline** (included in each notebook).
5. Fill `report/REPORT_OUTLINE.md` → export PDF after training.

## Dataset

[Hugging Face — Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

```python
from src.data import load_raw_datasets
reviews, meta = load_raw_datasets(category="All_Beauty")
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Out of memory | Reduce `MAX_SAMPLES` in notebook 4 |
| Broken image links | Re-run M0 download; some URLs expire |
| Diffusion slow | Use GPU; reduce `num_inference_steps` to 15 |
| Demo fallback values | Run notebooks 1–3 first to create checkpoints |

## Authors

Khazar University — Deep Learning Capstone
