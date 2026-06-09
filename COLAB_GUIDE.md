# Colab overnight run plan

Run these cells once at the top of Colab, then execute notebooks 00 → 06 in order.

```python
# 1) Mount Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# 2) Upload project OR clone from GitHub
# !git clone https://github.com/YOUR_USER/Project_Neural.git
%cd /content/Project_Neural

# 3) Install deps
!pip install -q -r requirements.txt

# 4) Enable GPU: Runtime > Change runtime type > T4 GPU
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices("GPU"))
```

## Suggested schedule (one evening)

| Time | Action |
|------|--------|
| 0:00 | Start `00_eda.ipynb` — leave image download running |
| 1:00 | Run `01_tabular_mlp.ipynb` |
| 1:20 | Run `02_vision_cnn_transfer.ipynb` |
| 2:10 | Run `03_text_embeddings.ipynb` |
| 2:35 | Run `04_transformers.ipynb` |
| 3:25 | Run `05_llm_rag_finetune.ipynb` |
| 3:45 | Run `06_diffusion.ipynb` |
| 4:00 | `python app/app.py` or `%run app/app.py` |

## After training

1. Download `data/checkpoints/` and `data/artifacts/` from Colab
2. Fill metrics into `report/REPORT_OUTLINE.md`
3. Export PDF report
4. Record 3–5 min demo screencast
