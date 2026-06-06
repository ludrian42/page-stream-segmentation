# Page Stream Segmentation — MSc Research

Experiments on document page stream segmentation (PSS) using the
[DocSplit benchmark](https://huggingface.co/datasets/amazon/doc_split)
(poly_seq / large variant).

## Research goal

Evaluate and compare approaches to the Page Stream Segmentation task,
progressing from classical OCR pipelines through vision-based encoders
to Multimodal Large Language Models (MLLMs).

## Repository structure

```text
src/assets/
├── run.py                          # CLI pipeline runner (supervised mode)
├── models.py                       # Pydantic data models
└── services/
    ├── base_ocr.py                 # Abstract OCR interface
    ├── tesseract_ocr.py            # Tesseract engine (CPU)
    ├── easyocr_ocr.py              # EasyOCR engine (GPU)
    ├── asset_creator.py            # Orchestrates OCR + asset writing
    ├── asset_writer.py             # Saves images and OCR text to disk
    └── pdf_loader.py               # Discovers and validates raw PDFs

notebooks/
├── 00_eda_poly_seq.ipynb           # Exploratory Data Analysis — DocSplit benchmark
├── 01_create_assets.ipynb          # Asset creation pipeline (interactive)
└── 02_ocr_evaluation.ipynb         # OCR engine comparison: quality & speed
```

## Data

All data lives **outside** this repository and is not tracked by git.

| Path | Contents |
|------|----------|
| `../data/raw_data/` | Source PDFs from RVL-CDIP-N-MP, organised by document type |
| `../data/assets/` | Generated assets: page images (PNG) + OCR text files |
| `../data/ocr_eval/` | Evaluation results: `results.csv` and benchmark charts |

Source dataset: [RVL-CDIP-N-MP](https://huggingface.co/datasets/jordyvl/rvl_cdip_n_mp)
Benchmark: [DocSplit](https://huggingface.co/datasets/amazon/doc_split) (Amazon, 2026)

## Running the OCR pipeline

The pipeline is designed to run in **tmux** to survive VS Code disconnects,
using a supervised mode that isolates memory between partitions:

```bash
# start a tmux session
tmux new -s ocr

cd page-stream-segmentation

# run Tesseract (CPU)
python src/assets/run.py --engine tesseract --supervise

# run EasyOCR (GPU — RTX 4050)
python src/assets/run.py --engine easyocr --supervise

# resume after a crash — checkpoint is loaded automatically
python src/assets/run.py --engine tesseract --supervise
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--engine` | `tesseract` | OCR engine: `tesseract` or `easyocr` |
| `--stop-after` | `30` | Documents per partition (reduce if OOM) |
| `--dpi` | `200` | Page rendering DPI |
| `--limit` | `None` | Cap total documents (useful for smoke tests) |

## Prerequisites

```bash
# System (WSL / Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Python — install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Progress

- [x] EDA — DocSplit benchmark (poly_seq / large)
- [x] OCR pipeline — Tesseract (CPU) & EasyOCR (GPU)
- [x] OCR evaluation — speed, confidence
- [ ] MLLM inference — Qwen VL (candidate) on DocSplit poly_seq
- [ ] Results analysis & comparison
