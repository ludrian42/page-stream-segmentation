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
    ├── doctr_ocr.py                # docTR engine (GPU, transformer-based)
    ├── asset_creator.py            # Orchestrates OCR + asset writing
    ├── asset_writer.py             # Saves images and OCR text to disk
    └── pdf_loader.py               # Discovers and validates raw PDFs

notebooks/
├── 00_eda_poly_seq.ipynb           # Exploratory Data Analysis — DocSplit benchmark
└── 01_ocr_evaluation.ipynb         # OCR engine comparison: quality & speed
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

The pipeline runs in **tmux** to survive VS Code disconnects. Each engine uses
a supervised mode that spawns isolated child processes to manage memory.

```bash
# start a tmux session
tmux new -s ocr
cd page-stream-segmentation

# step 1 — run all three engines (each resumes automatically from checkpoint)
python src/assets/run.py --engine tesseract --supervise
python src/assets/run.py --engine easyocr   --supervise
python src/assets/run.py --engine doctr     --supervise

# resume after a crash — checkpoint is loaded automatically
python src/assets/run.py --engine tesseract --supervise
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--engine` | `tesseract` | OCR engine: `tesseract`, `easyocr`, or `doctr` |
| `--stop-after` | `30` | Documents per partition (reduce if OOM) |
| `--dpi` | `200` | Page rendering DPI |
| `--limit` | `None` | Cap total documents (useful for smoke tests) |

## Evaluation

After running all engines, open `notebooks/01_ocr_evaluation.ipynb` and
**Restart & Run All**. The notebook benchmarks all three engines on a
stratified sample (3 docs × 3 pages per document type) and produces five
charts saved to `../data/ocr_eval/charts/`.

## Prerequisites

```bash
# System (WSL / Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Python — install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "numpy<2.0"   # required for docTR / EasyOCR ABI compatibility
```

## Progress

- [x] EDA — DocSplit benchmark (poly_seq / large)
- [x] OCR pipeline — Tesseract (CPU), EasyOCR (GPU), docTR (GPU, transformer-based)
- [x] OCR evaluation — speed, confidence, CER
- [ ] MLLM inference — Qwen VL (candidate) on DocSplit poly_seq
- [ ] Results analysis & comparison
