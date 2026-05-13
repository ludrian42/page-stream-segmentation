# Page Stream Segmentation — MSc Research

Experiments on document packet splitting using the
[DocSplit benchmark](https://huggingface.co/datasets/amazon/doc_split)
(poly_seq / large variant).

## Research goal
Evaluate and compare deep learning approaches to the Page Stream Segmentation (PSS)
task, with a focus on vision-based models (DiT, ViT) on the DocSplit benchmark.

## Repository structure

```text
notebooks/
├── 00_eda_poly_seq.ipynb       # Exploratory Data Analysis
├── 01_create_assets.ipynb      # Asset creation pipeline
├── 02_create_benchmarks.ipynb  # Benchmark generation
└── 03_analyze_benchmarks.ipynb # Benchmark analysis
```

## Data
Benchmark data comes from the DocSplit dataset (Amazon, 2026).
Source PDFs: [RVL-CDIP-N-MP](https://huggingface.co/datasets/jordyvl/rvl_cdip_n_mp).
Data is not included in this repository.

## Status
- [x] EDA — poly_seq / large
- [ ] Baseline model
- [ ] DiT experiments
- [ ] ViT experiments