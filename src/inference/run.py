"""
Qwen PSS inference CLI — designed to run on PLGrid (SLURM).

Reads the benchmark CSV produced by src/benchmarks/run.py, runs Qwen2.5-VL
pairwise inference on every spliced document, saves predictions to JSON,
and prints evaluation metrics.

Usage (local smoke-test):
    python src/inference/run.py \\
        --benchmark data/benchmark/poly_seq_large/test.csv \\
        --assets    data/assets \\
        --output    data/pss_results/qwen7b \\
        --limit     5

Usage (PLGrid — called from SLURM script):
    python src/inference/run.py \\
        --benchmark data/benchmark/poly_seq_large/test.csv \\
        --assets    data/assets \\
        --output    data/pss_results/qwen7b \\
        --model     Qwen/Qwen2.5-VL-7B-Instruct \\
        --batch-size 1
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Allow imports from src/
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_REPO_ROOT = Path(__file__).resolve().parents[3]  # magisterka/

from inference.qwen_pss import QwenPSSEngine
from evaluation.pss_metrics import evaluate_document, aggregate_metrics, print_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen2.5-VL pairwise PSS inference on DocSplit benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark",
                   default=str(_REPO_ROOT / "data" / "benchmarks" / "poly_seq" / "large" / "test.csv"),
                   help="Path to benchmark CSV")
    p.add_argument("--assets",
                   default=str(_REPO_ROOT / "data" / "assets"),
                   help="Root directory of page image assets")
    p.add_argument("--output",
                   default=str(_REPO_ROOT / "data" / "pss_results" / "qwen"),
                   help="Output directory for predictions and metrics")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace model ID for Qwen VL")
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size",
                   help="Pairs per forward pass (reduce if OOM)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N spliced documents (smoke-test)")
    p.add_argument("--resume", action="store_true",
                   help="Skip documents whose predictions are already saved")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_image_path(raw: str, assets: Path) -> Path:
    """Resolve image_path from the CSV to an absolute path under assets.

    The CSV may contain absolute paths generated on a different machine
    (e.g. /home/ludrian/projects/magisterka/data/assets/resume/foo.pdf/…).
    We strip everything up to and including the 'assets' component and
    re-root it under the user-supplied assets directory.  If the path is
    already relative, we join it with assets directly.
    """
    p = Path(raw)
    if p.is_absolute():
        parts = p.parts
        try:
            idx = parts.index("assets")
            return assets / Path(*parts[idx + 1:])
        except ValueError:
            pass  # 'assets' not found — fall through and use as-is
    return assets / p


def load_benchmark(csv_path: str, assets_root: str):
    """Load benchmark CSV and group pages by parent_doc_name.

    Returns
    -------
    docs : dict[str, list[dict]]
        Ordered pages per spliced document.
    """
    import csv

    docs = defaultdict(list)
    assets = Path(assets_root)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Resolve image path — handles both absolute (cross-machine) and
            # relative paths by stripping the original 'assets' prefix and
            # re-rooting under the user-supplied assets_root.
            img_path = _resolve_image_path(row["image_path"], assets)
            row["abs_image_path"] = str(img_path)
            docs[row["parent_doc_name"]].append(row)

    # Sort pages within each document by page number
    for doc_id in docs:
        docs[doc_id].sort(key=lambda r: int(r["page"]))

    logger.info(f"Loaded {len(docs)} spliced documents from {csv_path}")
    return dict(docs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(output_dir / "inference.log", level="DEBUG", rotation="50 MB")

    logger.info(f"Model      : {args.model}")
    logger.info(f"Benchmark  : {args.benchmark}")
    logger.info(f"Assets     : {args.assets}")
    logger.info(f"Output     : {args.output}")

    # Load benchmark
    docs = load_benchmark(args.benchmark, args.assets)
    doc_ids = list(docs.keys())
    if args.limit:
        doc_ids = doc_ids[: args.limit]
        logger.info(f"Limiting to {args.limit} documents")

    # Initialize engine (lazy — model loads on first predict call)
    engine = QwenPSSEngine(model_name=args.model, batch_size=args.batch_size)

    per_doc_metrics = []

    for doc_id in tqdm(doc_ids, desc="Spliced docs", unit="doc"):
        pred_path = pred_dir / f"{doc_id}.json"

        # Resume mode: skip already-processed documents
        if args.resume and pred_path.exists():
            logger.info(f"[SKIP] {doc_id} (already exists)")
            existing = json.loads(pred_path.read_text())
            if "metrics" in existing:
                per_doc_metrics.append(existing["metrics"])
            continue

        pages = docs[doc_id]
        image_paths = [p["abs_image_path"] for p in pages]
        gt_labels   = [p["group_id"] for p in pages]
        gt_ordinals = [int(p["local_doc_id_page_ordinal"]) for p in pages]

        # Check all images exist
        missing = [p for p in image_paths if not Path(p).exists()]
        if missing:
            logger.warning(f"[SKIP] {doc_id}: {len(missing)} images missing")
            continue

        logger.info(f"Processing {doc_id} ({len(pages)} pages) …")

        try:
            boundaries = engine.predict_sequence_from_paths(image_paths)
        except Exception as exc:
            logger.error(f"[FAIL] {doc_id}: {exc}")
            continue

        # Evaluate
        doc_metrics = evaluate_document(gt_labels, gt_ordinals, boundaries)
        per_doc_metrics.append(doc_metrics)

        logger.info(
            f"  Boundary F1={doc_metrics['bnd_f1']:.3f}  "
            f"V-measure={doc_metrics['v_measure']:.3f}  "
            f"Ordering={doc_metrics['ordering']:.3f}"
        )

        # Persist predictions
        result = {
            "doc_id":     doc_id,
            "n_pages":    len(pages),
            "boundaries": [bool(b) for b in boundaries],
            "gt_labels":  gt_labels,
            "metrics":    doc_metrics,
        }
        pred_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Final aggregated metrics
    if per_doc_metrics:
        agg = aggregate_metrics(per_doc_metrics)
        print_metrics(agg)

        summary_path = output_dir / "metrics_summary.json"
        summary_path.write_text(
            json.dumps({"model": args.model, "n_docs": len(per_doc_metrics),
                        "metrics": agg}, indent=2, ensure_ascii=False)
        )
        logger.info(f"Summary saved → {summary_path}")
    else:
        logger.warning("No documents evaluated — check paths and benchmark CSV.")


if __name__ == "__main__":
    main()
