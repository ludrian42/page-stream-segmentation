"""
Asset Creation CLI — run inside tmux, not in a notebook.

Usage:
    # Supervised mode — RECOMMENDED: isolates memory between partitions
    python src/assets/run.py --engine tesseract --supervise
    python src/assets/run.py --engine easyocr --supervise

    # Custom partition size
    python src/assets/run.py --engine tesseract --supervise --stop-after 20

    # Single partition (used internally by the supervisor)
    python src/assets/run.py --engine tesseract --stop-after 30

    # Quick smoke-test
    python src/assets/run.py --engine tesseract --limit 5 --supervise
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.pdf_loader import PdfLoader
from services.asset_writer import AssetWriter
from services.asset_creator import AssetCreator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, engine: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run-{engine}.log"
    logger.remove()
    logger.add(
        sys.stderr, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(log_file, level="DEBUG", rotation="50 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.info(f"Logs: {log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build page assets (images + OCR text) from raw PDF documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--engine", choices=["tesseract", "easyocr", "doctr"], default="tesseract",
                   help="OCR engine to use")
    p.add_argument("--raw-data", default="data/raw_data",
                   help="Path to raw PDF directory, relative to the repository root")
    p.add_argument("--output", default="data/assets",
                   help="Output path for generated assets, relative to the repository root")
    p.add_argument("--dpi", type=int, default=200,
                   help="Page rendering DPI (200 is sufficient for OCR and uses less RAM than 300)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total number of documents (None = all); useful for testing")
    p.add_argument("--stop-after", type=int, default=30, dest="stop_after",
                   help="Documents per partition — process N docs then exit cleanly")
    p.add_argument("--supervise", action="store_true",
                   help="Supervisor mode: spawn partitions as isolated child processes "
                        "to release memory between runs")
    # Tesseract options
    p.add_argument("--tess-lang", default="eng",
                   help="Tesseract language code(s), e.g. 'eng' or 'eng+pol'")
    # EasyOCR options
    p.add_argument("--easyocr-langs", default="en",
                   help="Comma-separated EasyOCR language codes, e.g. 'en,pl'")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU for GPU-based engines (EasyOCR, Surya)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Supervisor mode
# ---------------------------------------------------------------------------

def supervise(args: argparse.Namespace) -> None:
    """Spawn run.py as a child process with --stop-after N.

    Each partition is an isolated process: the OS reclaims all memory after
    it exits before the next partition starts. The loop continues until the
    child returns exit code 0 (all documents processed).
    """
    child_cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--engine",        args.engine,
        "--raw-data",      args.raw_data,
        "--output",        args.output,
        "--dpi",           str(args.dpi),
        "--stop-after",    str(args.stop_after),
        "--tess-lang",     args.tess_lang,
        "--easyocr-langs", args.easyocr_langs,
    ]
    if args.limit:
        child_cmd += ["--limit", str(args.limit)]
    if args.no_gpu:
        child_cmd.append("--no-gpu")

    partition = 0
    logger.info(f"Supervisor started | chunk={args.stop_after} docs | engine={args.engine}")
    logger.info(f"Child command: {' '.join(child_cmd)}")

    while True:
        partition += 1
        logger.info(f"--- Partition #{partition} ---")

        result = subprocess.run(child_cmd)

        if result.returncode == 0:
            logger.info("All documents processed. Done.")
            sys.exit(0)

        if result.returncode == 2:
            logger.info(f"Partition #{partition} complete. Freeing memory, starting next...")
            time.sleep(1)
            continue

        logger.error(f"Child exited with unexpected code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Worker mode (single partition)
# ---------------------------------------------------------------------------

def build_ocr(args: argparse.Namespace):
    if args.engine == "tesseract":
        from services.tesseract_ocr import TesseractOcr
        return TesseractOcr(lang=args.tess_lang, dpi=args.dpi)
    if args.engine == "easyocr":
        from services.easyocr_ocr import EasyOcrEngine
        langs = [l.strip() for l in args.easyocr_langs.split(",")]
        return EasyOcrEngine(langs=langs, gpu=not args.no_gpu)
    if args.engine == "doctr":
        from services.doctr_ocr import DocTROcrEngine
        return DocTROcrEngine(gpu=not args.no_gpu)
    raise ValueError(f"Unknown engine: {args.engine}")


def run_partition(args: argparse.Namespace) -> None:
    """Process one partition of documents and exit with code 0 or 2.

    Exit codes:
        0 — all documents in the full list are now complete
        2 — this partition is done, but more documents remain
    """
    repo_root     = Path(__file__).resolve().parents[3]
    raw_data_path = repo_root / args.raw_data
    output_path   = repo_root / args.output
    log_dir       = repo_root / "logs"

    setup_logging(log_dir, args.engine)
    logger.info(f"engine={args.engine} | dpi={args.dpi} | partition_size={args.stop_after}")

    loader    = PdfLoader(raw_data_path=str(raw_data_path))
    documents = loader.get_all_documents()
    if args.limit:
        documents = documents[: args.limit]

    cp_path     = output_path / f".checkpoint-{args.engine}.json"
    done_set    = set()
    failed_docs: list[str] = []

    if cp_path.exists():
        try:
            data        = json.loads(cp_path.read_text())
            done_set    = set(data.get("completed", []))
            failed_docs = data.get("failed", [])
            logger.info(f"Checkpoint loaded: {len(done_set)} already done")
        except Exception:
            pass

    pending = [d for d in documents if f"{d.doc_type}/{d.doc_name}" not in done_set]

    if not pending:
        logger.info("All documents already processed.")
        sys.exit(0)

    batch = pending[: args.stop_after]
    logger.info(f"This partition: {len(batch)} docs | remaining after: {len(pending) - len(batch)}")

    ocr     = build_ocr(args)
    writer  = AssetWriter(output_base_path=str(output_path), image_dpi=args.dpi)
    creator = AssetCreator(writer=writer, ocr=ocr)

    progress = tqdm(batch, unit="doc", desc=f"[{args.engine}]", dynamic_ncols=True)

    for doc in progress:
        key = f"{doc.doc_type}/{doc.doc_name}"
        try:
            creator.create_assets(doc)
            done_set.add(key)
        except Exception as exc:
            logger.error(f"Failed: {key} — {exc}")
            if key not in failed_docs:
                failed_docs.append(key)
        finally:
            gc.collect()
        progress.set_postfix(ok=len(done_set), err=len(failed_docs))

    # Save checkpoint
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text(
        json.dumps({"completed": sorted(done_set), "failed": sorted(failed_docs)}, indent=2),
        encoding="utf-8",
    )

    all_done = len(done_set) >= len(documents)
    logger.info(f"Partition complete | total done: {len(done_set)}/{len(documents)}")
    sys.exit(0 if all_done else 2)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.supervise:
        supervise(args)
    else:
        run_partition(args)
