"""
Generate data/metadata/document_mapping.csv by scanning data/assets/.

Expected asset structure:
    data/assets/{doc_type}/{filename}.pdf/pages/{page_num}/page-{page_num}.png

Output CSV columns: type, doc_name, filename, pages

Usage (from page-stream-segmentation/):
    python src/benchmarks/generate_mapping.py
"""

import argparse
import csv
from pathlib import Path

# Repo root = magisterka/ (3 levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def generate_mapping(assets_path: Path, output_path: Path) -> int:
    rows = []

    for doc_type_dir in sorted(assets_path.iterdir()):
        if not doc_type_dir.is_dir():
            continue
        doc_type = doc_type_dir.name

        for filename_dir in sorted(doc_type_dir.iterdir()):
            if not filename_dir.is_dir():
                continue
            filename = filename_dir.name   # e.g. "269633-budget.pdf"

            pages_dir = filename_dir / "pages"
            if not pages_dir.exists():
                continue

            page_count = sum(
                1 for p in pages_dir.iterdir()
                if p.is_dir() and any(p.glob("*.png"))
            )

            if page_count == 0:
                continue

            doc_name = filename.removesuffix(".pdf")

            rows.append({
                "type":     doc_type,
                "doc_name": doc_name,
                "filename": filename,
                "pages":    page_count,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "doc_name", "filename", "pages"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=str(_REPO_ROOT / "data" / "assets"))
    p.add_argument("--output", default=str(_REPO_ROOT / "data" / "metadata" / "document_mapping.csv"))
    args = p.parse_args()

    assets_path = Path(args.assets)
    output_path = Path(args.output)

    if not assets_path.exists():
        print(f"ERROR: assets path not found: {assets_path}")
        return

    n = generate_mapping(assets_path, output_path)
    print(f"Generated: {output_path}")
    print(f"  {n} documents found")


if __name__ == "__main__":
    main()
