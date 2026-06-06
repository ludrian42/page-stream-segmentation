import json
from pathlib import Path
from typing import List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from models import Document
from services.base_ocr import BaseOcr
from services.asset_writer import AssetWriter


class AssetCreator:
    """Creates page-image and OCR-text assets from PDF documents.

    Supports checkpoint-based resumption: already-processed documents are
    skipped on restart so a crash does not require restarting from scratch.

    Checkpoint location: {output_base_path}/.checkpoint-{ocr_name}.json
    """

    def __init__(self, writer: AssetWriter, ocr: BaseOcr):
        self.writer = writer
        self.ocr = ocr

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def create_assets(self, doc: Document) -> None:
        """Process one document: render images → OCR → write to disk."""
        logger.trace(f"Processing: {doc.doc_type}/{doc.doc_name}")

        with open(doc.absolute_filepath, "rb") as f:
            pdf_bytes = f.read()

        images     = self.writer.page_images_from_pdf(pdf_bytes)
        ocr_results = self.ocr.extract_pages(images)

        self.writer.save_document_assets(
            doc_type=doc.doc_type,
            doc_name=doc.doc_name,
            filename=doc.filename,
            pdf_bytes=pdf_bytes,
            ocr_results=ocr_results,
            ocr_name=self.ocr.name,
        )
        logger.trace(f"Done: {doc.doc_type}/{doc.doc_name}")

    def create_all(
        self,
        documents: List[Document],
        batch_size: int = 50,
        limit: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> dict:
        """Process documents in batches with checkpoint-based resumption.

        Args:
            documents:        All documents to process.
            batch_size:       Documents per batch before flushing the checkpoint.
            limit:            Cap total documents (useful for dry-runs).
            checkpoint_path:  Path to checkpoint JSON. Defaults to
                              {writer.output_base_path}/.checkpoint-{ocr_name}.json

        Returns:
            Dict with keys 'successful', 'failed', 'skipped', 'failed_docs'.
        """
        if limit and limit > 0:
            documents = documents[:limit]

        cp_path = Path(
            checkpoint_path
            or self.writer.output_base_path / f".checkpoint-{self.ocr.name}.json"
        )
        checkpoint    = self._load_checkpoint(cp_path)
        completed_set = set(checkpoint.get("completed", []))
        failed_set    = set(checkpoint.get("failed", []))

        pending  = [d for d in documents if self._doc_key(d) not in completed_set]
        skipped  = len(documents) - len(pending)
        successful = len(completed_set)
        failed     = len(failed_set)
        failed_docs = list(failed_set)

        logger.info(
            f"[{self.ocr.name}] Total: {len(documents)} | "
            f"Pending: {len(pending)} | Skipped (done): {skipped}"
        )

        for batch_start in range(0, len(pending), batch_size):
            batch     = pending[batch_start: batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(pending))
            logger.info(f"Batch {batch_start + 1}–{batch_end} / {len(pending)}")

            for doc in batch:
                key = self._doc_key(doc)
                try:
                    self.create_assets(doc)
                    completed_set.add(key)
                    successful += 1
                    failed_set.discard(key)
                except Exception as exc:
                    logger.error(f"Failed: {key} — {exc}")
                    failed_set.add(key)
                    failed += 1
                    if key not in failed_docs:
                        failed_docs.append(key)

            self._save_checkpoint(cp_path, completed_set, failed_set)
            logger.info(f"Checkpoint saved — successful: {successful}, failed: {failed}")

        return {
            "successful": successful,
            "failed":     failed,
            "skipped":    skipped,
            "failed_docs": failed_docs,
        }

    @staticmethod
    def _doc_key(doc: Document) -> str:
        return f"{doc.doc_type}/{doc.doc_name}"

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info(
                    f"Checkpoint loaded: {len(data.get('completed', []))} completed, "
                    f"{len(data.get('failed', []))} failed"
                )
                return data
            except Exception as e:
                logger.warning(f"Could not read checkpoint ({e}), starting fresh")
        return {"completed": [], "failed": []}

    @staticmethod
    def _save_checkpoint(path: Path, completed: set, failed: set) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"completed": sorted(completed), "failed": sorted(failed)},
                indent=2,
            ),
            encoding="utf-8",
        )
