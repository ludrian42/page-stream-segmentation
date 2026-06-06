import gc
from pathlib import Path
from typing import List

import fitz
from PIL import Image
from loguru import logger

from services.base_ocr import OcrPageResult


class AssetWriter:
    """Writes document assets (page images and OCR text) to disk.

    Directory layout::

        {output_base_path}/{doc_type}/{filename}/
            original/{filename}              ← original PDF
            pages/
                0001/
                    page-0001.png
                    page-0001-{ocr_name}.txt
                0002/
                    ...
    """

    def __init__(
        self,
        output_base_path: str = "data/assets",
        image_dpi: int = 200,
    ):
        self.output_base_path = Path(output_base_path)
        self.image_dpi = image_dpi

    def save_document_assets(
        self,
        doc_type: str,
        doc_name: str,
        filename: str,
        pdf_bytes: bytes,
        ocr_results: List[OcrPageResult],
        ocr_name: str,
    ) -> None:
        """Save all assets for one document.

        Args:
            doc_type:    Document category (e.g. 'budget').
            doc_name:    Stem of the PDF filename used as identifier.
            filename:    Full PDF filename (e.g. '269633-budget.pdf').
            pdf_bytes:   Raw PDF bytes.
            ocr_results: Per-page OCR results (one entry per page).
            ocr_name:    Engine identifier used in text file names
                         (e.g. 'tesseract', 'easyocr').
        """
        doc_dir      = self.output_base_path / doc_type / filename
        original_dir = doc_dir / "original"
        pages_dir    = doc_dir / "pages"
        original_dir.mkdir(parents=True, exist_ok=True)

        # Save original PDF
        (original_dir / filename).write_bytes(pdf_bytes)

        # Render pages and save alongside OCR text
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for page_idx in range(len(pdf_doc)):
                page_num_str = f"{page_idx + 1:04d}"
                page_dir     = pages_dir / page_num_str
                page_dir.mkdir(parents=True, exist_ok=True)

                # Page image — skip if already rendered by a previous engine run
                img_path = page_dir / f"page-{page_num_str}.png"
                if not img_path.exists():
                    pix = pdf_doc[page_idx].get_pixmap(dpi=self.image_dpi)
                    pix.save(str(img_path))
                    del pix  # release pixmap memory immediately

                # OCR text
                if page_idx < len(ocr_results):
                    text_path = page_dir / f"page-{page_num_str}-{ocr_name}.txt"
                    text_path.write_text(ocr_results[page_idx].text, encoding="utf-8")
        finally:
            pdf_doc.close()

        logger.debug(f"Saved assets: {doc_type}/{doc_name} ({ocr_name})")

    def page_images_from_pdf(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Render all pages of a PDF to PIL Images.

        Releases each PyMuPDF pixmap immediately after conversion to keep
        peak memory usage low.
        """
        images: List[Image.Image] = []
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for page in pdf_doc:
                pix = page.get_pixmap(dpi=self.image_dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                del pix
        finally:
            pdf_doc.close()
        return images

    def assets_exist(self, doc_type: str, filename: str, ocr_name: str) -> bool:
        """Return True if OCR text files already exist for this document."""
        pages_dir  = self.output_base_path / doc_type / filename / "pages"
        if not pages_dir.exists():
            return False
        return len(list(pages_dir.glob(f"**/*-{ocr_name}.txt"))) > 0
