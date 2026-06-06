import time
from typing import List

import pytesseract
from PIL import Image
from loguru import logger

from services.base_ocr import BaseOcr, OcrPageResult


class TesseractOcr(BaseOcr):
    """OCR engine using Tesseract via pytesseract.

    Runs on CPU. Provides per-word confidence scores via HOCR output.

    Prerequisites (WSL / Ubuntu):
        sudo apt-get install tesseract-ocr tesseract-ocr-eng
        pip install pytesseract
    """

    def __init__(self, lang: str = "eng", dpi: int = 200):
        """
        Args:
            lang: Tesseract language code(s), e.g. 'eng' or 'eng+pol'.
            dpi:  DPI hint passed to Tesseract (should match image rendering DPI).
        """
        self.lang = lang
        self.dpi = dpi
        self._config = f"--dpi {dpi} --oem 3 --psm 3"
        self._verify_installation()

    @property
    def name(self) -> str:
        return "tesseract"

    def _verify_installation(self) -> None:
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            raise RuntimeError(
                "Tesseract not found. Install with: "
                "sudo apt-get install tesseract-ocr tesseract-ocr-eng"
            ) from e

    def extract_page(self, image: Image.Image) -> OcrPageResult:
        t0 = time.perf_counter()

        text: str = pytesseract.image_to_string(
            image, lang=self.lang, config=self._config
        )
        confidence = self._mean_confidence(image)
        elapsed = time.perf_counter() - t0
        words = [w for w in text.split() if w]

        return OcrPageResult(
            text=text.strip(),
            word_count=len(words),
            confidence=confidence,
            elapsed_seconds=elapsed,
        )

    def _mean_confidence(self, image: Image.Image) -> float:
        """Return mean word confidence (0–1). Returns -1.0 on failure."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self._config,
                output_type=pytesseract.Output.DICT,
            )
            confs = [
                int(c) for c in data["conf"]
                if str(c).lstrip("-").isdigit() and int(c) >= 0
            ]
            return round(sum(confs) / len(confs) / 100.0, 4) if confs else -1.0
        except Exception:
            return -1.0

    def extract_pages(self, images: List[Image.Image]) -> List[OcrPageResult]:
        """Sequential extraction (Tesseract is not thread-safe per process)."""
        return [self.extract_page(img) for img in images]
