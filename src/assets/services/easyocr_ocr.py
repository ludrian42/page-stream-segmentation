import time
from typing import List, Optional

import numpy as np
from PIL import Image
from loguru import logger

from services.base_ocr import BaseOcr, OcrPageResult


class EasyOcrEngine(BaseOcr):
    """OCR engine using EasyOCR.

    Uses GPU (CUDA) automatically when available — on RTX 4050 this provides
    a significant speedup over CPU mode. The EasyOCR Reader is initialised
    lazily on the first call to avoid loading the model at import time.

    Prerequisites:
        pip install easyocr
    """

    def __init__(
        self,
        langs: Optional[List[str]] = None,
        gpu: bool = True,
        detail_level: int = 1,
    ):
        """
        Args:
            langs:        Language codes for EasyOCR, default ['en'].
            gpu:          Use CUDA GPU if available; falls back to CPU silently.
            detail_level: 0 = text only (faster), 1 = with bounding box + confidence.
        """
        self.langs = langs or ["en"]
        self.gpu = gpu
        self.detail_level = detail_level
        self._reader = None  # lazy initialisation

    @property
    def name(self) -> str:
        return "easyocr"

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    "EasyOCR not installed. Run: pip install easyocr"
                ) from e

            logger.info(
                f"Initialising EasyOCR reader "
                f"(langs={self.langs}, gpu={self.gpu}) — first call may take a moment"
            )
            self._reader = easyocr.Reader(self.langs, gpu=self.gpu)
            logger.info("EasyOCR reader ready")
        return self._reader

    def extract_page(self, image: Image.Image) -> OcrPageResult:
        reader = self._get_reader()
        img_array = np.array(image.convert("RGB"))

        t0 = time.perf_counter()
        results = reader.readtext(img_array, detail=self.detail_level)
        elapsed = time.perf_counter() - t0

        if self.detail_level == 0:
            # results is List[str]
            text = "\n".join(results)
            confidence = -1.0
        else:
            # results is List[Tuple[bbox, text, conf]]
            texts, confs = [], []
            for _bbox, txt, conf in results:
                texts.append(txt)
                confs.append(conf)
            text = "\n".join(texts)
            confidence = round(sum(confs) / len(confs), 4) if confs else -1.0

        words = [w for w in text.split() if w]
        return OcrPageResult(
            text=text.strip(),
            word_count=len(words),
            confidence=confidence,
            elapsed_seconds=elapsed,
        )

    def extract_pages(self, images: List[Image.Image]) -> List[OcrPageResult]:
        """Sequential extraction — EasyOCR reader is not thread-safe."""
        return [self.extract_page(img) for img in images]
