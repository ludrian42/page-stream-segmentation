from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class OcrPageResult:
    """Result of OCR on a single page."""
    text: str
    word_count: int
    confidence: float  # 0.0–1.0; -1.0 if the engine does not provide confidence
    elapsed_seconds: float


class BaseOcr(ABC):
    """Abstract OCR interface. Each implementation operates on PIL Images."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in file names, e.g. 'tesseract' or 'easyocr'."""

    @abstractmethod
    def extract_page(self, image: Image.Image) -> OcrPageResult:
        """Run OCR on a single page image and return a structured result."""

    def extract_pages(self, images: List[Image.Image]) -> List[OcrPageResult]:
        """Run OCR on a list of page images (default: sequential)."""
        return [self.extract_page(img) for img in images]
