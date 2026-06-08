import time
from typing import List

import numpy as np
from PIL import Image
from loguru import logger

from services.base_ocr import BaseOcr, OcrPageResult


class DocTROcrEngine(BaseOcr):
    """OCR engine using docTR (https://github.com/mindee/doctr).

    docTR is a transformer-based OCR library built on PyTorch. It combines
    a text detection model (DBNet) with a text recognition model (CRNN/ViT).
    Both models are loaded lazily on the first call and reused across all
    subsequent pages in the same process partition.

    Prerequisites:
        pip install python-doctr[torch]
        # PyTorch with CUDA must already be installed (in requirements.txt)
    """

    def __init__(self, gpu: bool = True):
        """
        Args:
            gpu: Use CUDA GPU if available. docTR auto-detects CUDA via PyTorch.
                 Set to False to force CPU inference.
        """
        self.gpu = gpu
        self._model = None

    @property
    def name(self) -> str:
        return "doctr"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if not self.gpu:
            import os  # noqa: PLC0415
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            from doctr.models import ocr_predictor  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "docTR not installed. Run: pip install python-doctr[torch]"
            ) from e

        logger.info("Loading docTR OCR predictor (detection + recognition)…")
        self._model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        if self.gpu:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("docTR running on GPU (CUDA)")
            else:
                logger.info("docTR running on CPU (CUDA not available)")
        else:
            logger.info("docTR running on CPU (forced)")

    def extract_page(self, image: Image.Image) -> OcrPageResult:
        self._load_model()

        # docTR expects a list of numpy arrays (uint8 RGB)
        img_array = np.array(image.convert("RGB"))

        t0 = time.perf_counter()
        result = self._model([img_array])
        elapsed = time.perf_counter() - t0

        texts = []
        confs = []

        # result.pages[0].blocks → lines → words
        for block in result.pages[0].blocks:
            for line in block.lines:
                line_words = []
                for word in line.words:
                    if word.value:
                        line_words.append(word.value)
                        confs.append(float(word.confidence))
                if line_words:
                    texts.append(" ".join(line_words))

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
        """Sequential extraction — docTR predictor is not thread-safe."""
        return [self.extract_page(img) for img in images]
