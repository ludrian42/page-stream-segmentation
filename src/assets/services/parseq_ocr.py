import time
from typing import List

import numpy as np
from PIL import Image
from loguru import logger

from services.base_ocr import BaseOcr, OcrPageResult


class ParseqOcrEngine(BaseOcr):
    """OCR engine using docTR with PARSeq recognition model.

    Architecture
    ------------
    Detection  : DBNet (db_resnet50) — identical to the docTR engine.
    Recognition: PARSeq (Permutation Autoregressive Sequence) — a Vision
                 Transformer trained with permuted language modelling.

    Ablation study value
    --------------------
    Comparing ParseqOcrEngine with DocTROcrEngine isolates the effect of the
    recognition head:

        docTR  = DBNet  +  CRNN / VGG16-BN  (CTC decoder, purely convolutional)
        parseq = DBNet  +  ViT              (transformer encoder, permutation
                                             autoregressive decoder)

    Both share the same detection backbone, so any difference in results is
    attributable solely to the recognizer architecture.

    PARSeq reference
    ----------------
    Bautista & Atienza, "Scene Text Recognition with Permuted Autoregressive
    Sequence Models", ECCV 2022. https://arxiv.org/abs/2207.06966

    Prerequisites
    -------------
        pip install python-doctr[torch]>=0.8.0
        # PyTorch with CUDA must already be installed.
        # PARSeq weights (~28 MB) are downloaded from HuggingFace on first run.
    """

    def __init__(self, gpu: bool = True):
        """
        Args:
            gpu: Use CUDA GPU if available.
        """
        self.gpu = gpu
        self._model = None

    @property
    def name(self) -> str:
        return "parseq"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if not self.gpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            from doctr.models import ocr_predictor
        except ImportError as e:
            raise ImportError(
                "docTR not installed. Run: pip install python-doctr[torch]"
            ) from e

        logger.info("Loading docTR+PARSeq predictor (db_resnet50 + parseq)…")
        self._model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="parseq",
            pretrained=True,
        )

        if self.gpu:
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("PARSeq running on GPU (CUDA)")
            else:
                logger.info("PARSeq running on CPU (CUDA not available)")
        else:
            logger.info("PARSeq running on CPU (forced)")

    def extract_page(self, image: Image.Image) -> OcrPageResult:
        self._load_model()

        img_array = np.array(image.convert("RGB"))

        t0 = time.perf_counter()
        result = self._model([img_array])
        elapsed = time.perf_counter() - t0

        texts = []
        confs = []

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
        return [self.extract_page(img) for img in images]
