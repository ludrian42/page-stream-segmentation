import time
from typing import List

import numpy as np
from PIL import Image
from loguru import logger

from services.base_ocr import BaseOcr, OcrPageResult


class TrOCROcrEngine(BaseOcr):
    """OCR engine using Microsoft TrOCR (Transformer-based OCR).

    Architecture
    ------------
    Detection  : docTR DBNet (db_resnet50) — same backbone as the docTR engine,
                 enabling a fair ablation study of the *recognition* component.
    Recognition: TrOCR — ViT image encoder + autoregressive Transformer decoder
                 (VisionEncoderDecoderModel from HuggingFace Transformers).

    Unlike CRNN-based recognizers (used in the docTR pipeline), TrOCR frames
    recognition as a sequence-to-sequence problem, attending to the full image
    patch sequence before generating each output token.  This makes it more
    robust to irregular fonts, dense text, and complex layouts.

    Reference
    ---------
    Li et al., "TrOCR: Transformer-based Optical Character Recognition with
    Pre-trained Models", AAAI 2023. https://arxiv.org/abs/2109.10282

    Prerequisites
    -------------
        pip install transformers sentencepiece
        # PyTorch with CUDA must already be installed (in requirements.txt)
        # Model weights (~400 MB) are downloaded from HuggingFace on first run.
    """

    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(self, gpu: bool = True, model_name: str = DEFAULT_MODEL):
        """
        Args:
            gpu: Use CUDA GPU if available.
            model_name: HuggingFace model identifier.
                        Options:
                          "microsoft/trocr-base-printed"   — faster, smaller
                          "microsoft/trocr-large-printed"  — higher accuracy
        """
        self.gpu = gpu
        self.model_name = model_name
        self._det_model = None   # docTR ocr_predictor (detection only)
        self._processor = None   # TrOCRProcessor
        self._recog = None       # VisionEncoderDecoderModel
        self._device = None

    # ------------------------------------------------------------------
    # BaseOcr interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "trocr"

    def _load_model(self) -> None:
        if self._recog is not None:
            return

        import torch
        from doctr.models import ocr_predictor
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._device = torch.device(
            "cuda" if (self.gpu and torch.cuda.is_available()) else "cpu"
        )

        # --- Detection backbone (docTR DBNet) ---
        # We reuse the docTR ocr_predictor purely for text-line detection.
        # Its CRNN recognition output is discarded; TrOCR handles recognition.
        logger.info("Loading TrOCR detection backbone (db_resnet50)…")
        self._det_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        if self.gpu and torch.cuda.is_available():
            self._det_model = self._det_model.cuda()

        # --- TrOCR recognition ---
        logger.info(f"Loading TrOCR recognizer ({self.model_name})…")
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._recog = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._recog = self._recog.to(self._device)
        self._recog.eval()

        if self.gpu and torch.cuda.is_available():
            logger.info(f"TrOCR running on GPU ({self._device})")
        else:
            logger.info("TrOCR running on CPU")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_lines(self, img_rgb: Image.Image) -> List[Image.Image]:
        """Run docTR DBNet and return a list of line-level PIL Image crops.

        TrOCR was pre-trained on *text line* images, so feeding it full-page
        images would degrade accuracy.  We therefore crop each detected line
        and pass crops to TrOCR individually.
        """
        img_array = np.array(img_rgb)
        h, w = img_array.shape[:2]

        det_result = self._det_model([img_array])

        crops: List[Image.Image] = []
        for block in det_result.pages[0].blocks:
            for line in block.lines:
                (x1r, y1r), (x2r, y2r) = line.geometry
                # Convert normalised coords → pixels; add small padding
                x1 = max(0, int(x1r * w) - 4)
                y1 = max(0, int(y1r * h) - 4)
                x2 = min(w, int(x2r * w) + 4)
                y2 = min(h, int(y2r * h) + 4)
                if x2 > x1 and y2 > y1:
                    crops.append(img_rgb.crop((x1, y1, x2, y2)))
        return crops

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_page(self, image: Image.Image) -> OcrPageResult:
        import torch

        self._load_model()

        img_rgb = image.convert("RGB")

        t0 = time.perf_counter()

        # 1. Detect text lines
        crops = self._detect_lines(img_rgb)

        texts: List[str] = []
        confs: List[float] = []
        BATCH = 8  # TrOCR base fits ~8 line crops per GPU batch on 6 GB VRAM

        with torch.no_grad():
            for i in range(0, len(crops), BATCH):
                batch = crops[i : i + BATCH]

                # TrOCRProcessor resizes each crop to 384×384 internally
                encoding = self._processor(
                    images=batch,
                    return_tensors="pt",
                    padding=True,
                )
                pixel_values = encoding.pixel_values.to(self._device)

                # 2. Generate token sequences
                outputs = self._recog.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                decoded = self._processor.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )

                # 3. Confidence = mean max-probability across generated tokens
                #    (proxy for token-level certainty; comparable across batches)
                if outputs.scores:
                    score_tensor = torch.stack(
                        [
                            torch.softmax(s, dim=-1).max(dim=-1).values
                            for s in outputs.scores
                        ],
                        dim=1,
                    )  # (batch, seq_len)
                    batch_confs = score_tensor.mean(dim=1).cpu().tolist()
                else:
                    batch_confs = [0.85] * len(batch)

                for text, conf in zip(decoded, batch_confs):
                    if text.strip():
                        texts.append(text.strip())
                        confs.append(float(conf))

        elapsed = time.perf_counter() - t0

        full_text = "\n".join(texts)
        words = [w for w in full_text.split() if w]
        confidence = round(sum(confs) / len(confs), 4) if confs else -1.0

        return OcrPageResult(
            text=full_text.strip(),
            word_count=len(words),
            confidence=confidence,
            elapsed_seconds=elapsed,
        )

    def extract_pages(self, images: List[Image.Image]) -> List[OcrPageResult]:
        """Sequential extraction — model state is not thread-safe."""
        return [self.extract_page(img) for img in images]
