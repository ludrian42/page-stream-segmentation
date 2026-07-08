"""
Qwen2.5-VL pairwise Page Stream Segmentation engine.

For each pair of consecutive pages (i, i+1), the model answers:
"Do both pages belong to the same document? YES / NO"
(prompt is intentionally sent in Polish: TAK = YES, NIE = NO)

A "NIE" response indicates a document boundary between page i and i+1.

Reference model: Qwen/Qwen2.5-VL-7B-Instruct (fits on 1× A100 40 GB)
Alternative:     Qwen/Qwen2.5-VL-72B-Instruct (needs 2–4× A100 80 GB)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from loguru import logger


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PAIR_PROMPT = (
    "Oto dwie kolejne strony ze zeskanowanego archiwum dokumentów.\n\n"
    "Czy obie strony należą do tego samego dokumentu?\n"
    "Odpowiedz wyłącznie jednym słowem: TAK lub NIE."
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QwenPSSEngine:
    """Pairwise PSS classifier using Qwen2.5-VL.

    Usage
    -----
    engine = QwenPSSEngine()
    # predict for a full sequence of page images:
    boundaries = engine.predict_sequence(images)
    # boundaries[i] == True  →  boundary BEFORE page i+1
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 8,
        batch_size: int = 1,
    ):
        """
        Args:
            model_name:     HuggingFace model ID.
            device:         "cuda" / "cpu" / None (auto-detect).
            torch_dtype:    None → bfloat16 on GPU, float32 on CPU.
            max_new_tokens: Limit output length (TAK/NIE needs only 1–3 tokens).
            batch_size:     How many pairs to process in one forward pass.
                            Set to 1 if you hit OOM; increase for speed.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        # Resolve device and dtype
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if torch_dtype is None:
            self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        else:
            self._dtype = torch_dtype

        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading {self.model_name} on {self._device} ({self._dtype}) …")
        t0 = time.perf_counter()

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            device_map="auto",          # distributes across all available GPUs
            attn_implementation="flash_attention_2" if self._device == "cuda" else "eager",
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(self.model_name)

        logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _build_message(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> dict:
        """Build a single Qwen chat message for a page pair."""
        return {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text",  "text":  _PAIR_PROMPT},
            ],
        }

    def _predict_batch(
        self,
        pairs: List[Tuple[Image.Image, Image.Image]],
    ) -> List[bool]:
        """Run inference on a batch of image pairs.

        Returns
        -------
        List of booleans: True = boundary (pages belong to DIFFERENT documents).
        """
        from qwen_vl_utils import process_vision_info

        messages_batch = [self._build_message(a, b) for a, b in pairs]

        # Tokenise all messages in the batch
        texts = [
            self._processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages_batch
        ]

        # Collect all vision inputs
        image_inputs_list = []
        for msg in messages_batch:
            img_inputs, _ = process_vision_info([msg])
            image_inputs_list.extend(img_inputs)

        inputs = self._processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,            # greedy — deterministic for thesis
                temperature=None,
                top_p=None,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        responses = self._processor.batch_decode(
            generated, skip_special_tokens=True
        )

        boundaries = []
        for resp in responses:
            text = resp.strip().upper()
            # "NIE" → boundary; "TAK" → same document
            is_boundary = text.startswith("NIE")
            boundaries.append(is_boundary)
            logger.debug(f"  raw='{resp.strip()}'  boundary={is_boundary}")

        return boundaries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_pair(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> bool:
        """Predict whether two pages belong to DIFFERENT documents.

        Returns True if a boundary exists between img1 and img2.
        """
        self._load()
        return self._predict_batch([(img1, img2)])[0]

    def predict_sequence(
        self,
        images: List[Image.Image],
    ) -> List[bool]:
        """Predict document boundaries for a full page sequence.

        Args:
            images: Ordered list of page images (N pages).

        Returns:
            List of N-1 booleans.
            boundaries[i] == True  →  page i+1 starts a new document.
        """
        if len(images) < 2:
            return []

        self._load()

        pairs = [(images[i], images[i + 1]) for i in range(len(images) - 1)]
        all_boundaries: List[bool] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            logger.debug(
                f"Processing pairs {start+1}–{start+len(batch)} / {len(pairs)}"
            )
            results = self._predict_batch(batch)
            all_boundaries.extend(results)

            # Free intermediate GPU memory between large batches
            if self._device == "cuda" and self.batch_size > 1:
                torch.cuda.empty_cache()

        return all_boundaries

    def predict_sequence_from_paths(
        self,
        image_paths: List[str],
    ) -> List[bool]:
        """Convenience wrapper: load images from disk, then predict."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.predict_sequence(images)
