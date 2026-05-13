# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import sys
import io
import tempfile
import shutil
import re
from typing import List, Optional
from contextlib import redirect_stdout
from loguru import logger
from PIL import Image

# Pin model revision for reproducibility and supply chain security
# Update README.md 9f30c71 commited on Nov 3, 2025
# False positive, high entropy string is acutally a commit hash required to remediate B615
MODEL_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"  # nosec SECRET-HEX-HIGH-ENTROPY-STRING


class DeepSeekOcr:
    """DeepSeek OCR for language documents."""

    def __init__(
        self, 
        model_name: str = "deepseek-ai/DeepSeek-OCR", 
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        """Initialize DeepSeek OCR from Hugging Face.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('cuda' or 'cpu')
            cache_dir: Optional cache directory for model downloads (use larger disk if needed)
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Verify CUDA availability
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU. Performance will be significantly slower.")
                device = "cpu"
            
            logger.info(f"Loading DeepSeek model: {model_name} on {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained( # nosec B615 - False positive, see MODEL_REVISION is set to a specific version hash
                model_name, 
                trust_remote_code=True,
                cache_dir=cache_dir,
                revision=MODEL_REVISION
            )
            
            self.model = AutoModel.from_pretrained(  # nosec B615 - False positive, see MODEL_REVISION is set to a specific version hash
                model_name,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                revision=MODEL_REVISION
            )
            
            self.model = self.model.eval()
            if device == "cuda":
                self.model = self.model.cuda()
            
            self.device = device
            logger.info(f"DeepSeek model loaded successfully on {device}")
        except ImportError as e:
            logger.error(f"Failed to import dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model: {e}")
            raise

    def extract_text_from_images(self, images: List[Image.Image]) -> List[str]:
        """Extract text from page images using DeepSeek OCR.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of markdown text per page
        """
        texts = []
        temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
        
        try:
            for idx, image in enumerate(images):
                if not isinstance(image, Image.Image):
                    logger.warning(f"Page {idx + 1} is not a valid PIL Image, skipping")
                    texts.append("")
                    continue
                    
                try:
                    # Suppress model debug output
                    with redirect_stdout(io.StringIO()):
                        self.model.infer(
                            self.tokenizer,
                            prompt="<image>\n<|grounding|>Convert the document to markdown.",
                            image_file=image,
                            output_path=temp_dir,
                            base_size=1024,
                            image_size=640,
                            crop_mode=True,
                            save_results=True
                        )
                    
                    # Read result from saved file
                    result_file = os.path.join(temp_dir, 'result.mmd')
                    if os.path.exists(result_file):
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = f.read()
                        
                        # Clean markup tags
                        clean_text = re.sub(r'<\|ref\|>text<\|/ref\|>', '', result)
                        clean_text = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', clean_text)
                        clean_text = clean_text.strip()
                        
                        texts.append(clean_text)
                        
                        # Delete all output files after reading
                        for item in os.listdir(temp_dir):
                            item_path = os.path.join(temp_dir, item)
                            try:
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                            except Exception as e:
                                logger.debug(f"Failed to clean up {item_path}: {e}")
                        
                        logger.info(f"DeepSeek OCR completed for page {idx + 1}")
                    else:
                        logger.warning(f"No result file found for page {idx + 1}")
                        texts.append("")
                        
                except Exception as e:
                    logger.error(f"DeepSeek OCR error on page {idx + 1}: {e}")
                    texts.append("")
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
        
        return texts


