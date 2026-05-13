# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from pathlib import Path
from typing import List
from PIL import Image
import fitz
from loguru import logger


class AssetWriter:
    """Writes document assets (PDF, images, OCR text) to disk."""

    def __init__(self, output_base_path: str = ".data/rvl-cdip-nmp-hf/rvl-cdip-nmp-assets"):
        self.output_base_path = Path(output_base_path)

    def save_document_assets(
        self,
        doc_type: str,
        doc_name: str,
        filename: str,
        pdf_bytes: bytes,
        text_pages: List[str]
    ):
        """Save all assets for a document: original PDF, page images, and OCR text."""
        
        # Create directory structure: {doc_type}/{filename}/
        doc_dir = self.output_base_path / doc_type / filename
        original_dir = doc_dir / "original"
        pages_dir = doc_dir / "pages"
        
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original PDF
        original_pdf_path = original_dir / filename
        with open(original_pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Extract and save page images
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            page_num_str = f"{page_num + 1:04d}"
            
            # Create page directory
            page_dir = pages_dir / page_num_str
            page_dir.mkdir(parents=True, exist_ok=True)
            
            # Save page image
            pix = page.get_pixmap(dpi=300)
            img_path = page_dir / f"page-{page_num_str}.png"
            pix.save(str(img_path))
            
            # Save OCR text
            if page_num < len(text_pages):
                text_path = page_dir / f"page-{page_num_str}-textract.md"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_pages[page_num])
        
        pdf_doc.close()
        logger.debug(f"Saved assets for {doc_type}/{doc_name}")
