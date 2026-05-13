# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import Document
from services.textract_ocr import TextractOcr
from services.deepseek_ocr import DeepSeekOcr
from services.asset_writer import AssetWriter


class AssetCreator:
    """Creates assets from PDFs: extracts images and OCR text."""

    def __init__(self, writer: AssetWriter, ocr):
        """Initialize AssetCreator.
        
        Args:
            writer: AssetWriter instance
            ocr: OCR service (TextractOcr or DeepSeekOcr)
        """
        self.writer = writer
        self.ocr = ocr
        self.is_deepseek = isinstance(ocr, DeepSeekOcr)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def create_assets(self, doc: Document) -> None:
        """Create assets for a single document with retry logic."""
        try:
            logger.trace(f"Processing: {doc.doc_name}")
            
            # Read PDF
            with open(doc.absolute_filepath, 'rb') as f:
                pdf_bytes = f.read()
            
            # Extract text via OCR
            if self.is_deepseek:
                # DeepSeek needs images, extract them first
                from src.idp_labs.core.util.pdf_util import PdfUtil
                images = PdfUtil.get_pages_images_from_pdf(pdf_path=doc.absolute_filepath)
                text_pages = self.ocr.extract_text_from_images(images)
            else:
                # Textract uses PDF bytes
                text_pages = self.ocr.extract_text_from_pdf(pdf_bytes)
            
            # Write all assets
            self.writer.save_document_assets(
                doc_type=doc.doc_type,
                doc_name=doc.doc_name,
                filename=doc.filename,
                pdf_bytes=pdf_bytes,
                text_pages=text_pages
            )
            
            logger.trace(f"Completed: {doc.doc_name}")
            
        except Exception as e:
            logger.exception(f"Error processing {doc.doc_name}: {e}")
            raise

    def create_all(
        self,
        documents: List[Document],
        workers: Optional[int] = None,
        limit: Optional[int] = None
    ) -> dict:
        """Create assets for all documents with concurrent execution.
        
        Returns:
            Dict with 'successful', 'failed', and 'failed_docs' keys.
        """
        if workers is None:
            workers = os.cpu_count() or 1
        
        docs_to_process = documents[:limit] if limit and limit > 0 else documents
        
        logger.info(f"Processing {len(docs_to_process)} documents with {workers} workers")
        
        successful = 0
        failed = 0
        failed_docs = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_doc = {
                executor.submit(self.create_assets, doc): doc
                for doc in docs_to_process
            }
            
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    future.result()
                    successful += 1
                except Exception as e:
                    failed += 1
                    failed_docs.append(doc.doc_name)
                    logger.error(f"Failed: {doc.doc_name} - {e}")
        
        logger.info(f"Completed: {successful} successful, {failed} failed")
        
        return {
            'successful': successful,
            'failed': failed,
            'failed_docs': failed_docs
        }
