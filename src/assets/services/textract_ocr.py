# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List
from loguru import logger
from textractor import Textractor
from textractor.data.constants import TextractFeatures


class TextractOcr:
    """Extracts text from PDFs using AWS Textract with S3 upload."""

    def __init__(self, s3_bucket: str, s3_prefix: str = 'textract-temp'):
        """Initialize Textract OCR.
        
        Args:
            s3_bucket: S3 bucket for temporary uploads
            s3_prefix: S3 prefix for uploads
        """
        self.textractor = Textractor()
        self.s3_upload_path = f"s3://{s3_bucket}/{s3_prefix}"
        logger.info(f"Textract S3 upload path: {self.s3_upload_path}")

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> List[str]:
        """Extract text from PDF using Textract with S3 upload.
        
        Returns:
            List of markdown text, one per page.
        """
        # Use start_document_analysis with S3 upload
        lazy_document = self.textractor.start_document_analysis(
            file_source=pdf_bytes,
            s3_upload_path=self.s3_upload_path,
            features=[TextractFeatures.LAYOUT],
            save_image=False,
        )
        
        # Wait for completion
        _ = lazy_document.response
        pages = lazy_document.document.pages
        
        # Convert to markdown
        page_texts = []
        for page in pages:
            try:
                page_content = page.to_markdown()
            except Exception:
                page_content = page.get_text()
            page_texts.append(page_content)
        
        return page_texts if page_texts else ['']
