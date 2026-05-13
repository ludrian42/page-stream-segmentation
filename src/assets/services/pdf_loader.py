# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from pathlib import Path
from typing import List, Optional
import csv
from loguru import logger
import fitz

from models import Document


class PdfLoader:
    """Loads PDF files from raw data directory."""

    def __init__(self, raw_data_path: str = ".data/rvl-cdip-mp/raw_data", max_file_size_mb: float = 150):
        """Initialize PDF loader.
        
        Args:
            raw_data_path: Path to raw data directory
            max_file_size_mb: Maximum file size in MB for valid PDFs
        """
        self.raw_data_path = Path(raw_data_path)
        self.max_file_size_mb = max_file_size_mb

    def validate_pdf(self, file_path: Path) -> tuple[bool, str, int]:
        """Validate if a PDF is readable and within size limits.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, status_message, page_count)
        """
        try:
            # Check file size
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            if size_bytes == 0:
                return False, "ZERO_SIZE", 0
            
            if size_mb > self.max_file_size_mb:
                return False, f"TOO_LARGE ({size_mb:.1f}MB)", 0
            
            # Check PDF header
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b"%PDF":
                    return False, "INVALID_PDF_HEADER", 0
            
            # Test readability with PyMuPDF
            doc = fitz.open(str(file_path))
            
            if doc.is_encrypted:
                doc.close()
                return False, "ENCRYPTED_PDF", 0
            
            page_count = doc.page_count
            if page_count == 0:
                doc.close()
                return False, "NO_PAGES", 0
            
            # Try to access first page
            page = doc[0]
            page.get_text()
            doc.close()
            
            return True, "VALID", page_count
            
        except Exception as e:
            return False, f"ERROR: {str(e)[:50]}", 0

    def get_all_documents(self, exclude_types: Optional[List[str]] = None) -> List[Document]:
        """Get all valid PDF documents from raw data directory.
        
        Args:
            exclude_types: List of document types to exclude (e.g., ['language'])
        """
        if exclude_types is None:
            exclude_types = []
            
        documents = []
        
        for doc_type_dir in self.raw_data_path.iterdir():
            if not doc_type_dir.is_dir():
                continue
                
            doc_type = doc_type_dir.name
            
            if doc_type in exclude_types:
                logger.info(f"Skipping excluded type: {doc_type}")
                continue
            
            for pdf_file in doc_type_dir.glob("*.pdf"):
                is_valid, status, page_count = self.validate_pdf(pdf_file)
                
                if not is_valid:
                    logger.warning(f"Skipping invalid PDF {pdf_file}: {status}")
                    continue
                
                doc = Document(
                    doc_type=doc_type,
                    doc_name=pdf_file.stem,
                    filename=pdf_file.name,
                    absolute_filepath=str(pdf_file.absolute()),
                    page_count=page_count
                )
                documents.append(doc)
                    
        logger.info(f"Loaded {len(documents)} valid documents")
        return documents

    def save_document_mapping(self, documents: List[Document], output_path: str):
        """Save document mapping to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['type', 'doc_name', 'filename', 'pages', 'validation_status'])
            writer.writeheader()
            for doc in documents:
                writer.writerow({
                    'type': doc.doc_type,
                    'doc_name': doc.doc_name,
                    'filename': doc.filename,
                    'pages': doc.page_count,
                    'validation_status': 'VALID'
                })
        
        logger.info(f"Saved document mapping to {output_path}")
