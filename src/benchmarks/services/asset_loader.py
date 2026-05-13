# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import csv
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from models import DocumentAsset, PageAsset


class AssetLoader:
    """Loads document assets from create_assets output using document_mapping.csv."""

    def __init__(self, assets_path: str, mapping_csv: str = None):
        """Initialize asset loader.
        
        Args:
            assets_path: Path to assets directory
            mapping_csv: Path to document_mapping.csv (default: data/metadata/document_mapping.csv)
        """
        self.assets_path = Path(assets_path)
        
        if mapping_csv is None:
            mapping_csv = "data/metadata/document_mapping.csv"
        
        self.mapping_csv = Path(mapping_csv)
        
        if not self.mapping_csv.exists():
            raise FileNotFoundError(f"Document mapping not found: {self.mapping_csv}")

    def load_all_documents(self, doc_types: List[str] = None) -> Dict[str, List[DocumentAsset]]:
        """Load all document assets grouped by type from CSV.
        
        Args:
            doc_types: Optional list of document types to load. If None, loads all.
            
        Returns:
            Dict mapping doc_type to list of DocumentAsset objects.
        """
        documents_by_type = {}
        
        # Read document mapping CSV
        with open(self.mapping_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_type = row['type']
                doc_name = row['doc_name']
                filename = row['filename']
                page_count = int(row['pages'])
                
                # Filter by doc_types if specified
                if doc_types and doc_type not in doc_types:
                    continue
                
                # Build document asset
                doc_asset = self._create_document_asset(
                    doc_type=doc_type,
                    doc_name=doc_name,
                    filename=filename,
                    page_count=page_count
                )
                
                if doc_type not in documents_by_type:
                    documents_by_type[doc_type] = []
                
                documents_by_type[doc_type].append(doc_asset)
        
        total = sum(len(docs) for docs in documents_by_type.values())
        logger.info(f"Loaded {total} documents across {len(documents_by_type)} types from CSV")
        
        return documents_by_type

    def _create_document_asset(
        self,
        doc_type: str,
        doc_name: str,
        filename: str,
        page_count: int
    ) -> DocumentAsset:
        """Create DocumentAsset with page information.
        
        Args:
            doc_type: Document category
            doc_name: Document identifier
            filename: PDF filename
            page_count: Number of pages
            
        Returns:
            DocumentAsset with populated pages
        """
        pages = []
        
        for page_num in range(1, page_count + 1):
            page_num_str = f"{page_num:04d}"
            
            # Construct paths based on known structure
            page_dir = self.assets_path / doc_type / filename / "pages" / page_num_str
            image_path = page_dir / f"page-{page_num_str}.png"
            text_path = page_dir / f"page-{page_num_str}-textract.md"
            
            page_asset = PageAsset(
                page_num=page_num,
                image_path=str(image_path),
                text_path=str(text_path),
                text_content=None  # Loaded on demand
            )
            pages.append(page_asset)
        
        return DocumentAsset(
            doc_type=doc_type,
            doc_name=doc_name,
            filename=filename,
            page_count=page_count,
            pages=pages
        )
