# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Represents a source document used in splicing."""

    doc_type: str = Field(..., description="Document category/type")
    doc_name: str = Field(..., description="Source document identifier")
    pages: List[int] = Field(..., description="Page numbers used from this document")


class GroundTruthPage(BaseModel):
    """Ground truth for a single page in spliced document."""

    page_num: int = Field(..., ge=1, description="Page number in spliced document")
    doc_type: str = Field(..., description="Document category/type")
    source_doc: str = Field(..., description="Source document identifier")
    source_page: int = Field(..., ge=1, description="Page number in source document")


class SplicedDocument(BaseModel):
    """Represents a spliced benchmark document."""

    spliced_doc_id: str = Field(..., description="Unique identifier for spliced document")
    source_documents: List[SourceDocument] = Field(..., description="Source documents used")
    ground_truth: List[GroundTruthPage] = Field(..., description="Ground truth page mappings")
    total_pages: int = Field(..., gt=0, description="Total pages in spliced document")


class BenchmarkSet(BaseModel):
    """Collection of spliced documents for a benchmark."""

    benchmark_name: str = Field(..., description="Benchmark identifier")
    strategy: str = Field(..., description="Shuffling strategy used")
    split: str = Field(..., description="Dataset split: train, test, or validation")
    created_at: str = Field(..., description="Creation timestamp")
    documents: List[SplicedDocument] = Field(..., description="Spliced documents")
    statistics: Dict[str, int] = Field(default_factory=dict, description="Benchmark statistics")


class DocumentAsset(BaseModel):
    """Represents a loaded document asset."""

    doc_type: str = Field(..., description="Document category/type")
    doc_name: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="PDF filename")
    page_count: int = Field(..., gt=0, description="Number of pages")
    pages: List['PageAsset'] = Field(default_factory=list, description="Page assets")


class PageAsset(BaseModel):
    """Represents a single page asset."""

    page_num: int = Field(..., ge=1, description="Page number")
    image_path: str = Field(..., description="Path to page image")
    text_path: str = Field(..., description="Path to OCR text")
    text_content: Optional[str] = Field(None, description="Loaded text content")
