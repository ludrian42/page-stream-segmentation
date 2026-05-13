# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in the dataset."""

    doc_type: str = Field(..., description="Document category/type")
    doc_name: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="PDF filename")
    absolute_filepath: str = Field(..., description="Full path to PDF file")
    page_count: int = Field(..., gt=0, description="Number of pages in document")
