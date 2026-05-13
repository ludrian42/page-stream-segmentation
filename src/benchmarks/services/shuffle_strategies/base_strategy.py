# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import ABC, abstractmethod
from typing import List, Dict
import random

from models import DocumentAsset, SplicedDocument


class BaseStrategy(ABC):
    """Abstract base class for shuffle strategies."""

    def __init__(
        self,
        min_pages: int = 17,
        max_pages: int = 20,
        random_seed: int = 42
    ):
        self.min_pages = min_pages
        self.max_pages = max_pages
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)  # nosec B311 - non-cryptographic use for benchmark shuffling

    @abstractmethod
    def generate(
        self,
        documents_by_type: Dict[str, List[DocumentAsset]],
        doc_names_for_split: Dict[str, List[str]],
        num_spliced_docs: int
    ) -> List[SplicedDocument]:
        """Generate spliced documents using this strategy.
        
        Args:
            documents_by_type: All available documents grouped by type
            doc_names_for_split: Document names to use for this split
            num_spliced_docs: Number of spliced documents to generate
            
        Returns:
            List of SplicedDocument objects
        """
        pass

    def _get_available_docs(
        self,
        documents_by_type: Dict[str, List[DocumentAsset]],
        doc_names_for_split: Dict[str, List[str]]
    ) -> Dict[str, List[DocumentAsset]]:
        """Filter documents to only those in the split."""
        available = {}
        
        for doc_type, doc_names in doc_names_for_split.items():
            if doc_type not in documents_by_type:
                continue
            
            doc_name_set = set(doc_names)
            available[doc_type] = [
                doc for doc in documents_by_type[doc_type]
                if doc.doc_name in doc_name_set
            ]
        
        return available

    def _get_random_doc(
        self,
        available_docs: Dict[str, List[DocumentAsset]],
        doc_type: str = None
    ) -> DocumentAsset:
        """Get a random document, optionally from a specific type."""
        if doc_type:
            if doc_type not in available_docs or not available_docs[doc_type]:
                raise ValueError(f"No documents available for type: {doc_type}")
            return self.rng.choice(available_docs[doc_type])
        else:
            all_docs = [doc for docs in available_docs.values() for doc in docs]
            if not all_docs:
                raise ValueError("No documents available")
            return self.rng.choice(all_docs)

    def _get_random_pages(
        self,
        doc: DocumentAsset,
        num_pages: int = None
    ) -> List[int]:
        """Get random page numbers from a document."""
        if num_pages is None:
            num_pages = self.rng.randint(self.min_pages, self.max_pages)
        
        num_pages = min(num_pages, doc.page_count)
        return self.rng.sample(range(1, doc.page_count + 1), num_pages)
