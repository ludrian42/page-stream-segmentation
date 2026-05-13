# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Dict, List
from datetime import datetime
from loguru import logger

from models import DocumentAsset, BenchmarkSet
from services.shuffle_strategies.base_strategy import BaseStrategy


class BenchmarkGenerator:
    """Orchestrates benchmark generation."""

    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def generate_for_split(
        self,
        documents_by_type: Dict[str, List[DocumentAsset]],
        doc_names_for_split: Dict[str, List[str]],
        num_spliced_docs: int,
        split_name: str,
        benchmark_name: str
    ) -> BenchmarkSet:
        """Generate benchmark set for a specific split.
        
        Args:
            documents_by_type: All available documents grouped by type
            doc_names_for_split: Document names to use for this split
            num_spliced_docs: Number of spliced documents to generate
            split_name: Name of the split (train, test, validation)
            benchmark_name: Name of the benchmark
            
        Returns:
            BenchmarkSet object
        """
        logger.info(f"Generating {num_spliced_docs} documents for {split_name} split")
        
        spliced_documents = self.strategy.generate(
            documents_by_type=documents_by_type,
            doc_names_for_split=doc_names_for_split,
            num_spliced_docs=num_spliced_docs
        )
        
        # Calculate statistics
        statistics = self._calculate_statistics(spliced_documents)
        
        benchmark_set = BenchmarkSet(
            benchmark_name=benchmark_name,
            strategy=self.strategy.__class__.__name__,
            split=split_name,
            created_at=datetime.now().isoformat(),
            documents=spliced_documents,
            statistics=statistics
        )
        
        logger.info(f"Generated benchmark set with {len(spliced_documents)} documents")
        return benchmark_set

    def _calculate_statistics(self, spliced_documents: List) -> Dict[str, int]:
        """Calculate statistics for the benchmark set."""
        total_pages = sum(doc.total_pages for doc in spliced_documents)
        total_source_docs = sum(len(doc.source_documents) for doc in spliced_documents)
        
        doc_types = set()
        for doc in spliced_documents:
            for source in doc.source_documents:
                doc_types.add(source.doc_type)
        
        return {
            'total_spliced_documents': len(spliced_documents),
            'total_pages': total_pages,
            'total_source_documents': total_source_docs,
            'unique_doc_types': len(doc_types),
            'avg_pages_per_document': int(total_pages / len(spliced_documents)) if spliced_documents else 0
        }
