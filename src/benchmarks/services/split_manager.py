# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import random
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from loguru import logger

from models import DocumentAsset


class SplitManager:
    """Manages train/test/validation splits for documents."""

    def __init__(
        self,
        train_ratio: float = 0.6,
        test_ratio: float = 0.25,
        validation_ratio: float = 0.15,
        random_seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.random_seed = random_seed

        if abs(train_ratio + test_ratio + validation_ratio - 1.0) > 0.001:
            raise ValueError("Split ratios must sum to 1.0")

    def create_split(
        self,
        documents_by_type: Dict[str, List[DocumentAsset]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """Create stratified train/test/validation split.

        Returns:
            Dict with structure: {split_name: {doc_type: [doc_names]}}
        """
        random.seed(self.random_seed)

        splits = {
            'train': {},
            'test': {},
            'validation': {}
        }

        for doc_type, documents in documents_by_type.items():
            doc_names = [doc.doc_name for doc in documents]
            random.shuffle(doc_names)

            total = len(doc_names)
            train_end = int(total * self.train_ratio)
            test_end = train_end + int(total * self.test_ratio)

            splits['train'][doc_type] = doc_names[:train_end]
            splits['test'][doc_type] = doc_names[train_end:test_end]
            splits['validation'][doc_type] = doc_names[test_end:]

        logger.info(f"Created split: train={self._count_docs(splits['train'])}, "
                   f"test={self._count_docs(splits['test'])}, "
                   f"val={self._count_docs(splits['validation'])}")

        return splits

    def save_split(self, splits: Dict, output_path: str):
        """Save split mapping to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        split_data = {
            'created_at': datetime.now().isoformat(),
            'split_config': {
                'train_ratio': self.train_ratio,
                'test_ratio': self.test_ratio,
                'validation_ratio': self.validation_ratio,
                'random_seed': self.random_seed,
                'stratified': True
            },
            'splits': splits,
            'statistics': {
                'train': self._get_statistics(splits['train']),
                'test': self._get_statistics(splits['test']),
                'validation': self._get_statistics(splits['validation'])
            }
        }

        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        logger.info(f"Saved split mapping to {output_path}")

    def load_split(self, split_path: str) -> Dict[str, Dict[str, List[str]]]:
        """Load split mapping from JSON."""
        with open(split_path, 'r') as f:
            split_data = json.load(f)

        logger.info(f"Loaded split mapping from {split_path}")
        return split_data['splits']

    def _count_docs(self, split: Dict[str, List[str]]) -> int:
        """Count total documents in a split."""
        return sum(len(docs) for docs in split.values())

    def _get_statistics(self, split: Dict[str, List[str]]) -> Dict[str, int]:
        """Get statistics for a split."""
        stats = {'total': self._count_docs(split)}
        for doc_type, docs in split.items():
            stats[doc_type] = len(docs)
        return stats
