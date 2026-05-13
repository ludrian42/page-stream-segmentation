# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
import json
from pathlib import Path
from loguru import logger

from models import BenchmarkSet


class BenchmarkWriter:
    """Writes benchmark datasets to disk as CSV files."""

    def __init__(self, output_base_path: str, assets_path: str = "data/assets"):
        self.output_base_path = Path(output_base_path)
        self.assets_path = Path(assets_path)

    def save_benchmark_set(
        self,
        benchmark_set: BenchmarkSet,
        split: str
    ):
        """Save a benchmark set to CSV and ground truth JSON files.
        
        Args:
            benchmark_set: The benchmark set to save
            split: Split name (train, test, or validation)
        """
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        output_path = self.output_base_path / f"{split}.csv"
        json_dir = self.output_base_path / "ground_truth_json" / split
        json_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate group IDs (AA, AB, AC, etc.)
        def generate_group_id(index):
            """Generate group_id like AA, AB, AC, ..., AZ, BA, BB, etc."""
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            first_char = alphabet[index // 26 % 26]
            second_char = alphabet[index % 26]
            return first_char + second_char
        
        # Flatten ground truth pages into CSV rows
        rows = []
        
        # Track unique source documents across all packets for group_id assignment
        source_doc_to_group_id = {}
        group_id_counter = 0
        
        for doc_idx, doc in enumerate(benchmark_set.documents):
            # Track page ordinal per source document
            source_doc_ordinals = {}
            # Track local_doc_id per source document within this packet
            source_doc_to_local_id = {}
            local_id_counter = {}
            
            for gt_page in doc.ground_truth:
                source_key = gt_page.source_doc
                
                # Assign group_id per unique source document
                if source_key not in source_doc_to_group_id:
                    source_doc_to_group_id[source_key] = generate_group_id(group_id_counter)
                    group_id_counter += 1
                
                group_id = source_doc_to_group_id[source_key]
                
                # Increment ordinal for this source document
                if source_key not in source_doc_ordinals:
                    source_doc_ordinals[source_key] = 0
                source_doc_ordinals[source_key] += 1
                
                # Assign local_doc_id per source document
                if source_key not in source_doc_to_local_id:
                    doc_type_prefix = gt_page.doc_type.replace(" ", "-").lower()
                    if doc_type_prefix not in local_id_counter:
                        local_id_counter[doc_type_prefix] = 0
                    local_id_counter[doc_type_prefix] += 1
                    source_doc_to_local_id[source_key] = f"{doc_type_prefix}-{local_id_counter[doc_type_prefix]:02d}"
                
                local_doc_id = source_doc_to_local_id[source_key]
                
                # Build paths
                image_path = f"{self.assets_path}/{gt_page.doc_type}/{gt_page.source_doc}.pdf/pages/{gt_page.source_page:04d}/page-{gt_page.source_page:04d}.png"
                text_path = f"{self.assets_path}/{gt_page.doc_type}/{gt_page.source_doc}.pdf/pages/{gt_page.source_page:04d}/page-{gt_page.source_page:04d}-textract.md"
                
                rows.append({
                    'doc_type': gt_page.doc_type,
                    'original_doc_name': gt_page.source_doc,
                    'parent_doc_name': doc.spliced_doc_id,
                    'local_doc_id': local_doc_id,
                    'page': gt_page.page_num,
                    'image_path': image_path,
                    'text_path': text_path,
                    'group_id': group_id,
                    'local_doc_id_page_ordinal': source_doc_ordinals[source_key]
                })
            
            # Generate ground truth JSON for this parent document
            self._save_ground_truth_json(doc, json_dir)
        
        # Write CSV
        if rows:
            fieldnames = ['doc_type', 'original_doc_name', 'parent_doc_name', 'local_doc_id', 
                         'page', 'image_path', 'text_path', 'group_id', 'local_doc_id_page_ordinal']
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        logger.info(f"Saved {len(benchmark_set.documents)} spliced documents ({len(rows)} pages) to {output_path}")
        logger.info(f"Saved {len(benchmark_set.documents)} ground truth JSON files to {json_dir}")
    
    def _save_ground_truth_json(self, doc, json_dir: Path):
        """Generate ground truth JSON for a single parent document."""
        # Group pages by source document (doc_type, source_doc)
        subdocs = {}
        for gt_page in doc.ground_truth:
            key = (gt_page.doc_type, gt_page.source_doc)
            if key not in subdocs:
                subdocs[key] = []
            subdocs[key].append(gt_page)
        
        # Build subdocuments structure
        subdocuments = []
        
        for subdoc_idx, ((doc_type, source_doc), pages) in enumerate(subdocs.items()):
            group_id = self._generate_group_id(subdoc_idx)
            
            # Count occurrences of this doc_type so far
            doc_type_count = sum(1 for s in subdocuments if s['doc_type_id'] == doc_type) + 1
            local_doc_id = f"{doc_type}-{doc_type_count:02d}"
            
            page_ordinals = [p.page_num for p in pages]
            
            subdoc_pages = []
            for idx, gt_page in enumerate(pages, 1):
                image_path = f"{self.assets_path}/{gt_page.doc_type}/{gt_page.source_doc}.pdf/pages/{gt_page.source_page:04d}/page-{gt_page.source_page:04d}.png"
                text_path = f"{self.assets_path}/{gt_page.doc_type}/{gt_page.source_doc}.pdf/pages/{gt_page.source_page:04d}/page-{gt_page.source_page:04d}-textract.md"
                
                subdoc_pages.append({
                    "page": gt_page.page_num,
                    "original_doc_name": gt_page.source_doc,
                    "image_path": image_path,
                    "text_path": text_path,
                    "local_doc_id_page_ordinal": idx
                })
            
            subdocuments.append({
                "doc_type_id": doc_type,
                "page_ordinals": page_ordinals,
                "local_doc_id": local_doc_id,
                "group_id": group_id,
                "pages": subdoc_pages
            })
        
        # Build final JSON structure
        ground_truth = {
            "doc_id": doc.spliced_doc_id,
            "total_pages": doc.total_pages,
            "subdocuments": subdocuments
        }
        
        # Write JSON file
        json_path = json_dir / f"{doc.spliced_doc_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)
    
    def _generate_group_id(self, index):
        """Generate group_id like AA, AB, AC, ..., AZ, BA, BB, etc."""
        result = ""
        index += 1
        while index > 0:
            index -= 1
            result = chr(65 + (index % 26)) + result
            index //= 26
        return result or "A"
