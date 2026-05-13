# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict
import uuid
from loguru import logger

from models import DocumentAsset, SplicedDocument, SourceDocument, GroundTruthPage
from services.shuffle_strategies.base_strategy import BaseStrategy


class MonoSeq(BaseStrategy):
    """DocSplit-Mono-Seq: Single category document concatenation sequentially.
    
    Creates document packets by concatenating entire documents from the same category
    while preserving original page order. Tests boundary detection without 
    category transitions as discriminative signals.
    """

    def generate(
        self,
        documents_by_type: Dict[str, List[DocumentAsset]],
        doc_names_for_split: Dict[str, List[str]],
        num_spliced_docs: int
    ) -> List[SplicedDocument]:
        
        available_docs = self._get_available_docs(documents_by_type, doc_names_for_split)
        
        # Filter out language category for large datasets (min_pages >= 20)
        if self.min_pages >= 20 and "language" in available_docs:
            del available_docs["language"]
        spliced_documents = []
        
        for i in range(num_spliced_docs):
            # Pick a random document type
            doc_type = self.rng.choice(list(available_docs.keys()))
            
            # Set target page count
            target_pages = self.rng.randint(self.min_pages, self.max_pages)
            
            # Keep adding entire documents until reaching target
            source_documents = []
            ground_truth = []
            current_page = 1
            used_docs = set()
            
            while current_page - 1 < target_pages:
                # Get available docs of this type that haven't been used
                available = [d for d in available_docs[doc_type] 
                           if d.doc_name not in used_docs and d.page_count <= self.max_pages]
                
                if not available:
                    break
                
                # Pick a random document
                doc = self.rng.choice(available)
                used_docs.add(doc.doc_name)
                
                # Check if adding this doc would exceed max_pages
                if current_page - 1 + doc.page_count > self.max_pages:
                    continue
                
                # Add all pages from this document in original order
                pages = list(range(1, doc.page_count + 1))
                
                source_documents.append(SourceDocument(
                    doc_type=doc.doc_type,
                    doc_name=doc.doc_name,
                    pages=pages
                ))
                
                # Add ground truth for each page
                for source_page in pages:
                    ground_truth.append(GroundTruthPage(
                        page_num=current_page,
                        doc_type=doc.doc_type,
                        source_doc=doc.doc_name,
                        source_page=source_page
                    ))
                    current_page += 1
            
            # Only add if we have at least min_pages
            if current_page - 1 >= self.min_pages:
                spliced_doc = SplicedDocument(
                    spliced_doc_id=str(uuid.uuid4()),
                    source_documents=source_documents,
                    ground_truth=ground_truth,
                    total_pages=current_page - 1
                )
                spliced_documents.append(spliced_doc)
        
        logger.info(f"Generated {len(spliced_documents)} DocSplit-Mono-Seq documents")
        return spliced_documents
