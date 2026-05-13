# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict
import uuid
from loguru import logger

from models import DocumentAsset, SplicedDocument, SourceDocument, GroundTruthPage
from services.shuffle_strategies.base_strategy import BaseStrategy


class PolySeq(BaseStrategy):
    """DocSplit-Poly-Seq: Multi category documents concatenation sequentially.
    
    Creates document packets by concatenating documents from different categories
    without repetition while preserving page ordering. Simulates heterogeneous 
    document assembly scenarios such as medical claims processing.
    """

    def generate(
        self,
        documents_by_type: Dict[str, List[DocumentAsset]],
        doc_names_for_split: Dict[str, List[str]],
        num_spliced_docs: int
    ) -> List[SplicedDocument]:
        
        available_docs = self._get_available_docs(documents_by_type, doc_names_for_split)
        doc_types = list(available_docs.keys())
        
        if len(doc_types) < 2:
            raise ValueError("Need at least 2 document types for multi-category strategy")
        
        spliced_documents = []
        
        for i in range(num_spliced_docs):
            # Set target page count
            target_pages = self.rng.randint(self.min_pages, self.max_pages)
            
            # Keep adding entire documents from different types until reaching target
            source_documents = []
            ground_truth = []
            current_page = 1
            used_types = set()
            
            while current_page - 1 < target_pages:
                # Get available types not yet used
                available_types = [t for t in doc_types if t not in used_types]
                
                if not available_types:
                    break
                
                # Pick a random type
                doc_type = self.rng.choice(available_types)
                used_types.add(doc_type)
                
                # Pick a random document from this type
                available = [d for d in available_docs[doc_type] if d.page_count <= self.max_pages]
                if not available:
                    continue
                
                doc = self.rng.choice(available)
                
                if current_page - 1 + doc.page_count > self.max_pages:
                    continue
                
                # Add all pages in original order
                pages = list(range(1, doc.page_count + 1))
                
                source_documents.append(SourceDocument(
                    doc_type=doc.doc_type,
                    doc_name=doc.doc_name,
                    pages=pages
                ))
                
                for source_page in pages:
                    ground_truth.append(GroundTruthPage(
                        page_num=current_page,
                        doc_type=doc.doc_type,
                        source_doc=doc.doc_name,
                        source_page=source_page
                    ))
                    current_page += 1
            
            if current_page - 1 >= self.min_pages:
                spliced_doc = SplicedDocument(
                    spliced_doc_id=str(uuid.uuid4()),
                    source_documents=source_documents,
                    ground_truth=ground_truth,
                    total_pages=current_page - 1
                )
                spliced_documents.append(spliced_doc)
        
        logger.info(f"Generated {len(spliced_documents)} DocSplit-Poly-Seq documents")
        return spliced_documents
