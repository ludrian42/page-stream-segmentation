# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict
import uuid
from loguru import logger

from models import DocumentAsset, SplicedDocument, SourceDocument, GroundTruthPage
from services.shuffle_strategies.base_strategy import BaseStrategy


class PolyRand(BaseStrategy):
    """DocSplit-Poly-Rand: Multi category document pages randomization.
    
    Similar to Poly-Seq but applies complete randomization across all pages,
    representing maximum entropy scenarios. Stress-tests model robustness under
    worst-case conditions where no structural assumptions hold.
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
            
            # Collect entire documents from different types
            selected_docs = []
            used_types = set()
            total_pages = 0
            
            while total_pages < target_pages:
                available_types = [t for t in doc_types if t not in used_types]
                if not available_types:
                    break
                
                doc_type = self.rng.choice(available_types)
                used_types.add(doc_type)
                
                available = [d for d in available_docs[doc_type] if d.page_count <= self.max_pages]
                if not available:
                    continue
                
                doc = self.rng.choice(available)
                
                if total_pages + doc.page_count > self.max_pages:
                    continue
                
                selected_docs.append(doc)
                total_pages += doc.page_count
            
            # Collect all pages from selected documents
            all_pages = []
            for doc in selected_docs:
                for page_num in range(1, doc.page_count + 1):
                    all_pages.append({
                        'doc_type': doc.doc_type,
                        'doc_name': doc.doc_name,
                        'source_page': page_num
                    })
            
            # Fully shuffle all pages
            self.rng.shuffle(all_pages)
            
            # Build source documents and ground truth
            source_docs_dict = {}
            ground_truth = []
            
            for idx, page_info in enumerate(all_pages, start=1):
                doc_key = (page_info['doc_type'], page_info['doc_name'])
                
                if doc_key not in source_docs_dict:
                    source_docs_dict[doc_key] = []
                
                source_docs_dict[doc_key].append(page_info['source_page'])
                
                ground_truth.append(GroundTruthPage(
                    page_num=idx,
                    doc_type=page_info['doc_type'],
                    source_doc=page_info['doc_name'],
                    source_page=page_info['source_page']
                ))
            
            source_documents = [
                SourceDocument(doc_type=doc_type, doc_name=doc_name, pages=pages)
                for (doc_type, doc_name), pages in source_docs_dict.items()
            ]
            
            if len(all_pages) >= self.min_pages:
                spliced_doc = SplicedDocument(
                    spliced_doc_id=str(uuid.uuid4()),
                    source_documents=source_documents,
                    ground_truth=ground_truth,
                    total_pages=len(all_pages)
                )
                spliced_documents.append(spliced_doc)
        
        logger.info(f"Generated {len(spliced_documents)} DocSplit-Poly-Rand documents")
        return spliced_documents
