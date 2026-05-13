# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Create Assets from Raw PDFs

Processes raw PDFs and creates structured assets:
- Page images (PNG at 300 DPI)
- OCR text (AWS Textract)

Usage:
    python main.py
    python main.py --workers 20 --limit 100
"""
import os
import argparse
from loguru import logger

from services.pdf_loader import PdfLoader
from services.textract_ocr import TextractOcr
from services.deepseek_ocr import DeepSeekOcr
from services.asset_writer import AssetWriter
from services.asset_creator import AssetCreator


def main():
    parser = argparse.ArgumentParser(description='Create assets from raw PDFs')
    parser.add_argument('--raw-data-path', default='../raw_data')
    parser.add_argument('--output-path', default='../processed_assets')
    parser.add_argument('--metadata-path', default='../metadata')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save-mapping', action='store_true')
    parser.add_argument('--use-deepseek-for-language', action='store_true', help='Use DeepSeek OCR for language docs (default: Textract)')
    parser.add_argument('--s3-bucket', default=os.getenv('DOCSPLIT_S3_BUCKET'), help='S3 bucket for Textract temporary uploads')
    parser.add_argument('--s3-prefix', default='textract-temp', help='S3 prefix for uploads')

    args = parser.parse_args()

    logger.info("Creating assets from PDFs")

    # Load all PDFs
    loader = PdfLoader(raw_data_path=args.raw_data_path)
    documents = loader.get_all_documents()

    successful_docs = []

    if args.use_deepseek_for_language:
        # Separate language documents for DeepSeek processing
        language_docs = [doc for doc in documents if doc.doc_type == 'language']
        other_docs = [doc for doc in documents if doc.doc_type != 'language']

        # Process non-language documents with Textract
        if other_docs:
            logger.info(f"Processing {len(other_docs)} non-language documents with Textract")
            ocr = TextractOcr(s3_bucket=args.s3_bucket, s3_prefix=args.s3_prefix)
            writer = AssetWriter(output_base_path=args.output_path)
            creator = AssetCreator(writer, ocr)

            results = creator.create_all(
                documents=other_docs,
                workers=args.workers,
                limit=args.limit
            )

            logger.info(f"Textract completed: {results}")
            if results['failed'] > 0:
                logger.warning(f"Textract failed: {results['failed_docs']}")

            failed_names = set(results['failed_docs'])
            successful_docs.extend([doc for doc in other_docs if doc.doc_name not in failed_names])

        # Process language documents with DeepSeek
        if language_docs:
            logger.info(f"Processing {len(language_docs)} language documents with DeepSeek OCR")
            try:
                deepseek_ocr = DeepSeekOcr()
                writer = AssetWriter(output_base_path=args.output_path)
                creator = AssetCreator(writer, deepseek_ocr)

                results = creator.create_all(
                    documents=language_docs,
                    workers=args.workers,
                    limit=args.limit
                )

                logger.info(f"DeepSeek completed: {results}")
                if results['failed'] > 0:
                    logger.warning(f"DeepSeek failed: {results['failed_docs']}")

                failed_names = set(results['failed_docs'])
                successful_docs.extend([doc for doc in language_docs if doc.doc_name not in failed_names])
            except Exception as e:
                logger.error(f"DeepSeek OCR initialization failed: {e}")
    else:
        # Process ALL documents with Textract
        logger.info(f"Processing {len(documents)} documents with Textract")
        ocr = TextractOcr(s3_bucket=args.s3_bucket, s3_prefix=args.s3_prefix)
        writer = AssetWriter(output_base_path=args.output_path)
        creator = AssetCreator(writer, ocr)

        results = creator.create_all(
            documents=documents,
            workers=args.workers,
            limit=args.limit
        )

        logger.info(f"Textract completed: {results}")
        if results['failed'] > 0:
            logger.warning(f"Textract failed: {results['failed_docs']}")

        failed_names = set(results['failed_docs'])
        successful_docs.extend([doc for doc in documents if doc.doc_name not in failed_names])

    # Save mapping AFTER processing with only successful documents
    if args.save_mapping:
        mapping_path = f"{args.metadata_path}/document_mapping.csv"
        loader.save_document_mapping(successful_docs, output_path=mapping_path)
        logger.info(f"Saved mapping for {len(successful_docs)} successful documents")


if __name__ == '__main__':
    main()
