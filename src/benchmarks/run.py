# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Create Benchmarks from Assets

Generates document splitting benchmarks from structured assets.

Usage:
    python main.py --strategy multi_category_concat
    python main.py --strategy single_category_concat --num-docs-train 500
"""

import argparse
from pathlib import Path
from loguru import logger

from services.asset_loader import AssetLoader
from services.split_manager import SplitManager
from services.benchmark_generator import BenchmarkGenerator
from services.benchmark_writer import BenchmarkWriter
from services.shuffle_strategies import get_strategy, STRATEGIES


def main():
    parser = argparse.ArgumentParser(description='Create benchmarks from assets')

    # Paths
    parser.add_argument('--assets-path', default='data/assets',
                       help='Path to assets from create_assets')
    parser.add_argument('--output-path', default='data/benchmarks',
                       help='Output path for benchmarks')
    parser.add_argument('--split-mapping', default='data/metadata/split_mapping.json',
                       help='Path to split mapping JSON (created if not exists)')

    # Strategy
    parser.add_argument('--strategy', choices=list(STRATEGIES.keys()) + ['all'], default='all',
                       help='Shuffle strategy to use (default: all strategies)')

    # Split configuration
    parser.add_argument('--num-docs-train', type=int, default=800,
                       help='Number of spliced documents for training')
    parser.add_argument('--num-docs-test', type=int, default=500,
                       help='Number of spliced documents for testing')
    parser.add_argument('--num-docs-val', type=int, default=200,
                       help='Number of spliced documents for validation')

    # Strategy parameters
    parser.add_argument('--size', choices=['small', 'large'], default='small',
                       help='Benchmark size: small (5-20 pages) or large (20-500 pages)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set page ranges based on size
    if args.size == 'small':
        min_pages, max_pages = 5, 20
    else:  # large
        min_pages, max_pages = 20, 500

    # Determine which strategies to run
    if args.strategy == 'all':
        strategies_to_run = list(STRATEGIES.keys())
        logger.info(f"Creating benchmarks for all strategies: {strategies_to_run}")
    else:
        strategies_to_run = [args.strategy]
        logger.info(f"Creating benchmark with strategy: {args.strategy}")

    logger.info(f"Size: {args.size} ({min_pages}-{max_pages} pages)")

    # Load assets
    loader = AssetLoader(assets_path=args.assets_path)
    documents_by_type = loader.load_all_documents()

    if not documents_by_type:
        logger.error("No documents loaded. Check assets path.")
        return

    # Create or load split
    split_manager = SplitManager(random_seed=args.random_seed)

    if args.split_mapping and Path(args.split_mapping).exists():
        logger.info(f"Loading existing split from {args.split_mapping}")
        splits = split_manager.load_split(args.split_mapping)
    else:
        logger.info("Creating new split")
        splits = split_manager.create_split(documents_by_type)

        # Save split mapping to metadata folder
        split_path = Path(args.split_mapping)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_manager.save_split(splits, str(split_path))

    # Run for each strategy
    for strategy_name in strategies_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing strategy: {strategy_name}")
        logger.info(f"{'='*60}\n")

        # Initialize strategy
        strategy = get_strategy(
            strategy_name,
            min_pages=min_pages,
            max_pages=max_pages,
            random_seed=args.random_seed
        )

        # Initialize generator and writer
        generator = BenchmarkGenerator(strategy=strategy)
        writer = BenchmarkWriter(
            output_base_path=str(Path(args.output_path) / strategy_name / args.size),
            assets_path=args.assets_path
        )

        # Generate benchmarks for each split
        split_configs = [
            ('train', args.num_docs_train),
            ('test', args.num_docs_test),
            ('validation', args.num_docs_val)
        ]

        for split_name, num_docs in split_configs:
            if num_docs <= 0:
                logger.info(f"Skipping {split_name} (num_docs=0)")
                continue

            logger.info(f"Generating {split_name} benchmark...")

            benchmark_set = generator.generate_for_split(
                documents_by_type=documents_by_type,
                doc_names_for_split=splits[split_name],
                num_spliced_docs=num_docs,
                split_name=split_name,
                benchmark_name=strategy_name
            )

            writer.save_benchmark_set(benchmark_set, split_name)

            logger.info(f"Completed {split_name}: {benchmark_set.statistics}")

    logger.info("\n" + "="*60)
    logger.info("All benchmark creation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
