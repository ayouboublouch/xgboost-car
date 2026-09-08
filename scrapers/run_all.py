#!/usr/bin/env python3
"""
Multi-Source Scraper Orchestrator for Autohouse.ma MLOps Pipeline
-----------------------------------------------------------------
CLI runner supporting:
  python scrapers/run_all.py --source moteur --max-pages 60
  python scrapers/run_all.py --source wandaloo --max-pages 30
  python scrapers/run_all.py --source avito --max-pages 10
  python scrapers/run_all.py --source all --max-pages 30

Enforces:
- Default primary source: 'moteur'
- Validation: DataFrame must have at least 50 rows before writing to disk
- Loud failure: If a source yields 0 rows, raises RuntimeError(f"Scraper for {source} returned 0 rows. Likely blocked by anti-bot.")
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Type

# Ensure repository root and scrapers directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

import pandas as pd

try:
    from scrapers.base import BaseScraper, SCHEMA_FIELDS, consolidate_daily_scrapes, purge_empty_raw_files
except ImportError:
    from base import BaseScraper, SCHEMA_FIELDS, consolidate_daily_scrapes, purge_empty_raw_files

def get_scraper_class(source: str) -> Type[BaseScraper]:
    """Lazy loader for scrapers to ensure smooth CLI operation and resilience."""
    if source == "moteur":
        try:
            from scrapers.moteur_scraper import MoteurScraper
            return MoteurScraper
        except ImportError:
            from moteur_scraper import MoteurScraper
            return MoteurScraper
    elif source == "wandaloo":
        try:
            from scrapers.wandaloo_scraper import WandalooScraper
            return WandalooScraper
        except ImportError:
            from wandaloo_scraper import WandalooScraper
            return WandalooScraper
    elif source == "avito":
        try:
            from scrapers.avito_scraper import AvitoScraper
            return AvitoScraper
        except ImportError:
            from avito_scraper import AvitoScraper
            return AvitoScraper
    elif source == "kifal":
        try:
            from scrapers.kifal_scraper import KifalScraper
            return KifalScraper
        except ImportError:
            from kifal_scraper import KifalScraper
            return KifalScraper
    elif source == "siaracash":
        try:
            from scrapers.siaracash_scraper import SiaraCashScraper
            return SiaraCashScraper
        except ImportError:
            from siaracash_scraper import SiaraCashScraper
            return SiaraCashScraper
    else:
        raise ValueError(f"Unknown scraper source: '{source}'")

AVAILABLE_SOURCES = ["moteur", "wandaloo", "avito", "kifal", "siaracash"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.runner")


def run():
    parser = argparse.ArgumentParser(description="Autohouse.ma Multi-Source Scraper Orchestrator")
    parser.add_argument(
        "--source",
        type=str,
        default="moteur",
        choices=["moteur", "wandaloo", "avito", "kifal", "siaracash", "all"],
        help="Target platform to scrape (default: moteur)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum pages to scrape per provider (default: 30)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help="Minimum validated rows required before disk persistence (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw Parquet & CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    targets = AVAILABLE_SOURCES if args.source == "all" else [args.source]

    logger.info("==========================================================")
    logger.info("Executing Multi-Source Scrapers: %s", targets)
    logger.info("Primary source: %s | Max pages: %d | Min rows: %d", args.source, args.max_pages, args.min_rows)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("==========================================================")

    summary = {}

    for src in targets:
        try:
            cls = get_scraper_class(src)
        except Exception as e:
            logger.warning("Could not load scraper for '%s': %s. Skipping.", src, e)
            continue

        logger.info("\n>>> Starting provider: %s ...", src)
        records_count = 0
        scraper = cls(output_dir=args.output_dir)
        try:
            df = scraper.scrape(max_pages=args.max_pages)
            if df is not None and not df.empty:
                records_count = len(df)
        except Exception as e:
            logger.warning(">>> Provider %s encountered exception: %s", src, e)
            df = None

        if df is None or len(df) == 0:
            logger.warning(">>> Provider %s yielded 0 records. Writing nothing.", src)
            records_count = 0

        summary[src] = records_count
        logger.info(">>> Provider %s finished with %d verified records.", src, records_count)

    logger.info("\n==========================================================")
    logger.info("Scraping Summary:")
    total = 0
    for src, count in summary.items():
        logger.info("  - %-12s: %5d records", src, count)
        total += count
    logger.info("Total harvested across all sources: %d records", total)
    logger.info("==========================================================")

    # Purge empty/dummy files and consolidate daily output into a single file
    purge_empty_raw_files(output_path)
    consolidate_daily_scrapes(output_path)


if __name__ == "__main__":
    run()
