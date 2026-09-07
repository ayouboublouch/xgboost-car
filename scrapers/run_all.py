#!/usr/bin/env python3
"""
Multi-Source Scraper Orchestrator for Autohouse.ma MLOps Pipeline
-----------------------------------------------------------------
CLI runner supporting:
  python scrapers/run_all.py --source all --max-pages 10
  python scrapers/run_all.py --source avito --max-pages 15
  python scrapers/run_all.py --source moteur --max-pages 15
  python scrapers/run_all.py --source wandaloo --max-pages 15
  python scrapers/run_all.py --source kifal --max-pages 15
  python scrapers/run_all.py --source siaracash --max-pages 15
"""

import argparse
import logging
import sys
from typing import Dict, Type

from scrapers.base import BaseScraper
from scrapers.avito_scraper import AvitoScraper
from scrapers.moteur_scraper import MoteurScraper
from scrapers.wandaloo_scraper import WandalooScraper
from scrapers.kifal_scraper import KifalScraper
from scrapers.siaracash_scraper import SiaraCashScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.runner")

REGISTRY: Dict[str, Type[BaseScraper]] = {
    "avito": AvitoScraper,
    "moteur": MoteurScraper,
    "wandaloo": WandalooScraper,
    "kifal": KifalScraper,
    "siaracash": SiaraCashScraper,
}


def run():
    parser = argparse.ArgumentParser(description="Autohouse.ma Multi-Source Scraper Orchestrator")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "avito", "moteur", "wandaloo", "kifal", "siaracash"],
        help="Target platform to scrape or 'all' (default: all)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages to scrape per provider (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw Parquet & CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    targets = list(REGISTRY.keys()) if args.source == "all" else [args.source]

    logger.info("==========================================================")
    logger.info("Executing Multi-Source Scrapers: %s", targets)
    logger.info("Max pages per source: %d | Output dir: %s", args.max_pages, args.output_dir)
    logger.info("==========================================================")

    summary = {}

    for src in targets:
        cls = REGISTRY[src]
        logger.info("\n>>> Starting provider: %s ...", src)
        try:
            scraper = cls(output_dir=args.output_dir)
            df = scraper.scrape(max_pages=args.max_pages)
            summary[src] = len(df)
            logger.info(">>> Provider %s completed with %d records.", src, len(df))
        except Exception as e:
            logger.error(">>> Provider %s failed with error: %s", src, e, exc_info=True)
            summary[src] = 0

    logger.info("\n==========================================================")
    logger.info("Scraping Summary:")
    total = 0
    for src, count in summary.items():
        logger.info("  - %-12s: %5d records", src, count)
        total += count
    logger.info("Total harvested across all sources: %d records", total)
    logger.info("==========================================================")


if __name__ == "__main__":
    run()
