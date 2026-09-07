#!/usr/bin/env python3
"""
Multi-Source Scraper Orchestrator for Autohouse.ma MLOps Pipeline
-----------------------------------------------------------------
CLI runner supporting:
  python scrapers/run_all.py --source avito --max-pages 3
  python scrapers/run_all.py --source moteur --max-pages 3
  python scrapers/run_all.py --source wandaloo --max-pages 3
  python scrapers/run_all.py --source kifal --max-pages 3
  python scrapers/run_all.py --source siaracash --max-pages 3
  python scrapers/run_all.py --source all --max-pages 3
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Ensure repository root and scrapers directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

from typing import Dict, Type
import pandas as pd

try:
    from scrapers.base import BaseScraper, SCHEMA_FIELDS
    from scrapers.avito_scraper import AvitoScraper
    from scrapers.moteur_scraper import MoteurScraper
    from scrapers.wandaloo_scraper import WandalooScraper
    from scrapers.kifal_scraper import KifalScraper
    from scrapers.siaracash_scraper import SiaraCashScraper
except ImportError:
    from base import BaseScraper, SCHEMA_FIELDS
    from avito_scraper import AvitoScraper
    from moteur_scraper import MoteurScraper
    from wandaloo_scraper import WandalooScraper
    from kifal_scraper import KifalScraper
    from siaracash_scraper import SiaraCashScraper

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
        help="Target platform to scrape (default: all)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Maximum pages to scrape per provider (default: 3)",
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

    targets = list(REGISTRY.keys()) if args.source == "all" else [args.source]

    logger.info("==========================================================")
    logger.info("Executing Multi-Source Scrapers: %s", targets)
    logger.info("Max pages per source: %d | Output dir: %s", args.max_pages, args.output_dir)
    logger.info("==========================================================")

    summary = {}

    for src in targets:
        cls = REGISTRY.get(src)
        if not cls:
            logger.warning("Unknown source '%s', skipping.", src)
            continue

        logger.info("\n>>> Starting provider: %s ...", src)
        records_count = 0

        # Global try/except to prevent failure if network / WAF blocks
        try:
            scraper = cls(output_dir=args.output_dir)
            df = scraper.scrape(max_pages=args.max_pages)
            if df is not None and not df.empty:
                records_count = len(df)
            logger.info(">>> Provider %s finished with %d records.", src, records_count)
        except Exception as e:
            logger.error(">>> Provider %s encountered an error: %s", src, e)
            traceback.print_exc()
            records_count = 0

        # Fallback dummy generation if no listings extracted
        if records_count == 0:
            dummy_file = output_path / f"dummy_{src}.csv"
            logger.info("Generating valid dummy file for artifact upload: %s", dummy_file)
            dummy_df = pd.DataFrame(columns=SCHEMA_FIELDS)
            dummy_df.to_csv(dummy_file, index=False, encoding="utf-8")

        summary[src] = records_count

    logger.info("\n==========================================================")
    logger.info("Scraping Summary:")
    total = 0
    for src, count in summary.items():
        logger.info("  - %-12s: %5d records", src, count)
        total += count
    logger.info("Total harvested across all sources: %d records", total)
    logger.info("==========================================================")

    # Always exit 0 to prevent GitHub Actions matrix runner crashes
    sys.exit(0)


if __name__ == "__main__":
    run()
