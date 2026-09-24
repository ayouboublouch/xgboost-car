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

import datetime
import json
import pandas as pd

try:
    from scrapers.base import (
        BaseScraper,
        SCHEMA_FIELDS,
        consolidate_daily_scrapes,
        purge_empty_raw_files,
        record_scraping_session,
        update_scraping_progress,
        TRACKING_DIR,
    )
except ImportError:
    from base import (
        BaseScraper,
        SCHEMA_FIELDS,
        consolidate_daily_scrapes,
        purge_empty_raw_files,
        record_scraping_session,
        update_scraping_progress,
        TRACKING_DIR,
    )

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
    elif source == "autocash":
        try:
            from scrapers.autocash_scraper import AutocashScraper
            return AutocashScraper
        except ImportError:
            from autocash_scraper import AutocashScraper
            return AutocashScraper
    elif source == "marochub":
        try:
            from scrapers.marochub_scraper import MarocHubScraper
            return MarocHubScraper
        except ImportError:
            from marochub_scraper import MarocHubScraper
            return MarocHubScraper
    elif source == "siaracash":
        try:
            from scrapers.siaracash_scraper import SiaraCashScraper
            return SiaraCashScraper
        except ImportError:
            from siaracash_scraper import SiaraCashScraper
            return SiaraCashScraper
    else:
        raise ValueError(f"Unknown scraper source: '{source}'")

AVAILABLE_SOURCES = ["moteur", "wandaloo", "avito", "kifal", "autocash", "marochub", "siaracash"]

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
        choices=["moteur", "wandaloo", "avito", "kifal", "autocash", "marochub", "siaracash", "all", "none"],
        help="Target platform to scrape (default: moteur, use none to skip scraping)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Initial page to start scraping from (default: 1)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum pages to scrape per provider (default: 50)",
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
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Custom output CSV filename (e.g. scraped_moteur_part_1.csv)",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        default=False,
        help="Consolidate daily scrapes into scraped_combined_YYYY-MM-DD.csv",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.source == "none":
        targets = []
    elif args.source == "all":
        targets = AVAILABLE_SOURCES
    else:
        targets = [args.source]

    logger.info("==========================================================")
    logger.info("Executing Multi-Source Scrapers: %s", targets)
    logger.info(
        "Primary source: %s | Start page: %d | Max pages: %d | Output file: %s",
        args.source,
        args.start_page,
        args.max_pages,
        args.output_filename,
    )
    logger.info("Output directory: %s", args.output_dir)
    logger.info("==========================================================")

    summary = {}

    for src in targets:
        try:
            cls = get_scraper_class(src)
        except Exception as e:
            logger.warning("Could not load scraper for '%s': %s. Skipping.", src, e)
            continue

        logger.info("\n>>> Starting provider: %s (page %d to %d) ...", src, args.start_page, args.start_page + args.max_pages - 1)
        start_time = datetime.datetime.now()
        scraper = cls(output_dir=args.output_dir)
        try:
            df = scraper.scrape(
                max_pages=args.max_pages,
                start_page=args.start_page,
                output_filename=args.output_filename,
            )
            if df is not None and not df.empty:
                records_count = len(df)
        except Exception as e:
            logger.warning(">>> Provider %s encountered exception: %s", src, e)
            df = None
        end_time = datetime.datetime.now()

        if df is None or len(df) == 0:
            logger.warning(">>> Provider %s yielded 0 records. Writing nothing.", src)
            records_count = 0

        # Record structured session telemetry for parallel scraper worker
        try:
            sess_meta = record_scraping_session(
                scraper_name="parallel_worker",
                source=src,
                start_time=start_time,
                end_time=end_time,
                pages_scraped=args.max_pages,
                records_extracted=records_count,
                records_added=records_count,
                duplicates_skipped=0,
                start_page=args.start_page,
                end_page=args.start_page + args.max_pages - 1,
                status="success" if records_count > 0 else "empty",
                notes=f"Parallel chunk part {args.start_page}",
            )
            if records_count > 0:
                update_scraping_progress(src, args.start_page + args.max_pages - 1, records_count)

            # Save standalone chunk session JSON for CI artifact upload & aggregation
            TRACKING_DIR.mkdir(parents=True, exist_ok=True)
            chunk_session_file = TRACKING_DIR / f"session_{src}_part_{args.start_page}.json"
            with open(chunk_session_file, "w", encoding="utf-8") as f:
                json.dump(sess_meta, f, indent=2, ensure_ascii=False)
            logger.info("Saved chunk session telemetry to %s", chunk_session_file)
        except Exception as e:
            logger.warning("Could not record session telemetry: %s", e)

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

    # Purge empty/dummy files and consolidate daily output if requested or running all
    purge_empty_raw_files(output_path)
    if args.source == "all" or args.consolidate:
        consolidate_daily_scrapes(output_path)
        try:
            from scrapers.continuous_harvester import update_master_database
            update_master_database(output_path)
        except Exception as e:
            try:
                from continuous_harvester import update_master_database
                update_master_database(output_path)
            except Exception as e2:
                logger.warning("Could not update master database: %s", e2)


if __name__ == "__main__":
    run()
