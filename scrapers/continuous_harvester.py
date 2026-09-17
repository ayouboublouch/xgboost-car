#!/usr/bin/env python3
"""
24/7 Continuous Car Harvester for Autohouse.ma MLOps Pipeline
------------------------------------------------------------
Implements:
- Continuous rolling ingestion over Moteur.ma and Wandaloo.ma.
- Strictly respects --duration-hours with an automatic safety stop:
    stop_time = datetime.now() + timedelta(hours=duration_hours - 0.15)
- Automated deduplication against persistent register: data/tracking/seen_listing_ids.txt
- Checkpoint flushing every 50 new listings to: data/raw/scraped_continuous_{YYYY-MM-DD}.csv
- Sanitizes multiline descriptions (escapes \r and \n into spaces)
- Enforces strict 10-digit phone strings starting with '0'
- Updates and consolidates data/raw/scraped_master_database.parquet with listing_id deduplication
"""

import argparse
import datetime
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

import pandas as pd

try:
    from scrapers.base import (
        SCHEMA_FIELDS,
        TRACKING_DIR,
        SEEN_IDS_FILE,
        load_seen_listing_ids,
        append_seen_listing_ids,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
    )
    from scrapers.moteur_scraper import MoteurScraper
    from scrapers.wandaloo_scraper import WandalooScraper
except ImportError:
    from base import (
        SCHEMA_FIELDS,
        TRACKING_DIR,
        SEEN_IDS_FILE,
        load_seen_listing_ids,
        append_seen_listing_ids,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
    )
    from moteur_scraper import MoteurScraper
    from wandaloo_scraper import WandalooScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("harvester.continuous")


def sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize text fields, enforce 10-digit seller phones, and harmonize brand/model."""
    # 1. Escape multiline descriptions and titles
    if rec.get("description_raw"):
        rec["description_raw"] = (
            str(rec["description_raw"])
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        if rec["description_raw"].lower() in ("nan", "none"):
            rec["description_raw"] = ""

    if rec.get("title_raw"):
        rec["title_raw"] = (
            str(rec["title_raw"])
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        if rec["title_raw"].lower() in ("nan", "none"):
            rec["title_raw"] = ""

    # 2. Strict 10-digit seller phone formatting
    if rec.get("seller_phone"):
        p = str(rec["seller_phone"]).replace(".0", "").strip()
        if p and p.lower() not in ("nan", "none", "<na>"):
            p = p.zfill(10)
            if len(p) == 10 and p.startswith("0") and p[1] in "567":
                rec["seller_phone"] = p
                if not rec.get("seller_phone_hash"):
                    rec["seller_phone_hash"] = hash_phone(p)
            else:
                rec["seller_phone"] = None
        else:
            rec["seller_phone"] = None

    # 3. Clean brand & model
    b, m, tr = clean_brand_and_model(
        rec.get("brand"),
        rec.get("model"),
        title_raw=str(rec.get("title_raw") or ""),
        trim=str(rec.get("trim") or ""),
    )
    rec["brand"] = b
    rec["model"] = m
    rec["trim"] = tr

    return rec


def flush_checkpoint(
    records: List[Dict[str, Any]], target_csv: Path, seen_ids: Set[str]
) -> int:
    """Flush accumulated in-memory records to checkpoint CSV and append new IDs to persistent register."""
    if not records:
        return 0

    sanitized = [sanitize_record(r) for r in records]
    df_new = pd.DataFrame(sanitized)

    for col in SCHEMA_FIELDS:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[SCHEMA_FIELDS].copy()

    # Format string columns
    for col in ["description_raw", "title_raw"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].fillna("").astype(str).apply(
                lambda s: s.replace("\r", " ").replace("\n", " ").strip() if s.lower() not in ("nan", "none") else ""
            )

    if "seller_phone" in df_new.columns:
        def _fmt_phone(p):
            if pd.isna(p) or p is None:
                return None
            s = str(p).replace(".0", "").strip()
            if not s or s.lower() in ("nan", "none", "<na>"):
                return None
            s = s.zfill(10)
            return s if len(s) == 10 and s.startswith("0") and s[1] in "567" else None
        df_new["seller_phone"] = df_new["seller_phone"].apply(_fmt_phone)

    target_csv.parent.mkdir(parents=True, exist_ok=True)

    if target_csv.exists():
        try:
            df_existing = pd.read_csv(target_csv, low_memory=False, dtype={"seller_phone": str, "listing_id": str})
            combined = pd.concat([df_existing, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["listing_id"], keep="last")
        except Exception as e:
            logger.warning("Could not read existing checkpoint %s: %s", target_csv.name, e)
            combined = df_new
    else:
        combined = df_new

    # Enforce exact column order
    combined = combined[SCHEMA_FIELDS]
    combined.to_csv(target_csv, index=False, encoding="utf-8")

    # Update persistent seen register on disk
    new_lids = [str(r.get("listing_id")).strip() for r in records if r.get("listing_id")]
    append_seen_listing_ids(new_lids)
    seen_ids.update(new_lids)

    logger.info(
        "Checkpoint flushed: +%d new records written to %s (Total batch file: %d rows | Total seen IDs: %d)",
        len(records),
        target_csv.name,
        len(combined),
        len(seen_ids),
    )
    return len(records)


def update_master_database(raw_dir: Path) -> Path:
    """
    Merge all raw scraped files (scraped_combined_*.csv, scraped_continuous_*.csv)
    into data/raw/scraped_master_database.parquet with listing_id deduplication.
    """
    raw_dir = Path(raw_dir)
    master_parquet = raw_dir / "scraped_master_database.parquet"
    frames = []

    # Read existing master if available
    if master_parquet.exists():
        try:
            df_m = pd.read_parquet(master_parquet)
            if not df_m.empty and len(df_m) > 0:
                frames.append(df_m)
                logger.info("Loaded %d records from existing master parquet", len(df_m))
        except Exception as e:
            logger.warning("Could not read existing master parquet: %s", e)

    # Load all raw CSV batches
    for f in sorted(raw_dir.glob("scraped_*.csv")):
        try:
            df_c = pd.read_csv(f, low_memory=False, dtype={"seller_phone": str, "listing_id": str})
            if not df_c.empty and len(df_c) > 0:
                frames.append(df_c)
        except Exception as e:
            logger.warning("Could not read %s: %s", f.name, e)

    if not frames:
        logger.warning("No scraped data found to build master database.")
        return master_parquet

    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate strictly on listing_id, keeping the latest record
    if "listing_id" in merged.columns:
        merged["listing_id"] = merged["listing_id"].astype(str).str.strip()
        merged = merged.drop_duplicates(subset=["listing_id"], keep="last")
    else:
        merged = merged.drop_duplicates()

    # Enforce schema fields
    for col in SCHEMA_FIELDS:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[SCHEMA_FIELDS].copy()

    # Sanitize multiline strings and enforce string types for Parquet serialization
    for col in merged.columns:
        if merged[col].dtype == "object":
            merged[col] = merged[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" and str(x).lower() not in ("nan", "none", "<na>") else None
            )

    try:
        merged.to_parquet(master_parquet, index=False, engine="pyarrow")
        logger.info(
            "Master database updated successfully: %s (%d unique listings)",
            master_parquet.name,
            len(merged),
        )
    except Exception as e:
        logger.error("Failed to write master database parquet: %s", e)

    return master_parquet


def run_continuous_harvest(
    duration_hours: float = 5.0,
    output_dir: str = "data/raw",
    batch_size: int = 50,
) -> None:
    """Run continuous harvesting loop until stop_time is reached."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()
    # Safety cutoff: stop 0.15 hours (~9 minutes) before duration expires to guarantee clean flush & commit
    safety_margin = 0.15 if duration_hours >= 0.5 else (duration_hours * 0.1)
    stop_time = start_time + datetime.timedelta(hours=duration_hours - safety_margin)

    logger.info("==================================================================")
    logger.info("Starting 24/7 Continuous Car Harvester")
    logger.info("Requested duration: %.2f hours | Safety cutoff: %.2f hours", duration_hours, safety_margin)
    logger.info("Scheduled stop time: %s", stop_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Batch checkpoint interval: every %d new records", batch_size)
    logger.info("Output directory: %s", output_path)
    logger.info("==================================================================")

    # Initialize persistent seen register
    seen_ids: Set[str] = load_seen_listing_ids()
    logger.info("Loaded %d previously seen listing IDs from persistent register.", len(seen_ids))

    # Initialize scraper engines
    moteur = MoteurScraper(output_dir=str(output_path))
    wandaloo = WandalooScraper(output_dir=str(output_path))

    # Sync scrapers seen_ids
    moteur.seen_ids = seen_ids
    wandaloo.seen_ids = seen_ids

    moteur_page = 1
    wandaloo_page = 1
    consecutive_empty_moteur = 0
    consecutive_empty_wandaloo = 0

    total_harvested = 0
    batch_buffer: List[Dict[str, Any]] = []

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    continuous_csv = output_path / f"scraped_continuous_{today_str}.csv"

    iteration = 0
    while datetime.datetime.now() < stop_time:
        iteration += 1
        now = datetime.datetime.now()
        remaining_secs = (stop_time - now).total_seconds()
        logger.info(
            "\n--- [Iter %d | Remaining: %.1f min] Moteur page %d | Wandaloo page %d ---",
            iteration,
            remaining_secs / 60.0,
            moteur_page,
            wandaloo_page,
        )

        # 1. Harvest from Moteur.ma
        try:
            moteur_batch = moteur.scrape_page(moteur_page)
            new_moteur = [r for r in moteur_batch if r.get("listing_id") not in seen_ids]
            if new_moteur:
                consecutive_empty_moteur = 0
                for r in new_moteur:
                    lid = str(r.get("listing_id")).strip()
                    seen_ids.add(lid)
                    batch_buffer.append(r)
                logger.info("[Moteur] Harvested %d fresh listings from page %d", len(new_moteur), moteur_page)
            else:
                consecutive_empty_moteur += 1
                logger.debug("[Moteur] 0 fresh listings on page %d (all seen or empty)", moteur_page)

            # Advance or wrap pagination
            moteur_page += 1
            if consecutive_empty_moteur >= 5 or moteur_page > 150:
                logger.info("[Moteur] Resetting pagination to page 1 for incoming ads.")
                moteur_page = 1
                consecutive_empty_moteur = 0

        except Exception as e:
            logger.warning("[Moteur] Error scraping page %d: %s", moteur_page, e)
            moteur_page += 1

        # Checkpoint if batch buffer reached target size
        if len(batch_buffer) >= batch_size:
            flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
            total_harvested += flushed
            batch_buffer.clear()

        # Check time before next source
        if datetime.datetime.now() >= stop_time:
            logger.info("Safety stop time reached during Moteur scrape. Exiting loop.")
            break

        time.sleep(random.uniform(1.0, 2.5))

        # 2. Harvest from Wandaloo.ma
        try:
            wandaloo_batch = wandaloo.scrape_page(wandaloo_page)
            new_wandaloo = [r for r in wandaloo_batch if r.get("listing_id") not in seen_ids]
            if new_wandaloo:
                consecutive_empty_wandaloo = 0
                for r in new_wandaloo:
                    lid = str(r.get("listing_id")).strip()
                    seen_ids.add(lid)
                    batch_buffer.append(r)
                logger.info("[Wandaloo] Harvested %d fresh listings from page %d", len(new_wandaloo), wandaloo_page)
            else:
                consecutive_empty_wandaloo += 1
                logger.debug("[Wandaloo] 0 fresh listings on page %d (all seen or empty)", wandaloo_page)

            # Advance or wrap pagination
            wandaloo_page += 1
            if consecutive_empty_wandaloo >= 5 or wandaloo_page > 80:
                logger.info("[Wandaloo] Resetting pagination to page 1 for incoming ads.")
                wandaloo_page = 1
                consecutive_empty_wandaloo = 0

        except Exception as e:
            logger.warning("[Wandaloo] Error scraping page %d: %s", wandaloo_page, e)
            wandaloo_page += 1

        # Checkpoint if batch buffer reached target size
        if len(batch_buffer) >= batch_size:
            flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
            total_harvested += flushed
            batch_buffer.clear()

        time.sleep(random.uniform(1.0, 2.5))

    # Final flush of any remaining in-memory records
    if batch_buffer:
        logger.info("Performing final flush of %d buffered listings ...", len(batch_buffer))
        flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
        total_harvested += flushed
        batch_buffer.clear()

    # Update master database parquet
    logger.info("Updating consolidated master parquet database ...")
    update_master_database(output_path)

    elapsed = (datetime.datetime.now() - start_time).total_seconds() / 3600.0
    logger.info("==================================================================")
    logger.info("Continuous harvester finished in %.2f hours.", elapsed)
    logger.info("Total fresh listings ingested in this session: %d", total_harvested)
    logger.info("Total persistent seen IDs in register: %d", len(seen_ids))
    logger.info("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Autohouse.ma 24/7 Continuous Car Harvester")
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=5.0,
        help="Duration in hours to continuously harvest (default: 5.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for raw continuous scrapes (default: data/raw)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of records before flushing checkpoint (default: 50)",
    )
    args = parser.parse_args()

    run_continuous_harvest(
        duration_hours=args.duration_hours,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
