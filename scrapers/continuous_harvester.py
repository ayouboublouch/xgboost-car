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
        PROGRESS_FILE,
        load_scraping_progress,
        update_scraping_progress,
        record_scraping_session,
        load_seen_listing_ids,
        append_seen_listing_ids,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
    )
    from scrapers.moteur_scraper import MoteurScraper
    from scrapers.wandaloo_scraper import WandalooScraper
    from scrapers.avito_scraper import AvitoScraper
except ImportError:
    from base import (
        SCHEMA_FIELDS,
        TRACKING_DIR,
        SEEN_IDS_FILE,
        PROGRESS_FILE,
        load_scraping_progress,
        update_scraping_progress,
        record_scraping_session,
        load_seen_listing_ids,
        append_seen_listing_ids,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
    )
    from moteur_scraper import MoteurScraper
    from wandaloo_scraper import WandalooScraper
    from avito_scraper import AvitoScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("harvester.continuous")

CURSOR_MOTEUR_FILE = TRACKING_DIR / "last_page_moteur.txt"
CURSOR_WANDALOO_FILE = TRACKING_DIR / "last_page_wandaloo.txt"
CURSOR_AVITO_FILE = TRACKING_DIR / "last_page_avito.txt"


def load_page_cursor(filepath: Path, default: int = 1) -> int:
    """Read last scraped page index from persistent cursor file. Defaults to 1 if missing or invalid."""
    if filepath.exists():
        try:
            val = filepath.read_text(encoding="utf-8").strip()
            page = int(val)
            if page >= 1:
                return page
        except Exception as e:
            logger.warning("Could not read cursor file %s: %s. Using default %d", filepath.name, e, default)
    return default


def save_page_cursor(filepath: Path, page: int) -> None:
    """Save current page index to persistent cursor file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(str(page), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save cursor to %s: %s", filepath.name, e)


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
    Merge all raw scraped files (scraped_combined_*.csv, scraped_continuous_*.csv, avito_local_*.csv)
    into data/raw/scraped_master_database.parquet with listing_id deduplication.
    Excludes scraped_master_database.parquet itself to avoid recursive empty self-reads.
    """
    raw_dir = Path(raw_dir)
    master_parquet = raw_dir / "scraped_master_database.parquet"
    dfs = []

    # Match all scraped_combined_*.csv, scraped_continuous_*.csv, scraped_*_part_*.csv, and avito/moteur/wandaloo files
    candidate_files = sorted(list(set(
        list(raw_dir.glob("scraped_combined_*.csv"))
        + list(raw_dir.glob("scraped_continuous_*.csv"))
        + list(raw_dir.glob("scraped_*_part_*.csv"))
        + list(raw_dir.glob("avito_local_*.csv"))
        + list(raw_dir.glob("moteur_*.csv"))
        + list(raw_dir.glob("wandaloo_*.csv"))
        + list(raw_dir.glob("avito_*.csv"))
    )))

    for f in candidate_files:
        try:
            if f.stat().st_size <= 500:
                logger.info("Skipping small/empty raw file: %s", f.name)
                continue
            df_c = pd.read_csv(f, low_memory=False, on_bad_lines="skip", dtype={"listing_id": str})
            if not df_c.empty and len(df_c) > 0:
                dfs.append(df_c)
                logger.info("Loaded %d records from %s", len(df_c), f.name)
        except Exception as e:
            logger.warning("Could not read %s: %s", f.name, e)

    if not dfs:
        logger.warning("No scraped data found to build master database.")
        return master_parquet

    master_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate strictly on listing_id, keeping the latest record
    if "listing_id" in master_df.columns:
        master_df["listing_id"] = master_df["listing_id"].astype(str).str.strip()
        master_df = master_df.drop_duplicates(subset=["listing_id"], keep="last")
    else:
        master_df = master_df.drop_duplicates()

    # Assert non-empty before saving
    assert len(master_df) > 0, "Master database DataFrame is empty after deduplication"

    # Enforce schema fields
    for col in SCHEMA_FIELDS:
        if col not in master_df.columns:
            master_df[col] = None
    master_df = master_df[SCHEMA_FIELDS].copy()

    # Sanitize multiline strings and enforce string types for Parquet serialization
    for col in ["description_raw", "title_raw", "model"]:
        if col in master_df.columns:
            master_df[col] = (
                master_df[col]
                .astype(str)
                .str.replace(r"[\r\n\t]+", " ", regex=True)
                .str.strip()
            )
            master_df.loc[master_df[col].isin(["nan", "None", "", "<NA>"]), col] = None

    for col in master_df.columns:
        if master_df[col].dtype == "object":
            master_df[col] = master_df[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" and str(x).lower() not in ("nan", "none", "<na>") else None
            )

    try:
        master_df.to_parquet(master_parquet, index=False, engine="pyarrow", compression="snappy")
        logger.info(
            "Master database updated successfully: %s (%d unique listings)",
            master_parquet.name,
            len(master_df),
        )
    except Exception as e:
        logger.error("Failed to write master database parquet: %s", e)

    return master_parquet


def run_continuous_harvest(
    duration_hours: float = 3.5,
    output_dir: str = "data/raw",
    batch_size: int = 50,
) -> None:
    """Run continuous harvesting loop until stop_time is reached."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()
    # 20-minute safety buffer (3.2h when duration is 3.5h) to guarantee clean flush & commit before runner SIGTERM
    if duration_hours >= 3.5:
        stop_time = start_time + datetime.timedelta(hours=3.2)
    elif duration_hours > 0.5:
        stop_time = start_time + datetime.timedelta(hours=duration_hours - (20.0 / 60.0))
    else:
        stop_time = start_time + datetime.timedelta(hours=duration_hours * 0.85)

    safety_buffer_mins = (start_time + datetime.timedelta(hours=duration_hours) - stop_time).total_seconds() / 60.0

    logger.info("==================================================================")
    logger.info("Starting 24/7 Continuous Car Harvester")
    logger.info("Requested duration: %.2f hours | Safety buffer: %.1f minutes", duration_hours, safety_buffer_mins)
    logger.info("Scheduled safety stop time: %s", stop_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Batch checkpoint interval: every %d new records", batch_size)
    logger.info("Output directory: %s", output_path)
    logger.info("==================================================================")

    # Initialize persistent seen register
    seen_ids: Set[str] = load_seen_listing_ids()
    logger.info("Loaded %d previously seen listing IDs from persistent register.", len(seen_ids))

    # Initialize persistent page cursors
    moteur_page = load_page_cursor(CURSOR_MOTEUR_FILE, default=1)
    wandaloo_page = load_page_cursor(CURSOR_WANDALOO_FILE, default=1)
    avito_page = load_page_cursor(CURSOR_AVITO_FILE, default=1)
    logger.info("Resuming Moteur.ma pagination from cursor: page %d", moteur_page)
    logger.info("Resuming Wandaloo.ma pagination from cursor: page %d", wandaloo_page)
    logger.info("Resuming Avito.ma pagination from cursor: page %d", avito_page)

    moteur_start_page = moteur_page
    wandaloo_start_page = wandaloo_page
    avito_start_page = avito_page

    # Initialize scraper engines
    moteur = MoteurScraper(output_dir=str(output_path))
    wandaloo = WandalooScraper(output_dir=str(output_path))
    avito = AvitoScraper(output_dir=str(output_path))

    # Sync scrapers seen_ids
    moteur.seen_ids = seen_ids
    wandaloo.seen_ids = seen_ids
    avito.seen_ids = seen_ids

    total_harvested = 0
    batch_buffer: List[Dict[str, Any]] = []

    moteur_added = 0
    wandaloo_added = 0
    avito_added = 0

    moteur_extracted = 0
    wandaloo_extracted = 0
    avito_extracted = 0

    moteur_pages = 0
    wandaloo_pages = 0
    avito_pages = 0

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    continuous_csv = output_path / f"scraped_continuous_{today_str}.csv"

    iteration = 0
    while datetime.datetime.now() < stop_time:
        iteration += 1
        now = datetime.datetime.now()
        remaining_secs = (stop_time - now).total_seconds()
        logger.info(
            "\n--- [Iter %d | Remaining: %.1f min] Moteur p.%d | Wandaloo p.%d | Avito p.%d ---",
            iteration,
            remaining_secs / 60.0,
            moteur_page,
            wandaloo_page,
            avito_page,
        )

        # 1. Harvest from Moteur.ma
        try:
            moteur_batch = moteur.scrape_page(moteur_page)
            moteur_pages += 1
            if not moteur_batch:
                logger.info("[Moteur] Page %d returned 0 listings (hit end of catalog or empty). Wrapping back to page 1.", moteur_page)
                moteur_page = 1
                save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)
            else:
                moteur_extracted += len(moteur_batch)
                new_moteur = [r for r in moteur_batch if r.get("listing_id") not in seen_ids]
                if new_moteur:
                    for r in new_moteur:
                        lid = str(r.get("listing_id")).strip()
                        seen_ids.add(lid)
                        batch_buffer.append(r)
                    moteur_added += len(new_moteur)
                    logger.info("[Moteur] Harvested %d fresh listings from page %d (Buffer: %d)", len(new_moteur), moteur_page, len(batch_buffer))
                    update_scraping_progress("moteur", moteur_page, len(new_moteur))
                else:
                    logger.info("[Moteur] Page %d had %d listings, all already seen. Advancing deeper into catalog...", moteur_page, len(moteur_batch))
                    update_scraping_progress("moteur", moteur_page, 0)

                # Advance cursor deeper into historical listings
                moteur_page += 1
                if moteur_page > 400:
                    logger.info("[Moteur] Reached deep catalog limit (page %d). Wrapping back to page 1.", moteur_page)
                    moteur_page = 1
                save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)

        except Exception as e:
            logger.warning("[Moteur] Error scraping page %d: %s", moteur_page, e)
            moteur_page += 1
            save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)

        # Checkpoint if batch buffer reached target size
        if len(batch_buffer) >= batch_size:
            flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
            total_harvested += flushed
            batch_buffer.clear()
            save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)
            save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)
            save_page_cursor(CURSOR_AVITO_FILE, avito_page)

        # Check time before next source
        if datetime.datetime.now() >= stop_time:
            logger.info("Safety stop time reached during Moteur scrape. Exiting loop.")
            break

        time.sleep(random.uniform(1.0, 2.5))

        # 2. Harvest from Wandaloo.ma
        try:
            wandaloo_batch = wandaloo.scrape_page(wandaloo_page)
            wandaloo_pages += 1
            if not wandaloo_batch:
                logger.info("[Wandaloo] Page %d returned 0 listings (hit end of catalog or empty). Wrapping back to page 1.", wandaloo_page)
                wandaloo_page = 1
                save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)
            else:
                wandaloo_extracted += len(wandaloo_batch)
                new_wandaloo = [r for r in wandaloo_batch if r.get("listing_id") not in seen_ids]
                if new_wandaloo:
                    for r in new_wandaloo:
                        lid = str(r.get("listing_id")).strip()
                        seen_ids.add(lid)
                        batch_buffer.append(r)
                    wandaloo_added += len(new_wandaloo)
                    logger.info("[Wandaloo] Harvested %d fresh listings from page %d (Buffer: %d)", len(new_wandaloo), wandaloo_page, len(batch_buffer))
                    update_scraping_progress("wandaloo", wandaloo_page, len(new_wandaloo))
                else:
                    logger.info("[Wandaloo] Page %d had %d listings, all already seen. Advancing deeper into catalog...", wandaloo_page, len(wandaloo_batch))
                    update_scraping_progress("wandaloo", wandaloo_page, 0)

                # Advance cursor deeper into historical listings
                wandaloo_page += 1
                if wandaloo_page > 200:
                    logger.info("[Wandaloo] Reached deep catalog limit (page %d). Wrapping back to page 1.", wandaloo_page)
                    wandaloo_page = 1
                save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)

        except Exception as e:
            logger.warning("[Wandaloo] Error scraping page %d: %s", wandaloo_page, e)
            wandaloo_page += 1
            save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)

        # Checkpoint if batch buffer reached target size
        if len(batch_buffer) >= batch_size:
            flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
            total_harvested += flushed
            batch_buffer.clear()
            save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)
            save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)
            save_page_cursor(CURSOR_AVITO_FILE, avito_page)

        # Check time before next source
        if datetime.datetime.now() >= stop_time:
            logger.info("Safety stop time reached during Wandaloo scrape. Exiting loop.")
            break

        time.sleep(random.uniform(1.0, 2.5))

        # 3. Harvest from Avito.ma
        try:
            avito_batch = avito.scrape_page(avito_page)
            avito_pages += 1
            if not avito_batch:
                logger.info("[Avito] Page %d returned 0 listings (empty or challenge). Wrapping back to page 1.", avito_page)
                avito_page = 1
                save_page_cursor(CURSOR_AVITO_FILE, avito_page)
            else:
                avito_extracted += len(avito_batch)
                new_avito = [r for r in avito_batch if r.get("listing_id") not in seen_ids]
                if new_avito:
                    for r in new_avito:
                        lid = str(r.get("listing_id")).strip()
                        seen_ids.add(lid)
                        batch_buffer.append(r)
                    avito_added += len(new_avito)
                    logger.info("[Avito] Harvested %d fresh listings from page %d (Buffer: %d)", len(new_avito), avito_page, len(batch_buffer))
                    update_scraping_progress("avito", avito_page, len(new_avito))
                else:
                    logger.info("[Avito] Page %d had %d listings, all already seen. Advancing...", avito_page, len(avito_batch))
                    update_scraping_progress("avito", avito_page, 0)

                # Advance cursor deeper into catalog
                avito_page += 1
                if avito_page > 300:
                    logger.info("[Avito] Reached deep catalog limit (page %d). Wrapping back to page 1.", avito_page)
                    avito_page = 1
                save_page_cursor(CURSOR_AVITO_FILE, avito_page)

        except Exception as e:
            logger.warning("[Avito] Error scraping page %d: %s", avito_page, e)
            avito_page += 1
            save_page_cursor(CURSOR_AVITO_FILE, avito_page)

        # Checkpoint if batch buffer reached target size
        if len(batch_buffer) >= batch_size:
            flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
            total_harvested += flushed
            batch_buffer.clear()
            save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)
            save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)
            save_page_cursor(CURSOR_AVITO_FILE, avito_page)

        time.sleep(random.uniform(1.0, 2.5))

    # Final flush of any remaining in-memory records
    if batch_buffer:
        logger.info("Performing final flush of %d buffered listings ...", len(batch_buffer))
        flushed = flush_checkpoint(batch_buffer, continuous_csv, seen_ids)
        total_harvested += flushed
        batch_buffer.clear()

    # Save final page cursors to disk
    save_page_cursor(CURSOR_MOTEUR_FILE, moteur_page)
    save_page_cursor(CURSOR_WANDALOO_FILE, wandaloo_page)
    save_page_cursor(CURSOR_AVITO_FILE, avito_page)

    # Record detailed scraping session telemetry
    session_end = datetime.datetime.now()
    if moteur_pages > 0:
        record_scraping_session(
            scraper_name="continuous_harvester",
            source="moteur",
            start_time=start_time,
            end_time=session_end,
            pages_scraped=moteur_pages,
            records_extracted=moteur_extracted,
            records_added=moteur_added,
            duplicates_skipped=max(0, moteur_extracted - moteur_added),
            start_page=moteur_start_page,
            end_page=moteur_page,
            status="success",
        )
    if wandaloo_pages > 0:
        record_scraping_session(
            scraper_name="continuous_harvester",
            source="wandaloo",
            start_time=start_time,
            end_time=session_end,
            pages_scraped=wandaloo_pages,
            records_extracted=wandaloo_extracted,
            records_added=wandaloo_added,
            duplicates_skipped=max(0, wandaloo_extracted - wandaloo_added),
            start_page=wandaloo_start_page,
            end_page=wandaloo_page,
            status="success",
        )
    if avito_pages > 0:
        record_scraping_session(
            scraper_name="continuous_harvester",
            source="avito",
            start_time=start_time,
            end_time=session_end,
            pages_scraped=avito_pages,
            records_extracted=avito_extracted,
            records_added=avito_added,
            duplicates_skipped=max(0, avito_extracted - avito_added),
            start_page=avito_start_page,
            end_page=avito_page,
            status="success",
        )

    # Update master database parquet
    logger.info("Updating consolidated master parquet database ...")
    update_master_database(output_path)

    elapsed = (datetime.datetime.now() - start_time).total_seconds() / 3600.0
    logger.info("==================================================================")
    logger.info("Continuous harvester finished in %.2f hours.", elapsed)
    logger.info("Total fresh listings ingested in this session: %d", total_harvested)
    logger.info("Total persistent seen IDs in register: %d", len(seen_ids))
    logger.info("Current persistent cursors: Moteur=%d | Wandaloo=%d | Avito=%d", moteur_page, wandaloo_page, avito_page)
    logger.info("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Autohouse.ma 24/7 Continuous Car Harvester")
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=3.5,
        help="Duration in hours to continuously harvest (default: 3.5)",
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
