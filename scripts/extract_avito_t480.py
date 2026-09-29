#!/usr/bin/env python3
"""
Extract 25K+ Avito Car Listings from T480 Repository Cache (pages.tar.gz & details.tar.gz)
Aligns 100% with the 24 standardized fields defined in the Autohouse.ma Cahier des Charges.
Saves to:
  - data/raw/avito_t480_complete.parquet
  - data/raw/avito_t480_complete.csv
Then triggers update_master_database to consolidate into scraped_master_database.parquet.
"""

import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scrapers.base import (
    SCHEMA_FIELDS,
    clean_brand_and_model,
    extract_moroccan_phone,
    hash_phone,
    infer_moroccan_region,
)
from scrapers.continuous_harvester import update_master_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("avito.t480_extractor")

PAGES_DIR = ROOT_DIR / "scratch" / "t480" / "pages"
DETAILS_DIR = ROOT_DIR / "scratch" / "t480" / "details"
OUTPUT_DIR = ROOT_DIR / "data" / "raw"


def normalize_fuel(fuel_raw: Optional[str]) -> str:
    if not fuel_raw:
        return ""
    f_low = str(fuel_raw).lower()
    if "diesel" in f_low:
        return "Diesel"
    if "essence" in f_low:
        return "Essence"
    if "hybride" in f_low:
        return "Hybride"
    if "electrique" in f_low or "électrique" in f_low:
        return "Electrique"
    return str(fuel_raw).strip().title()


def normalize_gearbox(gb_raw: Optional[str]) -> str:
    if not gb_raw:
        return ""
    gb_low = str(gb_raw).lower()
    if "auto" in gb_low:
        return "Automatique"
    if "man" in gb_low:
        return "Manuelle"
    return str(gb_raw).strip().title()


def normalize_customs(origin_raw: Optional[str]) -> str:
    if not origin_raw:
        return "Dédouanée"
    o_low = str(origin_raw).lower()
    if "ww" in o_low or "maroc" in o_low:
        return "WW au Maroc"
    if "pas" in o_low and "dédouanée" in o_low:
        return "Pas encore dédouanée"
    if "dédouan" in o_low or "dedouan" in o_low:
        return "Dédouanée"
    if "import" in o_low:
        return "Importée neuve"
    return str(origin_raw).strip()


def normalize_condition(cond_raw: Optional[str]) -> str:
    if not cond_raw:
        return "Occasion"
    c_low = str(cond_raw).lower()
    if "excellent" in c_low:
        return "Excellent"
    if "très bon" in c_low or "tres bon" in c_low:
        return "Très bon"
    if "bon" in c_low:
        return "Bon"
    if "correct" in c_low:
        return "Correct"
    if "pièce" in c_low or "piece" in c_low or "endomm" in c_low:
        return "Pour pièces"
    if "neuf" in c_low:
        return "Neuf"
    return str(cond_raw).strip().title()


def extract_record(ad: Dict[str, Any], detail: Dict[str, Any], date_fallback: str) -> Optional[Dict[str, Any]]:
    """Transform search ad + detail page into a unified Cahier des Charges record."""
    cat = str(ad.get("category") or "")
    # Restrict to car listings
    if cat and not cat.startswith("Voitures"):
        return None

    lid = str(ad.get("list_id") or ad.get("id") or "").strip()
    if not lid:
        return None

    det_fields = detail.get("details", {}) if isinstance(detail, dict) else {}

    # Title & Description
    title_raw = str(ad.get("title") or "").strip()
    desc_raw = str(ad.get("description") or "").replace("\r", " ").replace("\n", " ").strip()

    # Brand, Model, Trim resolution
    brand = det_fields.get("brand") or ""
    model = det_fields.get("model") or ""
    brand_c, model_c, trim_c = clean_brand_and_model(brand, model, title_raw=title_raw)

    # Year
    year = ad.get("year")
    if year is None:
        y_match = re.search(r"\b(19[89]\d|20[0-2]\d)\b", title_raw)
        year = int(y_match.group(1)) if y_match else None
    else:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None

    # Mileage
    km = ad.get("mileage_km")
    try:
        mileage_km = float(km) if km is not None else None
    except (ValueError, TypeError):
        mileage_km = None

    # Price
    price_val = detail.get("detail_price") if detail else None
    if price_val is None:
        price_val = ad.get("price")
    try:
        price_mad = float(price_val) if price_val is not None and float(price_val) > 0 else None
    except (ValueError, TypeError):
        price_mad = None

    # Fuel & Gearbox
    fuel_type = normalize_fuel(ad.get("fuel") or det_fields.get("fuel"))
    transmission = normalize_gearbox(ad.get("gearbox") or det_fields.get("bv"))

    # Fiscal Power
    pf = det_fields.get("pfiscale")
    fiscal_power_cv = str(pf).strip() if pf else ""

    # Doors Count
    doors_val = det_fields.get("doors")
    doors_count = None
    if doors_val:
        d_m = re.search(r"\b(\d)\b", str(doors_val))
        if d_m:
            doors_count = float(d_m.group(1))

    # Customs & Condition
    customs_status = normalize_customs(det_fields.get("v_origin"))
    condition = normalize_condition(det_fields.get("auto_condition"))

    # First Owner / Owners Count
    fo = str(det_fields.get("first_owner") or "").lower()
    owners_count = "1" if "oui" in fo else ""

    # Seller Type
    is_pro = ad.get("is_professional") or ad.get("seller_type") in ("PRO", "COMMERCIAL")
    seller_type = "Professionnel" if is_pro else "Particulier"

    # Seller Phone & Hash
    raw_phone = detail.get("phone") or ad.get("seller_phone")
    seller_phone = extract_moroccan_phone(raw_phone)
    seller_phone_hash = hash_phone(seller_phone) if seller_phone else None

    # City & Region
    city = str(ad.get("location") or "Casablanca").strip().title()
    region = infer_moroccan_region(city)

    # Date Posted & Date Scraped
    exact_date = detail.get("date_posted_exact") if detail else None
    if exact_date and len(str(exact_date)) >= 10:
        date_posted = str(exact_date)[:10]
    else:
        date_posted = date_fallback[:10]

    date_scraped = date_fallback

    # Photos count
    img_cnt = ad.get("image_count")
    try:
        photos_count = float(img_cnt) if img_cnt is not None else 1.0
    except (ValueError, TypeError):
        photos_count = 1.0

    url = str(ad.get("url") or f"https://www.avito.ma/fr/vi/{lid}.htm")

    record = {
        "listing_id": lid,
        "url": url,
        "source": "avito",
        "date_posted": date_posted,
        "date_scraped": date_scraped,
        "title_raw": title_raw,
        "brand": brand_c or "Autre",
        "model": model_c or "Autre",
        "trim": trim_c or "",
        "year": year,
        "mileage_km": mileage_km,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "fiscal_power_cv": fiscal_power_cv,
        "customs_status": customs_status,
        "condition": condition,
        "owners_count": owners_count,
        "doors_count": doors_count,
        "seller_type": seller_type,
        "seller_phone": seller_phone,
        "seller_phone_hash": seller_phone_hash,
        "city": city,
        "region": region,
        "price_mad": price_mad,
        "photos_count": photos_count,
        "description_raw": desc_raw,
    }
    return record


def main():
    if not PAGES_DIR.exists():
        logger.error("Pages directory %s does not exist! Please run download_t480_cache.py first.", PAGES_DIR)
        sys.exit(1)

    page_files = sorted(PAGES_DIR.glob("page_*.json"))
    logger.info("Found %d page files in %s", len(page_files), PAGES_DIR)

    date_fallback = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Index detail files in memory for fast O(1) lookup
    logger.info("Indexing detail files from %s...", DETAILS_DIR)
    detail_cache = {}
    if DETAILS_DIR.exists():
        for det_path in DETAILS_DIR.glob("*.json"):
            detail_cache[det_path.stem] = det_path
    logger.info("Indexed %d detail files", len(detail_cache))

    all_records = []
    seen_ids = set()

    for idx, page_path in enumerate(page_files, start=1):
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                ads = json.load(f)
        except Exception as e:
            logger.warning("Error reading %s: %s", page_path.name, e)
            continue

        if not isinstance(ads, list):
            continue

        for ad in ads:
            lid = str(ad.get("list_id") or ad.get("id") or "").strip()
            if not lid or lid in seen_ids:
                continue

            # Load detail JSON if available
            detail = {}
            if lid in detail_cache:
                try:
                    with open(detail_cache[lid], "r", encoding="utf-8") as f:
                        detail = json.load(f)
                except Exception:
                    detail = {}

            rec = extract_record(ad, detail, date_fallback)
            if rec:
                all_records.append(rec)
                seen_ids.add(lid)

        if idx % 100 == 0 or idx == len(page_files):
            logger.info("Processed %d/%d pages -> %d unique car records extracted so far", idx, len(page_files), len(all_records))

    logger.info("Extraction complete! Total unique car listings extracted: %d", len(all_records))

    df = pd.DataFrame(all_records)
    df = df[SCHEMA_FIELDS]

    # Save to data/raw/avito_t480_complete.parquet and .csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / "avito_t480_complete.parquet"
    csv_path = OUTPUT_DIR / "avito_t480_complete.csv"

    logger.info("Saving %d records to %s ...", len(df), parquet_path.name)
    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    logger.info("Saving %d records to %s ...", len(df), csv_path.name)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info("Successfully exported %s (%d rows, %.2f MB)", parquet_path.name, len(df), parquet_path.stat().st_size / 1e6)
    logger.info("Successfully exported %s (%d rows, %.2f MB)", csv_path.name, len(df), csv_path.stat().st_size / 1e6)

    # Now update master database
    logger.info("Consolidating into master database: data/raw/scraped_master_database.parquet ...")
    master_path = update_master_database(OUTPUT_DIR)
    logger.info("Master database update complete -> %s", master_path)


if __name__ == "__main__":
    main()
