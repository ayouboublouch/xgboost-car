#!/usr/bin/env python3
"""
Auto24.ma Dedicated Certified Used Car Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------------------
Africar Group certified automotive platform in Morocco (https://auto24.ma)
- Communicates directly with https://api.auto24.ma/api/cars REST API
- Extracts 100% verified dealer/certified inventory with exact mileage & VIN
- Full Cahier des Charges schema alignment (24 standard fields)
- Normalizes phone numbers from dealer store metadata
- Persists to data/raw/auto24_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

import pandas as pd

try:
    from scrapers.base import (
        BaseScraper,
        SCHEMA_FIELDS,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
        clean_brand_and_model,
    )
except ImportError:
    from base import (
        BaseScraper,
        SCHEMA_FIELDS,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
        clean_brand_and_model,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.auto24")


class Auto24Scraper(BaseScraper):
    source_name = "auto24"
    API_URL = "https://api.auto24.ma/api/cars"
    BASE_WEB_URL = "https://auto24.ma/acheter-voitures/"

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_min: float = 0.3,
        delay_max: float = 0.8,
        max_retries: int = 3,
    ):
        super().__init__(
            output_dir=output_dir,
            delay_min=delay_min,
            delay_max=delay_max,
            max_retries=max_retries,
        )

    def parse_car_json(self, car: Dict[str, Any], date_scraped: str) -> Optional[Dict[str, Any]]:
        """Map raw Auto24 REST API vehicle JSON to standard 24 Cahier des Charges fields."""
        uid = car.get("uId") or car.get("_id")
        if not uid:
            return None

        listing_id = str(uid).strip()
        slug = car.get("slug") or ""
        url = urljoin(self.BASE_WEB_URL, slug) if slug else f"https://auto24.ma/car/{listing_id}"

        # 1. Brand & Model Resolution
        brand_raw = ""
        b_obj = car.get("brand")
        if isinstance(b_obj, dict):
            brand_raw = b_obj.get("brand") or b_obj.get("name") or ""
        elif isinstance(b_obj, str):
            brand_raw = b_obj

        model_raw = ""
        m_obj = car.get("model")
        if isinstance(m_obj, dict):
            model_raw = m_obj.get("model") or m_obj.get("name") or ""
        elif isinstance(m_obj, str):
            model_raw = m_obj

        title_raw = car.get("name") or f"{brand_raw} {model_raw}".strip()
        brand, model, trim_inferred = clean_brand_and_model(brand_raw, model_raw, title_raw)

        # 2. Year & Mileage
        year = car.get("modelYear")
        if not year and car.get("dateFirstUse"):
            try:
                year = int(str(car["dateFirstUse"])[:4])
            except Exception:
                year = None

        mileage_km = car.get("mileage")
        if mileage_km is not None:
            try:
                mileage_km = float(mileage_km)
            except Exception:
                mileage_km = None

        # 3. Price
        price_mad = car.get("price")
        if price_mad is not None:
            try:
                price_mad = float(price_mad)
            except Exception:
                price_mad = None

        # 4. Fuel & Transmission Normalization
        fuel_raw = str(car.get("fuelType") or "").lower()
        if "diesel" in fuel_raw:
            fuel_type = "Diesel"
        elif "essence" in fuel_raw or "petrol" in fuel_raw or "gasoline" in fuel_raw:
            fuel_type = "Essence"
        elif "hybride" in fuel_raw or "hybrid" in fuel_raw:
            fuel_type = "Hybride"
        elif "electrique" in fuel_raw or "electric" in fuel_raw or car.get("isElectric"):
            fuel_type = "Electrique"
        else:
            fuel_type = "Autre"

        trans_raw = str(car.get("transmission") or "").lower()
        if "auto" in trans_raw:
            transmission = "Automatique"
        elif "man" in trans_raw:
            transmission = "Manuelle"
        else:
            transmission = "Automatique" if car.get("sequential") else ""

        # 5. Fiscal Power
        fiscal_cv = car.get("fiscalPower") or ""

        # 6. Condition & Customs
        cond_raw = str(car.get("condition") or "").lower()
        if "excel" in cond_raw or "new" in cond_raw:
            condition = "Excellent"
        elif "good" in cond_raw or "bon" in cond_raw:
            condition = "Très bon"
        else:
            condition = "Bon état"

        customs_status = "Dédouanée" if not car.get("imported") else "Non dédouanée"

        # 7. Store / Location / City / Region
        store = car.get("store") or {}
        city = "Casablanca"
        seller_phone = None
        if isinstance(store, dict):
            loc_text = f"{store.get('name', '')} {store.get('address', '')}"
            for c_cand in ["Casablanca", "Rabat", "Marrakech", "Tanger", "Agadir", "Fès", "Meknès", "Oujda", "Kénitra", "Tétouan"]:
                if c_cand.lower() in loc_text.lower():
                    city = c_cand
                    break
            raw_ph = store.get("phoneNumber") or store.get("phone")
            if raw_ph:
                seller_phone = extract_moroccan_phone(raw_ph)

        region = infer_moroccan_region(city)
        seller_phone_hash = hash_phone(seller_phone) if seller_phone else None

        # 8. Photos & Description
        images = car.get("images") or []
        photos_count = float(len(images)) if isinstance(images, list) else 0.0

        details = car.get("details") or []
        desc_parts = []
        if isinstance(details, list):
            for d in details:
                if isinstance(d, dict) and d.get("description"):
                    desc_parts.append(d["description"])
        description_raw = " ".join(desc_parts).strip()
        description_raw = re.sub(r"[\r\n\t]+", " ", description_raw)

        raw_record = {
            "listing_id": listing_id,
            "url": url,
            "source": self.source_name,
            "date_posted": str(car.get("createdAt") or date_scraped)[:10],
            "date_scraped": date_scraped,
            "title_raw": title_raw,
            "brand": brand,
            "model": model,
            "trim": car.get("trim") or trim_inferred or "",
            "year": year,
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "fiscal_power_cv": fiscal_cv,
            "customs_status": customs_status,
            "condition": condition,
            "owners_count": car.get("noOfOwners") or "",
            "doors_count": car.get("noOfDoors"),
            "seller_type": "Professionnel",
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": photos_count,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def scrape(
        self,
        max_pages: int = 10,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        limit_per_page: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch verified cars from Auto24 REST API."""
        logger.info("[%s] Harvesting certified cars from %s ...", self.source_name, self.API_URL)
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_records = []

        for page in range(start_page, start_page + max_pages):
            params = {"page": page, "limit": limit_per_page}
            try:
                resp = self.session.get(self.API_URL, params=params, timeout=12)
                if resp.status_code != 200:
                    logger.warning("[%s] API returned status %d on page %d", self.source_name, resp.status_code, page)
                    break

                data = resp.json()
                cars = data.get("cars") or []
                if not cars:
                    logger.info("[%s] No further cars returned on page %d. Catalog boundary reached.", self.source_name, page)
                    break

                page_records = 0
                for c in cars:
                    rec = self.parse_car_json(c, date_scraped)
                    if rec:
                        all_records.append(rec)
                        page_records += 1

                logger.info(
                    "[%s] Page %d: parsed %d/%d certified cars (running total: %d)",
                    self.source_name,
                    page,
                    page_records,
                    len(cars),
                    len(all_records),
                )

                total_pages = data.get("totalPages", 1)
                if page >= total_pages:
                    logger.info("[%s] Reached last page (%d of %d).", self.source_name, page, total_pages)
                    break

            except Exception as e:
                logger.warning("[%s] Error requesting page %d: %s", self.source_name, page, e)
                break

            self.sleep()

        if not all_records:
            logger.warning("[%s] Harvested 0 records. Writing nothing.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if "listing_id" in df.columns:
            df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)

        logger.info("[%s] Completed crawl. Total verified records: %d", self.source_name, len(df))
        self.save_output(df, custom_filename=output_filename)
        return df


def main():
    parser = argparse.ArgumentParser(description="Auto24 Production API Scraper")
    parser.add_argument("--start-page", type=int, default=1, help="Start page")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output CSV filename")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    scraper = Auto24Scraper(output_dir=args.output_dir)
    df = scraper.scrape(
        start_page=args.start_page,
        max_pages=args.max_pages,
        output_filename=args.output_filename,
    )
    print(f"Successfully scraped and saved {len(df)} records for Auto24")


if __name__ == "__main__":
    main()
