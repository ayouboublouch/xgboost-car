#!/usr/bin/env python3
"""
Kifal-Auto.ma Dedicated Used Car Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------------
Inherits from BaseScraper:
- Targets Kifal-Auto live catalog (https://occasion.kifal.ma/annonces)
- Full Cahier des Charges schema alignment (24 standard fields)
- Certified inspected vehicle source (high feature completeness)
- Saves to data/raw/kifal_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import logging
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, unquote

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
import pandas as pd

try:
    from scrapers.base import (
        BaseScraper,
        SCHEMA_FIELDS,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
        KNOWN_BRANDS,
    )
except ImportError:
    from base import (
        BaseScraper,
        SCHEMA_FIELDS,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
        KNOWN_BRANDS,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.kifal")

DEFAULT_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://occasion.kifal.ma/",
}


class KifalScraper(BaseScraper):
    source_name = "kifal"
    BASE_URL = "https://occasion.kifal.ma/annonces"

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_min: float = 0.5,
        delay_max: float = 1.2,
        max_retries: int = 3,
    ):
        super().__init__(
            output_dir=output_dir,
            delay_min=delay_min,
            delay_max=delay_max,
            max_retries=max_retries,
        )

    def parse_card(self, card_elem: BeautifulSoup, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse individual car card from Kifal-Auto listing page."""
        link = card_elem.find("a", href=re.compile(r"/annonce/"))
        if not link:
            return None

        href = link.get("href", "")
        if not href:
            return None
        full_url = urljoin("https://occasion.kifal.ma", href)

        # ID from URL: e.g. _7337_VEH00011V6.htm or _7337.htm
        m_id = re.search(r"_(\d+)(?:_VEH|\.htm)", full_url)
        if not m_id:
            m_id = re.search(r"[-_/](\d+)(?:\.html|\.htm|/|$)", full_url)
        listing_id = m_id.group(1) if m_id else full_url.split("/")[-1].replace(".htm", "")
        if not listing_id:
            return None

        if listing_id in self.seen_ids:
            return None

        card_text = card_elem.get_text(separator=" ", strip=True)

        # URL format breakdown:
        # /annonce/{BRAND}_{MODEL}_{YEAR}_{FUEL}_{TRANSMISSION}_{CITY}_{ID}_{VEHID}.htm
        url_match = re.search(r"/annonce/([^_]+)_([^_]+)_(\d{4})_([^_]+)_([^_]+)_([^_]+)_(\d+)", full_url)

        brand_from_url = unquote(url_match.group(1)).replace("-", " ") if url_match else ""
        model_from_url = unquote(url_match.group(2)) if url_match else ""
        year_from_url = int(url_match.group(3)) if url_match else None
        fuel_from_url = unquote(url_match.group(4)) if url_match else ""
        trans_from_url = unquote(url_match.group(5)) if url_match else ""
        city_from_url = unquote(url_match.group(6)) if url_match else ""

        # Brand resolution
        brand = brand_from_url
        if not brand:
            for b in KNOWN_BRANDS:
                if re.search(rf"\b{re.escape(b)}\b", card_text, re.IGNORECASE):
                    brand = b
                    break
        if not brand:
            brand = "Autre"

        # Model resolution
        model = model_from_url
        if not model:
            tokens = [t for t in card_text.split() if t and t.lower() != brand.lower()]
            model = tokens[0] if tokens else "Autre"

        # Trim
        trim = ""
        # Often in the text as full version description
        for part in card_text.split("|"):
            p_strip = part.strip()
            if any(k in p_strip.lower() for k in ["tdi", "hdi", "dci", "pack", "amg", "line", "quattro", "exclusive", "edition", "bva"]):
                trim = p_strip
                break

        # Year
        year = year_from_url
        if not year:
            y_m = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", card_text)
            year = int(y_m.group(1)) if y_m else None

        # Mileage
        km_m = re.search(r"(\d[\d\s\xa0]*)\s*km", card_text, re.I)
        mileage_km = None
        if km_m:
            raw_km = re.sub(r"[\s\xa0,.]", "", km_m.group(1))
            if raw_km.isdigit():
                mileage_km = float(raw_km)

        # Price
        p_m = re.search(r"(\d[\d\s\xa0]*)\s*(?:Dh|DH|MAD)", card_text)
        price_mad = None
        if p_m:
            raw_p = re.sub(r"[\s\xa0,.]", "", p_m.group(1))
            if raw_p.isdigit():
                price_mad = float(raw_p)

        # Fuel
        fuel_type = fuel_from_url.capitalize()
        if not fuel_type:
            c_lower = card_text.lower()
            if "diesel" in c_lower:
                fuel_type = "Diesel"
            elif "essence" in c_lower:
                fuel_type = "Essence"
            elif "hybride" in c_lower:
                fuel_type = "Hybride"
            elif "electrique" in c_lower or "électrique" in c_lower:
                fuel_type = "Electrique"

        # Transmission
        transmission = trans_from_url.capitalize()
        if not transmission:
            c_lower = card_text.lower()
            if "auto" in c_lower:
                transmission = "Automatique"
            elif "man" in c_lower:
                transmission = "Manuelle"

        # City & Region
        city = city_from_url.title() if city_from_url else "Casablanca"
        region = infer_moroccan_region(city)

        # Photos count from img tags
        imgs = card_elem.find_all("img")
        photos_count = float(len(imgs)) if imgs else 1.0

        raw_record = {
            "listing_id": str(listing_id),
            "url": full_url,
            "source": self.source_name,
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": f"{brand} {model}".strip(),
            "brand": brand,
            "model": model,
            "trim": trim,
            "year": year,
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "fiscal_power_cv": "",
            "customs_status": "WW au Maroc",
            "condition": "Inspecté / Certifié",
            "owners_count": "",
            "doors_count": None,
            "seller_type": "Professionnel",
            "seller_phone": None,
            "seller_phone_hash": None,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": photos_count,
            "description_raw": card_text,
        }
        return self.validate_and_format_record(raw_record)

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one catalog page from Kifal-Auto."""
        url = f"{self.BASE_URL}?page={page_num}" if page_num > 1 else self.BASE_URL
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS, timeout=(10, 20))
        if not html:
            # Fallback to requests directly
            try:
                import requests as std_requests
                resp = std_requests.get(url, headers=DEFAULT_BROWSER_HEADERS, verify=False, timeout=15)
                if resp.status_code == 200:
                    html = resp.text
            except Exception as e:
                logger.debug("[%s] Requests fallback error: %s", self.source_name, e)

        if not html:
            logger.warning("[%s] Failed to fetch page %d", self.source_name, page_num)
            return []

        soup = BeautifulSoup(html, "html.parser")
        items = []

        # Target card containers
        cards = soup.find_all("div", class_="d-md-flex")
        if not cards:
            cards = soup.find_all("div", class_=re.compile(r"item-card9|card", re.I))

        for card in cards:
            rec = self.parse_card(card, date_scraped)
            if rec:
                items.append(rec)

        logger.info("[%s] Page %d: parsed %d listings", self.source_name, page_num, len(items))
        return items

    def scrape(
        self,
        max_pages: int = 15,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Crawl Kifal used car listings up to max_pages starting from start_page."""
        end_page = start_page + max_pages - 1
        logger.info("[%s] Starting crawl for pages %d to %d (max %d pages) ...", self.source_name, start_page, end_page, max_pages)
        all_records = []

        consecutive_empty = 0
        for p in range(start_page, start_page + max_pages):
            batch = self.scrape_page(p)
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    logger.info("[%s] Consecutive empty pages encountered at page %d. Stopping.", self.source_name, p)
                    break
            else:
                consecutive_empty = 0
                all_records.extend(batch)
            self.sleep()

        if len(all_records) == 0:
            logger.warning("[%s] Crawl complete. 0 records harvested. Writing nothing.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if "listing_id" in df.columns:
            initial_len = len(df)
            df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)
            logger.info("[%s] Deduplicated from %d to %d unique listings.", self.source_name, initial_len, len(df))

        logger.info("[%s] Crawl complete. Total records: %d", self.source_name, len(df))
        self.save_output(df, custom_filename=output_filename)
        return df


def main():
    parser = argparse.ArgumentParser(description="Kifal-Auto Production Scraper")
    parser.add_argument("--start-page", type=int, default=1, help="Initial page to scrape (default: 1)")
    parser.add_argument("--max-pages", type=int, default=15, help="Max pages to scrape (default: 15)")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output CSV filename")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    scraper = KifalScraper(output_dir=args.output_dir)
    df = scraper.scrape(
        max_pages=args.max_pages,
        start_page=args.start_page,
        output_filename=args.output_filename,
    )
    print(f"Successfully scraped and saved {len(df)} records for Kifal-Auto.ma")


if __name__ == "__main__":
    main()
