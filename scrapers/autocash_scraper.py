#!/usr/bin/env python3
"""
Autocash.ma Dedicated Production Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------------
Inherits from BaseScraper:
- Targets Autocash.ma used car inventory (https://www.autocash.ma/fr/achat/voitures)
- Certified professional inspected car inventory
- Full Cahier des Charges schema alignment (24 standard fields)
- Saves to data/raw/autocash_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import logging
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
logger = logging.getLogger("scraper.autocash")

DEFAULT_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.autocash.ma/fr",
}

AUTOCASH_MARQUES = [
    "",  # All / general catalog
    "dacia", "renault", "peugeot", "volkswagen", "hyundai",
    "mercedes-benz", "audi", "toyota", "bmw", "citroen",
    "kia", "ford", "nissan", "fiat", "seat", "jeep",
    "land_rover", "changan", "opel", "skoda", "honda",
    "suzuki", "volvo", "porsche", "mitsubishi", "alfa-romeo",
    "jaguar", "ds", "mini", "cupra", "dfsk", "mg",
]


class AutocashScraper(BaseScraper):
    source_name = "autocash"
    BASE_URL = "https://www.autocash.ma/fr/achat/voitures"

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

    def parse_card(self, card_elem: BeautifulSoup, href: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse individual vehicle card on Autocash catalog."""
        m_id = re.search(r"/voiture/(\d+)", href)
        if not m_id:
            return None
        listing_id = m_id.group(1)

        if listing_id in self.seen_ids:
            return None

        full_url = urljoin("https://www.autocash.ma", href)
        card_text = card_elem.get_text(separator=" ", strip=True)

        # Year from MM/YYYY or YYYY
        year_match = re.search(r"\b(0[1-9]|1[0-2])/((?:19|20)\d\d)\b", card_text)
        if year_match:
            year = int(year_match.group(2))
        else:
            y_m = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", card_text)
            year = int(y_m.group(1)) if y_m else None

        # Mileage in km (e.g. 74 691 km)
        km_match = re.search(r"\b(\d{1,3}(?:[\s\xa0]\d{3})*|\d+)\s*km\b", card_text, re.I)
        mileage_km = None
        if km_match:
            raw_km = re.sub(r"[\s\xa0,.]", "", km_match.group(1))
            if raw_km.isdigit():
                mileage_km = float(raw_km)

        # Price in DH (e.g. 393 000 DH)
        price_mad = None
        p_match = re.search(r"\b(\d{1,3}(?:[\s\xa0]\d{3})*|\d+)\s*DH\b", card_text)
        if p_match:
            raw_p = re.sub(r"[\s\xa0,.]", "", p_match.group(1))
            if raw_p.isdigit():
                price_mad = float(raw_p)

        # Fuel
        fuel_type = ""
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
        transmission = ""
        if "automatique" in c_lower or "auto" in c_lower:
            transmission = "Automatique"
        elif "manuelle" in c_lower or "manuel" in c_lower:
            transmission = "Manuelle"

        # Brand and model resolution
        brand = ""
        model = ""
        trim = ""

        # Check against KNOWN_BRANDS
        for b in KNOWN_BRANDS:
            if re.search(rf"\b{re.escape(b)}\b", card_text, re.IGNORECASE):
                brand = b
                break

        # Tokens after brand
        if brand:
            m_after = re.search(rf"\b{re.escape(brand)}\b\s*([a-zA-Z0-9À-ÿ\.\-]+(?:\s+[a-zA-Z0-9À-ÿ\.\-]+)?)", card_text, re.IGNORECASE)
            if m_after:
                model = m_after.group(1).strip()
        else:
            brand = "Autre"
            model = "Autre"

        # City & Region
        city = "Casablanca"
        region = infer_moroccan_region(city)

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
            "customs_status": "Dédouanée",
            "condition": "Occasion",
            "owners_count": "",
            "doors_count": None,
            "seller_type": "Professionnel",
            "seller_phone": None,
            "seller_phone_hash": None,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": 1.0,
            "description_raw": card_text,
        }
        return self.validate_and_format_record(raw_record)

    def scrape_url(self, url: str) -> List[Dict[str, Any]]:
        """Fetch and parse one page/filter from Autocash."""
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS, timeout=(10, 20))
        if not html:
            try:
                import requests as std_requests
                resp = std_requests.get(url, headers=DEFAULT_BROWSER_HEADERS, verify=False, timeout=15)
                if resp.status_code == 200:
                    html = resp.text
            except Exception as e:
                logger.debug("[%s] Requests fallback error: %s", self.source_name, e)

        if not html:
            logger.warning("[%s] Failed to fetch: %s", self.source_name, url)
            return []

        soup = BeautifulSoup(html, "html.parser")
        links = soup.find_all("a", href=re.compile(r"/voiture/\d+"))

        records = []
        seen_links = set()

        for a in links:
            href = a.get("href")
            if not href or href in seen_links:
                continue
            seen_links.add(href)

            # Climb to card parent container
            parent = a
            for _ in range(6):
                if parent.parent:
                    parent = parent.parent
                    if any("rounded" in c for c in parent.get("class", [])):
                        break

            rec = self.parse_card(parent, href, date_scraped)
            if rec:
                records.append(rec)

        return records

    def scrape(
        self,
        max_pages: int = 15,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Crawl Autocash across marques up to max_pages."""
        logger.info("[%s] Starting crawl across marques (max %d queries) ...", self.source_name, max_pages)
        all_records = []

        marques_to_crawl = AUTOCASH_MARQUES[:max_pages]

        for idx, m in enumerate(marques_to_crawl, start=1):
            url = f"{self.BASE_URL}?marque={m}" if m else self.BASE_URL
            batch = self.scrape_url(url)
            logger.info("[%s] Query %d/%d ('%s'): parsed %d listings", self.source_name, idx, len(marques_to_crawl), m or "all", len(batch))
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
    parser = argparse.ArgumentParser(description="Autocash.ma Production Scraper")
    parser.add_argument("--max-pages", type=int, default=15, help="Max brand queries to scrape (default: 15)")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output CSV filename")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    scraper = AutocashScraper(output_dir=args.output_dir)
    df = scraper.scrape(
        max_pages=args.max_pages,
        output_filename=args.output_filename,
    )
    print(f"Successfully scraped and saved {len(df)} records for Autocash.ma")


if __name__ == "__main__":
    main()
