#!/usr/bin/env python3
"""
MarocHub.app Dedicated Production Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------------
Inherits from BaseScraper:
- Targets MarocHub used car marketplace (https://marochub.app/auto)
- Parses Next.js SSR payloads and card HTML
- Resolves seller WhatsApp and phone contact info
- Full Cahier des Charges schema alignment (24 standard fields)
- Saves to data/raw/marochub_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import json
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
logger = logging.getLogger("scraper.marochub")

DEFAULT_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://marochub.app/",
}

MAROCHUB_CATEGORIES = [
    "",  # Main auto feed
    "renault", "dacia", "peugeot", "volkswagen", "mercedes-benz",
    "hyundai", "toyota", "audi", "fiat", "honda", "seat", "citroen",
]


class MarocHubScraper(BaseScraper):
    source_name = "marochub"
    BASE_URL = "https://marochub.app/auto"

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
        import requests
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_BROWSER_HEADERS)

    def extract_phone_from_detail(self, detail_url: str) -> Optional[str]:
        """Fetch vehicle detail page to extract seller WhatsApp / phone."""
        if not detail_url:
            return None
        try:
            resp = self.session.get(detail_url, headers=DEFAULT_BROWSER_HEADERS, verify=False, timeout=6)
            if resp.status_code != 200:
                return None
            html = resp.text

            # 1. Check tel: links
            tel_matches = re.findall(r'href=["\']tel:([^"\']+)["\']', html, re.I)
            for t in tel_matches:
                p = extract_moroccan_phone(t)
                if p:
                    return p

            # 2. Check WhatsApp links
            wa_matches = re.findall(r'(?:wa\.me/|api\.whatsapp\.com/send\?phone=)(\+?\d+)', html, re.I)
            for w in wa_matches:
                p = extract_moroccan_phone(w)
                if p:
                    return p

            # 3. Regex search in body
            return extract_moroccan_phone(html)
        except Exception as e:
            logger.debug("[%s] Detail phone error for %s: %s", self.source_name, detail_url, e)
            return None

    def parse_vehicle_json(self, v: Dict[str, Any], date_scraped: str) -> Optional[Dict[str, Any]]:
        """Normalize vehicle dictionary from MarocHub Next.js JSON payload."""
        l_id = str(v.get("id") or "")
        if not l_id or l_id in self.seen_ids:
            return None

        # Build vehicle detail URL
        title = str(v.get("title") or "")
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
        short_id = l_id[:8]
        detail_url = f"https://marochub.app/vehicle/{slug}-{short_id}" if slug else f"https://marochub.app/vehicle/{short_id}"

        brand = str(v.get("brand") or "").strip()
        model = str(v.get("model") or "").strip()
        if not brand and title:
            tokens = title.split()
            brand = tokens[0] if tokens else "Autre"
            model = tokens[1] if len(tokens) > 1 else "Autre"

        price_raw = v.get("price")
        price_mad = float(price_raw) if price_raw is not None else None

        year_raw = v.get("year")
        year = int(year_raw) if year_raw is not None else None

        km_raw = v.get("mileage")
        mileage_km = float(km_raw) if km_raw is not None else None

        fuel_raw = str(v.get("fuel_type") or "").lower()
        fuel_type = "Diesel" if "diesel" in fuel_raw else ("Essence" if "essence" in fuel_raw else ("Hybride" if "hybride" in fuel_raw else ("Electrique" if "electrique" in fuel_raw else "")))

        trans_raw = str(v.get("transmission") or "").lower()
        transmission = "Automatique" if "auto" in trans_raw else ("Manuelle" if "man" in trans_raw else "")

        city = str(v.get("location_city") or "Casablanca").strip().title()
        region = infer_moroccan_region(city)

        imgs = v.get("images") or []
        photos_count = float(len(imgs)) if isinstance(imgs, list) else 1.0

        seller_phone = None
        # Attempt detail phone lookup for verified listings
        if detail_url:
            seller_phone = self.extract_phone_from_detail(detail_url)
        seller_phone_hash = hash_phone(seller_phone)

        raw_record = {
            "listing_id": l_id,
            "url": detail_url,
            "source": self.source_name,
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": title,
            "brand": brand,
            "model": model,
            "trim": "",
            "year": year,
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "fiscal_power_cv": "",
            "customs_status": "Dédouanée",
            "condition": "Occasion",
            "owners_count": "",
            "doors_count": None,
            "seller_type": "Particulier",
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": photos_count,
            "description_raw": title,
        }
        return self.validate_and_format_record(raw_record)

    def parse_card_html(self, href: str, title: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse vehicle card extracted from MarocHub HTML DOM."""
        short_id_m = re.search(r"/vehicle/.*?([a-f0-9]{8})$", href)
        listing_id = short_id_m.group(1) if short_id_m else href.split("/")[-1]
        if not listing_id or listing_id in self.seen_ids:
            return None

        full_url = urljoin("https://marochub.app", href)

        brand = ""
        for b in KNOWN_BRANDS:
            if re.search(rf"\b{re.escape(b)}\b", title, re.IGNORECASE):
                brand = b
                break
        if not brand:
            brand = "Autre"

        model = ""
        m_after = re.search(rf"\b{re.escape(brand)}\b\s*([a-zA-Z0-9À-ÿ\.\-]+)", title, re.IGNORECASE)
        if m_after:
            model = m_after.group(1).strip()
        else:
            model = "Autre"

        # Year
        y_m = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", title)
        year = int(y_m.group(1)) if y_m else None

        # Price
        p_m = re.search(r"(\d[\d\s\.]*)\s*(?:DH|MAD)\b", title, re.I)
        price_mad = None
        if p_m:
            raw_p = re.sub(r"[\s,.]", "", p_m.group(1))
            if raw_p.isdigit():
                price_mad = float(raw_p)

        # Phone lookup
        # Optional phone lookup (limited to avoid slow crawl)
        seller_phone = None
        if len(self.seen_ids) < 5:
            seller_phone = self.extract_phone_from_detail(full_url)
        seller_phone_hash = hash_phone(seller_phone)

        city = "Casablanca"
        region = infer_moroccan_region(city)

        raw_record = {
            "listing_id": listing_id,
            "url": full_url,
            "source": self.source_name,
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": title,
            "brand": brand,
            "model": model,
            "trim": "",
            "year": year,
            "mileage_km": None,
            "fuel_type": "",
            "transmission": "",
            "fiscal_power_cv": "",
            "customs_status": "Dédouanée",
            "condition": "Occasion",
            "owners_count": "",
            "doors_count": None,
            "seller_type": "Particulier",
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": 1.0,
            "description_raw": title,
        }
        return self.validate_and_format_record(raw_record)

    def scrape_url(self, url: str) -> List[Dict[str, Any]]:
        """Fetch MarocHub page and extract both SSR JSON objects and HTML cards."""
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = None
        try:
            resp = self.session.get(url, headers=DEFAULT_BROWSER_HEADERS, verify=False, timeout=12)
            if resp.status_code == 200:
                html = resp.text
        except Exception as e:
            logger.debug("[%s] Request error for %s: %s", self.source_name, url, e)
            html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS, timeout=(10, 20))

        if not html:
            logger.warning("[%s] Failed to fetch: %s", self.source_name, url)
            return []

        records = []
        parsed_ids = set()

        # 1. Parse JSON objects from script payloads
        for s in re.finditer(r'initialFeatured\\?":(\[.*?\])\s*,\s*\\?"initial', html):
            raw_json = s.group(1).replace(r'\"', '"').replace(r'\\/', '/')
            try:
                vehicles = json.loads(raw_json)
                for v in vehicles:
                    rec = self.parse_vehicle_json(v, date_scraped)
                    if rec and rec["listing_id"] not in parsed_ids:
                        records.append(rec)
                        parsed_ids.add(rec["listing_id"])
            except Exception as e:
                logger.debug("[%s] Error parsing initialFeatured: %s", self.source_name, e)

        # 2. Parse HTML cards with /vehicle/ links
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=re.compile(r"/vehicle/")):
            href = a.get("href", "")
            title = a.get_text(separator=" ", strip=True) or a.get("aria-label", "") or a.get("title", "")
            if not title and a.parent:
                title = a.parent.get_text(separator=" ", strip=True)
            rec = self.parse_card_html(href, title, date_scraped)
            if rec and rec["listing_id"] not in parsed_ids:
                records.append(rec)
                parsed_ids.add(rec["listing_id"])

        return records

    def scrape(
        self,
        max_pages: int = 10,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Crawl MarocHub across categories and main feed up to max_pages."""
        logger.info("[%s] Starting crawl across %d categories/pages ...", self.source_name, max_pages)
        all_records = []

        categories = MAROCHUB_CATEGORIES[:max_pages]

        for idx, cat in enumerate(categories, start=1):
            url = f"{self.BASE_URL}/{cat}" if cat else self.BASE_URL
            batch = self.scrape_url(url)
            logger.info("[%s] Category %d/%d ('%s'): parsed %d listings", self.source_name, idx, len(categories), cat or "main", len(batch))
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
    parser = argparse.ArgumentParser(description="MarocHub.app Production Scraper")
    parser.add_argument("--max-pages", type=int, default=10, help="Max category feeds to scrape (default: 10)")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output CSV filename")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    scraper = MarocHubScraper(output_dir=args.output_dir)
    df = scraper.scrape(
        max_pages=args.max_pages,
        output_filename=args.output_filename,
    )
    print(f"Successfully scraped and saved {len(df)} records for MarocHub.app")


if __name__ == "__main__":
    main()
