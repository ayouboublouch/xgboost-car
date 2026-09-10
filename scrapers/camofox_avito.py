#!/usr/bin/env python3
"""
Camoufox Avito.ma Local Stealth Scraper
=======================================
Optional standalone local crawler utilizing Camoufox (stealth Firefox browser)
to bypass Cloudflare Turnstile / Bot Management on Moroccan Avito (avito.ma).

Designed strictly for local residential runs to prevent IP bans and Cloudflare
datacenter blocks without adding heavy browser dependencies to the GitHub Actions runner.

Installation:
    pip install camoufox
    camoufox fetch

Usage:
    python scrapers/camofox_avito.py --max-pages 5 --headless
"""

import argparse
import datetime
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

try:
    from scrapers.base import BaseScraper, SCHEMA_FIELDS, extract_moroccan_phone, hash_phone
except ImportError:
    from base import BaseScraper, SCHEMA_FIELDS, extract_moroccan_phone, hash_phone

# Optional stealth engine import: zero CI bloat
try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    CAMOUFOX_AVAILABLE = False

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.camoufox_avito")


class CamoufoxAvitoScraper(BaseScraper):
    source_name = "avito"
    BASE_URL = "https://www.avito.ma/fr/maroc/voitures-%C3%A0_vendre"

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_min: float = 2.0,
        delay_max: float = 4.0,
        headless: bool = True,
    ):
        super().__init__(output_dir=output_dir, delay_min=delay_min, delay_max=delay_max)
        self.headless = headless

    def parse_listing_card(self, item_html: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse raw HTML listing card container into normalized 24-field dictionary."""
        # Extract listing URL and ID
        link_m = re.search(r'href=["\'](https://www\.avito\.ma/fr/[^"\']*_(\d+)\.htm)["\']', item_html)
        if not link_m:
            link_m = re.search(r'href=["\'](/fr/[^"\']*_(\d+)\.htm)["\']', item_html)
            if link_m:
                url = "https://www.avito.ma" + link_m.group(1)
                listing_id = link_m.group(2)
            else:
                return None
        else:
            url = link_m.group(1)
            listing_id = link_m.group(2)

        # Title
        title_m = re.search(r'<(?:h3|h2|p|span)[^>]*class=["\'][^"\']*(?:title|heading)[^"\']*["\'][^>]*>(.*?)</', item_html, re.DOTALL | re.IGNORECASE)
        title_raw = title_m.group(1).strip() if title_m else ""
        title_raw = re.sub(r'<[^>]+>', '', title_raw).strip()

        # Price in MAD
        price_mad = None
        price_m = re.search(r'<(?:span|p|div)[^>]*class=["\'][^"\']*(?:price|prix)[^"\']*["\'][^>]*>(.*?)</', item_html, re.DOTALL | re.IGNORECASE)
        if price_m:
            p_text = re.sub(r'<[^>]+>', '', price_m.group(1)).strip()
            p_clean = re.sub(r'[\s\xa0,.]', '', p_text)
            try:
                price_mad = float(re.search(r'\d+', p_clean).group(0))
            except Exception:
                price_mad = None

        # Year
        year = None
        y_m = re.search(r'\b(19[8-9]\d|20[0-2]\d)\b', item_html)
        if y_m:
            year = int(y_m.group(1))

        # Mileage km
        mileage_km = None
        km_m = re.search(r'(\d[\d\s\xa0]*)\s*(?:km|kms)\b', item_html, re.IGNORECASE)
        if km_m:
            try:
                mileage_km = float(re.sub(r'[\s\xa0,.]', '', km_m.group(1)))
            except ValueError:
                mileage_km = None

        # Fuel type
        fuel_type = ""
        item_lower = item_html.lower()
        if "diesel" in item_lower:
            fuel_type = "Diesel"
        elif "essence" in item_lower:
            fuel_type = "Essence"
        elif "hybride" in item_lower:
            fuel_type = "Hybride"
        elif "electrique" in item_lower or "électrique" in item_lower:
            fuel_type = "Electrique"

        # Transmission
        transmission = ""
        if "automatique" in item_lower:
            transmission = "Automatique"
        elif "manuelle" in item_lower:
            transmission = "Manuelle"

        # City
        city_m = re.search(r'<(?:span|p|div)[^>]*class=["\'][^"\']*(?:location|city|ville)[^"\']*["\'][^>]*>(.*?)</', item_html, re.DOTALL | re.IGNORECASE)
        city = re.sub(r'<[^>]+>', '', city_m.group(1)).strip() if city_m else ""

        # Brand / Model tokens
        brand = ""
        model = ""
        if title_raw:
            tokens = [t for t in re.sub(r'[^a-zA-Z0-9À-ÿ\s]', ' ', title_raw).split() if t]
            if tokens:
                brand = tokens[0].capitalize()
                model = " ".join(tokens[1:]).capitalize() if len(tokens) > 1 else ""
        if not brand:
            brand = "Autre"
        if not model:
            model = "Autre"

        # Extract seller phone numbers from tel:, wa.me, data-phone, and container text
        seller_phone = None
        tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', item_html, re.IGNORECASE)
        if tel_m:
            seller_phone = extract_moroccan_phone(tel_m.group(1))

        if not seller_phone:
            wa_m = re.search(r'wa\.me/(\+?212\d{9}|0[5-7]\d{8}|\d+)', item_html, re.IGNORECASE)
            if wa_m:
                seller_phone = extract_moroccan_phone(wa_m.group(1))

        if not seller_phone:
            dp_m = re.search(r'data-phone=["\']([^"\']+)["\']', item_html, re.IGNORECASE)
            if dp_m:
                seller_phone = extract_moroccan_phone(dp_m.group(1))

        if not seller_phone:
            seller_phone = extract_moroccan_phone(item_html)

        seller_phone_hash = hash_phone(seller_phone)

        raw_record = {
            "listing_id": listing_id,
            "url": url,
            "source": self.source_name,
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": title_raw,
            "brand": brand,
            "model": model,
            "trim": "",
            "year": year,
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "fiscal_power_cv": "",
            "customs_status": "",
            "condition": "",
            "owners_count": "",
            "doors_count": None,
            "seller_type": "Particulier",
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": "",
            "price_mad": price_mad,
            "photos_count": 1.0,
            "description_raw": "",
        }
        return self.validate_and_format_record(raw_record)

    def scrape(self, max_pages: int = 5, **kwargs) -> pd.DataFrame:
        """Crawl Avito.ma using stealth Camoufox browser."""
        if not CAMOUFOX_AVAILABLE:
            logger.error(
                "Camoufox is not installed in the active environment.\n"
                "To use the local Cloudflare bypass crawler, install it locally:\n"
                "    pip install camoufox\n"
                "    camoufox fetch\n"
            )
            return pd.DataFrame()

        logger.info(
            "[%s] Launching Camoufox stealth browser (headless=%s) for %d pages...",
            self.source_name,
            self.headless,
            max_pages,
        )

        all_records: List[Dict[str, Any]] = []
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with Camoufox(headless=self.headless) as browser:
                page = browser.new_page()

                for p in range(1, max_pages + 1):
                    page_url = f"{self.BASE_URL}?o={p}" if p > 1 else self.BASE_URL
                    logger.info("[%s] Navigating to page %d: %s", self.source_name, p, page_url)

                    try:
                        page.goto(page_url, wait_until="domcontentloaded", timeout=45000)
                        time.sleep(3.0)  # Allow dynamic listing render and Turnstile resolution
                    except Exception as e:
                        logger.warning("[%s] Page %d navigation error: %s", self.source_name, p, e)
                        continue

                    page_html = page.content()
                    if "Attention Required! | Cloudflare" in page_html or "cf-browser-verification" in page_html:
                        logger.warning("[%s] Cloudflare challenge encountered on page %d. Solving...", self.source_name, p)
                        time.sleep(5.0)
                        page_html = page.content()

                    # Extract card containers
                    cards = re.split(r'(?=<a[^>]+href=["\'][^"\']*_(\d+)\.htm)', page_html)
                    page_records = 0
                    for c in cards[1:]:
                        rec = self.parse_listing_card(c, date_scraped)
                        if rec:
                            all_records.append(rec)
                            page_records += 1

                    logger.info("[%s] Page %d parsed: %d listings harvested.", self.source_name, p, page_records)
                    self.sleep()

        except Exception as e:
            logger.error("[%s] Unexpected Camoufox execution error: %s", self.source_name, e)

        if len(all_records) == 0:
            logger.warning("[%s] Crawl complete. 0 records harvested.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info("[%s] Local crawl finished. Total verified records: %d", self.source_name, len(df))
        self.save_output(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Camoufox Avito.ma Local Stealth Scraper")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages to scrape (default: 3)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--no-headless", action="store_true", help="Launch browser with GUI visible for debugging")
    args = parser.parse_args()

    scraper = CamoufoxAvitoScraper(
        output_dir=args.output_dir,
        headless=not args.no_headless,
    )
    df = scraper.scrape(max_pages=args.max_pages)
    print(f"Scraped {len(df)} records for Avito.ma using Camoufox.")


if __name__ == "__main__":
    main()
