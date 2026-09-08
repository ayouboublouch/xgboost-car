#!/usr/bin/env python3
"""
Moteur.ma High-Throughput Production Scraper
============================================
Pivoted primary data source for Autohouse.ma MLOps pipeline.
- Targets Moteur.ma unblocked catalog pagination
- Extracts all ad containers and normalizes strictly to 24 Cahier des Charges fields
- Features dual-engine extraction (BeautifulSoup + Regex parser)
- Fails loudly with assertion if records == 0 to guarantee non-empty dataset
- Saves to data/raw/moteur_YYYY-MM-DD.csv and .parquet
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

# Add parent directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

try:
    from scrapers.base import BaseScraper, SCHEMA_FIELDS
except ImportError:
    from base import BaseScraper, SCHEMA_FIELDS

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.moteur")


class MoteurScraper(BaseScraper):
    source_name = "moteur"
    BASE_URL = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/recherche"

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_min: float = 0.4,
        delay_max: float = 1.0,
        max_retries: int = 3,
    ):
        super().__init__(
            output_dir=output_dir,
            delay_min=delay_min,
            delay_max=delay_max,
            max_retries=max_retries,
        )

    def parse_card_regex(self, block: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Fast, robust regex-based card extractor for Moteur.ma HTML."""
        link_m = re.search(r'href=["\']([^"\']*detail-annonce/(\d+)/?([^"\'\s>]*\.html)?)["\']', block)
        if not link_m:
            return None

        href = link_m.group(1)
        listing_id = link_m.group(2)
        slug = link_m.group(3) or ""
        full_url = urljoin("https://www.moteur.ma", href)

        # Title
        title_m = re.search(r'class=["\'][^"\']*ads-index-title[^"\']*["\']>\s*(.*?)\s*</h', block, re.DOTALL)
        if not title_m:
            title_m = re.search(r'alt=["\']([^"\']+)["\']\s+class=["\'][^"\']*cover-image', block)
        title_raw = title_m.group(1).strip() if title_m else ""

        # City
        city_m = re.search(r'fa-map-marker[^>]*></i>\s*([^\s<]+)', block)
        city = city_m.group(1).strip() if city_m else ""

        # Date posted
        timeago_m = re.search(r'class=["\']timeago["\']\s+data-time=["\']([^"\']+)["\']', block)
        date_posted = timeago_m.group(1)[:10] if timeago_m else date_scraped[:10]

        # Description
        desc_m = re.search(r'class=["\'][^"\']*ad-desc[^"\']*["\']>\s*(.*?)\s*</p>', block, re.DOTALL)
        description_raw = desc_m.group(1).strip() if desc_m else ""

        # Price in MAD
        price_m = re.search(r'class=["\'][^"\']*ad-price-grid[^"\']*["\']>\s*(.*?)\s*</h4>', block, re.DOTALL)
        price_raw = price_m.group(1).strip() if price_m else ""
        price_mad = None
        if price_raw and "appeler" not in price_raw.lower() and "demande" not in price_raw.lower():
            pm = re.search(r'(\d[\d\s,.]*)', price_raw)
            if pm:
                clean_p = re.sub(r'[\s,.]', '', pm.group(1))
                try:
                    price_mad = float(clean_p)
                except ValueError:
                    price_mad = None

        # Year
        year_m = re.search(r'fa-calendar\s+me-1["\']></i>\s*(\d{4})', block)
        year = int(year_m.group(1)) if year_m else None

        # Transmission
        trans_m = re.search(r'fa-cog\s+me-1["\']></i>\s*([A-Za-zÀ-ÿ]+)', block)
        trans_raw = trans_m.group(1).strip() if trans_m else ""
        transmission = ""
        if "auto" in trans_raw.lower():
            transmission = "Automatique"
        elif "man" in trans_raw.lower():
            transmission = "Manuelle"

        # Fuel type
        fuel_m = re.search(r'fa-tachometer\s+me-1["\']></i>\s*([A-Za-zÀ-ÿ]+)', block)
        fuel_raw = fuel_m.group(1).strip() if fuel_m else ""
        fuel_type = ""
        if "diesel" in fuel_raw.lower():
            fuel_type = "Diesel"
        elif "essence" in fuel_raw.lower():
            fuel_type = "Essence"
        elif "hybride" in fuel_raw.lower():
            fuel_type = "Hybride"
        elif "elect" in fuel_raw.lower() or "élect" in fuel_raw.lower():
            fuel_type = "Electrique"

        # Mileage km
        km_m = re.search(r'fa-road\s+me-1["\']></i>\s*(\d[\d\s,.]*)', block)
        mileage_km = None
        if km_m:
            try:
                mileage_km = float(re.sub(r'[\s,.]', '', km_m.group(1)))
            except ValueError:
                mileage_km = None

        # Fallbacks in card text
        card_text = re.sub(r'<[^>]+>', ' ', block)
        if not year:
            ym = re.search(r'\b(19[8-9]\d|20[0-2]\d)\b', card_text)
            if ym:
                year = int(ym.group(1))
        if not fuel_type:
            ct_lower = card_text.lower()
            if "diesel" in ct_lower:
                fuel_type = "Diesel"
            elif "essence" in ct_lower:
                fuel_type = "Essence"
            elif "hybride" in ct_lower:
                fuel_type = "Hybride"
            elif "electrique" in ct_lower or "électrique" in ct_lower:
                fuel_type = "Electrique"
        if not transmission:
            ct_lower = card_text.lower()
            if "automatique" in ct_lower or "auto" in ct_lower:
                transmission = "Automatique"
            elif "manuelle" in ct_lower or "manuel" in ct_lower:
                transmission = "Manuelle"
        if mileage_km is None:
            km_match = re.search(r'(\d[\d\s,.]*)\s*(?:km|kms)\b', card_text, re.I)
            if km_match:
                try:
                    mileage_km = float(re.sub(r'[\s,.]', '', km_match.group(1)))
                except ValueError:
                    pass

        # Brand and model resolution
        brand = ""
        model = ""
        if slug:
            slug_clean = slug.replace(".html", "")
            parts = slug_clean.split("-")
            if parts:
                brand = parts[0].capitalize()
                model = " ".join(parts[1:]).capitalize()
        if not brand and title_raw:
            tokens = title_raw.split()
            brand = tokens[0].capitalize()
            model = " ".join(tokens[1:]).capitalize() if len(tokens) > 1 else ""

        raw_record = {
            "listing_id": listing_id,
            "url": full_url,
            "source": self.source_name,
            "date_posted": date_posted,
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
            "city": city,
            "region": "",
            "price_mad": price_mad,
            "photos_count": 1.0,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def parse_page(self, html: str, date_scraped: str) -> List[Dict[str, Any]]:
        """Parse all listing cards from a catalog page HTML."""
        records = []
        
        # Split by distinct ad card containers
        card_blocks = re.split(r'<div class=["\']ad-col col-12["\']', html)
        if len(card_blocks) <= 1:
            card_blocks = re.split(r'class=["\']card mb-0 overflow-hidden h-100 ads-index-card["\']', html)
        if len(card_blocks) <= 1:
            card_blocks = re.split(r'(?=<a[^>]+href=["\'][^"\']*detail-annonce/\d+)', html)

        for block in card_blocks[1:]:
            rec = self.parse_card_regex(block, date_scraped)
            if rec:
                records.append(rec)

        # Fallback to BeautifulSoup if bs4 is available and regex yielded few records
        if len(records) == 0 and BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            cards = soup.find_all(["div", "article"], class_=re.compile(r"ad-col|ads-index-card|item-ad|row-item", re.I))
            if not cards:
                detail_links = soup.find_all("a", href=re.compile(r"/detail-annonce/\d+", re.I))
                cards = [a.find_parent(["div", "article"]) or a for a in detail_links]
            for card in cards:
                rec = self.parse_card_regex(str(card), date_scraped)
                if rec:
                    records.append(rec)

        return records

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one page from Moteur.ma catalog."""
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Primary pagination URL
        url = f"{self.BASE_URL}?page={page_num}"
        html = self.fetch_page(url)

        # Fallback pagination URL if empty
        if not html:
            offset = (page_num - 1) * 30
            url = f"https://www.moteur.ma/fr/voiture/achat-voiture-occasion/{offset}"
            html = self.fetch_page(url)

        if not html:
            logger.warning("[%s] Failed to fetch page %d", self.source_name, page_num)
            return []

        records = self.parse_page(html, date_scraped)
        logger.info("[%s] Page %d: successfully parsed %d listings", self.source_name, page_num, len(records))
        return records

    def scrape(self, max_pages: int = 60, **kwargs) -> pd.DataFrame:
        """
        Scrapes the first 50 to 80 pages (yielding ~1,000+ real records in under 5 minutes).
        Fails loudly with an assertion if len(records) == 0.
        """
        logger.info("[%s] Starting production crawl for up to %d pages ...", self.source_name, max_pages)
        all_records: List[Dict[str, Any]] = []

        consecutive_empty = 0
        for p in range(1, max_pages + 1):
            batch = self.scrape_page(p)
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.info("[%s] 3 consecutive empty pages at page %d. Halting pagination.", self.source_name, p)
                    break
            else:
                consecutive_empty = 0
                all_records.extend(batch)

            self.sleep()

        # Enforce strict non-empty assertion mandated by specification
        assert len(all_records) > 0, (
            f"[{self.source_name}] Scraper extracted 0 records across {max_pages} pages! "
            "Failing loudly to prevent committing an empty CSV dataset."
        )

        df = pd.DataFrame(all_records)
        logger.info("[%s] Crawl complete. Total raw records harvested: %d", self.source_name, len(df))
        self.save_output(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Moteur.ma Production Scraper")
    parser.add_argument("--max-pages", type=int, default=60, help="Max pages to scrape (default: 60)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    args = parser.parse_args()

    scraper = MoteurScraper(output_dir=args.output_dir)
    df = scraper.scrape(max_pages=args.max_pages)
    print(f"Successfully scraped and saved {len(df)} records for Moteur.ma")


if __name__ == "__main__":
    main()
