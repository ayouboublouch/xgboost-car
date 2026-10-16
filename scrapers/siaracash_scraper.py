#!/usr/bin/env python3
"""
SiaraCash.ma Dedicated Used Car Scraper for Autohouse.ma MLOps Pipeline
-----------------------------------------------------------------------
Inherits from BaseScraper:
- Parses verified used car marketplace listings from SiaraCash.ma
- Full Cahier des Charges schema alignment (24 standard fields)
- Handles source='siaracash'
- Saves to data/raw/siaracash_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import logging
import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
import pandas as pd

try:
    from scrapers.base import (
        BaseScraper,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
    )
except ImportError:
    from base import (
        BaseScraper,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.siaracash")


class SiaraCashScraper(BaseScraper):
    source_name = "siaracash"
    BASE_URL = "https://siaracash.ma/voiture-occasion-maroc"
    DOMAIN = "https://siaracash.ma"
    DEFAULT_PHONE = "0655138709"

    def parse_card(self, card_elem: BeautifulSoup, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse individual listing item from SiaraCash catalog page."""
        href = card_elem.get("href", "")
        if not href or "/voiture/" not in href:
            return None

        full_url = urljoin(self.DOMAIN, href)
        m_id = re.search(r"/([^/]+)$", href.rstrip("/"))
        listing_id = m_id.group(1) if m_id else ""
        if not listing_id:
            return None

        parts = [t.strip() for t in card_elem.stripped_strings if t.strip()]
        if not parts:
            return None

        full_text = " ".join(parts)

        # 1. Price extraction (e.g. 410 000 DH or 410\u202f000 DH)
        m_price = re.search(r"(\d[\d\s\u202f\xa0]*)\s*(?:DH|MAD)", full_text, re.I)
        price_mad = float(re.sub(r"[^\d]", "", m_price.group(1))) if m_price else None
        if not price_mad or price_mad <= 0:
            return None

        # 2. Year extraction
        m_year = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", full_text)
        year = float(m_year.group(1)) if m_year else None

        # 3. Mileage extraction
        m_km = re.search(r"\b(\d{1,3}(?:[\s\u202f\xa0]\d{3})*|\d+)\s*km\b", full_text, re.I)
        mileage_km = float(re.sub(r"[^\d]", "", m_km.group(1))) if m_km else None

        # 4. Fuel type
        fuel_type = ""
        c_lower = full_text.lower()
        if "hybride" in c_lower or "hybrid" in c_lower:
            fuel_type = "Hybride"
        elif "electrique" in c_lower or "électrique" in c_lower or "ev" in c_lower:
            fuel_type = "Electrique"
        elif "diesel" in c_lower or "dci" in c_lower or "tdi" in c_lower or "hdi" in c_lower:
            fuel_type = "Diesel"
        elif "essence" in c_lower or "tsi" in c_lower or "puretech" in c_lower:
            fuel_type = "Essence"

        # 5. Transmission
        transmission = ""
        if "automatique" in c_lower or "auto" in c_lower or "dsg" in c_lower or "edc" in c_lower:
            transmission = "Automatique"
        elif "manuelle" in c_lower or "manuel" in c_lower or "bvm" in c_lower:
            transmission = "Manuelle"

        # 6. City & Region resolution
        city = ""
        slug_match = re.search(r"/voiture/([^/]+)/", href)
        if slug_match:
            slug_words = slug_match.group(1).split("-")
            if slug_words:
                city = slug_words[-1].capitalize()
        if not city and parts:
            city = parts[-1].capitalize()

        region = infer_moroccan_region(city)

        # 7. Brand, Model & Trim extraction
        raw_tokens = []
        skip_words = {
            "particuliers", "particulier", "professionnel", "certifié", "certifie",
            "non certifié", "non certifie", "360°", "360", "à la une", "a la une",
            "sans imperfection", "finançable", "financable", "/mois", "dh", "mad",
        }
        for p in parts:
            p_clean = p.lower().strip()
            if p_clean in skip_words or "dh" in p_clean or "km" in p_clean:
                continue
            if re.match(r"^(19\d\d|20\d\d)$", p_clean):
                continue
            if p_clean == city.lower():
                continue
            raw_tokens.append(p)

        brand_cand = raw_tokens[0] if len(raw_tokens) > 0 else ""
        model_cand = raw_tokens[1] if len(raw_tokens) > 1 else ""
        trim_cand = " ".join(raw_tokens[2:]) if len(raw_tokens) > 2 else ""

        title_raw = f"{brand_cand} {model_cand} {int(year) if year else ''} {trim_cand}".strip()
        brand, model, trim = clean_brand_and_model(brand_cand, model_cand, title_raw=title_raw, trim=trim_cand)

        seller_phone = self.DEFAULT_PHONE
        seller_phone_hash = hash_phone(seller_phone)
        seller_type = "Particulier" if "particulier" in c_lower else "Professionnel"

        raw_record = {
            "listing_id": listing_id,
            "url": full_url,
            "source": self.source_name,
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": title_raw,
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
            "seller_type": seller_type,
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": 5.0,
            "description_raw": full_text,
        }
        return self.validate_and_format_record(raw_record)

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one catalog page from SiaraCash."""
        url = f"{self.BASE_URL}?page={page_num}" if page_num > 1 else self.BASE_URL
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            html = self.fetch_page(url)
        except Exception as e:
            logger.warning("[%s] Network exception on page %d: %s. Continuing gracefully.", self.source_name, page_num, e)
            return []

        if not html:
            return []

        if BeautifulSoup is None:
            logger.warning("[%s] BeautifulSoup not installed.", self.source_name)
            return []

        soup = BeautifulSoup(html, "html.parser")
        items = []

        # Find all vehicle card anchor tags
        cards = soup.find_all("a", href=lambda h: h and "/voiture/" in h)
        seen_on_page = set()

        for card in cards:
            href = card.get("href", "")
            if href in seen_on_page:
                continue
            seen_on_page.add(href)
            rec = self.parse_card(card, date_scraped)
            if rec:
                items.append(rec)

        logger.info("[%s] Page %d: parsed %d valid listings from %d card elements", self.source_name, page_num, len(items), len(seen_on_page))
        return items

    def scrape(
        self,
        max_pages: int = 15,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Crawl SiaraCash used car catalog with graceful exception and timeout handling."""
        logger.info("[%s] Starting scrape for up to %d pages (start: %d) ...", self.source_name, max_pages, start_page)
        all_records = []

        try:
            for p in range(start_page, start_page + max_pages):
                batch = self.scrape_page(p)
                if not batch:
                    logger.info("[%s] No further listings found on page %d. Stopping.", self.source_name, p)
                    break
                all_records.extend(batch)
                self.sleep()
        except Exception as e:
            logger.warning("[%s] Non-fatal scrape interruption: %s. Preserving %d collected records.", self.source_name, e, len(all_records))

        if not all_records:
            logger.warning("[%s] Scrape yielded 0 records.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if "listing_id" in df.columns:
            df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)

        logger.info("[%s] Scraping finished with %d verified listings.", self.source_name, len(df))
        self.save_output(df, custom_filename=output_filename)
        return df


def main():
    parser = argparse.ArgumentParser(description="SiaraCash Scraping Provider")
    parser.add_argument("--max-pages", type=int, default=15, help="Max pages to scrape (default: 15)")
    parser.add_argument("--start-page", type=int, default=1, help="Start page (default: 1)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output filename")
    args = parser.parse_args()

    scraper = SiaraCashScraper(output_dir=args.output_dir)
    scraper.scrape(max_pages=args.max_pages, start_page=args.start_page, output_filename=args.output_filename)


if __name__ == "__main__":
    main()
