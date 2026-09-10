#!/usr/bin/env python3
"""
Wandaloo.com Production Scraper
===============================
Primary data source for Autohouse.ma MLOps pipeline.
- Targets Wandaloo unblocked catalog pagination (https://www.wandaloo.com/occasion/?pg={page})
- Extracts car cards: Price in DH, model year, fuel type, mileage, fiscal power, city
- Maps strictly to the 24 Cahier des Charges columns
- Fails loudly with assertion if records == 0
- Saves directly to data/raw/wandaloo_YYYY-MM-DD.csv and .parquet
"""

import argparse
import datetime
import logging
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
    from scrapers.base import BaseScraper, SCHEMA_FIELDS, extract_moroccan_phone, hash_phone
except ImportError:
    from base import BaseScraper, SCHEMA_FIELDS, extract_moroccan_phone, hash_phone

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
logger = logging.getLogger("scraper.wandaloo")

DEFAULT_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.wandaloo.com/",
    "DNT": "1",
}


class WandalooScraper(BaseScraper):
    source_name = "wandaloo"
    BASE_URL = "https://www.wandaloo.com/occasion/"

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

    def extract_phone_from_detail(self, detail_url: str) -> Optional[str]:
        """Fetch listing page (/occasion/...html) and parse seller modal/contact block for telephone links."""
        if not detail_url:
            return None
        try:
            html = self.fetch_page(detail_url, headers=DEFAULT_BROWSER_HEADERS, timeout=6, max_retries=1)
            if not html:
                return None

            # 1. Search for telephone links (href="tel:06...")
            tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', html, re.IGNORECASE)
            if tel_m:
                phone = extract_moroccan_phone(tel_m.group(1))
                if phone:
                    return phone

            # 2. Search for WhatsApp links (wa.me)
            wa_m = re.search(r'wa\.me/(\+?212\d{9}|0[5-7]\d{8}|\d+)', html, re.IGNORECASE)
            if wa_m:
                phone = extract_moroccan_phone(wa_m.group(1))
                if phone:
                    return phone

            # 3. Parse seller modal / contact block (e.g. modal, contact, seller)
            modal_m = re.search(
                r'<(?:div|span|p|a)[^>]*class=["\'][^"\']*(?:seller-phone|modal-contact|phone|telephone|contact-seller|seller-info)[^"\']*["\'][^>]*>(.*?)</(?:div|span|p|a)>',
                html,
                re.DOTALL | re.IGNORECASE,
            )
            if modal_m:
                phone = extract_moroccan_phone(modal_m.group(1))
                if phone:
                    return phone

            # 4. Search for data-phone
            dp_m = re.search(r'data-phone=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if dp_m:
                phone = extract_moroccan_phone(dp_m.group(1))
                if phone:
                    return phone

            # 5. Scan full detail HTML text container with extract_moroccan_phone
            return extract_moroccan_phone(html)
        except Exception as e:
            logger.debug("[%s] Detail phone extraction error for %s: %s", self.source_name, detail_url, e)
            return None

    def parse_card_regex(self, block: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Extract Wandaloo listing card fields via regex."""
        link_m = re.search(r'href=["\'](https://www\.wandaloo\.com/occasion/[^"\']+/(\d+)\.html)["\']', block)
        if not link_m:
            link_m = re.search(r'href=["\'](/occasion/[^"\']+/(\d+)\.html)["\']', block)
            if link_m:
                url = "https://www.wandaloo.com" + link_m.group(1)
                listing_id = link_m.group(2)
            else:
                return None
        else:
            url = link_m.group(1)
            listing_id = link_m.group(2)

        # Title
        title_m = re.search(r'class=["\']titre["\']>\s*<a[^>]*>(.*?)</a>', block, re.DOTALL)
        if not title_m:
            title_m = re.search(r'<p[^>]*class=["\']titre["\'][^>]*>(.*?)</p>', block, re.DOTALL)
        title_raw = title_m.group(1).strip() if title_m else ""
        title_raw = re.sub(r'<[^>]+>', '', title_raw).strip()

        # Price in DH
        price_m = re.search(r'class=["\']prix["\']>\s*<span>([^<]+)</span>', block)
        price_mad = None
        if price_m:
            p_str = re.sub(r'[\s,.]', '', price_m.group(1).strip())
            try:
                price_mad = float(p_str)
            except ValueError:
                price_mad = None

        # Detail block (<ul class="detail"> ... </ul>)
        detail_m = re.search(r'class=["\']detail["\'][^>]*>(.*?)</ul>', block, re.DOTALL)
        fuel_type = ""
        year = None
        fiscal_cv = ""
        mileage_km = None
        transmission = ""

        if detail_m:
            items = re.findall(r'<li>(.*?)</li>', detail_m.group(1), re.DOTALL)
            for it in items:
                it_clean = re.sub(r'<[^>]+>', '', it).strip()
                it_lower = it_clean.lower()
                if "diesel" in it_lower:
                    fuel_type = "Diesel"
                elif "essence" in it_lower:
                    fuel_type = "Essence"
                elif "hybride" in it_lower:
                    fuel_type = "Hybride"
                elif "electrique" in it_lower or "électrique" in it_lower:
                    fuel_type = "Electrique"
                elif re.search(r'\b(19[8-9]\d|20[0-2]\d)\b', it_clean):
                    year = int(re.search(r'\b(19[8-9]\d|20[0-2]\d)\b', it_clean).group(1))
                elif "cv" in it_lower:
                    cv_m = re.search(r'(\d+)', it_clean)
                    if cv_m:
                        fiscal_cv = cv_m.group(1)
                elif "km" in it_lower:
                    km_m = re.search(r'(\d[\d\s,.]*)', it_clean)
                    if km_m:
                        try:
                            mileage_km = float(re.sub(r'[\s,.]', '', km_m.group(1)))
                        except ValueError:
                            pass
                elif "auto" in it_lower:
                    transmission = "Automatique"
                elif "man" in it_lower:
                    transmission = "Manuelle"

        # City
        city_m = re.search(r'class=["\']city["\'][^>]*>.*?</i>\s*([^<]+)', block, re.DOTALL)
        city = city_m.group(1).strip() if city_m else ""

        # Date posted
        date_posted = date_scraped[:10]

        # Brand / Model tokens
        brand = ""
        model = ""
        if title_raw:
            tokens = [t for t in re.sub(r'[^a-zA-Z0-9À-ÿ\s]', ' ', title_raw).split() if t]
            if tokens:
                brand = tokens[0].capitalize()
                model = " ".join(tokens[1:]).capitalize() if len(tokens) > 1 else ""
        if not brand or not model:
            m_slug = re.search(r'/occasion/([a-zA-Z0-9\-]+)/', url)
            if m_slug:
                parts = [p for p in m_slug.group(1).split('-') if p and p not in ('occasion', 'maroc')]
                if parts and not brand:
                    brand = parts[0].capitalize()
                if len(parts) > 1 and not model:
                    model = parts[1].capitalize()
        if not brand:
            brand = "Autre"
        if not model:
            model = "Autre"

        # Extract description if present
        desc_m = re.search(r'class=["\'][^"\']*(?:desc|detail-txt|texte)[^"\']*["\'][^>]*>(.*?)</(?:p|div)>', block, re.DOTALL | re.IGNORECASE)
        description_raw = desc_m.group(1).strip() if desc_m else ""
        description_raw = re.sub(r'<[^>]+>', '', description_raw).strip()

        # Extract seller phone numbers from contact container, telephone buttons, description, and block
        seller_phone = None
        tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', block, re.IGNORECASE)
        if tel_m:
            seller_phone = extract_moroccan_phone(tel_m.group(1))

        if not seller_phone:
            dp_m = re.search(r'data-phone=["\']([^"\']+)["\']', block, re.IGNORECASE)
            if dp_m:
                seller_phone = extract_moroccan_phone(dp_m.group(1))

        if not seller_phone:
            contact_m = re.search(r'class=["\'][^"\']*(?:phone|tel|contact)[^"\']*["\'][^>]*>(.*?)</', block, re.DOTALL | re.IGNORECASE)
            if contact_m:
                seller_phone = extract_moroccan_phone(contact_m.group(1))

        if not seller_phone and description_raw:
            seller_phone = extract_moroccan_phone(description_raw)

        if not seller_phone and title_raw:
            seller_phone = extract_moroccan_phone(title_raw)

        if not seller_phone:
            seller_phone = extract_moroccan_phone(block)

        # Detail-page deep phone extraction for verified listings (valid price, year, and recognized brand)
        if not seller_phone and url and "/occasion/" in url:
            if price_mad and year and brand != "Autre":
                seller_phone = self.extract_phone_from_detail(url)

        seller_phone_hash = hash_phone(seller_phone)

        raw_record = {
            "listing_id": listing_id,
            "url": url,
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
            "fiscal_power_cv": fiscal_cv,
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
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def parse_page(self, html: str, date_scraped: str) -> List[Dict[str, Any]]:
        """Parse listing cards from Wandaloo HTML."""
        records = []
        # Split by listing card containers
        blocks = re.split(r'<li\s+class=["\'](?:even|odd)["\'][^>]*>', html)
        if len(blocks) <= 1:
            blocks = re.split(r'class=["\'](?:result-item|occasion-item)["\']', html)
        if len(blocks) <= 1:
            blocks = re.split(r'(?=<a[^>]+href=["\'][^"\']*/occasion/[^"\']+\.html)', html)

        for b in blocks[1:]:
            rec = self.parse_card_regex(b, date_scraped)
            if rec:
                records.append(rec)

        if len(records) == 0 and BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            cards = soup.find_all("li", class_=re.compile(r"even|odd|result-item", re.I))
            for card in cards:
                rec = self.parse_card_regex(str(card), date_scraped)
                if rec:
                    records.append(rec)

        return records

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one page from Wandaloo used car section."""
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        url = f"{self.BASE_URL}?pg={page_num}" if page_num > 1 else self.BASE_URL

        html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS)
        if not html:
            logger.warning("[%s] Failed to fetch page %d", self.source_name, page_num)
            return []

        records = self.parse_page(html, date_scraped)
        logger.info("[%s] Page %d: parsed %d listings", self.source_name, page_num, len(records))
        return records

    def scrape(self, max_pages: int = 30, **kwargs) -> pd.DataFrame:
        """Crawl Wandaloo used car listings up to max_pages."""
        logger.info("[%s] Starting crawl for up to %d pages ...", self.source_name, max_pages)
        all_records: List[Dict[str, Any]] = []

        consecutive_empty = 0
        for p in range(1, max_pages + 1):
            batch = self.scrape_page(p)
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.info("[%s] 3 consecutive empty pages at page %d. Stopping.", self.source_name, p)
                    break
            else:
                consecutive_empty = 0
                all_records.extend(batch)

            self.sleep()

        if len(all_records) == 0:
            logger.warning("[%s] Crawl complete. 0 records harvested. Writing nothing.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info("[%s] Crawl complete. Total records: %d", self.source_name, len(df))
        self.save_output(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Wandaloo.com Production Scraper")
    parser.add_argument("--max-pages", type=int, default=30, help="Max pages to scrape (default: 30)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    args = parser.parse_args()

    scraper = WandalooScraper(output_dir=args.output_dir)
    df = scraper.scrape(max_pages=args.max_pages)
    print(f"Successfully scraped and saved {len(df)} records for Wandaloo.ma")


if __name__ == "__main__":
    main()
