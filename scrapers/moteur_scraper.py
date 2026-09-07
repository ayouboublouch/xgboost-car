#!/usr/bin/env python3
"""
Moteur.ma Dedicated Used Car Scraper for Autohouse.ma MLOps Pipeline
--------------------------------------------------------------------
Inherits from BaseScraper:
- Parses listing cards and detail elements from Moteur.ma
- Full Cahier des Charges schema alignment (24 standard fields)
- Handles source='moteur' and notes seller_type reliability constraints
- Saves to data/raw/moteur_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import logging
import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd

from scrapers.base import BaseScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.moteur")


class MoteurScraper(BaseScraper):
    source_name = "moteur"
    BASE_URL = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/"

    def parse_card(self, card_elem: BeautifulSoup, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse individual listing item from Moteur.ma listing page."""
        # Find detail link
        link = card_elem.find("a", href=re.compile(r"/detail-annonce/(\d+)/"))
        if not link:
            link = card_elem.find("a", href=re.compile(r"detail-annonce"))
        if not link:
            return None

        href = link.get("href", "")
        m_id = re.search(r"/detail-annonce/(\d+)/", href)
        if not m_id:
            m_id = re.search(r"/(\d+)/", href)
        listing_id = m_id.group(1) if m_id else ""
        if not listing_id:
            return None

        full_url = urljoin("https://www.moteur.ma", href)

        # Title
        title_elem = card_elem.find(["h3", "h2", "div", "span"], class_=re.compile(r"title|titre", re.I))
        title_raw = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)

        # Split brand / model if possible
        tokens = title_raw.split()
        brand = tokens[0] if tokens else ""
        model = " ".join(tokens[1:]) if len(tokens) > 1 else ""

        # Price
        price_elem = card_elem.find(string=re.compile(r"(\d[\d\s\xa0]*)\s*(DH|MAD)", re.I))
        price_mad = price_elem if price_elem else None
        if not price_mad:
            p_elem = card_elem.find(["div", "span", "p"], class_=re.compile(r"price|prix", re.I))
            if p_elem:
                price_mad = p_elem.get_text(strip=True)

        # Details block: Year, Mileage, Fuel, Transmission, City
        card_text = card_elem.get_text(separator=" ", strip=True)

        # Year
        year_match = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", card_text)
        year = year_match.group(1) if year_match else None

        # Mileage
        km_match = re.search(r"(\d[\d\s\xa0]*)\s*(km|kms)", card_text, re.I)
        mileage_km = km_match.group(1) if km_match else None

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

        # Fiscal power
        fiscal_cv = ""
        fp_match = re.search(r"(\d+)\s*(cv|ch)", c_lower)
        if fp_match:
            fiscal_cv = fp_match.group(1)

        # City
        city = ""
        city_elem = card_elem.find(["span", "div", "li"], class_=re.compile(r"city|ville|location", re.I))
        if city_elem:
            city = city_elem.get_text(strip=True)

        # Seller type: Note known Moteur bias in Cahier des Charges (often defaults to Professionnel)
        seller_type = "Professionnel" if "pro" in c_lower or "garage" in c_lower else "Particulier"

        raw_record = {
            "listing_id": listing_id,
            "url": full_url,
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
            "fiscal_power_cv": fiscal_cv,
            "customs_status": "",
            "condition": "",
            "owners_count": "",
            "doors_count": None,
            "seller_type": seller_type,
            "city": city,
            "region": "",
            "price_mad": price_mad,
            "photos_count": 0.0,
            "description_raw": "",
        }
        return self.validate_and_format_record(raw_record)

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one catalog page from Moteur.ma."""
        offset = (page_num - 1) * 30
        url = f"{self.BASE_URL}{offset}" if offset > 0 else self.BASE_URL
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = self.fetch_page(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        items = []

        # Find ad container elements
        cards = soup.find_all("div", class_=re.compile(r"row-item|item-ad|block-item|detail-annonce", re.I))
        if not cards:
            # Fallback to article or general container
            cards = soup.find_all(["div", "article"], class_=re.compile(r"annonce|occasion", re.I))

        for card in cards:
            rec = self.parse_card(card, date_scraped)
            if rec:
                items.append(rec)

        logger.info("[%s] Page %d: parsed %d listings", self.source_name, page_num, len(items))
        return items

    def scrape(self, max_pages: int = 15, **kwargs) -> pd.DataFrame:
        """Crawl Moteur.ma catalog up to max_pages."""
        logger.info("[%s] Starting scrape for up to %d pages ...", self.source_name, max_pages)
        all_records = []

        for p in range(1, max_pages + 1):
            batch = self.scrape_page(p)
            if not batch:
                logger.info("[%s] No further listings found on page %d. Stopping.", self.source_name, p)
                break
            all_records.extend(batch)
            self.sleep()

        df = pd.DataFrame(all_records)
        self.save_output(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Moteur.ma Scraping Provider")
    parser.add_argument("--max-pages", type=int, default=15, help="Max pages to scrape (default: 15)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    scraper = MoteurScraper(output_dir=args.output_dir)
    scraper.scrape(max_pages=args.max_pages)


if __name__ == "__main__":
    main()
