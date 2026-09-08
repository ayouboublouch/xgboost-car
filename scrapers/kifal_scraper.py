#!/usr/bin/env python3
"""
Kifal-Auto.ma Dedicated Used Car Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------------
Inherits from BaseScraper:
- Parses inspected used car listings from Kifal-Auto.ma
- Full Cahier des Charges schema alignment (24 standard fields)
- Certified inspected data source (high feature completeness)
- Saves to data/raw/kifal_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import json
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
    from scrapers.base import BaseScraper
except ImportError:
    from base import BaseScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.kifal")


class KifalScraper(BaseScraper):
    source_name = "kifal"
    BASE_URL = "https://kifal-auto.ma/voitures-occasion"

    def parse_card(self, card_elem: BeautifulSoup, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse individual car card from Kifal-Auto listing page."""
        link = card_elem.find("a", href=re.compile(r"/voiture[s]?-occasion/|/ad/|/detail/"))
        if not link:
            link = card_elem.find("a", href=True)
        if not link:
            return None

        href = link.get("href", "")
        m_id = re.search(r"[-_/](\d+)(?:\.html|/|$)", href)
        listing_id = m_id.group(1) if m_id else re.sub(r"\W+", "_", href.strip("/").split("/")[-1])
        if not listing_id:
            return None

        full_url = urljoin("https://kifal-auto.ma", href)

        # Title
        title_elem = card_elem.find(["h2", "h3", "h4", "div"], class_=re.compile(r"title|name|modele", re.I))
        title_raw = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)

        tokens = title_raw.split()
        brand = tokens[0] if tokens else ""
        model = tokens[1] if len(tokens) > 1 else ""

        # Price
        price_elem = card_elem.find(["div", "span", "p"], class_=re.compile(r"price|prix", re.I))
        price_mad = price_elem.get_text(strip=True) if price_elem else None
        if not price_mad:
            m_p = re.search(r"(\d[\d\s\xa0]*)\s*(DH|MAD)", card_elem.get_text(), re.I)
            if m_p:
                price_mad = m_p.group(1)

        card_text = card_elem.get_text(separator=" ", strip=True)

        # Year
        year_match = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", card_text)
        year = year_match.group(1) if year_match else None

        # Mileage
        km_match = re.search(r"(\d[\d\s\xa0]*)\s*km", card_text, re.I)
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
        city_elem = card_elem.find(["span", "div"], class_=re.compile(r"city|ville|loc", re.I))
        if city_elem:
            city = city_elem.get_text(strip=True)

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
            "customs_status": "WW au Maroc",
            "condition": "Inspecté / Certifié",
            "owners_count": "",
            "doors_count": None,
            "seller_type": "Professionnel",
            "city": city,
            "region": "",
            "price_mad": price_mad,
            "photos_count": 0.0,
            "description_raw": "",
        }
        return self.validate_and_format_record(raw_record)

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one catalog page from Kifal-Auto."""
        url = f"{self.BASE_URL}?page={page_num}" if page_num > 1 else self.BASE_URL
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = self.fetch_page(url)
        if not html:
            return []

        # Check for Next.js or JSON-LD first
        soup = BeautifulSoup(html, "html.parser")
        items = []

        script = soup.find("script", id="__NEXT_DATA__")
        if script and script.string:
            try:
                data = json.loads(script.string)
                # Walk through objects for car lists
                props = data.get("props", {}).get("pageProps", {})
                cars = props.get("cars") or props.get("vehicles") or props.get("listings") or []
                for c in cars:
                    if isinstance(c, dict):
                        l_id = str(c.get("id") or c.get("slug") or "")
                        if l_id:
                            rec = {
                                "listing_id": l_id,
                                "url": urljoin("https://kifal-auto.ma", c.get("url") or f"/voiture-occasion/{l_id}"),
                                "source": self.source_name,
                                "date_posted": date_scraped[:10],
                                "date_scraped": date_scraped,
                                "title_raw": f"{c.get('make', '')} {c.get('model', '')}",
                                "brand": c.get("make") or c.get("brand") or "",
                                "model": c.get("model") or "",
                                "trim": c.get("trim") or "",
                                "year": c.get("year"),
                                "mileage_km": c.get("mileage") or c.get("mileage_km"),
                                "fuel_type": c.get("fuel") or c.get("fuel_type") or "",
                                "transmission": c.get("transmission") or c.get("gearbox") or "",
                                "fiscal_power_cv": str(c.get("fiscal_power") or c.get("power") or ""),
                                "customs_status": "WW au Maroc",
                                "condition": "Inspecté / Certifié",
                                "owners_count": "",
                                "doors_count": c.get("doors"),
                                "seller_type": "Professionnel",
                                "city": c.get("city") or "",
                                "region": "",
                                "price_mad": c.get("price") or c.get("price_mad"),
                                "photos_count": float(len(c.get("images", []))),
                                "description_raw": c.get("description") or "",
                            }
                            formatted = self.validate_and_format_record(rec)
                            if formatted:
                                items.append(formatted)
            except Exception as e:
                logger.debug("[%s] JSON parse failed: %s", self.source_name, e)

        # Fallback to HTML DOM cards
        if not items:
            cards = soup.find_all(["div", "article"], class_=re.compile(r"car-card|vehicle-card|listing-item|card", re.I))
            for card in cards:
                rec = self.parse_card(card, date_scraped)
                if rec:
                    items.append(rec)

        logger.info("[%s] Page %d: parsed %d listings", self.source_name, page_num, len(items))
        return items

    def scrape(self, max_pages: int = 15, **kwargs) -> pd.DataFrame:
        """Crawl Kifal used car listings up to max_pages."""
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
    parser = argparse.ArgumentParser(description="Kifal-Auto Scraping Provider")
    parser.add_argument("--max-pages", type=int, default=15, help="Max pages to scrape (default: 15)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    scraper = KifalScraper(output_dir=args.output_dir)
    scraper.scrape(max_pages=args.max_pages)


if __name__ == "__main__":
    main()
