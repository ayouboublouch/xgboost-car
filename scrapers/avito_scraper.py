#!/usr/bin/env python3
"""
Avito.ma Dedicated Used Car Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------
Inherits from BaseScraper:
- Next.js __NEXT_DATA__ JSON extraction with robust BeautifulSoup DOM fallback
- Full Cahier des Charges schema alignment (24 standard fields)
- TLS impersonation & randomized delays
- Matrix iteration across letters (A-Z) and years (2022-2026)
"""

import argparse
import datetime
import json
import logging
import re
import string
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd

try:
    from scrapers.base import BaseScraper, SCHEMA_FIELDS
except ImportError:
    from base import BaseScraper, SCHEMA_FIELDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.avito")


class AvitoScraper(BaseScraper):
    source_name = "avito"
    BASE_URL = "https://www.avito.ma/fr/maroc/voitures-%C3%A0_vendre"

    def _extract_next_data(self, html: str) -> Optional[Dict[str, Any]]:
        """Extract Next.js __NEXT_DATA__ JSON payload from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            script = soup.find("script", id="__NEXT_DATA__")
            if script and script.string:
                return json.loads(script.string)
        except Exception as e:
            logger.debug("Failed to extract __NEXT_DATA__: %s", e)
        return None

    def _parse_params_dict(self, params_list_or_dict: Any) -> Dict[str, Any]:
        """Normalize ad parameters from Next.js data."""
        parsed = {}
        if isinstance(params_list_or_dict, list):
            for item in params_list_or_dict:
                if isinstance(item, dict):
                    key = item.get("id") or item.get("name") or item.get("key")
                    val = (
                        item.get("valueLabel")
                        or item.get("value")
                        or item.get("val")
                        or item.get("title")
                    )
                    if key:
                        parsed[str(key).lower()] = val
        elif isinstance(params_list_or_dict, dict):
            for k, v in params_list_or_dict.items():
                if isinstance(v, dict):
                    val = v.get("valueLabel") or v.get("value") or v.get("val")
                else:
                    val = v
                parsed[str(k).lower()] = val
        return parsed

    def _extract_ad_from_json(self, ad: Dict[str, Any], date_scraped: str) -> Optional[Dict[str, Any]]:
        """Parse structured ad item from Next.js payload into raw dictionary."""
        listing_id = str(ad.get("id") or ad.get("ad_id") or "")
        if not listing_id:
            return None

        title_raw = ad.get("subject") or ad.get("title") or ""
        url = ad.get("url") or ad.get("canonical_url") or ""
        if url and not url.startswith("http"):
            url = urljoin("https://www.avito.ma", url)
        if not url:
            url = f"https://www.avito.ma/fr/ad_{listing_id}.htm"

        # Price
        raw_price = ad.get("price")
        price_val = None
        if isinstance(raw_price, dict):
            price_val = raw_price.get("value") or raw_price.get("amount")
        elif raw_price is not None:
            price_val = raw_price

        # Date posted
        date_posted = None
        raw_date = ad.get("date") or ad.get("time") or ad.get("created_at")
        if raw_date:
            try:
                if isinstance(raw_date, (int, float)):
                    date_posted = datetime.datetime.fromtimestamp(raw_date).strftime("%Y-%m-%d")
                elif isinstance(raw_date, str):
                    date_posted = raw_date[:10]
            except Exception:
                date_posted = None

        # Location
        location = ad.get("location") or {}
        city = ""
        region = ""
        if isinstance(location, dict):
            city = location.get("city") or location.get("cityName") or ""
            region = location.get("region") or location.get("regionName") or ""
        elif isinstance(location, str):
            city = location

        # Seller type
        user_info = ad.get("user") or {}
        seller_type = "Particulier"
        if isinstance(user_info, dict):
            if user_info.get("type") in ("pro", "shop", "dealer") or user_info.get("is_pro"):
                seller_type = "Professionnel"
        elif ad.get("is_shop") or ad.get("is_pro"):
            seller_type = "Professionnel"

        # Photos count
        images = ad.get("images") or ad.get("photos") or []
        photos_count = float(len(images)) if isinstance(images, list) else 0.0

        # Description
        description_raw = ad.get("body") or ad.get("description") or ""

        # Parse parameters
        params = self._parse_params_dict(ad.get("params") or ad.get("attributes") or [])

        brand = params.get("brand") or params.get("marque") or ""
        model = params.get("model") or params.get("modele") or ""
        trim = params.get("trim") or params.get("finition") or ""
        year = params.get("regdate") or params.get("annee_modele") or params.get("year") or params.get("annee")
        mileage_km = params.get("mileage") or params.get("kilometrage") or params.get("mileage_km")
        fuel_type = params.get("fuel") or params.get("carburant") or params.get("fuel_type") or ""
        transmission = params.get("gearbox") or params.get("boite_de_vitesses") or params.get("transmission") or ""
        fiscal_power_cv = params.get("horse_power") or params.get("puissance_fiscale") or params.get("fiscal_power") or ""
        customs_status = params.get("customs") or params.get("statut_douanier") or params.get("dedouanee") or ""
        condition = params.get("condition") or params.get("etat") or ""
        owners_count = params.get("first_owner") or params.get("nombre_de_mains") or params.get("owners_count") or ""
        doors_count = params.get("doors") or params.get("nombre_de_portes") or params.get("doors_count")

        raw_dict = {
            "listing_id": listing_id,
            "url": url,
            "source": self.source_name,
            "date_posted": date_posted,
            "date_scraped": date_scraped,
            "title_raw": title_raw,
            "brand": brand,
            "model": model,
            "trim": trim,
            "year": year,
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "fiscal_power_cv": fiscal_power_cv,
            "customs_status": customs_status,
            "condition": condition,
            "owners_count": owners_count,
            "doors_count": doors_count,
            "seller_type": seller_type,
            "city": city,
            "region": region,
            "price_mad": price_val,
            "photos_count": photos_count,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_dict)

    def _extract_ads_from_dom(self, html: str, date_scraped: str) -> List[Dict[str, Any]]:
        """Fallback extraction using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        items = []

        links = soup.find_all("a", href=re.compile(r"/voitures(_d_occasion|)/.*_\d+\.htm"))
        seen_urls = set()

        for a in links:
            href = a.get("href", "")
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)

            m_id = re.search(r"_(\d+)\.htm", href)
            listing_id = m_id.group(1) if m_id else ""
            if not listing_id:
                continue

            full_url = urljoin("https://www.avito.ma", href)
            card = a.find_parent("div") or a

            title_elem = card.find(["h2", "h3", "span", "p"])
            title_raw = title_elem.get_text(strip=True) if title_elem else a.get_text(strip=True)

            price_match = re.search(r"(\d[\d\s\xa0]*)\s*(DH|MAD|DHS)", card.get_text(), re.IGNORECASE)
            price_mad = price_match.group(1) if price_match else None

            year_match = re.search(r"\b(20[0-2]\d)\b", card.get_text())
            year = year_match.group(1) if year_match else None

            km_match = re.search(r"(\d[\d\s\xa0]*)\s*km", card.get_text(), re.IGNORECASE)
            mileage_km = km_match.group(1) if km_match else None

            card_txt = card.get_text().lower()
            fuel_type = ""
            if "diesel" in card_txt:
                fuel_type = "Diesel"
            elif "essence" in card_txt:
                fuel_type = "Essence"
            elif "hybride" in card_txt:
                fuel_type = "Hybride"
            elif "electrique" in card_txt or "électrique" in card_txt:
                fuel_type = "Electrique"

            transmission = ""
            if "automatique" in card_txt:
                transmission = "Automatique"
            elif "manuelle" in card_txt:
                transmission = "Manuelle"

            record = {
                "listing_id": listing_id,
                "url": full_url,
                "source": self.source_name,
                "date_posted": date_scraped[:10],
                "date_scraped": date_scraped,
                "title_raw": title_raw,
                "brand": "",
                "model": "",
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
                "city": "",
                "region": "",
                "price_mad": price_mad,
                "photos_count": 0.0,
                "description_raw": "",
            }
            formatted = self.validate_and_format_record(record)
            if formatted:
                items.append(formatted)

        return items

    def _find_ads_recursive(self, obj: Any) -> List[Dict[str, Any]]:
        """Traverse arbitrary JSON to find lists of ad dictionaries."""
        ads = []
        if isinstance(obj, dict):
            if ("subject" in obj or "title" in obj) and ("id" in obj or "ad_id" in obj):
                ads.append(obj)
            else:
                for k, v in obj.items():
                    if k in ("ads", "items", "listings", "list") and isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict) and ("id" in item or "ad_id" in item):
                                ads.append(item)
                    elif isinstance(v, (dict, list)):
                        ads.extend(self._find_ads_recursive(v))
        elif isinstance(obj, list):
            for item in obj:
                ads.extend(self._find_ads_recursive(item))
        return ads

    def scrape_query(self, query: str, year: int, page: int = 1) -> List[Dict[str, Any]]:
        """Query Avito for letter + year combination."""
        params = {"q": query, "year_min": year, "year_max": year, "o": page}
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = self.fetch_page(self.BASE_URL, params=params)
        if not html:
            return []

        results = []
        next_data = self._extract_next_data(html)
        if next_data:
            ad_objects = self._find_ads_recursive(next_data)
            for ad_dict in ad_objects:
                parsed = self._extract_ad_from_json(ad_dict, date_scraped)
                if parsed:
                    if parsed["year"] is None:
                        parsed["year"] = float(year)
                    results.append(parsed)

        if not results:
            dom_ads = self._extract_ads_from_dom(html, date_scraped)
            for parsed in dom_ads:
                if parsed["year"] is None:
                    parsed["year"] = float(year)
                results.append(parsed)

        return results

    def scrape(self, max_pages: int = 1, letters: Optional[List[str]] = None, years: Optional[List[int]] = None, **kwargs) -> pd.DataFrame:
        """Run iteration matrix over Letters x Years."""
        letters = letters or list(string.ascii_uppercase)
        years = years or [2022, 2023, 2024, 2025, 2026]

        logger.info(
            "[%s] Starting matrix: %d letters x %d years (%d combinations), max_pages=%d",
            self.source_name,
            len(letters),
            len(years),
            len(letters) * len(years),
            max_pages,
        )

        all_records = []
        for year in years:
            for letter in letters:
                for p in range(1, max_pages + 1):
                    batch = self.scrape_query(query=letter, year=year, page=p)
                    all_records.extend(batch)
                    self.sleep()

        df = pd.DataFrame(all_records)
        self.save_output(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Avito.ma Scraping Provider")
    parser.add_argument("--max-pages", type=int, default=1, help="Max pages per query (default: 1)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run (A-B, 2024)")
    args = parser.parse_args()

    scraper = AvitoScraper(output_dir=args.output_dir)
    if args.dry_run:
        scraper.scrape(max_pages=1, letters=["A", "B"], years=[2024])
    else:
        scraper.scrape(max_pages=args.max_pages)


if __name__ == "__main__":
    main()
