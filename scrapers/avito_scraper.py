#!/usr/bin/env python3
"""
Avito.ma Dedicated Used Car Scraper for Autohouse.ma MLOps Pipeline
-------------------------------------------------------------------
Implements:
- Search matrix: Letters A to Z x Years [2022, 2023, 2024, 2025, 2026]
- Next.js __NEXT_DATA__ JSON extraction with robust BeautifulSoup DOM fallback
- Full Cahier des Charges schema alignment (24 standard fields)
- Anti-bot defense using curl_cffi (Chrome TLS impersonation) or rotating requests headers
- Rate limiting with randomized delays (1-3 seconds)
- Automated storage in data/raw/avito_YYYY-MM-DD.parquet / .csv
"""

import argparse
import datetime
import json
import logging
import os
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("avito_scraper")

# HTTP Client setup: prefer curl_cffi for TLS fingerprint impersonation, fallback to requests
USE_CURL_CFFI = False
try:
    from curl_cffi import requests as curl_requests
    USE_CURL_CFFI = True
    logger.info("Using curl_cffi with Chrome TLS impersonation")
except ImportError:
    import requests
    logger.info("curl_cffi not installed, using standard requests with browser headers")

from bs4 import BeautifulSoup
import pandas as pd

# Target schema mandated by Autohouse.ma Cahier des Charges
SCHEMA_FIELDS = [
    "listing_id",
    "url",
    "source",
    "date_posted",
    "date_scraped",
    "title_raw",
    "brand",
    "model",
    "trim",
    "year",
    "mileage_km",
    "fuel_type",
    "transmission",
    "fiscal_power_cv",
    "customs_status",
    "condition",
    "owners_count",
    "doors_count",
    "seller_type",
    "city",
    "region",
    "price_mad",
    "photos_count",
    "description_raw",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
]


class AvitoScraper:
    BASE_URL = "https://www.avito.ma/fr/maroc/voitures-%C3%A0_vendre"

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_min: float = 1.0,
        delay_max: float = 3.0,
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self.seen_ids: Set[str] = set()
        self.session = self._init_session()

    def _init_session(self):
        if USE_CURL_CFFI:
            s = curl_requests.Session(impersonate="chrome124")
        else:
            s = requests.Session()
        s.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.avito.ma/",
                "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": random.choice(USER_AGENTS),
            }
        )
        return s

    def _sleep(self):
        duration = random.uniform(self.delay_min, self.delay_max)
        time.sleep(duration)

    def fetch_page(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        for attempt in range(1, self.max_retries + 1):
            try:
                self.session.headers["User-Agent"] = random.choice(USER_AGENTS)
                response = self.session.get(url, params=params, timeout=20)

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.debug("Page not found (404): %s", url)
                    return None
                elif response.status_code in (403, 429):
                    wait = 3 * attempt + random.uniform(1.0, 3.0)
                    logger.warning(
                        "Status %d on %s. Backing off for %.1fs (attempt %d/%d)",
                        response.status_code,
                        url,
                        wait,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "Unexpected status %d for %s (attempt %d/%d)",
                        response.status_code,
                        url,
                        attempt,
                        self.max_retries,
                    )
            except Exception as e:
                logger.warning(
                    "Network error fetching %s: %s (attempt %d/%d)",
                    url,
                    e,
                    attempt,
                    self.max_retries,
                )
                time.sleep(2 * attempt)
        return None

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

    def _extract_ad_from_json(
        self, ad: Dict[str, Any], date_scraped: str
    ) -> Optional[Dict[str, Any]]:
        """Parse structured ad item from Next.js payload into Cahier des Charges schema."""
        listing_id = str(ad.get("id") or ad.get("ad_id") or "")
        if not listing_id or listing_id in self.seen_ids:
            return None
        self.seen_ids.add(listing_id)

        # Basic fields
        title_raw = ad.get("subject") or ad.get("title") or ""
        url = ad.get("url") or ad.get("canonical_url") or ""
        if url and not url.startswith("http"):
            url = urljoin("https://www.avito.ma", url)
        if not url:
            url = f"https://www.avito.ma/fr/ad_{listing_id}.htm"

        # Price
        price_val = None
        raw_price = ad.get("price")
        if isinstance(raw_price, dict):
            price_val = raw_price.get("value") or raw_price.get("amount")
        elif raw_price is not None:
            try:
                price_val = float(raw_price)
            except (ValueError, TypeError):
                price_val = None

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
        if not date_posted:
            date_posted = date_scraped[:10]

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

        year = None
        year_raw = (
            params.get("regdate")
            or params.get("annee_modele")
            or params.get("year")
            or params.get("annee")
        )
        if year_raw:
            m = re.search(r"(\d{4})", str(year_raw))
            if m:
                year = float(m.group(1))

        mileage_km = None
        km_raw = (
            params.get("mileage")
            or params.get("kilometrage")
            or params.get("mileage_km")
        )
        if km_raw:
            m = re.search(r"(\d+[\d\s]*)", str(km_raw).replace("\xa0", " "))
            if m:
                try:
                    mileage_km = float(re.sub(r"\s+", "", m.group(1)))
                except ValueError:
                    mileage_km = None

        fuel_type = (
            params.get("fuel")
            or params.get("carburant")
            or params.get("fuel_type")
            or ""
        )
        transmission = (
            params.get("gearbox")
            or params.get("boite_de_vitesses")
            or params.get("transmission")
            or ""
        )
        fiscal_power_cv = (
            params.get("horse_power")
            or params.get("puissance_fiscale")
            or params.get("fiscal_power")
            or ""
        )
        customs_status = (
            params.get("customs")
            or params.get("statut_douanier")
            or params.get("dedouanee")
            or ""
        )
        condition = (
            params.get("condition")
            or params.get("etat")
            or ""
        )
        owners_count = (
            params.get("first_owner")
            or params.get("nombre_de_mains")
            or params.get("owners_count")
            or ""
        )

        doors_count = None
        doors_raw = (
            params.get("doors")
            or params.get("nombre_de_portes")
            or params.get("doors_count")
        )
        if doors_raw:
            m = re.search(r"(\d+)", str(doors_raw))
            if m:
                doors_count = float(m.group(1))

        return {
            "listing_id": listing_id,
            "url": url,
            "source": "avito",
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
            "fiscal_power_cv": str(fiscal_power_cv) if fiscal_power_cv else "",
            "customs_status": customs_status,
            "condition": condition,
            "owners_count": str(owners_count) if owners_count else "",
            "doors_count": doors_count,
            "seller_type": seller_type,
            "city": city,
            "region": region,
            "price_mad": float(price_val) if price_val is not None else None,
            "photos_count": photos_count,
            "description_raw": description_raw,
        }

    def _extract_ads_from_dom(self, html: str, date_scraped: str) -> List[Dict[str, Any]]:
        """Fallback extraction using BeautifulSoup when __NEXT_DATA__ is unavailable."""
        soup = BeautifulSoup(html, "html.parser")
        items = []

        # Find car listing links
        links = soup.find_all("a", href=re.compile(r"/voitures(_d_occasion|)/.*_\d+\.htm"))
        seen_urls = set()

        for a in links:
            href = a.get("href", "")
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)

            # Extract listing_id from URL
            m_id = re.search(r"_(\d+)\.htm", href)
            listing_id = m_id.group(1) if m_id else ""
            if not listing_id or listing_id in self.seen_ids:
                continue
            self.seen_ids.add(listing_id)

            full_url = urljoin("https://www.avito.ma", href)
            card = a.find_parent("div") or a

            title_elem = card.find(["h2", "h3", "span", "p"])
            title_raw = title_elem.get_text(strip=True) if title_elem else a.get_text(strip=True)

            # Price extraction
            price_mad = None
            price_match = re.search(
                r"(\d[\d\s\xa0]*)\s*(DH|MAD|DHS)", card.get_text(), re.IGNORECASE
            )
            if price_match:
                clean_p = re.sub(r"[\s\xa0]", "", price_match.group(1))
                try:
                    price_mad = float(clean_p)
                except ValueError:
                    price_mad = None

            # Year extraction
            year = None
            year_match = re.search(r"\b(20[0-2]\d)\b", card.get_text())
            if year_match:
                year = float(year_match.group(1))

            # Mileage extraction
            mileage_km = None
            km_match = re.search(
                r"(\d[\d\s\xa0]*)\s*km", card.get_text(), re.IGNORECASE
            )
            if km_match:
                clean_km = re.sub(r"[\s\xa0]", "", km_match.group(1))
                try:
                    mileage_km = float(clean_km)
                except ValueError:
                    mileage_km = None

            # Fuel type detection
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

            items.append(
                {
                    "listing_id": listing_id,
                    "url": full_url,
                    "source": "avito",
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
            )

        return items

    def _find_ads_recursive(self, obj: Any) -> List[Dict[str, Any]]:
        """Traverse arbitrary JSON to find lists of ad dictionaries."""
        ads = []
        if isinstance(obj, dict):
            # Check if this dictionary represents an ad
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

    def scrape_query(
        self, query: str, year: int, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Query Avito for a specific letter and year."""
        params = {
            "q": query,
            "year_min": year,
            "year_max": year,
            "o": page,
        }
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = self.fetch_page(self.BASE_URL, params=params)
        if not html:
            return []

        results = []
        next_data = self._extract_next_data(html)
        if next_data:
            ad_objects = self._find_ads_recursive(next_data)
            logger.debug(
                "Found %d ad candidates in __NEXT_DATA__ for query='%s', year=%d",
                len(ad_objects),
                query,
                year,
            )
            for ad_dict in ad_objects:
                parsed = self._extract_ad_from_json(ad_dict, date_scraped)
                if parsed:
                    # Enforce year if missing
                    if parsed["year"] is None:
                        parsed["year"] = float(year)
                    results.append(parsed)

        # Fallback to DOM parsing if Next.js data gave zero ads
        if not results:
            dom_ads = self._extract_ads_from_dom(html, date_scraped)
            for parsed in dom_ads:
                if parsed["year"] is None:
                    parsed["year"] = float(year)
                results.append(parsed)

        return results

    def run_matrix(
        self,
        letters: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        max_pages: int = 1,
    ) -> pd.DataFrame:
        """Run the iteration matrix over Letters x Years."""
        letters = letters or list(string.ascii_uppercase)
        years = years or [2022, 2023, 2024, 2025, 2026]

        logger.info(
            "Starting Avito scraping iteration matrix: %d letters x %d years (%d combinations)",
            len(letters),
            len(years),
            len(letters) * len(years),
        )

        all_records: List[Dict[str, Any]] = []

        total_combinations = len(letters) * len(years)
        current = 0

        for year in years:
            for letter in letters:
                current += 1
                logger.info(
                    "[%d/%d] Scraping query='%s', year=%d ...",
                    current,
                    total_combinations,
                    letter,
                    year,
                )

                for p in range(1, max_pages + 1):
                    batch = self.scrape_query(query=letter, year=year, page=p)
                    all_records.extend(batch)
                    logger.info(
                        "  -> Found %d listings (total collected: %d)",
                        len(batch),
                        len(all_records),
                    )
                    self._sleep()

        df = pd.DataFrame(all_records)
        if df.empty:
            logger.warning("No listings collected. Creating empty DataFrame with schema.")
            df = pd.DataFrame(columns=SCHEMA_FIELDS)
        else:
            # Ensure all schema fields exist
            for col in SCHEMA_FIELDS:
                if col not in df.columns:
                    df[col] = None
            df = df[SCHEMA_FIELDS].drop_duplicates(subset=["listing_id"])

        self.save_output(df)
        return df

    def save_output(self, df: pd.DataFrame) -> None:
        """Save collected data to data/raw/avito_YYYY-MM-DD.parquet and .csv."""
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        parquet_path = self.output_dir / f"avito_{today_str}.parquet"
        csv_path = self.output_dir / f"avito_{today_str}.csv"

        # Save Parquet
        try:
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            logger.info("Saved %d records to Parquet: %s", len(df), parquet_path)
        except Exception as e:
            logger.error("Failed to save Parquet: %s", e)

        # Save CSV
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("Saved %d records to CSV: %s", len(df), csv_path)
        except Exception as e:
            logger.error("Failed to save CSV: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Avito.ma Scraping Pipeline")
    parser.add_argument(
        "--letters",
        type=str,
        default=",".join(string.ascii_uppercase),
        help="Comma-separated letters to iterate over (default: A-Z)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2022,2023,2024,2025,2026",
        help="Comma-separated years (default: 2022,2023,2024,2025,2026)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages per letter/year combination (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=1.0,
        help="Minimum randomized delay in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=3.0,
        help="Maximum randomized delay in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run limited test query (letters A-B, year 2024) for quick validation",
    )

    args = parser.parse_args()

    if args.dry_run:
        letters = ["A", "B"]
        years = [2024]
        logger.info("DRY RUN mode activated: letters=%s, years=%s", letters, years)
    else:
        letters = [l.strip().upper() for l in args.letters.split(",") if l.strip()]
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    scraper = AvitoScraper(
        output_dir=args.output_dir,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )
    df = scraper.run_matrix(letters=letters, years=years, max_pages=args.max_pages)
    logger.info("Avito scraping completed successfully. Total unique listings: %d", len(df))


if __name__ == "__main__":
    main()
