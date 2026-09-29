#!/usr/bin/env python3
"""
MarocAnnonces Dedicated Production Scraper for Autohouse.ma MLOps Pipeline
--------------------------------------------------------------------------
Inherits from BaseScraper:
- Targets MarocAnnonces used car marketplace (https://www.marocannonces.com/categorie/314/Voitures-occasion.html)
- Extracts card parameters and deep-inspects listing detail pages
- Decodes obfuscated seller phone numbers from phone_number.php base64 tokens
- Full Cahier des Charges schema alignment (24 standard fields + seller_phone & seller_phone_hash)
- Persists to data/raw/marocannonces_YYYY-MM-DD.parquet / .csv
"""

import argparse
import base64
import datetime
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote, urljoin

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
import requests

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
logger = logging.getLogger("scraper.marocannonces")

MONTH_MAP: Dict[str, int] = {
    "jan": 1, "fev": 2, "fév": 2, "mar": 3, "avr": 4, "mai": 5, "jui": 6, "juin": 6,
    "juil": 7, "aou": 8, "août": 8, "aout": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12, "déc": 12,
}


def parse_marocannonces_date(raw_date: Optional[str]) -> str:
    """Parse Moroccan relative date strings (e.g., '29 Sep-12:01', 'Aujourd'hui', 'Hier')."""
    now = datetime.datetime.now()
    if not raw_date or pd.isna(raw_date):
        return now.strftime("%Y-%m-%d")
    s = str(raw_date).lower().strip()
    if "aujourd" in s:
        return now.strftime("%Y-%m-%d")
    if "hier" in s:
        return (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.search(r"(\d{1,2})\s+([a-zéûà]+)", s)
    if m:
        try:
            day = int(m.group(1))
            mon_prefix = m.group(2)[:4].strip()
            month = None
            for k, v in MONTH_MAP.items():
                if mon_prefix.startswith(k):
                    month = v
                    break
            if month:
                year = now.year
                if month > now.month:
                    year -= 1
                return datetime.date(year, month, day).strftime("%Y-%m-%d")
        except Exception:
            pass
    return now.strftime("%Y-%m-%d")


class MarocAnnoncesScraper(BaseScraper):
    source_name = "marocannonces"
    BASE_URL = "https://www.marocannonces.com/categorie/314/Voitures-occasion"

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
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.marocannonces.com/",
            }
        )

    def fetch_index_page(self, page_num: int) -> Optional[str]:
        """Fetch index page HTML with retries."""
        if page_num <= 1:
            url = f"{self.BASE_URL}.html"
        else:
            url = f"{self.BASE_URL}/{page_num}.html"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, verify=False, timeout=12)
                if resp.status_code == 200 and resp.text:
                    return resp.text
                elif resp.status_code == 404:
                    logger.info("[%s] Page %d returned 404. End of catalog reached.", self.source_name, page_num)
                    return None
            except Exception as e:
                logger.debug("[%s] Index fetch attempt %d failed for page %d: %s", self.source_name, attempt, page_num, e)
            self.sleep()
        return None

    def fetch_detail_page(self, detail_url: str) -> Optional[str]:
        """Fetch listing detail page HTML with automatic URL encoding for non-ASCII characters."""
        if not detail_url:
            return None
        safe_url = quote(detail_url, safe=":/?=&%")
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(safe_url, verify=False, timeout=10)
                if resp.status_code == 200 and resp.text:
                    return resp.text
            except Exception as e:
                logger.debug("[%s] Detail fetch attempt %d failed for %s: %s", self.source_name, attempt, safe_url, e)
            self.sleep()
        return None

    def parse_listing(self, card_or_html: Any, date_scraped: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse card element, fetch detail page, and harmonize fields into Cahier des Charges schema."""
        if not date_scraped:
            date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        card = card_or_html
        a_tag = card.find("a", href=re.compile(r"annonce/(\d+)"))
        if not a_tag:
            return None

        href = a_tag.get("href", "")
        m_id = re.search(r"annonce/(\d+)", href)
        if not m_id:
            return None
        listing_id = m_id.group(1)

        # Skip if already encountered in this execution run
        if listing_id in self.seen_ids:
            return None

        canonical_url = href if href.startswith("http") else urljoin("https://www.marocannonces.com/", href.lstrip("/"))

        # Card baseline fields
        price_tag = card.find(class_=re.compile(r"price", re.I))
        price_card = self.clean_numeric(price_tag.get_text(strip=True)) if price_tag else None

        loc_tag = card.find(class_=re.compile(r"location", re.I))
        city_card = loc_tag.get_text(strip=True) if loc_tag else ""

        title_card = a_tag.get("title", "") or ""
        if not title_card:
            h3_tag = card.find("h3")
            if h3_tag:
                title_card = h3_tag.get_text(" ", strip=True)

        time_tag = card.find(class_=re.compile(r"time|date", re.I))
        raw_date_str = time_tag.get_text(" ", strip=True) if time_tag else ""
        date_posted = parse_marocannonces_date(raw_date_str)

        # Deep fetch detail page for granular specs
        detail_html = self.fetch_detail_page(canonical_url)
        detail_soup = BeautifulSoup(detail_html, "html.parser") if detail_html else None

        # Fallback values from card if detail fetch fails
        brand = None
        model = None
        year = None
        mileage_km = None
        fuel_type = None
        fiscal_power_cv = None
        seller_phone = None
        seller_type = "Particulier"
        photos_count = 1.0
        description_raw = title_card
        title_raw = title_card
        city = city_card

        if detail_soup:
            h1 = detail_soup.find("h1")
            if h1:
                title_raw = h1.get_text(" ", strip=True) or title_card

            # Extract specs from info and extraQuestionName ULs
            specs: Dict[str, str] = {}
            for ul in detail_soup.find_all("ul", class_=re.compile(r"info|extraQuestionName", re.I)):
                for sli in ul.find_all("li"):
                    txt = sli.get_text(" ", strip=True)
                    if ":" in txt:
                        k, v = txt.split(":", 1)
                        specs[k.strip().lower()] = v.strip()

            # Map specs dictionary
            brand = specs.get("marque")
            model = specs.get("modèle") or specs.get("modele")
            year = self.clean_year(specs.get("année") or specs.get("annee") or specs.get("année-modèle"))
            mileage_km = self.clean_numeric(specs.get("kilométrage") or specs.get("kilometrage"))
            fuel_type = specs.get("carburant")
            fiscal_power_cv = specs.get("puissance") or specs.get("puissance fiscale")
            if specs.get("ville"):
                city = specs.get("ville")

            # Date posted from detail if available
            if "publiée le" in specs:
                date_posted = parse_marocannonces_date(specs["publiée le"])
            elif "publiee le" in specs:
                date_posted = parse_marocannonces_date(specs["publiee le"])

            # Decode obfuscated phone number from base64 image parameter
            img_phone = detail_soup.find("img", src=re.compile(r"phone_number\.php\?phone=([A-Za-z0-9+/=]+)"))
            if img_phone:
                m_p = re.search(r"phone=([A-Za-z0-9+/=]+)", img_phone.get("src", ""))
                if m_p:
                    try:
                        raw_b64 = m_p.group(1)
                        decoded_phone = base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
                        seller_phone = extract_moroccan_phone(decoded_phone)
                    except Exception:
                        pass

            # Description
            desc_tag = detail_soup.find("div", class_="description") or detail_soup.find("div", class_="block")
            if desc_tag:
                description_raw = desc_tag.get_text(" ", strip=True)

            # Photos count
            photo_imgs = detail_soup.find_all("img", src=re.compile(r"user_images/"))
            if photo_imgs:
                photos_count = float(len(photo_imgs))

            # Seller type
            dl_tag = detail_soup.find("dl")
            if dl_tag and ("professionnel" in dl_tag.get_text().lower() or "boutique" in dl_tag.get_text().lower()):
                seller_type = "Professionnel"

        # Fallback phone from description/title if not found in base64 token
        if not seller_phone:
            seller_phone = extract_moroccan_phone(description_raw) or extract_moroccan_phone(title_raw)

        seller_phone_hash = hash_phone(seller_phone)

        # Transmission extraction from specs or description
        text_lower = f"{title_raw} {description_raw}".lower()
        transmission = ""
        if any(w in text_lower for w in ["automatique", " bva", "boite auto", "boîte auto"]):
            transmission = "Automatique"
        elif any(w in text_lower for w in ["manuelle", " manuel", " bvm"]):
            transmission = "Manuelle"

        # Fuel normalization
        fuel_norm = ""
        if fuel_type:
            f_low = fuel_type.lower()
            if "diesel" in f_low or "gazole" in f_low or "gasoil" in f_low:
                fuel_norm = "Diesel"
            elif "essence" in f_low:
                fuel_norm = "Essence"
            elif "hybride" in f_low or "hybrid" in f_low:
                fuel_norm = "Hybride"
            elif "electrique" in f_low or "électrique" in f_low:
                fuel_norm = "Electrique"

        # Fallback year from title if missing in specs
        if not year:
            year = self.clean_year(title_raw)

        # Customs status & condition
        customs_status = ""
        if "dédouanée" in text_lower or "dedouanee" in text_lower or "dédouané" in text_lower:
            customs_status = "Dédouanée"
        elif "non dédouanée" in text_lower:
            customs_status = "Non dédouanée"
        elif "ww au maroc" in text_lower or " ww" in text_lower:
            customs_status = "WW au Maroc"

        condition = "Occasion"
        if " neuf" in text_lower or "neuve" in text_lower:
            condition = "Neuf"

        owners_count = ""
        if "1er main" in text_lower or "premiere main" in text_lower or "première main" in text_lower or "1ère main" in text_lower:
            owners_count = "1"

        raw_record = {
            "listing_id": listing_id,
            "url": canonical_url,
            "source": self.source_name,
            "date_posted": date_posted,
            "date_scraped": date_scraped,
            "title_raw": title_raw,
            "brand": brand,
            "model": model,
            "trim": "",
            "year": year,
            "mileage_km": mileage_km,
            "fuel_type": fuel_norm,
            "transmission": transmission,
            "fiscal_power_cv": fiscal_power_cv or "",
            "customs_status": customs_status,
            "condition": condition,
            "owners_count": owners_count,
            "doors_count": None,
            "seller_type": seller_type,
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": infer_moroccan_region(city),
            "price_mad": price_card,
            "photos_count": photos_count,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def scrape(
        self,
        max_pages: int = 50,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Crawl MarocAnnonces from start_page for max_pages."""
        logger.info(
            "[%s] Starting crawl: start_page=%d, max_pages=%d ...",
            self.source_name,
            start_page,
            max_pages,
        )
        all_records = []
        consecutive_empty = 0
        seen_batch_ids: Set[str] = set()

        end_page = start_page + max_pages

        for page in range(start_page, end_page):
            html = self.fetch_index_page(page)
            if not html:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    logger.info("[%s] 2 consecutive empty pages encountered. Exiting.", self.source_name, page)
                    break
                continue

            soup = BeautifulSoup(html, "html.parser")
            cards = []
            for li in soup.find_all("li"):
                if li.find("a", href=re.compile(r"annonce/(\d+)")):
                    cards.append(li)

            if not cards:
                logger.info("[%s] Page %d contained 0 listing cards. Catalog bound reached.", self.source_name, page)
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                continue

            # Detect catalog loopback (MarocAnnonces loops back to page 1 on out-of-bounds)
            page_ids = set()
            for c in cards:
                a = c.find("a", href=re.compile(r"annonce/(\d+)"))
                if a:
                    m = re.search(r"annonce/(\d+)", a.get("href", ""))
                    if m:
                        page_ids.add(m.group(1))

            # If every listing on this page was already scraped in this session, we have looped
            if page_ids and page_ids.issubset(seen_batch_ids):
                logger.info(
                    "[%s] Page %d listings are an exact subset of previously scraped listings (catalog loopback). Stopping crawl.",
                    self.source_name,
                    page,
                )
                break

            consecutive_empty = 0
            page_records = 0

            for card in cards:
                record = self.parse_listing(card)
                if record:
                    all_records.append(record)
                    seen_batch_ids.add(record["listing_id"])
                    page_records += 1
                self.sleep()

            logger.info(
                "[%s] Page %d/%d complete: parsed %d valid listings (total: %d)",
                self.source_name,
                page,
                end_page - 1,
                page_records,
                len(all_records),
            )

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
    parser = argparse.ArgumentParser(description="MarocAnnonces Production Scraper")
    parser.add_argument("--start-page", type=int, default=1, help="Initial page to scrape (default: 1)")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to scrape (default: 50)")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output CSV filename")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    args = parser.parse_args()

    scraper = MarocAnnoncesScraper(output_dir=args.output_dir)
    df = scraper.scrape(
        start_page=args.start_page,
        max_pages=args.max_pages,
        output_filename=args.output_filename,
    )
    print(f"Successfully scraped and saved {len(df)} records for MarocAnnonces")


if __name__ == "__main__":
    main()
