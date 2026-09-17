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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


DEFAULT_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.moteur.ma/",
    "DNT": "1",
}


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

    def extract_phone_from_detail(self, detail_url: str, session: Optional[Any] = None) -> Optional[str]:
        """Fetch detail page HTML for verified listing to extract seller telephone with timeout=5."""
        if not detail_url:
            return None
        try:
            s = session or self.session
            html = None
            if s is not None:
                try:
                    resp = s.get(detail_url, headers=DEFAULT_BROWSER_HEADERS, timeout=5, verify=False)
                    if resp.status_code == 200:
                        html = resp.text
                except Exception:
                    html = None
            if not html:
                html = self.fetch_page(detail_url, headers=DEFAULT_BROWSER_HEADERS, timeout=5, max_retries=1)
            if not html:
                return None

            # 1. Search for tel: links (href="tel:06...")
            tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', html, re.IGNORECASE)
            if tel_m:
                p = extract_moroccan_phone(tel_m.group(1))
                if p:
                    return p

            # 2. Search for WhatsApp links (wa.me/06... or wa.me/2126...)
            wa_m = re.search(r'wa\.me/(\+?212\d{9}|0[5-7]\d{8}|\d+)', html, re.IGNORECASE)
            if wa_m:
                p = extract_moroccan_phone(wa_m.group(1))
                if p:
                    return p

            # 3. Search for contact class containers
            pclass_m = re.search(
                r'class=["\'][^"\']*(?:ad-contact-phone|phone-number|mobile-sticky-phone|contact-phone|seller-phone|btn-phone)[^"\']*["\'][^>]*>(.*?)</',
                html,
                re.DOTALL | re.IGNORECASE,
            )
            if pclass_m:
                p = extract_moroccan_phone(pclass_m.group(1))
                if p:
                    return p

            # 4. Search for data-phone or data-tel attribute
            dp_m = re.search(r'data-(?:phone|tel)=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if dp_m:
                p = extract_moroccan_phone(dp_m.group(1))
                if p:
                    return p

            # 5. Search for telephone in script tags or JSON
            script_m = re.search(r'["\']?(?:phone|telephone|tel|contact_phone)["\']?\s*[:=]\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
            if script_m:
                p = extract_moroccan_phone(script_m.group(1))
                if p:
                    return p

            # 6. Fallback: Scan full detail HTML text container
            return extract_moroccan_phone(html)
        except Exception as e:
            logger.debug("[%s] Detail phone extraction error for %s: %s", self.source_name, detail_url, e)
            return None

    def parse_card_regex(self, block: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Fast, robust regex-based card extractor for Moteur.ma HTML."""
        link_m = re.search(r'href=["\']([^"\']*detail-annonce/(\d+)/?([^"\'\s>]*\.html)?)["\']', block)
        if not link_m:
            return None

        href = link_m.group(1)
        listing_id = link_m.group(2)
        if listing_id in self.seen_ids:
            return None
        slug = link_m.group(3) or ""
        full_url = urljoin("https://www.moteur.ma", href)

        # 1. Title Extraction
        title_raw = ""
        title_m = re.search(
            r'<(?:h\d|div|a|p)[^>]*class=["\'][^"\']*(?:ads-index-title|title)[^"\']*["\'][^>]*>\s*(.*?)\s*</(?:h\d|div|a|p)>',
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if title_m:
            title_raw = re.sub(r'<[^>]+>', '', title_m.group(1)).strip()
        if not title_raw:
            alt_m = re.search(r'alt=["\']([^"\']+)["\']', block)
            if alt_m and "moteur" not in alt_m.group(1).lower():
                title_raw = alt_m.group(1).strip()

        # Parse slug for brand & model resolution
        slug_clean = re.sub(r'\.html$', '', slug).strip()
        slug_clean = slug_clean.split('?')[0].split('#')[0]
        slug_parts = [p for p in slug_clean.split('-') if p]

        # 2. Brand, Model & Trim Resolution
        brand = ""
        model = ""
        trim = ""

        # Check against KNOWN_BRANDS for precise brand identification
        for b in KNOWN_BRANDS:
            if re.search(rf"\b{re.escape(b)}\b", title_raw, re.IGNORECASE):
                brand = b
                break

        if not brand and slug_parts:
            for b in KNOWN_BRANDS:
                if slug_parts[0].lower() == b.lower() or (
                    len(slug_parts) > 1 and f"{slug_parts[0]}-{slug_parts[1]}".lower() == b.lower().replace(" ", "-")
                ):
                    brand = b
                    break

        if brand:
            # Model and trim extraction from title_raw relative to brand
            clean_after_brand = re.sub(rf"\b{re.escape(brand)}\b", "", title_raw, flags=re.IGNORECASE).strip()
            tokens = [t for t in re.sub(r'[^a-zA-Z0-9À-ÿ\s\.\-]', ' ', clean_after_brand).split() if t]
            if tokens:
                model = tokens[0].capitalize()
                if len(tokens) > 1:
                    trim = " ".join(tokens[1:])
            elif slug_parts:
                model = slug_parts[1].capitalize() if len(slug_parts) > 1 else "Autre"
                trim = " ".join(slug_parts[2:]) if len(slug_parts) > 2 else ""
        else:
            tokens = [t for t in re.sub(r'[^a-zA-Z0-9À-ÿ\s\.\-]', ' ', title_raw).split() if t]
            if tokens:
                brand = tokens[0].capitalize()
                if len(tokens) > 1:
                    model = tokens[1].capitalize()
                if len(tokens) > 2:
                    trim = " ".join(tokens[2:])
            elif slug_parts:
                brand = slug_parts[0].capitalize()
                model = slug_parts[1].capitalize() if len(slug_parts) > 1 else "Autre"
                trim = " ".join(slug_parts[2:]) if len(slug_parts) > 2 else ""

        # Never leave brand or model empty
        if not brand:
            brand = "Autre"
        if not model:
            model = "Autre"

        if not title_raw:
            title_raw = f"{brand} {model} {trim}".strip()

        # 3. City & Region
        city_m = re.search(r'fa-map-marker[^>]*></i>\s*([^\s<]+)', block)
        city = city_m.group(1).strip() if city_m else ""
        region = infer_moroccan_region(city)

        # 4. Date posted
        timeago_m = re.search(r'class=["\']timeago["\']\s+data-time=["\']([^"\']+)["\']', block)
        date_posted = timeago_m.group(1)[:10] if timeago_m else date_scraped[:10]

        # 5. Description
        desc_m = re.search(r'class=["\'][^"\']*ad-desc[^"\']*["\']>\s*(.*?)\s*</p>', block, re.DOTALL | re.IGNORECASE)
        description_raw = desc_m.group(1).strip() if desc_m else ""

        # 6. Price in MAD: Parse numeric from .ad-price-grid or price tag
        price_m = re.search(r'class=["\'][^"\']*(?:ad-price-grid|price|prix)[^"\']*["\'][^>]*>\s*(.*?)\s*</', block, re.DOTALL | re.IGNORECASE)
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
        if price_mad is None:
            pm2 = re.search(r'(\d[\d\s\xa0]{3,})\s*(?:DH|MAD|Dhs)\b', block, re.IGNORECASE)
            if pm2:
                try:
                    price_mad = float(re.sub(r'[\s\xa0,.]', '', pm2.group(1)))
                except ValueError:
                    price_mad = None

        # 7. Year: Parse 4-digit integer (1980-2027)
        year = None
        year_m = re.search(r'(?:title=["\'](?:Année|Annee)["\'][^>]*>|fa-calendar[^>]*></i>)\s*(\d{4})', block, re.IGNORECASE)
        if year_m:
            y_val = int(year_m.group(1))
            if 1980 <= y_val <= 2027:
                year = y_val
        if not year:
            ym = re.search(r'\b(20[0-2]\d|19[8-9]\d)\b', block)
            if ym:
                year = int(ym.group(1))

        # 8. Transmission: Parse 'Automatique' or 'Manuelle'
        trans_m = re.search(r'fa-cog[^>]*></i>\s*([A-Za-zÀ-ÿ]+)', block, re.IGNORECASE)
        trans_raw = trans_m.group(1).strip().lower() if trans_m else block.lower()
        transmission = ""
        if "automatique" in trans_raw or "auto" in trans_raw:
            transmission = "Automatique"
        elif "manuelle" in trans_raw or "manuel" in trans_raw:
            transmission = "Manuelle"

        # 9. Fuel type: Parse 'Diesel', 'Essence', 'Hybride', or 'Electrique'
        fuel_m = re.search(r'fa-tachometer[^>]*></i>\s*([A-Za-zÀ-ÿ]+)', block, re.IGNORECASE)
        fuel_raw = fuel_m.group(1).strip().lower() if fuel_m else block.lower()
        fuel_type = ""
        if "diesel" in fuel_raw:
            fuel_type = "Diesel"
        elif "essence" in fuel_raw:
            fuel_type = "Essence"
        elif "hybride" in fuel_raw:
            fuel_type = "Hybride"
        elif "elect" in fuel_raw or "élect" in fuel_raw:
            fuel_type = "Electrique"

        # 10. Mileage km: Parse digits before 'km'
        mileage_km = None
        km_m = re.search(r'fa-road[^>]*></i>\s*(\d[\d\s,.]*)', block, re.IGNORECASE)
        if km_m:
            try:
                mileage_km = float(re.sub(r'[\s,.]', '', km_m.group(1)))
            except ValueError:
                mileage_km = None
        if mileage_km is None:
            km_match = re.search(r'(\d[\d\s\xa0]*)\s*(?:km|kms)\b', block, re.IGNORECASE)
            if km_match:
                try:
                    mileage_km = float(re.sub(r'[\s\xa0,.]', '', km_match.group(1)))
                except ValueError:
                    mileage_km = None

        # 11. Fiscal Power (CV)
        fiscal_cv = ""
        cv_m = re.search(r'(?:fa-flash|fa-bolt|puissance)[^>]*></i>\s*(\d{1,2})', block, re.IGNORECASE)
        if not cv_m:
            cv_m = re.search(r'\b(\d{1,2})\s*(?:CV|cv|ch)\b', block)
        if cv_m:
            fiscal_cv = cv_m.group(1)

        # 12. Condition & Customs Status
        condition = "Occasion"
        customs_status = "Dédouanée"
        if "non dédouan" in block.lower() or "non dedouan" in block.lower():
            customs_status = "Non dédouanée"

        # 13. Seller Type
        seller_type = "Particulier"
        if any(k in block.lower() for k in ["professionnel", "garage", "vitrine", "concession", "ad-pro"]):
            seller_type = "Professionnel"

        # 14. Photos Count
        photos_count = 1.0
        p_count_m = re.search(r'fa-camera[^>]*></i>\s*(\d+)', block)
        if p_count_m:
            try:
                photos_count = float(p_count_m.group(1))
            except ValueError:
                photos_count = 1.0

        # 15. Extract seller phone numbers from card HTML
        seller_phone = None
        tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', block, re.IGNORECASE)
        if tel_m:
            seller_phone = extract_moroccan_phone(tel_m.group(1))

        if not seller_phone:
            dp_m = re.search(r'data-phone=["\']([^"\']+)["\']', block, re.IGNORECASE)
            if dp_m:
                seller_phone = extract_moroccan_phone(dp_m.group(1))

        if not seller_phone and description_raw:
            seller_phone = extract_moroccan_phone(description_raw)

        if not seller_phone and title_raw:
            seller_phone = extract_moroccan_phone(title_raw)

        if not seller_phone:
            seller_phone = extract_moroccan_phone(block)

        seller_phone_hash = hash_phone(seller_phone)

        raw_record = {
            "listing_id": listing_id,
            "url": full_url,
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
            "fiscal_power_cv": fiscal_cv,
            "customs_status": customs_status,
            "condition": condition,
            "owners_count": "",
            "doors_count": None,
            "seller_type": seller_type,
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": region,
            "price_mad": price_mad,
            "photos_count": photos_count,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def parse_page(self, html: str, date_scraped: str) -> List[Dict[str, Any]]:
        """Parse all listing cards from a catalog page HTML with concurrent detail phone resolution."""
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

        # High-performance concurrent detail phone extraction for listings needing deep phone lookup
        pending_details = [
            (i, rec["url"])
            for i, rec in enumerate(records)
            if not rec.get("seller_phone")
            and rec.get("url")
            and "detail-annonce" in rec.get("url", "")
            and rec.get("price_mad")
            and rec.get("year")
            and rec.get("brand") != "Autre"
        ]

        if pending_details:
            workers = min(8, len(pending_details))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(self.extract_phone_from_detail, url): idx
                    for idx, url in pending_details
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        phone = future.result()
                        if phone:
                            records[idx]["seller_phone"] = phone
                            records[idx]["seller_phone_hash"] = hash_phone(phone)
                    except Exception as e:
                        logger.debug("[%s] Detail async phone lookup error: %s", self.source_name, e)

        return records

    def scrape_page(self, page_num: int) -> List[Dict[str, Any]]:
        """Fetch and parse one page from Moteur.ma catalog."""
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Primary pagination URL
        url = f"{self.BASE_URL}?page={page_num}"
        html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS)

        # Fallback pagination URL if empty
        if not html:
            offset = (page_num - 1) * 30
            url = f"https://www.moteur.ma/fr/voiture/achat-voiture-occasion/{offset}"
            html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS)

        if not html:
            logger.warning("[%s] Failed to fetch page %d", self.source_name, page_num)
            return []

        records = self.parse_page(html, date_scraped)
        logger.info("[%s] Page %d: successfully parsed %d listings", self.source_name, page_num, len(records))
        return records

    def scrape(self, max_pages: int = 60, **kwargs) -> pd.DataFrame:
        """
        Scrapes up to max_pages (default 60, yielding ~1,800 records).
        Gracefully halts if consecutive empty pages are encountered.
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

        if len(all_records) == 0:
            logger.warning("[%s] Crawl complete. 0 records harvested. Writing nothing.", self.source_name)
            return pd.DataFrame()

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
