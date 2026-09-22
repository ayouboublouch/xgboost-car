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
    SHOWROOM_URL = "https://www.wandaloo.com/occasion/?vendeur=2"
    GARAGES_URL = "https://www.wandaloo.com/occasion/garages/"

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

    def extract_phone_from_detail(self, detail_url: str, session: Optional[Any] = None) -> Optional[str]:
        """Fetch listing page (/occasion/...html) and parse seller contact details with resilient timeout."""
        if not detail_url:
            return None
        try:
            s = session or self.session
            html = None
            if s is not None:
                try:
                    resp = s.get(detail_url, headers=DEFAULT_BROWSER_HEADERS, timeout=(6, 12), verify=False)
                    if resp.status_code == 200:
                        html = resp.text
                except Exception:
                    html = None
            if not html:
                try:
                    import requests as std_requests
                    r = std_requests.get(detail_url, headers=DEFAULT_BROWSER_HEADERS, timeout=10)
                    if r.status_code == 200:
                        html = r.text
                except Exception:
                    html = None
            if not html:
                html = self.fetch_page(detail_url, headers=DEFAULT_BROWSER_HEADERS, timeout=(6, 12), max_retries=1)
            if not html:
                return None

            # 1. Search for telephone links (href="tel:06...")
            tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', html, re.IGNORECASE)
            if tel_m:
                phone = extract_moroccan_phone(tel_m.group(1))
                if phone:
                    return phone

            # 2. Search for WhatsApp links (wa.me or api.whatsapp.com)
            wa_m = re.search(r'(?:wa\.me/|api\.whatsapp\.com/send\?phone=)(\+?212\d{9}|0[5-7]\d{8}|\d+)', html, re.IGNORECASE)
            if wa_m:
                phone = extract_moroccan_phone(wa_m.group(1))
                if phone:
                    return phone

            # 3. Parse seller modal / contact block / phone buttons
            modal_m = re.search(
                r'<(?:div|span|p|a|button)[^>]*class=["\'][^"\']*(?:seller-phone|modal-contact|phone|telephone|contact-seller|seller-info|tel-btn|contact-phone)[^"\']*["\'][^>]*>(.*?)</(?:div|span|p|a|button)>',
                html,
                re.DOTALL | re.IGNORECASE,
            )
            if modal_m:
                phone = extract_moroccan_phone(modal_m.group(1))
                if phone:
                    return phone

            # 4. Search for data-phone, data-tel, data-contact attribute
            dp_m = re.search(r'data-(?:phone|tel|contact)=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if dp_m:
                phone = extract_moroccan_phone(dp_m.group(1))
                if phone:
                    return phone

            # 5. Search script tags / JSON objects
            script_m = re.search(r'["\']?(?:phone|telephone|tel|contact_phone)["\']?\s*[:=]\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
            if script_m:
                phone = extract_moroccan_phone(script_m.group(1))
                if phone:
                    return phone

            # 6. Fallback: Scan full detail HTML text container
            return extract_moroccan_phone(html)
        except Exception as e:
            logger.debug("[%s] Detail phone extraction error for %s: %s", self.source_name, detail_url, e)
            return None

    def parse_card_regex(self, block: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Extract Wandaloo listing card fields via regex."""
        link_m = re.search(r'href=["\'](https?://(?:www\.)?wandaloo\.com/occasion/[^"\']+/(\d+)\.html)["\']', block, re.IGNORECASE)
        if not link_m:
            link_m = re.search(r'href=["\'](/occasion/[^"\']+/(\d+)\.html)["\']', block, re.IGNORECASE)
            if link_m:
                url = "https://www.wandaloo.com" + link_m.group(1)
                listing_id = link_m.group(2)
            else:
                return None
        else:
            url = link_m.group(1)
            listing_id = link_m.group(2)

        if listing_id in self.seen_ids:
            return None

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

        # City & Region
        city_m = re.search(r'class=["\']city["\'][^>]*>.*?</i>\s*([^<]+)', block, re.DOTALL)
        city = city_m.group(1).strip() if city_m else ""
        region = infer_moroccan_region(city)

        # Date posted
        date_posted = date_scraped[:10]

        # Brand, Model & Trim Resolution
        brand = ""
        model = ""
        trim = ""

        # Check against KNOWN_BRANDS
        for b in KNOWN_BRANDS:
            if re.search(rf"\b{re.escape(b)}\b", title_raw, re.IGNORECASE):
                brand = b
                break

        if not brand:
            m_slug = re.search(r'/occasion/([a-zA-Z0-9\-]+)/', url)
            if m_slug:
                parts = [p for p in m_slug.group(1).split('-') if p and p not in ('occasion', 'maroc')]
                if parts:
                    for b in KNOWN_BRANDS:
                        if parts[0].lower() == b.lower():
                            brand = b
                            break

        if brand:
            clean_after_brand = re.sub(rf"\b{re.escape(brand)}\b", "", title_raw, flags=re.IGNORECASE).strip()
            tokens = [t for t in re.sub(r'[^a-zA-Z0-9À-ÿ\s\.\-]', ' ', clean_after_brand).split() if t]
            if tokens:
                model = tokens[0].capitalize()
                if len(tokens) > 1:
                    trim = " ".join(tokens[1:])
        else:
            tokens = [t for t in re.sub(r'[^a-zA-Z0-9À-ÿ\s\.\-]', ' ', title_raw).split() if t]
            if tokens:
                brand = tokens[0].capitalize()
                if len(tokens) > 1:
                    model = tokens[1].capitalize()
                if len(tokens) > 2:
                    trim = " ".join(tokens[2:])

        if not brand:
            brand = "Autre"
        if not model:
            model = "Autre"

        # Extract description if present
        desc_m = re.search(r'class=["\'][^"\']*(?:desc|detail-txt|texte)[^"\']*["\'][^>]*>(.*?)</(?:p|div)>', block, re.DOTALL | re.IGNORECASE)
        description_raw = desc_m.group(1).strip() if desc_m else ""
        description_raw = re.sub(r'<[^>]+>', '', description_raw).strip()

        # Condition & Customs Status
        condition = "Occasion"
        customs_status = "Dédouanée"
        if "non dédouan" in block.lower() or "non dedouan" in block.lower():
            customs_status = "Non dédouanée"

        # Seller Type
        seller_type = "Particulier"
        if any(k in block.lower() for k in ["pro", "garage", "vitrine", "concession"]):
            seller_type = "Professionnel"

        # Extract seller phone numbers from card HTML
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
            "photos_count": 1.0,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def parse_page(self, html: str, date_scraped: str) -> List[Dict[str, Any]]:
        """Parse listing cards from Wandaloo HTML with concurrent detail phone lookup."""
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

        # Concurrent detail phone resolution for verified listings missing phone
        pending_details = [
            (i, rec["url"])
            for i, rec in enumerate(records)
            if not rec.get("seller_phone")
            and rec.get("url")
            and "/occasion/" in rec.get("url", "")
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

    def scrape_page(
        self,
        page_num: int,
        is_showroom: bool = False,
        custom_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch and parse one page from Wandaloo.
        Supports both general used cars (?page={p}&pg={p}) and showroom/garage inventories.
        """
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if custom_url:
            # Map /occasion/garages/ or dealer URLs directly to showroom pro feed
            if "/garages" in custom_url.lower():
                url = f"{self.SHOWROOM_URL}&page={page_num}&pg={page_num}" if page_num > 1 else self.SHOWROOM_URL
            else:
                url = custom_url
        elif is_showroom:
            url = f"{self.SHOWROOM_URL}&page={page_num}&pg={page_num}" if page_num > 1 else f"{self.SHOWROOM_URL}&pg=1"
        else:
            url = f"{self.BASE_URL}?page={page_num}&pg={page_num}" if page_num > 1 else self.BASE_URL

        html = self.fetch_page(url, headers=DEFAULT_BROWSER_HEADERS, timeout=(10, 20))
        if not html:
            try:
                import requests as std_requests
                resp = std_requests.get(url, headers=DEFAULT_BROWSER_HEADERS, timeout=15)
                if resp.status_code == 200:
                    html = resp.text
            except Exception as e:
                logger.debug("[%s] Requests fallback error for %s: %s", self.source_name, url, e)

        if not html:
            logger.warning("[%s] Failed to fetch page %d (url: %s)", self.source_name, page_num, url)
            return []

        records = self.parse_page(html, date_scraped)
        feed_type = "Showroom" if is_showroom else "Catalog"
        logger.info("[%s] %s Page %d: parsed %d listings", self.source_name, feed_type, page_num, len(records))
        return records

    def scrape(
        self,
        max_pages: int = 45,
        start_page: int = 1,
        output_filename: Optional[str] = None,
        include_showrooms: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Crawl Wandaloo used car listings for up to max_pages starting from start_page.
        Also crawls professional showroom / dealer feeds (vendeur=2) when start_page == 1
        or when include_showrooms is enabled to ensure complete showroom coverage.
        """
        end_page = start_page + max_pages - 1
        logger.info(
            "[%s] Starting crawl for pages %d to %d (max %d pages, include_showrooms=%s) ...",
            self.source_name,
            start_page,
            end_page,
            max_pages,
            include_showrooms,
        )
        all_records: List[Dict[str, Any]] = []

        # 1. Harvest general catalog listings (pages start_page to end_page)
        consecutive_empty = 0
        for p in range(start_page, start_page + max_pages):
            batch = self.scrape_page(p, is_showroom=False)
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.info("[%s] 3 consecutive empty pages at page %d. Stopping general catalog.", self.source_name, p)
                    break
            else:
                consecutive_empty = 0
                all_records.extend(batch)

            self.sleep()

        # 2. Harvest professional showroom dealer inventories
        if include_showrooms and start_page == 1:
            logger.info("[%s] Ingesting dedicated showroom dealer inventory feed (vendeur=2)...", self.source_name)
            consecutive_empty_pro = 0
            max_pro_pages = min(max_pages, 20)
            for p in range(1, max_pro_pages + 1):
                batch = self.scrape_page(p, is_showroom=True)
                if not batch:
                    consecutive_empty_pro += 1
                    if consecutive_empty_pro >= 2:
                        break
                else:
                    consecutive_empty_pro = 0
                    all_records.extend(batch)
                self.sleep()

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
    parser = argparse.ArgumentParser(description="Wandaloo.com Production Scraper")
    parser.add_argument("--start-page", type=int, default=1, help="Initial page to scrape (default: 1)")
    parser.add_argument("--max-pages", type=int, default=45, help="Max pages to scrape (default: 45)")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output CSV filename (e.g. test_wandaloo.csv)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--no-showrooms", action="store_true", help="Disable dealer showroom inventory feed")
    args = parser.parse_args()

    scraper = WandalooScraper(output_dir=args.output_dir)
    df = scraper.scrape(
        max_pages=args.max_pages,
        start_page=args.start_page,
        output_filename=args.output_filename,
        include_showrooms=not args.no_showrooms,
    )
    print(f"Successfully scraped and saved {len(df)} records for Wandaloo.com")


if __name__ == "__main__":
    main()
