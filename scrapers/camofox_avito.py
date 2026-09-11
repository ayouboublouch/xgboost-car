#!/usr/bin/env python3
"""
Camoufox Avito.ma Local Stealth Deep Scraper
=============================================
Local stealth crawler utilizing Camoufox (stealth Firefox browser)
to bypass Cloudflare Turnstile / Bot Management on Moroccan Avito (avito.ma),
performing two-stage deep extraction:
  1. Index Phase: Discovers individual listing URLs on search listing pages.
  2. Detail & Phone Phase: Opens each listing page, extracts rich vehicle
     specs from JSON-LD / DOM, and executes click-to-reveal phone extraction.

Designed strictly for local residential runs to prevent IP bans and Cloudflare
datacenter blocks without adding heavy browser dependencies to the GitHub Actions runner.

Installation:
    pip install camoufox
    camoufox fetch

Usage:
    python scrapers/camofox_avito.py --max-listings 5
    python scrapers/camofox_avito.py --max-pages 2
    python scrapers/camofox_avito.py --url "https://www.avito.ma/fr/..._58374652.htm"
"""

import argparse
import datetime
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
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
        BLACKLIST_PHONES,
    )
except ImportError:
    from base import (
        BaseScraper,
        SCHEMA_FIELDS,
        extract_moroccan_phone,
        hash_phone,
        BLACKLIST_PHONES,
    )

# Optional stealth engine import
try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    CAMOUFOX_AVAILABLE = False

import pandas as pd

# Platform customer care numbers blacklist
BLACKLIST_PHONES: Set[str] = {"0520428686", "0522000000", "0802000000"}

KNOWN_BRANDS = [
    "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "BYD", "Chery", "Chevrolet",
    "Chrysler", "Citroën", "Citroen", "Cupra", "Dacia", "Daihatsu", "Dodge", "DS", "Ferrari",
    "Fiat", "Ford", "Geely", "GMC", "Great Wall", "Haval", "Honda", "Hummer",
    "Hyundai", "Infiniti", "Isuzu", "Iveco", "Jaguar", "Jeep", "Kia", "Lada", "Lamborghini",
    "Lancia", "Land Rover", "Lexus", "Maserati", "Mahindra", "Mazda", "Mercedes-Benz",
    "Mercedes", "MG", "Mini", "Mitsubishi", "Nissan", "Opel", "Peugeot", "Porsche",
    "Range Rover", "Renault", "Rolls-Royce", "Rover", "Saab", "Seat", "Skoda", "Smart",
    "Ssangyong", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.camoufox_avito")


class CamoufoxAvitoScraper(BaseScraper):
    source_name = "avito"
    BASE_URL = "https://www.avito.ma/fr/maroc/voitures_d_occasion--%C3%A0_vendre"

    def __init__(
        self,
        output_dir: str = "data/raw",
        delay_min: float = 1.0,
        delay_max: float = 2.0,
        headless: bool = True,
    ):
        super().__init__(output_dir=output_dir, delay_min=delay_min, delay_max=delay_max)
        self.headless = headless

    def extract_seller_phone_from_page(self, page: Any, description_raw: str = "") -> Optional[str]:
        """
        Extract verified seller phone number via click-to-reveal modal, WhatsApp link, or description.
        Filters against BLACKLIST_PHONES to prevent customer support numbers from being captured.
        """
        seller_phone: Optional[str] = None

        # 1. Locate Contacter le Vendeur / Appeler button
        btn_selectors = [
            'button:has-text("Contacter le Vendeur")',
            '[data-cy="call-button"]',
            'button:has-text("Appeler")',
            'button:has-text("Contacter")',
        ]

        clicked = False
        for sel in btn_selectors:
            try:
                buttons = page.query_selector_all(sel)
                for b in buttons:
                    if b.is_visible():
                        try:
                            b.click(timeout=3000)
                            clicked = True
                            break
                        except Exception:
                            try:
                                b.dispatch_event("click")
                                clicked = True
                                break
                            except Exception:
                                pass
                if clicked:
                    break
            except Exception:
                continue

        if clicked:
            time.sleep(1.5)
            try:
                page.wait_for_selector('a[href^="tel:"], .modal, [role="dialog"]', timeout=3000)
            except Exception:
                pass

            # Extract from a[href^="tel:"]
            try:
                tel_el = page.query_selector('a[href^="tel:"]')
                if tel_el:
                    href = tel_el.get_attribute("href")
                    candidate = extract_moroccan_phone(href)
                    if candidate and candidate not in BLACKLIST_PHONES:
                        seller_phone = candidate
            except Exception as e:
                logger.debug("Error reading tel element: %s", e)

            # Or parse text inside popup modal dialog
            if not seller_phone:
                try:
                    dialog_el = page.query_selector('[role="dialog"], .modal, div[class*="modal"]')
                    if dialog_el:
                        candidate = extract_moroccan_phone(dialog_el.inner_text())
                        if candidate and candidate not in BLACKLIST_PHONES:
                            seller_phone = candidate
                except Exception as e:
                    logger.debug("Error reading modal text: %s", e)

        # 2. WhatsApp icon / link
        if not seller_phone:
            try:
                wa_el = page.query_selector('a[href*="wa.me/"], a[href*="api.whatsapp.com"]')
                if wa_el:
                    wa_href = wa_el.get_attribute("href")
                    candidate = extract_moroccan_phone(wa_href)
                    if candidate and candidate not in BLACKLIST_PHONES:
                        seller_phone = candidate
            except Exception as e:
                logger.debug("Error checking WhatsApp link: %s", e)

        # 3. Fallback: parse description_raw
        if not seller_phone and description_raw:
            candidate = extract_moroccan_phone(description_raw)
            if candidate and candidate not in BLACKLIST_PHONES:
                seller_phone = candidate

        # Phone blacklist filter
        if seller_phone in BLACKLIST_PHONES:
            logger.warning("Extracted phone %s is in platform BLACKLIST_PHONES! Discarding.", seller_phone)
            seller_phone = None

        # Strict validation: 10 digits starting with 05, 06, or 07
        if seller_phone:
            if not (len(seller_phone) == 10 and seller_phone.startswith("0") and seller_phone[1] in "567"):
                seller_phone = None

        return seller_phone

    def _infer_brand_and_model(self, title_raw: str, url: str, breadcrumbs: List[str]) -> tuple:
        """Accurately deduce brand and model so neither is left as 'Autre' if discernible."""
        brand = ""
        model = ""
        haystack = f"{title_raw} {url} {' '.join(breadcrumbs)}"

        for b in KNOWN_BRANDS:
            # Case-insensitive whole word boundary match
            if re.search(rf"\b{re.escape(b)}\b", haystack, re.IGNORECASE):
                brand = b
                break

        if brand:
            # Extract model from title after the brand name
            m_match = re.search(rf"\b{re.escape(brand)}\s+([a-zA-Z0-9\-_]+(?:\s+[a-zA-Z0-9\-_]+)?)", title_raw, re.IGNORECASE)
            if m_match:
                model = m_match.group(1).strip()
            else:
                # Try from URL slug
                slug_m = re.search(rf"voitures_d_occasion/([^/]+)_(\d+)\.htm", url)
                if slug_m:
                    slug = slug_m.group(1).replace("_", " ")
                    slug_match = re.search(rf"\b{re.escape(brand)}\s+([a-zA-Z0-9\-_]+(?:\s+[a-zA-Z0-9\-_]+)?)", slug, re.IGNORECASE)
                    if slug_match:
                        model = slug_match.group(1).strip()

        if not brand and title_raw:
            tokens = [t for t in re.sub(r"[^a-zA-Z0-9À-ÿ\s]", " ", title_raw).split() if t]
            if tokens:
                brand = tokens[0].capitalize()
                if len(tokens) > 1:
                    model = " ".join(tokens[1:3]).capitalize()

        return brand or "Autre", model or "Autre"

    def scrape_listing_page(self, page: Any, listing_url: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        """Open listing detail page, extract core specs via LD+JSON & DOM, and reveal seller phone."""
        id_m = re.search(r"_(\d+)\.htm", listing_url)
        listing_id = id_m.group(1) if id_m else None
        if not listing_id:
            return None

        try:
            page.goto(listing_url, timeout=30000, wait_until="domcontentloaded")
            time.sleep(1.5)
        except Exception as e:
            logger.warning("[%s] Failed to open listing page %s: %s", self.source_name, listing_url, e)
            return None

        html = page.content()

        # Handle Cloudflare Challenge if any
        if "Attention Required! | Cloudflare" in html or "cf-browser-verification" in html:
            logger.warning("[%s] Cloudflare verification encountered. Waiting 5s...", self.source_name)
            time.sleep(5.0)
            html = page.content()

        # Parse JSON-LD scripts
        title_raw = ""
        brand = ""
        model = ""
        description_raw = ""
        price_mad = None
        photos_count = 1.0
        city = ""
        breadcrumbs: List[str] = []

        scripts = re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL)
        for s in scripts:
            try:
                data = json.loads(s.strip())
                if isinstance(data, dict):
                    if data.get("@type") in ("Car", "Product", "Vehicle"):
                        title_raw = data.get("name") or title_raw
                        b_val = data.get("brand")
                        if isinstance(b_val, dict):
                            brand = b_val.get("name") or brand
                        elif isinstance(b_val, str):
                            brand = b_val
                        description_raw = data.get("description") or description_raw
                        offers = data.get("offers")
                        if isinstance(offers, dict):
                            price_mad = offers.get("price")
                        elif isinstance(offers, list) and offers:
                            price_mad = offers[0].get("price")
                        images = data.get("image")
                        if isinstance(images, list):
                            photos_count = float(len(images))
                        elif isinstance(images, str):
                            photos_count = 1.0
                    elif data.get("@type") == "BreadcrumbList":
                        items = data.get("itemListElement") or []
                        for it in items:
                            if isinstance(it, dict) and it.get("name"):
                                breadcrumbs.append(it["name"])
                        if len(items) >= 3 and isinstance(items[2], dict):
                            city = items[2].get("name") or city
            except Exception:
                pass

        # Parse key-value specifications from DOM
        specs: Dict[str, str] = {}
        try:
            elements = page.query_selector_all('li, div[class*="sc-"]')
            for el in elements:
                txt = el.inner_text().strip()
                lines = [l.strip() for l in txt.split("\n") if l.strip()]
                if len(lines) == 2 and len(lines[0]) < 35 and len(lines[1]) < 35:
                    specs[lines[1].lower()] = lines[0]
                    specs[lines[0].lower()] = lines[1]
        except Exception:
            pass

        # Map specifications
        year_raw = specs.get("année-modèle") or specs.get("annee-modele") or specs.get("année") or specs.get("annee")
        mileage_raw = specs.get("kilométrage") or specs.get("kilometrage")
        transmission_raw = specs.get("boite de vitesses") or specs.get("boîte de vitesses") or specs.get("boite")
        fuel_raw = specs.get("type de carburant") or specs.get("carburant")
        fiscal_cv_raw = specs.get("puissance fiscale") or specs.get("puissance")
        doors_raw = specs.get("nombre de portes") or specs.get("portes")
        owners_raw = specs.get("première main") or specs.get("premiere main")
        condition_raw = specs.get("état") or specs.get("etat")
        customs_raw = specs.get("dédouané") or specs.get("dedouane")
        if not brand or brand == "Autre":
            brand = specs.get("marque") or brand
        if not model or model == "Autre":
            model = specs.get("modèle") or specs.get("modele") or model

        # Fallback for title
        if not title_raw:
            h1 = page.query_selector("h1")
            title_raw = h1.inner_text().strip() if h1 else ""

        # Normalize city from URL if missing
        if not city:
            city_m = re.search(r"/fr/([^/]+)/voitures_d_occasion/", listing_url)
            if city_m:
                city = city_m.group(1).replace("_", " ").title()

        # Deduce brand & model accurately (never leave as 'Autre' if discernible)
        if not brand or brand == "Autre" or not model or model == "Autre":
            inferred_b, inferred_m = self._infer_brand_and_model(title_raw, listing_url, breadcrumbs)
            if not brand or brand == "Autre":
                brand = inferred_b
            if not model or model == "Autre":
                model = inferred_m

        # Fallback for price from DOM if not in JSON-LD
        if price_mad is None or price_mad == 0:
            try:
                for sel in ['[data-cy*="price"]', 'p[class*="price"]', 'span[class*="price"]', 'div[class*="price"]']:
                    p_el = page.query_selector(sel)
                    if p_el:
                        p_val = self.clean_numeric(p_el.inner_text())
                        if p_val and p_val > 1000:
                            price_mad = p_val
                            break
            except Exception:
                pass

        # Interactive click-to-reveal phone extraction
        seller_phone = self.extract_seller_phone_from_page(page, description_raw=description_raw)
        seller_phone_hash = hash_phone(seller_phone)

        raw_record = {
            "listing_id": listing_id,
            "url": listing_url,
            "source": self.source_name,
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": title_raw,
            "brand": brand,
            "model": model,
            "trim": "",
            "year": self.clean_year(year_raw) or self.clean_year(title_raw),
            "mileage_km": self.clean_numeric(mileage_raw),
            "fuel_type": fuel_raw or "",
            "transmission": transmission_raw or "",
            "fiscal_power_cv": fiscal_cv_raw or "",
            "customs_status": customs_raw or "",
            "condition": condition_raw or "",
            "owners_count": owners_raw or "",
            "doors_count": self.clean_numeric(doors_raw),
            "seller_type": "Particulier",
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
            "city": city,
            "region": "",
            "price_mad": self.clean_numeric(price_mad),
            "photos_count": photos_count,
            "description_raw": description_raw,
        }
        return self.validate_and_format_record(raw_record)

    def scrape(
        self,
        max_pages: int = 1,
        target_url: Optional[str] = None,
        max_listings: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Crawl Avito.ma using stealth Camoufox browser with two-stage deep extraction."""
        if not CAMOUFOX_AVAILABLE:
            logger.error(
                "Camoufox is not installed in the active environment.\n"
                "To use the local Cloudflare bypass crawler, install it locally:\n"
                "    pip install camoufox\n"
                "    camoufox fetch\n"
            )
            return pd.DataFrame()

        all_records: List[Dict[str, Any]] = []
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with Camoufox(headless=self.headless) as browser:
                page = browser.new_page()

                # Single listing targeted mode
                if target_url:
                    logger.info("[%s] Scraping single targeted listing: %s", self.source_name, target_url)
                    rec = self.scrape_listing_page(page, target_url, date_scraped)
                    if rec:
                        all_records.append(rec)
                        logger.info(
                            "[%s] (1/1) Scraped %s %s (%s) - Phone: %s - %s MAD",
                            self.source_name,
                            rec.get("brand"),
                            rec.get("model"),
                            int(rec.get("year")) if rec.get("year") else "N/A",
                            rec.get("seller_phone") or "None",
                            f"{int(rec.get('price_mad'))}" if rec.get("price_mad") else "N/A",
                        )
                else:
                    # Step A: Index Phase - discover listing URLs from search pages
                    all_listing_urls: List[str] = []
                    seen_urls: Set[str] = set()

                    for p in range(1, max_pages + 1):
                        search_url = f"{self.BASE_URL}?o={p}" if p > 1 else self.BASE_URL
                        logger.info("[%s] [Index Phase] Navigating to search page %d: %s", self.source_name, p, search_url)

                        try:
                            page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                            time.sleep(2.0)
                        except Exception as e:
                            logger.warning("[%s] Search page %d navigation error: %s", self.source_name, p, e)
                            continue

                        page_html = page.content()
                        if "Attention Required! | Cloudflare" in page_html or "cf-browser-verification" in page_html:
                            logger.warning("[%s] Cloudflare challenge encountered on search page %d. Waiting 5s...", self.source_name, p)
                            time.sleep(5.0)
                            page_html = page.content()

                        # Extract listing links
                        links = re.findall(
                            r'href=["\']((?:https://www\.avito\.ma)?/fr/[^"\']*_(\d+)\.htm)["\']',
                            page_html,
                        )
                        page_found = 0
                        for l_url, _ in links:
                            if not l_url.startswith("http"):
                                l_url = urljoin("https://www.avito.ma", l_url)
                            if l_url not in seen_urls:
                                seen_urls.add(l_url)
                                all_listing_urls.append(l_url)
                                page_found += 1

                        logger.info("[%s] [Index Phase] Page %d discovered %d new listing URLs (total: %d)", self.source_name, p, page_found, len(all_listing_urls))

                    total_discovered = len(all_listing_urls)
                    target_count = max_listings or total_discovered
                    logger.info("[%s] [Detail & Phone Phase] Deeply extracting up to %d verified listings...", self.source_name, target_count)

                    # Step B: Detail & Phone Phase - visit each listing URL
                    for l_url in all_listing_urls:
                        rec = self.scrape_listing_page(page, l_url, date_scraped)
                        if rec:
                            # Verify row quality
                            is_valid_row = bool(
                                rec.get("brand")
                                and rec.get("brand") != "Autre"
                                and rec.get("price_mad")
                                and rec.get("price_mad") > 0
                                and rec.get("seller_phone")
                            )

                            if is_valid_row or not max_listings:
                                all_records.append(rec)
                                idx = len(all_records)
                                logger.info(
                                    "[%s] (%d/%d) Scraped %s %s (%s) - Phone: %s - %s MAD",
                                    self.source_name,
                                    idx,
                                    target_count,
                                    rec.get("brand"),
                                    rec.get("model"),
                                    int(rec.get("year")) if rec.get("year") else "N/A",
                                    rec.get("seller_phone") or "None",
                                    f"{int(rec.get('price_mad'))}" if rec.get("price_mad") else "N/A",
                                )

                            if max_listings and len(all_records) >= max_listings:
                                break

                        # Polite delay between listing detail visits (1 to 2 seconds)
                        time.sleep(random.uniform(self.delay_min, self.delay_max))

        except Exception as e:
            logger.error("[%s] Unexpected Camoufox execution error: %s", self.source_name, e)

        if len(all_records) == 0:
            logger.warning("[%s] Crawl complete. 0 records harvested.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info("[%s] Local crawl finished. Total verified records: %d", self.source_name, len(df))
        self.save_output(df)
        return df

    def save_output(self, df: pd.DataFrame) -> None:
        """Persist harmonized DataFrame to data/raw/avito_local_{YYYY-MM-DD}.csv and Parquet."""
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        csv_path = self.output_dir / f"avito_local_{today_str}.csv"
        parquet_path = self.output_dir / f"avito_local_{today_str}.parquet"

        if df is None or df.empty or len(df) == 0:
            logger.warning("[%s] Scraper harvested 0 records. Writing nothing.", self.source_name)
            return

        # Drop any row where brand == 'Autre' and price_mad is missing
        initial_len = len(df)
        df = df[~((df["brand"] == "Autre") & (df["price_mad"].isna()))]
        if len(df) < initial_len:
            logger.info("[%s] Dropped %d invalid rows (brand=='Autre' and missing price).", self.source_name, initial_len - len(df))

        for col in SCHEMA_FIELDS:
            if col not in df.columns:
                df[col] = None
        df = df[SCHEMA_FIELDS].drop_duplicates(subset=["listing_id"])

        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("[%s] Saved %d records to %s", self.source_name, len(df), csv_path)
        except Exception as e:
            logger.error("[%s] Failed to save CSV: %s", self.source_name, e)

        try:
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            logger.info("[%s] Saved %d records to %s", self.source_name, len(df), parquet_path)
        except Exception as e:
            logger.error("[%s] Failed to save Parquet: %s", self.source_name, e)


def main():
    parser = argparse.ArgumentParser(description="Camoufox Avito.ma Local Stealth Deep Scraper")
    parser.add_argument("--max-pages", type=int, default=1, help="Max search pages to index (default: 1)")
    parser.add_argument("--max-listings", type=int, default=5, help="Max total listings to deeply scrape (default: 5)")
    parser.add_argument("--url", type=str, default=None, help="Scrape a specific listing URL directly")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--no-headless", action="store_true", help="Launch browser with GUI visible for debugging")
    args = parser.parse_args()

    scraper = CamoufoxAvitoScraper(
        output_dir=args.output_dir,
        headless=not args.no_headless,
    )
    df = scraper.scrape(
        max_pages=args.max_pages,
        target_url=args.url,
        max_listings=args.max_listings,
    )
    print(f"\nSuccessfully scraped {len(df)} deep records for Avito.ma using Camoufox.")


if __name__ == "__main__":
    main()
