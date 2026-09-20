#!/usr/bin/env python3
"""
Unified Local Stealth Camoufox Crawler for Moroccan Car Marketplaces
===================================================================
Scrapes Avito.ma, Moteur.ma, and Wandaloo.com using Camoufox (stealth Firefox).
Designed for residential execution to bypass Cloudflare/datacenter IP blocks,
extract rich vehicle specifications, reveal verified seller telephone numbers,
and automatically consolidate records into the master dataset.

Usage:
    python scrapers/camoufox_all.py --source all --max-listings 100 --max-pages 5
    python scrapers/camoufox_all.py --source avito --max-listings 50 --headless True
    python scrapers/camoufox_all.py --source moteur --max-listings 50
    python scrapers/camoufox_all.py --source wandaloo --max-listings 50
"""

import argparse
import datetime
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

import pandas as pd

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    CAMOUFOX_AVAILABLE = False

try:
    from scrapers.base import (
        SCHEMA_FIELDS,
        TRACKING_DIR,
        SEEN_IDS_FILE,
        load_seen_listing_ids,
        append_seen_listing_ids,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
        KNOWN_BRANDS,
    )
    from scrapers.continuous_harvester import update_master_database
except ImportError:
    from base import (
        SCHEMA_FIELDS,
        TRACKING_DIR,
        SEEN_IDS_FILE,
        load_seen_listing_ids,
        append_seen_listing_ids,
        clean_brand_and_model,
        extract_moroccan_phone,
        hash_phone,
        infer_moroccan_region,
        KNOWN_BRANDS,
    )
    from continuous_harvester import update_master_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.camoufox_all")

BLACKLIST_PHONES: Set[str] = {"0520428686", "0522000000", "0802000000"}


def clean_numeric(val: Any) -> Optional[float]:
    """Helper to extract clean float from strings like '120 000 DH' or '150 000 km'."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    s = str(val).replace("\xa0", " ").replace(" ", "").replace(",", ".").lower()
    s = re.sub(r"[^\d.]", "", s)
    try:
        f = float(s)
        return f if f > 0 else None
    except ValueError:
        return None


def clean_year(val: Any) -> Optional[float]:
    """Extract valid automobile manufacturing year (1970 - 2026)."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    m = re.search(r"\b(19[7-9]\d|20[0-2]\d)\b", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strictly sanitize raw record:
    1. Multiline description fix: escapes newlines and tabs into single spaces.
    2. Enforce 10-digit seller phone string starting with 0.
    3. Harmonize brand and model.
    """
    # 1. Multiline description and title fix
    if rec.get("description_raw"):
        rec["description_raw"] = re.sub(r"[\r\n\t]+", " ", str(rec["description_raw"])).strip()
        if rec["description_raw"].lower() in ("nan", "none"):
            rec["description_raw"] = ""

    if rec.get("title_raw"):
        rec["title_raw"] = re.sub(r"[\r\n\t]+", " ", str(rec["title_raw"])).strip()
        if rec["title_raw"].lower() in ("nan", "none"):
            rec["title_raw"] = ""

    # 2. Strict phone formatting
    if rec.get("seller_phone"):
        raw_p = str(rec["seller_phone"]).replace(".0", "").strip()
        p = extract_moroccan_phone(raw_p)
        if p and p not in BLACKLIST_PHONES and len(p) == 10 and p.startswith("0") and p[1] in "567":
            rec["seller_phone"] = p
            if not rec.get("seller_phone_hash"):
                rec["seller_phone_hash"] = hash_phone(p)
        else:
            rec["seller_phone"] = None
    else:
        rec["seller_phone"] = None

    # 3. Clean brand & model
    b, m, tr = clean_brand_and_model(
        rec.get("brand"),
        rec.get("model"),
        title_raw=str(rec.get("title_raw") or ""),
        trim=str(rec.get("trim") or ""),
    )
    rec["brand"] = b
    rec["model"] = m
    rec["trim"] = tr

    # Infer region if missing
    if not rec.get("region") and rec.get("city"):
        rec["region"] = infer_moroccan_region(rec["city"])

    return rec


def flush_checkpoint(records: List[Dict[str, Any]], target_csv: Path, seen_ids: Set[str]) -> int:
    """Flush accumulated records to daily continuous CSV and update tracking register."""
    if not records:
        return 0

    sanitized = [sanitize_record(r) for r in records]
    df_new = pd.DataFrame(sanitized)

    for col in SCHEMA_FIELDS:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[SCHEMA_FIELDS].copy()

    # Format string columns
    for col in ["description_raw", "title_raw"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].fillna("").astype(str).apply(
                lambda s: re.sub(r"[\r\n\t]+", " ", s).strip() if s.lower() not in ("nan", "none") else ""
            )

    if "seller_phone" in df_new.columns:
        def _fmt_phone(p):
            if pd.isna(p) or p is None:
                return None
            s = str(p).replace(".0", "").strip()
            if not s or s.lower() in ("nan", "none", "<na>"):
                return None
            s = s.zfill(10)
            return s if len(s) == 10 and s.startswith("0") and s[1] in "567" else None
        df_new["seller_phone"] = df_new["seller_phone"].apply(_fmt_phone)

    target_csv.parent.mkdir(parents=True, exist_ok=True)

    if target_csv.exists():
        try:
            df_existing = pd.read_csv(target_csv, low_memory=False, dtype={"seller_phone": str, "listing_id": str})
            combined = pd.concat([df_existing, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["listing_id"], keep="last")
        except Exception as e:
            logger.warning("Could not read existing checkpoint %s: %s", target_csv.name, e)
            combined = df_new
    else:
        combined = df_new

    combined = combined[SCHEMA_FIELDS]
    combined.to_csv(target_csv, index=False, encoding="utf-8")

    # Update persistent register
    new_lids = [str(r.get("listing_id")).strip() for r in records if r.get("listing_id")]
    append_seen_listing_ids(new_lids)
    seen_ids.update(new_lids)

    logger.info(
        "Checkpoint flushed: +%d new records written to %s (Total batch: %d rows | Total seen IDs: %d)",
        len(records),
        target_csv.name,
        len(combined),
        len(seen_ids),
    )
    return len(records)


# ==============================================================================
# 1. AVITO.MA EXTRACTOR
# ==============================================================================
class AvitoCamoufoxEngine:
    BASE_URL = "https://www.avito.ma/fr/maroc/voitures_d_occasion--%C3%A0_vendre"

    def __init__(self, page: Any, seen_ids: Set[str]):
        self.page = page
        self.seen_ids = seen_ids

    def extract_seller_phone_from_page(self, description_raw: str = "") -> Optional[str]:
        """Reveal and extract seller phone number via call button or WhatsApp link."""
        seller_phone: Optional[str] = None
        btn_selectors = [
            'button:has-text("Contacter le Vendeur")',
            '[data-cy="call-button"]',
            'button:has-text("Appeler")',
            'button:has-text("Contacter")',
        ]

        clicked = False
        for sel in btn_selectors:
            try:
                buttons = self.page.query_selector_all(sel)
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
                self.page.wait_for_selector('a[href^="tel:"], .modal, [role="dialog"]', timeout=3000)
            except Exception:
                pass

            try:
                tel_el = self.page.query_selector('a[href^="tel:"]')
                if tel_el:
                    href = tel_el.get_attribute("href")
                    candidate = extract_moroccan_phone(href)
                    if candidate and candidate not in BLACKLIST_PHONES:
                        seller_phone = candidate
            except Exception:
                pass

            if not seller_phone:
                try:
                    dialog_el = self.page.query_selector('[role="dialog"], .modal, div[class*="modal"]')
                    if dialog_el:
                        candidate = extract_moroccan_phone(dialog_el.inner_text())
                        if candidate and candidate not in BLACKLIST_PHONES:
                            seller_phone = candidate
                except Exception:
                    pass

        if not seller_phone:
            try:
                wa_el = self.page.query_selector('a[href*="wa.me/"], a[href*="api.whatsapp.com"]')
                if wa_el:
                    candidate = extract_moroccan_phone(wa_el.get_attribute("href"))
                    if candidate and candidate not in BLACKLIST_PHONES:
                        seller_phone = candidate
            except Exception:
                pass

        if not seller_phone and description_raw:
            candidate = extract_moroccan_phone(description_raw)
            if candidate and candidate not in BLACKLIST_PHONES:
                seller_phone = candidate

        if seller_phone and (len(seller_phone) != 10 or not seller_phone.startswith("0") or seller_phone[1] not in "567"):
            seller_phone = None

        return seller_phone

    def scrape_listing(self, url: str, date_scraped: str) -> Optional[Dict[str, Any]]:
        id_m = re.search(r"_(\d+)\.htm", url)
        listing_id = id_m.group(1) if id_m else None
        if not listing_id or listing_id in self.seen_ids:
            return None

        try:
            self.page.goto(url, timeout=30000, wait_until="domcontentloaded")
            time.sleep(1.2)
        except Exception as e:
            logger.warning("[Avito] Failed to load %s: %s", url, e)
            return None

        html = self.page.content()

        # Parse JSON-LD
        title_raw = ""
        brand = ""
        model = ""
        description_raw = ""
        price_mad = None
        photos_count = 1.0
        city = ""

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
                    elif data.get("@type") == "BreadcrumbList":
                        items = data.get("itemListElement") or []
                        if len(items) >= 3 and isinstance(items[2], dict):
                            city = items[2].get("name") or city
            except Exception:
                pass

        # Parse specs from DOM
        specs: Dict[str, str] = {}
        try:
            elements = self.page.query_selector_all('li, div[class*="sc-"]')
            for el in elements:
                txt = el.inner_text().strip()
                lines = [l.strip() for l in txt.split("\n") if l.strip()]
                if len(lines) == 2 and len(lines[0]) < 35 and len(lines[1]) < 35:
                    specs[lines[1].lower()] = lines[0]
                    specs[lines[0].lower()] = lines[1]
        except Exception:
            pass

        year_raw = specs.get("année-modèle") or specs.get("annee-modele") or specs.get("année") or specs.get("annee")
        mileage_raw = specs.get("kilométrage") or specs.get("kilometrage")
        transmission_raw = specs.get("boite de vitesses") or specs.get("boîte de vitesses") or specs.get("boite")
        fuel_raw = specs.get("type de carburant") or specs.get("carburant")
        fiscal_cv_raw = specs.get("puissance fiscale") or specs.get("puissance")
        doors_raw = specs.get("nombre de portes") or specs.get("portes")
        customs_raw = specs.get("dédouané") or specs.get("dedouane")
        condition_raw = specs.get("état") or specs.get("etat")

        if not brand or brand == "Autre":
            brand = specs.get("marque") or brand
        if not model or model == "Autre":
            model = specs.get("modèle") or specs.get("modele") or model

        if not title_raw:
            h1 = self.page.query_selector("h1")
            title_raw = h1.inner_text().strip() if h1 else ""

        if not city:
            city_m = re.search(r"/fr/([^/]+)/voitures_d_occasion/", url)
            if city_m:
                city = city_m.group(1).replace("_", " ").title()

        if price_mad is None or price_mad == 0:
            for sel in ['[data-cy*="price"]', 'p[class*="price"]', 'span[class*="price"]']:
                p_el = self.page.query_selector(sel)
                if p_el:
                    p_val = clean_numeric(p_el.inner_text())
                    if p_val and p_val > 1000:
                        price_mad = p_val
                        break

        # Reveal seller phone number
        seller_phone = self.extract_seller_phone_from_page(description_raw=description_raw)

        raw_record = {
            "listing_id": listing_id,
            "url": url,
            "source": "avito",
            "date_posted": date_scraped[:10],
            "date_scraped": date_scraped,
            "title_raw": title_raw,
            "brand": brand,
            "model": model,
            "trim": "",
            "year": clean_year(year_raw) or clean_year(title_raw),
            "mileage_km": clean_numeric(mileage_raw),
            "fuel_type": fuel_raw or "",
            "transmission": transmission_raw or "",
            "fiscal_power_cv": fiscal_cv_raw or "",
            "customs_status": customs_raw or "",
            "condition": condition_raw or "",
            "owners_count": "",
            "doors_count": clean_numeric(doors_raw),
            "seller_type": "Particulier",
            "seller_phone": seller_phone,
            "seller_phone_hash": hash_phone(seller_phone),
            "city": city,
            "region": infer_moroccan_region(city),
            "price_mad": clean_numeric(price_mad),
            "photos_count": photos_count,
            "description_raw": description_raw,
        }
        return sanitize_record(raw_record)

    def scrape(self, max_pages: int = 5, max_listings: int = 100) -> List[Dict[str, Any]]:
        logger.info("[Avito] Starting Camoufox crawl (max_pages=%d, max_listings=%d)...", max_pages, max_listings)
        results: List[Dict[str, Any]] = []
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for p in range(1, max_pages + 1):
            if len(results) >= max_listings:
                break
            url = f"{self.BASE_URL}?o={p}" if p > 1 else self.BASE_URL
            logger.info("[Avito] Loading search page %d: %s", p, url)
            try:
                self.page.goto(url, timeout=30000, wait_until="domcontentloaded")
                time.sleep(2.0)
            except Exception as e:
                logger.warning("[Avito] Failed to load search page %d: %s", p, e)
                continue

            links = self.page.query_selector_all('a[href*="/voitures_d_occasion/"]')
            listing_urls = []
            for l in links:
                href = l.get_attribute("href")
                if href and re.search(r"_(\d+)\.htm", href):
                    full_url = href if href.startswith("http") else f"https://www.avito.ma{href}"
                    if full_url not in listing_urls:
                        listing_urls.append(full_url)

            logger.info("[Avito] Page %d discovered %d listing candidates", p, len(listing_urls))

            for l_url in listing_urls:
                if len(results) >= max_listings:
                    break
                id_m = re.search(r"_(\d+)\.htm", l_url)
                if id_m and id_m.group(1) in self.seen_ids:
                    continue

                rec = self.scrape_listing(l_url, date_scraped)
                if rec and rec.get("price_mad") and rec.get("brand") and rec.get("brand") != "Autre":
                    results.append(rec)
                    self.seen_ids.add(str(rec["listing_id"]).strip())
                    logger.info(
                        "[Avito] (%d/%d) %s %s (%s) - %s MAD - Phone: %s",
                        len(results),
                        max_listings,
                        rec.get("brand"),
                        rec.get("model"),
                        rec.get("year"),
                        rec.get("price_mad"),
                        rec.get("seller_phone") or "N/A",
                    )
                time.sleep(random.uniform(0.8, 1.5))

        return results


# ==============================================================================
# 2. MOTEUR.MA EXTRACTOR
# ==============================================================================
class MoteurCamoufoxEngine:
    BASE_URL = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/recherche"

    def __init__(self, page: Any, seen_ids: Set[str]):
        self.page = page
        self.seen_ids = seen_ids

    def scrape(self, max_pages: int = 5, max_listings: int = 100) -> List[Dict[str, Any]]:
        logger.info("[Moteur] Starting Camoufox crawl (max_pages=%d, max_listings=%d)...", max_pages, max_listings)
        results: List[Dict[str, Any]] = []
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for p in range(1, max_pages + 1):
            if len(results) >= max_listings:
                break
            url = f"{self.BASE_URL}?page={p}"
            logger.info("[Moteur] Loading catalog page %d: %s", p, url)
            try:
                self.page.goto(url, timeout=30000, wait_until="domcontentloaded")
                time.sleep(2.0)
            except Exception as e:
                logger.warning("[Moteur] Failed to load catalog page %d: %s", p, e)
                continue

            html = self.page.content()
            soup = BeautifulSoup(html, "html.parser") if BS4_AVAILABLE else None

            # Split or query listing blocks
            cards = []
            if soup:
                cards = soup.find_all("div", class_=re.compile(r"ad-col|ads-index-card|item-ad", re.I))
                if not cards:
                    detail_links = soup.find_all("a", href=re.compile(r"/detail-annonce/\d+", re.I))
                    cards = [a.find_parent(["div", "article"]) or a for a in detail_links]

            raw_blocks = [str(c) for c in cards] if cards else re.split(r'<div class=["\']ad-col col-12["\']', html)[1:]

            logger.info("[Moteur] Page %d found %d listing blocks", p, len(raw_blocks))

            for block in raw_blocks:
                if len(results) >= max_listings:
                    break

                id_m = re.search(r'/detail-annonce/(\d+)', block)
                if not id_m:
                    continue
                listing_id = id_m.group(1).strip()
                if listing_id in self.seen_ids:
                    continue

                url_m = re.search(r'href=["\'](https?://[^"\']*detail-annonce/\d+[^"\']*)["\']', block)
                if not url_m:
                    url_m = re.search(r'href=["\'](/detail-annonce/\d+[^"\']*)["\']', block)
                    full_url = f"https://www.moteur.ma{url_m.group(1)}" if url_m else ""
                else:
                    full_url = url_m.group(1)

                title_m = re.search(r'class=["\']col-12 title_mark_model[^"\']*["\'][^>]*>(?:<[^>]+>)*([^<]+)', block, re.IGNORECASE)
                if not title_m:
                    title_m = re.search(r'<h3[^>]*>.*?<a[^>]*>([^<]+)</a>', block, re.DOTALL | re.IGNORECASE)
                title_raw = title_m.group(1).strip() if title_m else ""

                price_m = re.search(r'class=["\']PriceListing[^"\']*["\'][^>]*>(?:<[^>]+>)*([^<]+)', block, re.IGNORECASE)
                price_mad = clean_numeric(price_m.group(1)) if price_m else None

                year_m = re.search(r'<span[^>]*class=["\']text-meta[^"\']*["\'][^>]*>.*?(\d{4})', block, re.DOTALL | re.IGNORECASE)
                year = clean_year(year_m.group(1)) if year_m else clean_year(title_raw)

                km_m = re.search(r'(\d+[\s\d]*)\s*Km', block, re.IGNORECASE)
                mileage_km = clean_numeric(km_m.group(1)) if km_m else None

                fuel = "Diesel" if re.search(r'\bdiesel\b', block, re.I) else ("Essence" if re.search(r'\bessence\b', block, re.I) else "")
                trans = "Automatique" if re.search(r'\bautomatique\b', block, re.I) else ("Manuelle" if re.search(r'\bmanuelle\b', block, re.I) else "")
                cv_m = re.search(r'(\d+)\s*CV', block, re.IGNORECASE)
                fiscal_cv = cv_m.group(1) if cv_m else ""

                city_m = re.search(r'<span[^>]*class=["\']link-muted[^"\']*["\'][^>]*>(?:<[^>]+>)*([^<]+)', block, re.IGNORECASE)
                city = city_m.group(1).strip() if city_m else ""

                # Extract phone directly or open detail if needed
                tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', block)
                seller_phone = extract_moroccan_phone(tel_m.group(1)) if tel_m else None
                if not seller_phone:
                    dp_m = re.search(r'data-phone=["\']([^"\']+)["\']', block)
                    if dp_m:
                        seller_phone = extract_moroccan_phone(dp_m.group(1))

                # If missing phone and detail exists, fetch detail using page
                if not seller_phone and full_url and len(results) < max_listings:
                    try:
                        self.page.goto(full_url, timeout=20000, wait_until="domcontentloaded")
                        time.sleep(1.0)
                        d_html = self.page.content()
                        d_tel = re.search(r'href=["\']tel:([^"\']+)["\']', d_html)
                        if d_tel:
                            seller_phone = extract_moroccan_phone(d_tel.group(1))
                        if not seller_phone:
                            seller_phone = extract_moroccan_phone(d_html)
                    except Exception:
                        pass

                raw_record = {
                    "listing_id": listing_id,
                    "url": full_url,
                    "source": "moteur",
                    "date_posted": date_scraped[:10],
                    "date_scraped": date_scraped,
                    "title_raw": title_raw,
                    "brand": "",
                    "model": "",
                    "trim": "",
                    "year": year,
                    "mileage_km": mileage_km,
                    "fuel_type": fuel,
                    "transmission": trans,
                    "fiscal_power_cv": fiscal_cv,
                    "customs_status": "Dédouané",
                    "condition": "Occasion",
                    "owners_count": "",
                    "doors_count": None,
                    "seller_type": "Particulier",
                    "seller_phone": seller_phone,
                    "seller_phone_hash": hash_phone(seller_phone),
                    "city": city,
                    "region": infer_moroccan_region(city),
                    "price_mad": price_mad,
                    "photos_count": 1.0,
                    "description_raw": "",
                }
                rec = sanitize_record(raw_record)
                if rec.get("price_mad") and rec.get("brand") and rec.get("brand") != "Autre":
                    results.append(rec)
                    self.seen_ids.add(listing_id)
                    logger.info(
                        "[Moteur] (%d/%d) %s %s (%s) - %s MAD - Phone: %s",
                        len(results),
                        max_listings,
                        rec.get("brand"),
                        rec.get("model"),
                        rec.get("year"),
                        rec.get("price_mad"),
                        rec.get("seller_phone") or "N/A",
                    )

        return results


# ==============================================================================
# 3. WANDALOO.COM EXTRACTOR
# ==============================================================================
class WandalooCamoufoxEngine:
    BASE_URL = "https://www.wandaloo.com/occasion/"

    def __init__(self, page: Any, seen_ids: Set[str]):
        self.page = page
        self.seen_ids = seen_ids

    def scrape(self, max_pages: int = 5, max_listings: int = 100) -> List[Dict[str, Any]]:
        logger.info("[Wandaloo] Starting Camoufox crawl (max_pages=%d, max_listings=%d)...", max_pages, max_listings)
        results: List[Dict[str, Any]] = []
        date_scraped = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for p in range(1, max_pages + 1):
            if len(results) >= max_listings:
                break
            url = f"{self.BASE_URL}?pg={p}" if p > 1 else self.BASE_URL
            logger.info("[Wandaloo] Loading catalog page %d: %s", p, url)
            try:
                self.page.goto(url, timeout=30000, wait_until="domcontentloaded")
                time.sleep(2.0)
            except Exception as e:
                logger.warning("[Wandaloo] Failed to load catalog page %d: %s", p, e)
                continue

            html = self.page.content()
            soup = BeautifulSoup(html, "html.parser") if BS4_AVAILABLE else None

            cards = []
            if soup:
                cards = soup.find_all("li", class_=re.compile(r"even|odd|result-item", re.I))

            raw_blocks = [str(c) for c in cards] if cards else re.split(r'<li\s+class=["\'](?:even|odd)["\'][^>]*>', html)[1:]
            logger.info("[Wandaloo] Page %d found %d listing blocks", p, len(raw_blocks))

            for block in raw_blocks:
                if len(results) >= max_listings:
                    break

                url_m = re.search(r'href=["\'](/occasion/[^"\']+\.html)["\']', block)
                if not url_m:
                    url_m = re.search(r'href=["\'](https?://[^"\']*/occasion/[^"\']+\.html)["\']', block)
                if not url_m:
                    continue

                rel_url = url_m.group(1)
                full_url = rel_url if rel_url.startswith("http") else f"https://www.wandaloo.com{rel_url}"

                # Extract stable listing_id from URL
                id_m = re.search(r',(\d+)\.html', full_url)
                if not id_m:
                    id_m = re.search(r'-(\d+)\.html', full_url)
                listing_id = id_m.group(1).strip() if id_m else re.sub(r'\D', '', full_url)[-8:]

                if listing_id in self.seen_ids:
                    continue

                title_m = re.search(r'<h2[^>]*>.*?<a[^>]*>([^<]+)</a>', block, re.DOTALL | re.IGNORECASE)
                if not title_m:
                    title_m = re.search(r'class=["\']titre[^"\']*["\'][^>]*>.*?<a[^>]*>([^<]+)</a>', block, re.DOTALL | re.IGNORECASE)
                title_raw = title_m.group(1).strip() if title_m else ""

                price_m = re.search(r'class=["\']price[^"\']*["\'][^>]*>(?:<[^>]+>)*([^<]+)', block, re.IGNORECASE)
                price_mad = clean_numeric(price_m.group(1)) if price_m else None

                km_m = re.search(r'(\d+[\s\d]*)\s*Km', block, re.IGNORECASE)
                mileage_km = clean_numeric(km_m.group(1)) if km_m else None

                year_m = re.search(r'\b(19[89]\d|20[0-2]\d)\b', block)
                year = clean_year(year_m.group(1)) if year_m else clean_year(title_raw)

                fuel = "Diesel" if re.search(r'\bdiesel\b', block, re.I) else ("Essence" if re.search(r'\bessence\b', block, re.I) else "")
                trans = "Automatique" if re.search(r'\bautomatique\b', block, re.I) else ("Manuelle" if re.search(r'\bmanuelle\b', block, re.I) else "")
                city_m = re.search(r'class=["\']city[^"\']*["\'][^>]*>([^<]+)', block, re.I)
                city = city_m.group(1).strip() if city_m else ""

                # Extract phone directly or open detail page
                tel_m = re.search(r'href=["\']tel:([^"\']+)["\']', block)
                seller_phone = extract_moroccan_phone(tel_m.group(1)) if tel_m else None

                if not seller_phone and full_url and len(results) < max_listings:
                    try:
                        self.page.goto(full_url, timeout=20000, wait_until="domcontentloaded")
                        time.sleep(1.0)
                        d_html = self.page.content()
                        d_tel = re.search(r'href=["\']tel:([^"\']+)["\']', d_html)
                        if d_tel:
                            seller_phone = extract_moroccan_phone(d_tel.group(1))
                        if not seller_phone:
                            wa_m = re.search(r'(?:wa\.me/|api\.whatsapp\.com/send\?phone=)(\+?212\d{9}|0[5-7]\d{8})', d_html)
                            if wa_m:
                                seller_phone = extract_moroccan_phone(wa_m.group(1))
                    except Exception:
                        pass

                raw_record = {
                    "listing_id": listing_id,
                    "url": full_url,
                    "source": "wandaloo",
                    "date_posted": date_scraped[:10],
                    "date_scraped": date_scraped,
                    "title_raw": title_raw,
                    "brand": "",
                    "model": "",
                    "trim": "",
                    "year": year,
                    "mileage_km": mileage_km,
                    "fuel_type": fuel,
                    "transmission": trans,
                    "fiscal_power_cv": "",
                    "customs_status": "Dédouané",
                    "condition": "Occasion",
                    "owners_count": "",
                    "doors_count": None,
                    "seller_type": "Particulier",
                    "seller_phone": seller_phone,
                    "seller_phone_hash": hash_phone(seller_phone),
                    "city": city,
                    "region": infer_moroccan_region(city),
                    "price_mad": price_mad,
                    "photos_count": 1.0,
                    "description_raw": "",
                }
                rec = sanitize_record(raw_record)
                if rec.get("price_mad") and rec.get("brand") and rec.get("brand") != "Autre":
                    results.append(rec)
                    self.seen_ids.add(listing_id)
                    logger.info(
                        "[Wandaloo] (%d/%d) %s %s (%s) - %s MAD - Phone: %s",
                        len(results),
                        max_listings,
                        rec.get("brand"),
                        rec.get("model"),
                        rec.get("year"),
                        rec.get("price_mad"),
                        rec.get("seller_phone") or "N/A",
                    )

        return results


# ==============================================================================
# MAIN UNIFIED CRAWLER ORCHESTRATOR
# ==============================================================================
def run_unified_crawler(
    source: str = "all",
    max_listings: int = 100,
    max_pages: int = 5,
    headless: bool = True,
    output_dir: str = "data/raw",
) -> None:
    """Run unified stealth Camoufox crawler across Avito, Moteur, and Wandaloo."""
    if not CAMOUFOX_AVAILABLE:
        logger.error(
            "Camoufox is not installed in the active Python environment.\n"
            "Install it via:\n"
            "    pip install camoufox\n"
            "    camoufox fetch\n"
        )
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    continuous_csv = output_path / f"scraped_continuous_{today_str}.csv"

    # Pre-load persistent seen register
    seen_ids: Set[str] = load_seen_listing_ids()
    logger.info("Loaded %d previously seen listing IDs from persistent register.", len(seen_ids))

    sources_to_run = []
    if source == "all":
        sources_to_run = ["avito", "moteur", "wandaloo"]
    else:
        sources_to_run = [source]

    logger.info("==================================================================")
    logger.info("Starting Unified Camoufox Stealth Harvester")
    logger.info("Target Sources: %s", sources_to_run)
    logger.info("Limits per source: max_listings=%d, max_pages=%d, headless=%s", max_listings, max_pages, headless)
    logger.info("Output checkpoint: %s", continuous_csv)
    logger.info("==================================================================")

    all_harvested: List[Dict[str, Any]] = []

    try:
        with Camoufox(headless=headless) as browser:
            page = browser.new_page()

            for src in sources_to_run:
                logger.info("\n>>> Launching stealth crawler for source: %s ...", src)
                src_records = []
                try:
                    if src == "avito":
                        engine = AvitoCamoufoxEngine(page, seen_ids)
                        src_records = engine.scrape(max_pages=max_pages, max_listings=max_listings)
                    elif src == "moteur":
                        engine = MoteurCamoufoxEngine(page, seen_ids)
                        src_records = engine.scrape(max_pages=max_pages, max_listings=max_listings)
                    elif src == "wandaloo":
                        engine = WandalooCamoufoxEngine(page, seen_ids)
                        src_records = engine.scrape(max_pages=max_pages, max_listings=max_listings)
                except Exception as e:
                    logger.error("Error during crawling of source %s: %s", src, e)

                if src_records:
                    flushed = flush_checkpoint(src_records, continuous_csv, seen_ids)
                    all_harvested.extend(src_records)
                    logger.info(">>> Source %s finished: +%d verified records saved.", src, flushed)
                else:
                    logger.warning(">>> Source %s yielded 0 new records.", src)

    except Exception as e:
        logger.critical("Fatal error running Camoufox browser session: %s", e)

    # Rebuild consolidated master parquet database
    logger.info("\nUpdating consolidated master database (scraped_master_database.parquet)...")
    try:
        update_master_database(output_path)
    except Exception as e:
        logger.error("Failed to update master database parquet: %s", e)

    logger.info("==================================================================")
    logger.info("Unified Camoufox crawl complete!")
    logger.info("Total fresh records harvested across sources: %d", len(all_harvested))
    logger.info("Total persistent seen IDs in register: %d", len(seen_ids))
    logger.info("==================================================================")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Unified Camoufox Stealth Moroccan Car Marketplace Crawler")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "avito", "moteur", "wandaloo"],
        help="Target platform to crawl (default: all)",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=100,
        help="Max unique listings to harvest per platform (default: 100)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max catalog pages to crawl per platform (default: 5)",
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=True,
        help="Run browser in headless stealth mode (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for raw continuous scrapes (default: data/raw)",
    )
    args = parser.parse_args()

    run_unified_crawler(
        source=args.source,
        max_listings=args.max_listings,
        max_pages=args.max_pages,
        headless=args.headless,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
