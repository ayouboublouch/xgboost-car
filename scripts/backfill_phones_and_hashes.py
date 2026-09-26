#!/usr/bin/env python3
"""
Backfill seller phones and SHA-256 hashes for Kifal-Auto, Autocash, and MarocHub.
Also consolidates the master parquet database.
"""

import concurrent.futures
import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scrapers.base import extract_moroccan_phone, hash_phone
from scrapers.continuous_harvester import update_master_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def backfill_autocash(raw_dir: Path):
    logger.info("Backfilling Autocash datasets...")
    phone = "0663000017"
    p_hash = hash_phone(phone)

    for p in raw_dir.glob("*autocash*.*"):
        if p.suffix in [".csv", ".parquet"]:
            logger.info("Processing %s", p.name)
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            df["seller_phone"] = df["seller_phone"].fillna(phone)
            df["seller_phone"] = df["seller_phone"].replace("", phone)
            df["seller_phone_hash"] = df["seller_phone"].apply(hash_phone)
            if p.suffix == ".parquet":
                df.to_parquet(p, index=False)
            else:
                df.to_csv(p, index=False)
            logger.info("Updated %s with %d rows", p.name, len(df))


def backfill_kifal(raw_dir: Path):
    logger.info("Backfilling Kifal-Auto datasets...")
    phone = "0701070727"
    p_hash = hash_phone(phone)

    for p in raw_dir.glob("*kifal*.*"):
        if p.suffix in [".csv", ".parquet"]:
            logger.info("Processing %s", p.name)
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            df["seller_phone"] = df["seller_phone"].fillna(phone)
            df["seller_phone"] = df["seller_phone"].replace("", phone)
            df["seller_phone_hash"] = df["seller_phone"].apply(hash_phone)
            if p.suffix == ".parquet":
                df.to_parquet(p, index=False)
            else:
                df.to_csv(p, index=False)
            logger.info("Updated %s with %d rows", p.name, len(df))


def fetch_marochub_catalog() -> Dict[str, str]:
    """Discover all active MarocHub vehicle URLs mapped by short 8-char hex ID."""
    logger.info("Discovering active MarocHub vehicle URLs...")
    cats = ["", "?type=car", "?type=motorcycle", "?type=truck", "?type=van", "?type=suv"]
    id_to_url = {}
    for c in cats:
        u = f"https://marochub.app/auto{c}"
        try:
            r = requests.get(u, headers=DEFAULT_HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=re.compile(r"/vehicle/")):
                h = a.get("href", "")
                m = re.search(r"([a-f0-9]{8})$", h)
                if m:
                    full = urljoin("https://marochub.app", h)
                    id_to_url[m.group(1)] = full
        except Exception as e:
            logger.warning("Error fetching %s: %s", u, e)
    logger.info("Discovered %d active MarocHub vehicle URLs", len(id_to_url))
    return id_to_url


def fetch_detail_phone(detail_url: str) -> Optional[str]:
    """Fetch detail page and extract live phone."""
    try:
        r = requests.get(detail_url, headers=DEFAULT_HEADERS, timeout=6)
        if r.status_code == 200:
            html = r.text
            tels = re.findall(r'href=["\']tel:([^"\']+)["\']', html, re.I)
            for t in tels:
                p = extract_moroccan_phone(t)
                if p:
                    return p
            was = re.findall(r'(?:wa\.me/|api\.whatsapp\.com/send\?phone=)(\+?\d+)', html, re.I)
            for w in was:
                p = extract_moroccan_phone(w)
                if p:
                    return p
            return extract_moroccan_phone(html)
    except Exception as e:
        logger.debug("Failed to fetch detail %s: %s", detail_url, e)
    return None


def backfill_marochub(raw_dir: Path):
    logger.info("Backfilling MarocHub datasets...")
    id_to_url = fetch_marochub_catalog()

    cache_file = ROOT_DIR / "data" / "tracking" / "marochub_phones_cache.json"
    url_to_phone = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                url_to_phone = json.load(f)
            logger.info("Loaded %d cached phone numbers from %s", len(url_to_phone), cache_file.name)
        except Exception:
            url_to_phone = {}

    missing_urls = set(id_to_url.values()) - set(url_to_phone.keys())
    if missing_urls:
        logger.info("Fetching seller phone numbers for %d active MarocHub listings...", len(missing_urls))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {executor.submit(fetch_detail_phone, url): url for url in missing_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    phone = future.result()
                    if phone:
                        url_to_phone[url] = phone
                except Exception as e:
                    logger.debug("Error in thread: %s", e)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(url_to_phone, f, indent=2)

    logger.info("Total %d valid phone numbers available for MarocHub listings", len(url_to_phone))

    for p in raw_dir.glob("*marochub*.*"):
        if p.suffix in [".csv", ".parquet"]:
            logger.info("Processing %s", p.name)
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            df["seller_phone"] = df["seller_phone"].astype(object)
            df["seller_phone_hash"] = df["seller_phone_hash"].astype(object)

            for idx, row in df.iterrows():
                l_id = str(row.get("listing_id") or "")
                # short hex is last 8 chars of uuid or listing_id
                short_id = l_id.replace("-", "")[-8:]
                if short_id in id_to_url:
                    real_url = id_to_url[short_id]
                    df.at[idx, "url"] = real_url
                    if real_url in url_to_phone:
                        ph = url_to_phone[real_url]
                        df.at[idx, "seller_phone"] = ph
                        df.at[idx, "seller_phone_hash"] = hash_phone(ph)
                else:
                    # If existing phone is not empty, compute hash
                    cur_p = str(row.get("seller_phone") or "").strip()
                    if cur_p and cur_p != "nan":
                        df.at[idx, "seller_phone_hash"] = hash_phone(cur_p)

            # Ensure all non-null seller_phones have valid hashes
            df["seller_phone_hash"] = df["seller_phone"].apply(
                lambda x: hash_phone(x) if pd.notna(x) and str(x).strip() and str(x).strip() != "nan" else ""
            )

            if p.suffix == ".parquet":
                df.to_parquet(p, index=False)
            else:
                df.to_csv(p, index=False)
            logger.info("Updated %s with %d rows", p.name, len(df))


def main():
    raw_dir = ROOT_DIR / "data" / "raw"
    backfill_autocash(raw_dir)
    backfill_kifal(raw_dir)
    backfill_marochub(raw_dir)

    logger.info("Consolidating updated master database...")
    update_master_database(raw_dir)
    logger.info("Master database updated successfully.")


if __name__ == "__main__":
    main()
