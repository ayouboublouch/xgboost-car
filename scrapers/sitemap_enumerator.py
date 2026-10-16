#!/usr/bin/env python3
"""
Sitemap XML Harvester & Incremental ID Probing Engine
------------------------------------------------------
Implements:
1. Sitemap XML Harvesting:
   - Streams and parses sitemap.xml / sitemap_index.xml across Moroccan auto platforms
   - Bypasses search pagination caps and extracts unindexed listing URLs directly
2. Incremental ID Probing:
   - Exploits deterministic URL schemas:
     - MarocAnnonces: /categorie/314/Voitures-occasion/annonce/{id}/auto.html
     - Moteur.ma: /fr/voiture/achat-voiture-occasion/detail-annonce/{id}/...
   - Probes ID ranges backward concurrently to recover unindexed and archived listings
"""

import argparse
import datetime
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRAPERS_DIR = Path(__file__).resolve().parent
if str(SCRAPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_DIR))

import pandas as pd

try:
    from scrapers.base import (
        BaseScraper,
        SCHEMA_FIELDS,
        load_seen_listing_ids,
        append_seen_listing_ids,
    )
    from scrapers.marocannonces_scraper import MarocAnnoncesScraper
except ImportError:
    from base import (
        BaseScraper,
        SCHEMA_FIELDS,
        load_seen_listing_ids,
        append_seen_listing_ids,
    )
    from marocannonces_scraper import MarocAnnoncesScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper.enumerator")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class SitemapHarvester(BaseScraper):
    source_name = "sitemap_harvester"

    SITEMAP_REGISTRY = {
        "siaracash": "https://siaracash.ma/sitemap.xml",
        "auto24": "https://auto24.ma/sitemap.xml",
    }

    def fetch_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Fetch and parse XML sitemap, recursively traversing sub-sitemaps."""
        logger.info("[%s] Fetching sitemap XML from: %s", self.source_name, sitemap_url)
        content = self.fetch_page(sitemap_url, headers=DEFAULT_HEADERS)
        if not content:
            return []

        urls = []
        try:
            # Strip XML namespaces for simplified element matching
            content_clean = re.sub(r' xmlns="[^"]+"', "", content, count=1)
            root = ET.fromstring(content_clean)

            # Check if sitemapindex
            if root.tag.endswith("sitemapindex"):
                for sm in root.findall(".//sitemap"):
                    loc = sm.find("loc")
                    if loc is not None and loc.text:
                        sub_urls = self.fetch_sitemap_urls(loc.text.strip())
                        urls.extend(sub_urls)
            else:
                # urlset
                for u in root.findall(".//url"):
                    loc = u.find("loc")
                    if loc is not None and loc.text:
                        urls.append(loc.text.strip())
        except Exception as e:
            logger.warning("[%s] XML parsing error for %s: %s", self.source_name, sitemap_url, e)
            # Fallback to regex loc extraction
            matches = re.findall(r"<loc>(.*?)</loc>", content)
            urls = [m.strip() for m in matches if m.strip()]

        logger.info("[%s] Discovered %d URLs in sitemap %s", self.source_name, len(urls), sitemap_url)
        return urls

    def filter_car_urls(self, urls: List[str], source_hint: str) -> List[str]:
        """Filter URLs to retain only valid vehicle listing detail links."""
        filtered = []
        for u in urls:
            u_clean = u.split("?")[0].rstrip("/")
            if source_hint == "siaracash":
                if any(k in u_clean for k in ["/voiture-occasion-maroc/", "/annonce/", "/detail/", "/marque/"]):
                    filtered.append(u)
            elif source_hint == "auto24":
                if any(k in u_clean for k in ["/acheter-voitures/", "/car/"]):
                    filtered.append(u)
            else:
                if re.search(r"/\d+[-_]|/annonce/|/detail/|/voiture/", u_clean):
                    filtered.append(u)
        return list(dict.fromkeys(filtered))

    def scrape(self, **kwargs) -> pd.DataFrame:
        """Satisfy BaseScraper interface by returning discovered sitemap URLs as DataFrame."""
        source = kwargs.get("source", "siaracash")
        sitemap_url = self.SITEMAP_REGISTRY.get(source)
        if not sitemap_url:
            return pd.DataFrame()
        urls = self.fetch_sitemap_urls(sitemap_url)
        filtered = self.filter_car_urls(urls, source)
        return pd.DataFrame([{"url": u, "source": source} for u in filtered])


class IncrementalIDProber(BaseScraper):
    source_name = "id_prober"

    def __init__(self, output_dir: str = "data/raw", concurrency: int = 8):
        super().__init__(output_dir=output_dir)
        self.concurrency = concurrency
        self.maroc_scraper = MarocAnnoncesScraper(output_dir=output_dir)

    def find_latest_marocannonces_id(self) -> int:
        """Fetch page 1 of MarocAnnonces to identify current maximum listing ID."""
        url = "https://www.marocannonces.com/categorie/314/Voitures-occasion.html"
        html = self.fetch_page(url, headers=DEFAULT_HEADERS)
        if not html:
            return 10474000
        ids = [int(x) for x in re.findall(r"annonce/(\d+)", html)]
        if ids:
            max_id = max(ids)
            logger.info("[%s] Detected current latest MarocAnnonces listing ID: %d", self.source_name, max_id)
            return max_id
        return 10474000

    def probe_single_marocannonces_id(self, listing_id: int) -> Optional[Dict[str, Any]]:
        """Probe single MarocAnnonces ID with deterministic slug-independent URL."""
        url = f"https://www.marocannonces.com/categorie/314/Voitures-occasion/annonce/{listing_id}/auto.html"
        try:
            # Use short timeout to quickly identify active vs expired listings
            resp = self.session.get(url, headers=DEFAULT_HEADERS, timeout=(4, 8), allow_redirects=True)
            if resp.status_code == 200:
                if "annonce" in resp.url and "Voitures-occasion" in resp.url:
                    # Valid listing page
                    soup = None
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception:
                        pass
                    if soup:
                        rec = self.maroc_scraper.parse_listing(soup)
                        if rec:
                            rec["listing_id"] = str(listing_id)
                            rec["url"] = resp.url
                            return rec
        except Exception:
            pass
        return None

    def probe_marocannonces(
        self,
        start_id: Optional[int] = None,
        count: int = 200,
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Scan backward from start_id for count iterations using concurrent workers.
        Bypasses search engine indexing and pagination entirely.
        """
        if not start_id:
            start_id = self.find_latest_marocannonces_id()

        target_ids = list(range(start_id, max(1, start_id - count), -1))
        seen_ids = load_seen_listing_ids()
        # Filter out IDs already in our seen database
        unseen_ids = [i for i in target_ids if str(i) not in seen_ids]
        logger.info(
            "[%s] Probing MarocAnnonces IDs [%d down to %d] (%d total, %d unseen) with concurrency=%d ...",
            self.source_name,
            start_id,
            start_id - count,
            len(target_ids),
            len(unseen_ids),
            self.concurrency,
        )

        all_records = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_id = {executor.submit(self.probe_single_marocannonces_id, i): i for i in unseen_ids}
            done_count = 0
            for future in as_completed(future_to_id):
                done_count += 1
                rec = future.result()
                if rec:
                    all_records.append(rec)
                if done_count % 50 == 0 or done_count == len(unseen_ids):
                    logger.info(
                        "[%s] Probed %d/%d IDs (harvested %d active listings so far)",
                        self.source_name,
                        done_count,
                        len(unseen_ids),
                        len(all_records),
                    )

        if not all_records:
            logger.warning("[%s] ID probing harvested 0 listings.", self.source_name)
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if "listing_id" in df.columns:
            df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)

        logger.info("[%s] Completed ID probing. Total active listings recovered: %d", self.source_name, len(df))
        self.save_output(df, custom_filename=output_filename or f"probed_marocannonces_{datetime.date.today()}.csv")
        return df

    def scrape(self, max_pages: int = 100, count: Optional[int] = None, start_id: Optional[int] = None, output_filename: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Satisfy BaseScraper interface by running sequential ID probing."""
        total_probe = count or (max_pages * 20)
        return self.probe_marocannonces(start_id=start_id, count=total_probe, output_filename=output_filename)


def main():
    parser = argparse.ArgumentParser(description="Sitemap XML Harvester & Incremental ID Prober")
    parser.add_argument("--mode", type=str, choices=["sitemap", "id-probe"], default="id-probe", help="Operation mode")
    parser.add_argument("--source", type=str, default="marocannonces", choices=["marocannonces", "siaracash", "auto24"])
    parser.add_argument("--start-id", type=int, default=None, help="Start ID for sequential probing")
    parser.add_argument("--count", type=int, default=100, help="Number of sequential IDs to probe")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    if args.mode == "sitemap":
        harvester = SitemapHarvester(output_dir=args.output_dir)
        sitemap_url = harvester.SITEMAP_REGISTRY.get(args.source)
        if sitemap_url:
            urls = harvester.fetch_sitemap_urls(sitemap_url)
            filtered = harvester.filter_car_urls(urls, args.source)
            print(f"Total discovered vehicle URLs for {args.source}: {len(filtered)}")
            for u in filtered[:5]:
                print(f"  {u}")
    elif args.mode == "id-probe":
        prober = IncrementalIDProber(output_dir=args.output_dir, concurrency=args.concurrency)
        if args.source == "marocannonces":
            df = prober.probe_marocannonces(start_id=args.start_id, count=args.count)
            print(f"Successfully harvested {len(df)} records via sequential ID probing.")


if __name__ == "__main__":
    main()
