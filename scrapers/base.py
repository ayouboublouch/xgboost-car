#!/usr/bin/env python3
"""
Base Scraper Class for Autohouse.ma Multi-Source Ingestion Pipeline
-------------------------------------------------------------------
Enforces:
- Strict alignment with the 24 standardized fields defined in the Cahier des Charges
- Common session management with Chrome TLS fingerprint impersonation (curl_cffi)
- Randomized rate limiting to protect against IP bans on GitHub Actions
- Parquet & CSV persistence in data/raw/{source}_YYYY-MM-DD.parquet / .csv
"""

import abc
import datetime
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

USE_CURL_CFFI = False
requests = None
try:
    from curl_cffi import requests as curl_requests
    USE_CURL_CFFI = True
except ImportError:
    try:
        import requests
    except ImportError:
        requests = None

import pandas as pd

logger = logging.getLogger("scraper.base")

# Target schema mandated by Autohouse.ma Cahier des Charges (24 standardized fields)
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


class BaseScraper(abc.ABC):
    """Abstract Base Class for all Moroccan car platform scrapers."""

    source_name: str = "base"

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
        """Initialize HTTP session with TLS impersonation or standard headers."""
        s = None
        if USE_CURL_CFFI:
            s = curl_requests.Session(impersonate="chrome124")
        elif requests is not None:
            s = requests.Session()
        else:
            logger.warning("[%s] Neither curl_cffi nor requests installed; using urllib fallback.", self.source_name)
            return None

        s.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
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

    def sleep(self):
        """Randomized rate limiting delay."""
        duration = random.uniform(self.delay_min, self.delay_max)
        time.sleep(duration)

    def fetch_page(
        self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Fetch page content with retries, 15s timeout, and fallback."""
        default_ua = random.choice(USER_AGENTS)
        for attempt in range(1, self.max_retries + 1):
            if self.session is not None:
                try:
                    self.session.headers["User-Agent"] = default_ua
                    if headers:
                        self.session.headers.update(headers)

                    response = self.session.get(url, params=params, timeout=15)
                    if response.status_code == 200:
                        return response.text
                    elif response.status_code == 404:
                        logger.debug("[%s] 404 Not Found: %s", self.source_name, url)
                        return None
                    elif response.status_code in (403, 429):
                        wait = 2.0 * attempt + random.uniform(0.5, 1.5)
                        logger.warning(
                            "[%s] HTTP %d on %s. Backoff %.1fs (attempt %d/%d)",
                            self.source_name,
                            response.status_code,
                            url,
                            wait,
                            attempt,
                            self.max_retries,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "[%s] Unexpected HTTP %d for %s", self.source_name, response.status_code, url
                        )
                except Exception as e:
                    logger.warning(
                        "[%s] Request error for %s: %s (attempt %d/%d). Trying fallback...",
                        self.source_name,
                        url,
                        e,
                        attempt,
                        self.max_retries,
                    )
            
            # Fallback directly using standard requests or urllib if session had issue
            try:
                if requests is not None:
                    r = requests.get(url, params=params, headers={"User-Agent": default_ua}, timeout=15)
                    if r.status_code == 200:
                        return r.text
                else:
                    import urllib.request
                    import ssl
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    full_url = url
                    if params:
                        from urllib.parse import urlencode
                        sep = "&" if "?" in url else "?"
                        full_url = f"{url}{sep}{urlencode(params)}"
                    req = urllib.request.Request(
                        full_url,
                        headers={
                            "User-Agent": default_ua,
                            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                        }
                    )
                    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                        if resp.status == 200:
                            return resp.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            time.sleep(1.5 * attempt)
        return None

    def clean_numeric(self, val: Any) -> Optional[float]:
        """Helper to extract clean float from strings like '120 000 DH' or '150 000 km'."""
        if val is None or pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).replace("\xa0", " ")
        m = re.search(r"(\d+[\d\s]*)", s)
        if m:
            clean = re.sub(r"\s+", "", m.group(1))
            try:
                return float(clean)
            except ValueError:
                return None
        return None

    def clean_year(self, val: Any) -> Optional[float]:
        """Extract a plausible 4-digit automotive year (1980 - 2027)."""
        if val is None or pd.isna(val):
            return None
        s = str(val)
        m = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", s)
        if m:
            return float(m.group(1))
        return None

    def validate_and_format_record(self, raw_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and harmonize a raw dictionary into the strict 24-field Cahier des Charges schema."""
        listing_id = str(raw_record.get("listing_id") or "").strip()
        if not listing_id:
            return None

        # Check in-memory seen ids for this run
        if listing_id in self.seen_ids:
            return None
        self.seen_ids.add(listing_id)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Standardize record
        record: Dict[str, Any] = {
            "listing_id": listing_id,
            "url": str(raw_record.get("url") or "").strip(),
            "source": self.source_name,
            "date_posted": str(raw_record.get("date_posted") or now_str[:10])[:10],
            "date_scraped": str(raw_record.get("date_scraped") or now_str),
            "title_raw": str(raw_record.get("title_raw") or "").strip(),
            "brand": str(raw_record.get("brand") or "").strip(),
            "model": str(raw_record.get("model") or "").strip(),
            "trim": str(raw_record.get("trim") or "").strip(),
            "year": self.clean_year(raw_record.get("year")),
            "mileage_km": self.clean_numeric(raw_record.get("mileage_km")),
            "fuel_type": str(raw_record.get("fuel_type") or "").strip(),
            "transmission": str(raw_record.get("transmission") or "").strip(),
            "fiscal_power_cv": str(raw_record.get("fiscal_power_cv") or "").strip(),
            "customs_status": str(raw_record.get("customs_status") or "").strip(),
            "condition": str(raw_record.get("condition") or "").strip(),
            "owners_count": str(raw_record.get("owners_count") or "").strip(),
            "doors_count": self.clean_numeric(raw_record.get("doors_count")),
            "seller_type": str(raw_record.get("seller_type") or "Particulier").strip(),
            "city": str(raw_record.get("city") or "").strip(),
            "region": str(raw_record.get("region") or "").strip(),
            "price_mad": self.clean_numeric(raw_record.get("price_mad")),
            "photos_count": self.clean_numeric(raw_record.get("photos_count")) or 0.0,
            "description_raw": str(raw_record.get("description_raw") or "").strip(),
        }
        return record

    @abc.abstractmethod
    def scrape(self, max_pages: int = 10, **kwargs) -> pd.DataFrame:
        """Entry point to execute the scraper. Must be implemented by subclasses."""
        pass

    def save_output(self, df: pd.DataFrame) -> None:
        """Persist harmonized DataFrame to Parquet & CSV. Fails loudly on empty result."""
        if df is None or df.empty or len(df) == 0:
            raise RuntimeError(
                f"Scraper for {self.source_name} returned 0 rows. Likely blocked by anti-bot."
            )

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        parquet_path = self.output_dir / f"{self.source_name}_{today_str}.parquet"
        csv_path = self.output_dir / f"{self.source_name}_{today_str}.csv"

        # Enforce schema columns
        for col in SCHEMA_FIELDS:
            if col not in df.columns:
                df[col] = None
        df = df[SCHEMA_FIELDS].drop_duplicates(subset=["listing_id"])

        if len(df) == 0:
            raise RuntimeError(
                f"Scraper for {self.source_name} returned 0 rows after deduplication. Likely blocked by anti-bot."
            )

        try:
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            logger.info("[%s] Saved %d records to %s", self.source_name, len(df), parquet_path)
        except Exception as e:
            logger.error("[%s] Failed to save Parquet: %s", self.source_name, e)

        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("[%s] Saved %d records to %s", self.source_name, len(df), csv_path)
        except Exception as e:
            logger.error("[%s] Failed to save CSV: %s", self.source_name, e)
