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
import hashlib
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

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

import pandas as pd

logger = logging.getLogger("scraper.base")

# Target schema mandated by Autohouse.ma Cahier des Charges (standardized fields + phone & privacy hash)
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
    "seller_phone",
    "seller_phone_hash",
    "city",
    "region",
    "price_mad",
    "photos_count",
    "description_raw",
]


BLACKLIST_PHONES = {"0520428686", "0522000000", "0802000000"}


def extract_moroccan_phone(text: Any) -> Optional[str]:
    r"""
    Extract and normalize Moroccan customer/seller phone numbers.
    Pattern matches mobile and landline numbers (05, 06, 07):
    Regex: r'(?:(?:\+|00)212|0)\s*[5-7](?:[\s\.-]*\d{2}){4}'
    Normalizes to 10-digit format (e.g., '+212612345678' -> '0612345678').
    Filters out known platform support/customer service numbers (BLACKLIST_PHONES).
    """
    if text is None or pd.isna(text):
        return None
    s = str(text)
    pattern = r"(?:(?:\+|00)212|0)\s*[5-7](?:[\s\.-]*\d{2}){4}"
    match = re.search(pattern, s)
    if not match:
        return None

    raw_phone = match.group(0)
    # Remove all formatting characters (spaces, dots, hyphens)
    digits = re.sub(r"[\s\.\-]+", "", raw_phone)
    if digits.startswith("+212"):
        digits = "0" + digits[4:]
    elif digits.startswith("00212"):
        digits = "0" + digits[5:]

    if len(digits) == 10 and digits.startswith("0") and digits[1] in "567":
        if digits in BLACKLIST_PHONES:
            return None
        return digits
    return None


def hash_phone(phone: Optional[str]) -> Optional[str]:
    """Compute SHA-256 hash of normalized phone number for privacy-compliant repost deduplication."""
    if not phone or not isinstance(phone, str) or str(phone).strip() == "" or str(phone).lower() == "nan":
        return None
    return hashlib.sha256(phone.strip().encode("utf-8")).hexdigest()

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
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 15,
        max_retries: Optional[int] = None,
    ) -> Optional[str]:
        """Fetch page content with configurable retries and timeout."""
        default_ua = random.choice(USER_AGENTS)
        retries = max_retries if max_retries is not None else self.max_retries
        for attempt in range(1, retries + 1):
            if self.session is not None:
                try:
                    self.session.headers["User-Agent"] = default_ua
                    if headers:
                        self.session.headers.update(headers)

                    response = self.session.get(url, params=params, timeout=timeout, verify=False)
                    resp_len = len(response.text) if hasattr(response, "text") and response.text else 0
                    if response.status_code == 200:
                        if resp_len < 2000:
                            logger.warning(
                                "[%s] HTTP 200 on %s but small response (%d bytes). Headers: %s | Preview: %r",
                                self.source_name,
                                url,
                                resp_len,
                                dict(getattr(response, "headers", {})),
                                response.text[:300] if hasattr(response, "text") else "",
                            )
                        return response.text
                    elif response.status_code == 404:
                        logger.debug("[%s] 404 Not Found: %s", self.source_name, url)
                        return None
                    else:
                        logger.warning(
                            "[%s] HTTP %d on %s. Headers: %s | Preview: %r",
                            self.source_name,
                            response.status_code,
                            url,
                            dict(getattr(response, "headers", {})),
                            response.text[:300] if hasattr(response, "text") else "",
                        )
                        if response.status_code in (403, 429):
                            wait = 2.0 * attempt + random.uniform(0.5, 1.5)
                            time.sleep(wait)
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
                    r = requests.get(url, params=params, headers={"User-Agent": default_ua}, timeout=timeout, verify=False)
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
                    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
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

        # Extract and normalize Moroccan customer/seller phone
        seller_phone = None
        raw_phone = raw_record.get("seller_phone")
        if raw_phone:
            seller_phone = extract_moroccan_phone(raw_phone)
        if not seller_phone and raw_record.get("description_raw"):
            seller_phone = extract_moroccan_phone(raw_record.get("description_raw"))
        if not seller_phone and raw_record.get("title_raw"):
            seller_phone = extract_moroccan_phone(raw_record.get("title_raw"))

        # Discard if blacklisted or invalid format
        if seller_phone in BLACKLIST_PHONES:
            seller_phone = None

        if seller_phone:
            if not (len(seller_phone) == 10 and seller_phone.startswith("0") and seller_phone[1] in "567"):
                seller_phone = None

        # Compute or preserve SHA-256 hash of phone
        if not seller_phone:
            seller_phone_hash = None
        else:
            seller_phone_hash = raw_record.get("seller_phone_hash")
            if not seller_phone_hash or pd.isna(seller_phone_hash) or str(seller_phone_hash).strip() == "" or str(seller_phone_hash).lower() == "nan":
                seller_phone_hash = hash_phone(seller_phone)
            else:
                seller_phone_hash = str(seller_phone_hash).strip()

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
            "seller_phone": seller_phone,
            "seller_phone_hash": seller_phone_hash,
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
        """Persist harmonized DataFrame to Parquet & CSV."""
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        parquet_path = self.output_dir / f"{self.source_name}_{today_str}.parquet"
        csv_path = self.output_dir / f"{self.source_name}_{today_str}.csv"

        if df is None or df.empty or len(df) == 0:
            logger.warning("[%s] Scraper harvested 0 records. Writing nothing.", self.source_name)
            return

        # Enforce schema columns
        for col in SCHEMA_FIELDS:
            if col not in df.columns:
                df[col] = None
        df = df[SCHEMA_FIELDS].drop_duplicates(subset=["listing_id"])

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


def purge_empty_raw_files(raw_dir: Path) -> List[str]:
    """
    Scan all existing CSV and Parquet files in data/raw/.
    Delete any file with a size <= 500 bytes or containing only headers (e.g. avito_2026-09-07.csv).
    Also delete Parquet files containing 0 rows.
    Safeguards: Never delete used_car_training_combined.csv.
    """
    purged: List[str] = []
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return purged

    for file_path in raw_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.name == "used_car_training_combined.csv" or file_path.name.startswith("avito_local_"):
            continue

        should_delete = False
        if file_path.suffix == ".csv":
            size = file_path.stat().st_size
            if size <= 500:
                should_delete = True
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if len(lines) <= 1:
                        should_delete = True
                except Exception:
                    pass
        elif file_path.suffix == ".parquet":
            try:
                df_p = pd.read_parquet(file_path)
                if len(df_p) == 0:
                    should_delete = True
            except Exception:
                if file_path.stat().st_size <= 500:
                    should_delete = True

        if should_delete:
            try:
                file_path.unlink()
                purged.append(file_path.name)
                logger.info("Purged empty / header-only file: %s", file_path.name)
            except Exception as e:
                logger.warning("Could not delete %s: %s", file_path.name, e)

    return purged


def consolidate_daily_scrapes(
    raw_dir: Path, target_date: Optional[str] = None
) -> Optional[Path]:
    """
    Find all scraped batches generated for target_date across sources (Avito, Moteur, Wandaloo, etc.).
    Concatenate them into a single standardized file: data/raw/scraped_combined_YYYY-MM-DD.csv.
    Ensure deduplication on (listing_id, url) while merging.
    Remove individual small/partial temporary batch files for that day once consolidated.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None

    # First purge empty/dummy files
    purge_empty_raw_files(raw_dir)

    # Determine dates to consolidate if target_date is not specified
    dates_to_process = set()
    if target_date:
        dates_to_process.add(target_date)
    else:
        # Detect all dates present in format {source}_{YYYY-MM-DD}.csv
        for f in raw_dir.glob("*_*.csv"):
            if f.name.startswith("scraped_combined_") or f.name == "used_car_training_combined.csv":
                continue
            m = re.search(r"_(\d{4}-\d{2}-\d{2})\.csv$", f.name)
            if m:
                dates_to_process.add(m.group(1))

    consolidated_files = []

    for d in sorted(list(dates_to_process)):
        pattern = f"*_{d}.csv"
        matching_csvs = [
            f
            for f in raw_dir.glob(pattern)
            if not f.name.startswith("scraped_combined_")
            and not f.name.startswith("avito_local_")
            and f.name != "used_car_training_combined.csv"
        ]
        if not matching_csvs:
            continue

        frames = []
        files_to_remove = []

        out_csv = raw_dir / f"scraped_combined_{d}.csv"
        out_parquet = raw_dir / f"scraped_combined_{d}.parquet"

        # If a combined file already exists for date d, load it first to merge & deduplicate
        if out_csv.exists():
            try:
                existing_df = pd.read_csv(out_csv, low_memory=False)
                if not existing_df.empty and len(existing_df) > 0:
                    frames.append(existing_df)
            except Exception as e:
                logger.warning("Could not read existing combined file %s: %s", out_csv.name, e)

        for f in matching_csvs:
            try:
                df = pd.read_csv(f, low_memory=False)
                if not df.empty and len(df) > 0:
                    frames.append(df)
                    files_to_remove.append(f)
                    pq = f.with_suffix(".parquet")
                    if pq.exists():
                        files_to_remove.append(pq)
            except Exception as e:
                logger.warning("Could not read batch file %s: %s", f.name, e)

        if not frames:
            continue

        merged_df = pd.concat(frames, ignore_index=True)

        # Deduplicate on (listing_id, url) if available
        dedup_cols = []
        for c in ["listing_id", "url"]:
            if c in merged_df.columns:
                dedup_cols.append(c)

        if dedup_cols:
            merged_df = merged_df.drop_duplicates(subset=dedup_cols, keep="last")
        else:
            merged_df = merged_df.drop_duplicates()

        # Enforce SCHEMA_FIELDS order
        for col in SCHEMA_FIELDS:
            if col not in merged_df.columns:
                merged_df[col] = None
        merged_df = merged_df[SCHEMA_FIELDS]


        merged_df.to_csv(out_csv, index=False, encoding="utf-8")
        try:
            for col in merged_df.columns:
                if merged_df[col].dtype == "object":
                    merged_df[col] = merged_df[col].apply(
                        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" and str(x).lower() not in ("nan", "none", "<na>") else None
                    )
            merged_df.to_parquet(out_parquet, index=False, engine="pyarrow")
        except Exception:
            pass

        logger.info(
            "Consolidated %d batches for %s into %s (%d records)",
            len(frames),
            d,
            out_csv.name,
            len(merged_df),
        )
        consolidated_files.append(out_csv)

        # Clean up partial temporary batch files for that day
        for tmp_f in files_to_remove:
            try:
                if tmp_f.exists() and tmp_f.resolve() != out_csv.resolve() and tmp_f.resolve() != out_parquet.resolve():
                    tmp_f.unlink()
                    logger.info("Removed temporary batch file: %s", tmp_f.name)
            except Exception as e:
                logger.warning("Failed to remove temporary file %s: %s", tmp_f.name, e)

    return consolidated_files[0] if consolidated_files else None
