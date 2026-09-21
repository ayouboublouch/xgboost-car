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
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    s = str(text).strip()
    if s.endswith(".0"):
        s = s[:-2]
    # If 9 digits starting with 5, 6, 7 (e.g. from float representation), prepend 0
    if len(s) == 9 and s[0] in "567" and s.isdigit():
        s = "0" + s

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


MOROCCAN_CITY_REGIONS: Dict[str, str] = {
    "casablanca": "Casablanca-Settat",
    "mohammedia": "Casablanca-Settat",
    "settat": "Casablanca-Settat",
    "berrechid": "Casablanca-Settat",
    "el jadida": "Casablanca-Settat",
    "bouskoura": "Casablanca-Settat",
    "nouaceur": "Casablanca-Settat",
    "tit mellil": "Casablanca-Settat",
    "mediouna": "Casablanca-Settat",
    "benslimane": "Casablanca-Settat",
    "dar bouazza": "Casablanca-Settat",
    "had soualem": "Casablanca-Settat",
    "rabat": "Rabat-Salé-Kénitra",
    "salé": "Rabat-Salé-Kénitra",
    "sale": "Rabat-Salé-Kénitra",
    "kénitra": "Rabat-Salé-Kénitra",
    "kenitra": "Rabat-Salé-Kénitra",
    "témara": "Rabat-Salé-Kénitra",
    "temara": "Rabat-Salé-Kénitra",
    "skhirat": "Rabat-Salé-Kénitra",
    "khémisset": "Rabat-Salé-Kénitra",
    "khemisset": "Rabat-Salé-Kénitra",
    "sidi kacem": "Rabat-Salé-Kénitra",
    "sidi slimane": "Rabat-Salé-Kénitra",
    "marrakech": "Marrakech-Safi",
    "safi": "Marrakech-Safi",
    "essaouira": "Marrakech-Safi",
    "el kelaa des sraghna": "Marrakech-Safi",
    "benguerir": "Marrakech-Safi",
    "tanger": "Tanger-Tétouan-Al Hoceïma",
    "tangier": "Tanger-Tétouan-Al Hoceïma",
    "tétouan": "Tanger-Tétouan-Al Hoceïma",
    "tetouan": "Tanger-Tétouan-Al Hoceïma",
    "larache": "Tanger-Tétouan-Al Hoceïma",
    "al hoceima": "Tanger-Tétouan-Al Hoceïma",
    "al hoceïma": "Tanger-Tétouan-Al Hoceïma",
    "chaouen": "Tanger-Tétouan-Al Hoceïma",
    "chefchaouen": "Tanger-Tétouan-Al Hoceïma",
    "ksar el kebir": "Tanger-Tétouan-Al Hoceïma",
    "asilah": "Tanger-Tétouan-Al Hoceïma",
    "fès": "Fès-Meknès",
    "fes": "Fès-Meknès",
    "meknès": "Fès-Meknès",
    "meknes": "Fès-Meknès",
    "taza": "Fès-Meknès",
    "sefrou": "Fès-Meknès",
    "zouagha": "Fès-Meknès",
    "agadir": "Souss-Massa",
    "inezgane": "Souss-Massa",
    "ait melloul": "Souss-Massa",
    "taroudant": "Souss-Massa",
    "tiznit": "Souss-Massa",
    "oujda": "Oriental",
    "nador": "Oriental",
    "berkane": "Oriental",
    "taourirt": "Oriental",
    "driouch": "Oriental",
    "béni mellal": "Béni Mellal-Khénifra",
    "beni mellal": "Béni Mellal-Khénifra",
    "khouribga": "Béni Mellal-Khénifra",
    "khénifra": "Béni Mellal-Khénifra",
    "khenifra": "Béni Mellal-Khénifra",
    "fquih ben salah": "Béni Mellal-Khénifra",
    "ouarzazate": "Drâa-Tafilalet",
    "errachidia": "Drâa-Tafilalet",
    "zagora": "Drâa-Tafilalet",
    "tinghir": "Drâa-Tafilalet",
    "guelmim": "Guelmim-Oued Noun",
    "tan-tan": "Guelmim-Oued Noun",
    "laâyoune": "Laâyoune-Sakia El Hamra",
    "laayoune": "Laâyoune-Sakia El Hamra",
    "dakhla": "Dakhla-Oued Ed-Dahab",
}


def infer_moroccan_region(city: Optional[str]) -> str:
    """Map Moroccan city to official administrative region."""
    if not city or pd.isna(city):
        return ""
    c_lower = str(city).lower().strip()
    # Direct match
    if c_lower in MOROCCAN_CITY_REGIONS:
        return MOROCCAN_CITY_REGIONS[c_lower]
    # Substring search
    for k, v in MOROCCAN_CITY_REGIONS.items():
        if k in c_lower or c_lower in k:
            return v
    return ""


KNOWN_BRANDS: List[str] = [
    "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "BYD", "Chery", "Chevrolet",
    "Chrysler", "Citroën", "Citroen", "Cupra", "Dacia", "Daihatsu", "Dodge", "DS", "Ferrari",
    "Fiat", "Ford", "Geely", "GMC", "Great Wall", "Haval", "Honda", "Hummer",
    "Hyundai", "Infiniti", "Isuzu", "Iveco", "Jaguar", "Jeep", "Kia", "Lada", "Lamborghini",
    "Lancia", "Land Rover", "Lexus", "Maserati", "Mahindra", "Mazda", "Mercedes-Benz",
    "Mercedes", "MG", "Mini", "Mitsubishi", "Nissan", "Opel", "Peugeot", "Porsche",
    "Range Rover", "Renault", "Rolls-Royce", "Rover", "Saab", "Seat", "Skoda", "Smart",
    "Ssangyong", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
]

BRAND_TYPO_MAP: Dict[str, str] = {
    "peugeut": "Peugeot",
    "peugot": "Peugeot",
    "peugeot": "Peugeot",
    "porch": "Porsche",
    "porsche": "Porsche",
    "renoult": "Renault",
    "renault": "Renault",
    "volswagen": "Volkswagen",
    "vw": "Volkswagen",
    "volkswagen": "Volkswagen",
    "hyandai": "Hyundai",
    "hyanday": "Hyundai",
    "hyndai": "Hyundai",
    "hyondai": "Hyundai",
    "hyundai": "Hyundai",
    "mercedes": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "mercides": "Mercedes-Benz",
    "benz": "Mercedes-Benz",
    "maseratti": "Maserati",
    "maserati": "Maserati",
    "mitusubshi": "Mitsubishi",
    "mitsubishi": "Mitsubishi",
    "bently": "Bentley",
    "bentley": "Bentley",
    "bmw": "BMW",
    "bwm": "BMW",
    "bm": "BMW",
    "geep": "Jeep",
    "jeep": "Jeep",
    "cetroen": "Citroën",
    "citroen": "Citroën",
    "citroën": "Citroën",
    "alfa": "Alfa Romeo",
    "alfa romeo": "Alfa Romeo",
    "land": "Land Rover",
    "land rover": "Land Rover",
    "range": "Land Rover",
    "range rover": "Land Rover",
    "rover": "Land Rover",
    "skoda": "Skoda",
    "koda": "Skoda",
    "dacia": "Dacia",
    "audi": "Audi",
    "audia3": "Audi",
    "fiat": "Fiat",
    "fiát": "Fiat",
    "fiât": "Fiat",
    "ford": "Ford",
    "nissan": "Nissan",
    "toyota": "Toyota",
    "kia": "Kia",
    "seat": "Seat",
    "opel": "Opel",
    "suzuki": "Suzuki",
    "chevrolet": "Chevrolet",
    "honda": "Honda",
    "jaguar": "Jaguar",
    "cupra": "Cupra",
    "ds": "DS",
    "byd": "BYD",
    "geely": "Geely",
    "chery": "Chery",
    "haval": "Haval",
    "mg": "MG",
}

STOP_WORDS_MODEL: Set[str] = {
    "diesel", "essence", "hybride", "hybrid", "hybri", "electrique", "électrique", "electriq", "gpl",
    "manuelle", "manuel", "automatique", "auto", "bva", "bvm",
    "à", "au", "en", "pour", "sur", "avec", "sans",
    "vendre", "vente",
    "presque", "neuf", "neuve", "etat", "état", "très", "tres", "bon", "bonne", "occasion",
    "peinture", "peintures", "origine",
    "premier", "premiere", "première", "1ere", "1ère", "1er", "main",
    "diw", "dedouane", "dédouané", "dédouanée", "dedouanee", "douane",
    "import", "importe", "importé", "importee", "importée", "allemagne", "france",
    "modèle", "modele", "model",
    "derkaoui", "l3amra", "khawya",
    "options", "option", "tt", "toutes", "toute", "tout", "full",
    "maroc", "ww", "www",
    "km",
    "gtline", "gt-line", "sline", "s-line", "rline", "r-line", "amg-line",
    "allure", "business", "active", "feel", "shine", "intens", "zen", "confort", "titanium",
    "luxe", "exclusive", "prestige",
}


def clean_brand_and_model(
    brand: Optional[str],
    model: Optional[str],
    title_raw: Optional[str] = "",
    trim: Optional[str] = "",
) -> Tuple[str, str, str]:
    """
    Harmonize brand typos (e.g. 'Peugeut' -> 'Peugeot') and strip title noise
    (transmission, fuel, city, year, sale phrases) from model tokens.
    """
    b_str = str(brand or "").strip()
    m_str = str(model or "").strip()
    t_str = str(title_raw or "").strip()
    tr_str = str(trim or "").strip()

    # 1. Harmonize brand typos
    b_lower = b_str.lower()
    b_clean = BRAND_TYPO_MAP.get(b_lower, b_str)
    if not b_clean or b_clean.lower() in ("nan", "none", ""):
        b_clean = "Autre"

    # If brand itself is still unknown or Autre, try detecting from title_raw
    if b_clean == "Autre" and t_str:
        for typo, canonical in BRAND_TYPO_MAP.items():
            if re.search(rf"\b{re.escape(typo)}\b", t_str, re.IGNORECASE):
                b_clean = canonical
                break

    # 2. Check if model is the brand name or brand typo (e.g. model="Peugeut", brand="Peugeot")
    if (
        m_str.lower() in ("peugeut", "peugot", "peugeot")
        or m_str.lower() == b_clean.lower()
        or m_str.lower() in BRAND_TYPO_MAP
    ):
        # Recover real model from trim or title_raw
        if tr_str and tr_str.lower() not in ("nan", "none", ""):
            tr_tokens = tr_str.split()
            m_str = tr_tokens[0]
            tr_str = " ".join(tr_tokens[1:])
        elif t_str:
            t_clean = re.sub(
                rf"\b({re.escape(b_clean)}|peugeut|peugot|mercedes|benz|audi|bmw|renault|dacia|volkswagen|vw)\b",
                " ",
                t_str,
                flags=re.IGNORECASE,
            )
            t_tokens = [tok for tok in t_clean.split() if tok]
            if t_tokens:
                m_str = t_tokens[0]
                if len(t_tokens) > 1 and not tr_str:
                    tr_str = " ".join(t_tokens[1:])

    # 3. Strip title noise from model tokens
    s = re.sub(r"[\(\)\[\],;\*\?!\"/\\\|]", " ", m_str)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()

    kept_tokens = []
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        # 4-digit year check (allow Peugeot 2008, 3008, 5008 as first token)
        if re.match(r"^(19\d\d|20\d\d)$", tok_lower):
            if i == 0 and tok_lower in ("2008", "3008", "5008") and b_clean.lower() == "peugeot":
                kept_tokens.append(tok)
                continue
            else:
                break

        # Engine specs & power
        if re.match(r"^\d+[\.,]\d+$", tok_lower) or re.match(r"^\d+(cv|ch|eat\d*)$", tok_lower):
            break
        if tok_lower in ("hdi", "dci", "tdi", "cdi", "crdi", "tfsi", "tsi", "cv", "ch", "eat8", "4x4", "4motion", "4matic"):
            break

        # Stop words & Moroccan cities
        if tok_lower in STOP_WORDS_MODEL or tok_lower in MOROCCAN_CITY_REGIONS:
            break

        # Standalone lowercase 'a' (preposition in French, e.g. 'a vendre')
        if tok_lower == "a" and (i == 0 or (i > 0 and kept_tokens[-1].lower() != "classe")):
            if i + 1 < len(tokens) and tokens[i + 1].lower() in (
                "vendre", "casablanca", "rabat", "tanger", "marrakech", "agadir", "fes"
            ):
                break

        # Strip repeated brand name prefix
        if i == 0 and (tok_lower == b_clean.lower() or tok_lower in ("mercedes", "benz", "rover", "peugeut", "peugot")):
            continue

        kept_tokens.append(tok)
        if len(kept_tokens) >= 3 and not (kept_tokens[0].lower() in ("range", "série", "serie", "grand", "classe", "land")):
            break

    m_final = " ".join(kept_tokens).strip()
    if not m_final or m_final.lower() in ("nan", "none", ""):
        m_final = "Autre"

    m_final = m_final.title() if m_final.islower() else m_final
    if not tr_str or tr_str.lower() in ("nan", "none", ""):
        tr_str = ""

    return b_clean, m_final, tr_str


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
]

TRACKING_DIR = Path("data/tracking")
SEEN_IDS_FILE = TRACKING_DIR / "seen_listing_ids.txt"
PROGRESS_FILE = TRACKING_DIR / "scraping_progress.json"
SESSIONS_FILE = TRACKING_DIR / "scraping_sessions.jsonl"


def load_scraping_progress(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """Load per-platform scraping progress dictionary from JSON."""
    path = Path(filepath or PROGRESS_FILE)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read scraping progress from %s: %s", path, e)
    return {
        "moteur": {"last_page": 1, "total_scraped": 0, "last_updated": None},
        "wandaloo": {"last_page": 1, "total_scraped": 0, "last_updated": None},
        "avito": {"last_page": 1, "total_scraped": 0, "last_updated": None},
    }


def update_scraping_progress(
    source: str,
    last_page: int,
    count_added: int = 0,
    filepath: Optional[Path] = None,
) -> Dict[str, Any]:
    """Update and persist progress statistics for a given platform."""
    path = Path(filepath or PROGRESS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = load_scraping_progress(path)

    src_key = source.lower().replace(".ma", "").strip()
    if src_key not in progress:
        progress[src_key] = {"last_page": 1, "total_scraped": 0, "last_updated": None}

    current_total = progress[src_key].get("total_scraped", 0)
    progress[src_key]["last_page"] = max(last_page, progress[src_key].get("last_page", 1))
    progress[src_key]["total_scraped"] = current_total + count_added
    progress[src_key]["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        logger.info("Updated scraping progress for %s: last_page=%d | total=%d", src_key, progress[src_key]["last_page"], progress[src_key]["total_scraped"])
    except Exception as e:
        logger.warning("Could not write scraping progress to %s: %s", path, e)

    return progress


def record_scraping_session(
    scraper_name: str,
    source: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    pages_scraped: int,
    records_extracted: int,
    records_added: int,
    duplicates_skipped: int,
    start_page: int = 1,
    end_page: int = 1,
    status: str = "success",
    notes: str = "",
    filepath: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Log a structured scraping session to data/tracking/scraping_sessions.jsonl
    and record latest session metadata in data/tracking/scraping_progress.json.
    """
    path = Path(filepath or SESSIONS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_seen_listing_ids()
    duration_secs = round((end_time - start_time).total_seconds(), 2)

    session_data = {
        "session_id": f"sess_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{source}",
        "scraper_name": scraper_name,
        "source": source,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration_secs,
        "pages_scraped": pages_scraped,
        "page_range": {"start": start_page, "end": end_page},
        "records_extracted": records_extracted,
        "records_added": records_added,
        "duplicates_skipped": duplicates_skipped,
        "cumulative_seen_ids": len(seen_ids),
        "status": status,
        "notes": notes,
    }

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Could not append session data to %s: %s", path, e)

    # Also update scraping_progress.json with recent_sessions
    try:
        progress = load_scraping_progress()
        if "recent_sessions" not in progress or not isinstance(progress["recent_sessions"], list):
            progress["recent_sessions"] = []
        progress["recent_sessions"].append(session_data)
        # Keep only the last 20 sessions in progress.json to keep it compact
        progress["recent_sessions"] = progress["recent_sessions"][-20:]
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Could not update progress recent_sessions: %s", e)

    logger.info(
        "Session recorded: %s on %s (+%d new / %d skipped in %.1fs)",
        scraper_name,
        source,
        records_added,
        duplicates_skipped,
        duration_secs,
    )
    return session_data


def load_seen_listing_ids(filepath: Optional[Path] = None) -> Set[str]:
    """Load persistent set of previously seen listing IDs from disk."""
    path = Path(filepath or SEEN_IDS_FILE)
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logger.warning("Could not read seen listing IDs from %s: %s", path, e)
        return set()


def append_seen_listing_ids(new_ids: Any, filepath: Optional[Path] = None) -> None:
    """Atomically append new listing IDs to persistent disk register."""
    if not new_ids:
        return
    path = Path(filepath or SEEN_IDS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            for lid in new_ids:
                s = str(lid).strip()
                if s and s.lower() not in ("nan", "none"):
                    f.write(f"{s}\n")
    except Exception as e:
        logger.warning("Could not append seen listing IDs to %s: %s", path, e)


def init_seen_listing_ids(raw_dir: Path, filepath: Optional[Path] = None) -> Set[str]:
    """Scan all CSV files in raw_dir to initialize or backfill persistent seen IDs."""
    raw_dir = Path(raw_dir)
    seen_ids = set()
    if raw_dir.exists():
        for f in raw_dir.glob("*.csv"):
            if f.name == "used_car_training_combined.csv":
                continue
            try:
                df = pd.read_csv(f, low_memory=False, dtype={"listing_id": str})
                if "listing_id" in df.columns:
                    for lid in df["listing_id"].dropna():
                        s = str(lid).strip()
                        if s and s.lower() not in ("nan", "none", ""):
                            seen_ids.add(s)
            except Exception:
                pass
    path = Path(filepath or SEEN_IDS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for lid in sorted(seen_ids):
                f.write(f"{lid}\n")
    except Exception as e:
        logger.warning("Could not write initialized seen listing IDs: %s", e)
    return seen_ids


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
        self.seen_ids: Set[str] = load_seen_listing_ids()
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
        timeout: Any = (5, 10),
        max_retries: Optional[int] = None,
    ) -> Optional[str]:
        """Fetch page content with configurable retries and timeout (connect, read)."""
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
                    timeout_val = sum(timeout) if isinstance(timeout, (tuple, list)) else timeout
                    with urllib.request.urlopen(req, context=ctx, timeout=timeout_val) as resp:
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

        city_str = str(raw_record.get("city") or "").strip()
        region_str = str(raw_record.get("region") or "").strip()
        if not region_str and city_str:
            region_str = infer_moroccan_region(city_str)

        # Harmonize brand and refine model/trim
        brand_clean, model_clean, trim_clean = clean_brand_and_model(
            raw_record.get("brand"),
            raw_record.get("model"),
            title_raw=str(raw_record.get("title_raw") or ""),
            trim=str(raw_record.get("trim") or ""),
        )

        # Standardize record
        record: Dict[str, Any] = {
            "listing_id": listing_id,
            "url": str(raw_record.get("url") or "").strip(),
            "source": self.source_name,
            "date_posted": str(raw_record.get("date_posted") or now_str[:10])[:10],
            "date_scraped": str(raw_record.get("date_scraped") or now_str),
            "title_raw": str(raw_record.get("title_raw") or "").strip(),
            "brand": brand_clean,
            "model": model_clean,
            "trim": trim_clean,
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
            "city": city_str,
            "region": region_str,
            "price_mad": self.clean_numeric(raw_record.get("price_mad")),
            "photos_count": self.clean_numeric(raw_record.get("photos_count")) or 0.0,
            "description_raw": str(raw_record.get("description_raw") or "").strip(),
        }

        # 1. ESCAPE NEWLINES IN TEXT FIELDS
        record["description_raw"] = str(record.get("description_raw") or "").replace("\r", " ").replace("\n", " ").strip()
        record["title_raw"] = str(record.get("title_raw") or "").replace("\r", " ").replace("\n", " ").strip()
        if record["description_raw"].lower() in ("nan", "none"):
            record["description_raw"] = ""
        if record["title_raw"].lower() in ("nan", "none"):
            record["title_raw"] = ""

        # 2. FORCE SELLER PHONE AS STRING (10-digit string starting with 0)
        if record.get("seller_phone"):
            p_val = str(record["seller_phone"]).replace(".0", "").strip()
            if p_val and p_val.lower() not in ("nan", "none", "<na>"):
                record["seller_phone"] = p_val.zfill(10)
            else:
                record["seller_phone"] = None

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

        # Sanitize text fields and enforce string phone formatting
        if "description_raw" in df.columns:
            df["description_raw"] = df["description_raw"].fillna("").astype(str).apply(
                lambda s: s.replace("\r", " ").replace("\n", " ").strip() if s.lower() not in ("nan", "none") else ""
            )
        if "title_raw" in df.columns:
            df["title_raw"] = df["title_raw"].fillna("").astype(str).apply(
                lambda s: s.replace("\r", " ").replace("\n", " ").strip() if s.lower() not in ("nan", "none") else ""
            )
        if "seller_phone" in df.columns:
            def _fmt_phone(p):
                if pd.isna(p) or p is None:
                    return None
                s = str(p).replace(".0", "").strip()
                if not s or s.lower() in ("nan", "none", "<na>"):
                    return None
                s = s.zfill(10)
                return s if len(s) == 10 and s.startswith("0") else None
            df["seller_phone"] = df["seller_phone"].apply(_fmt_phone)

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

        # Update persistent seen IDs register
        if "listing_id" in df.columns:
            new_lids = [str(x).strip() for x in df["listing_id"].dropna() if str(x).strip()]
            append_seen_listing_ids(new_lids)
            self.seen_ids.update(new_lids)


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
        # Detect all dates present in format {source}_{YYYY-MM-DD}.csv or scraped_combined_{YYYY-MM-DD}.csv
        for f in raw_dir.glob("*.csv"):
            if f.name == "used_car_training_combined.csv":
                continue
            m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", f.name)
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
        out_csv = raw_dir / f"scraped_combined_{d}.csv"
        out_parquet = raw_dir / f"scraped_combined_{d}.parquet"

        if not matching_csvs:
            continue

        frames = []
        files_to_remove = []

        # If a combined file already exists for date d, load it first to merge & deduplicate
        if out_csv.exists():
            try:
                existing_df = pd.read_csv(out_csv, low_memory=False, dtype={"seller_phone": str, "listing_id": str})
                if not existing_df.empty and len(existing_df) > 0:
                    frames.append(existing_df)
            except Exception as e:
                logger.warning("Could not read existing combined file %s: %s", out_csv.name, e)

        for f in matching_csvs:
            try:
                df = pd.read_csv(f, low_memory=False, dtype={"seller_phone": str, "listing_id": str})
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
        merged_df = merged_df[SCHEMA_FIELDS].copy()

        # 1. ESCAPE NEWLINES IN TEXT FIELDS
        if "description_raw" in merged_df.columns:
            merged_df["description_raw"] = (
                merged_df["description_raw"]
                .fillna("")
                .astype(str)
                .apply(lambda s: s.replace("\r", " ").replace("\n", " ").strip() if s.lower() not in ("nan", "none") else "")
            )
        if "title_raw" in merged_df.columns:
            merged_df["title_raw"] = (
                merged_df["title_raw"]
                .fillna("")
                .astype(str)
                .apply(lambda s: s.replace("\r", " ").replace("\n", " ").strip() if s.lower() not in ("nan", "none") else "")
            )

        # 2. FORCE SELLER PHONE AS 10-DIGIT STRING STARTING WITH '0'
        if "seller_phone" in merged_df.columns:
            def _clean_phone_val(p):
                if pd.isna(p) or p is None:
                    return None
                s = str(p).replace(".0", "").strip()
                if not s or s.lower() in ("nan", "none", "<na>"):
                    return None
                s = s.zfill(10)
                if len(s) == 10 and s.startswith("0") and s[1] in "567" and s not in BLACKLIST_PHONES:
                    return s
                return None

            merged_df["seller_phone"] = merged_df["seller_phone"].apply(_clean_phone_val)
            if "seller_phone_hash" in merged_df.columns:
                def _resolve_hash(row):
                    h = row.get("seller_phone_hash")
                    if pd.notna(h) and str(h).strip() != "" and str(h).lower() not in ("nan", "none"):
                        return str(h).strip()
                    p = row.get("seller_phone")
                    if p:
                        return hash_phone(p)
                    return None
                merged_df["seller_phone_hash"] = merged_df.apply(_resolve_hash, axis=1)

        # 3. CLEAN BRAND & MODEL REFINEMENT
        def _apply_brand_model(row):
            b, m, tr = clean_brand_and_model(
                row.get("brand"),
                row.get("model"),
                title_raw=str(row.get("title_raw") or ""),
                trim=str(row.get("trim") or "")
            )
            return pd.Series([b, m, tr])

        merged_df[["brand", "model", "trim"]] = merged_df.apply(_apply_brand_model, axis=1)

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

        # Update persistent seen IDs register
        if "listing_id" in merged_df.columns:
            new_lids = [str(x).strip() for x in merged_df["listing_id"].dropna() if str(x).strip()]
            append_seen_listing_ids(new_lids)

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
