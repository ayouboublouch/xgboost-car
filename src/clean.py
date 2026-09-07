#!/usr/bin/env python3
"""
Autohouse.ma ML Pipeline - Phase 1: Data Ingestion, Cleaning & Validation
------------------------------------------------------------------------
Implements:
- Raw dataset aggregation (merges all data/raw/*.parquet and *.csv with used_car_training_combined.csv)
- Deduplication by listing_id and url
- Conversion of fiscal_power_cv into integer fiscal_power_int and ceiling flag
- Outlier detection (is_outlier) on price, year, and mileage
- Cross-source and repost grouping (repost_group_id) to avoid data leakage
- Parquet & CSV export to data/processed/cleaned_cars.parquet / .csv
"""

import argparse
import glob
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("clean")

TARGET_RAW_COLUMNS = [
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


def parse_fiscal_power(val: Any) -> tuple:
    """Extract integer fiscal power and detect if it is a bucket ceiling (e.g. '40 CV et plus')."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return np.nan, 0

    s = str(val).lower().strip()
    is_ceiling = 1 if ("plus" in s or ">" in s or "+" in s or "40" in s and "plus" in s) else 0

    m = re.search(r"(\d+)", s)
    if m:
        val_int = int(m.group(1))
        if val_int > 50:
            return np.nan, 0
        return float(val_int), is_ceiling
    return np.nan, 0


def generate_repost_group_id(row: pd.Series) -> str:
    """Generate a consistent repost_group_id if not already present."""
    existing = row.get("repost_group_id")
    if pd.notna(existing) and str(existing).strip() != "":
        return str(existing)

    brand = str(row.get("brand", "")).lower().strip()
    model = str(row.get("model", "")).lower().strip()
    year = str(row.get("year", ""))
    fuel = str(row.get("fuel_type", "")).lower().strip()
    city = str(row.get("city", "")).lower().strip()

    mileage = row.get("mileage_km")
    km_bucket = "unknown"
    if pd.notna(mileage):
        try:
            km_bucket = str(int(round(float(mileage) / 2500.0) * 2500))
        except (ValueError, TypeError):
            pass

    key = f"{brand}|{model}|{year}|{fuel}|{city}|{km_bucket}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def load_raw_datasets(raw_dir: Path) -> pd.DataFrame:
    """Scan and merge all raw parquet and csv datasets in data/raw."""
    frames = []

    # Parquet files
    parquet_files = sorted(list(raw_dir.glob("*.parquet")))
    for pf in parquet_files:
        try:
            df_p = pd.read_parquet(pf)
            logger.info("Loaded %d rows from %s", len(df_p), pf.name)
            frames.append(df_p)
        except Exception as e:
            logger.warning("Could not read %s: %s", pf.name, e)

    # CSV files
    csv_files = sorted(list(raw_dir.glob("*.csv")))
    for cf in csv_files:
        try:
            df_c = pd.read_csv(cf, low_memory=False)
            logger.info("Loaded %d rows from %s", len(df_c), cf.name)
            frames.append(df_c)
        except Exception as e:
            logger.warning("Could not read %s: %s", cf.name, e)

    if not frames:
        raise FileNotFoundError(f"No parquet or csv files found in {raw_dir}")

    merged = pd.concat(frames, ignore_index=True)
    logger.info("Combined total raw rows before deduplication: %d", len(merged))
    return merged


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Execute validation, outlier marking, type casting, and deduplication."""
    # Deduplicate primarily on listing_id, secondarily on url
    if "listing_id" in df.columns:
        df["listing_id"] = df["listing_id"].astype(str)
        df = df.drop_duplicates(subset=["listing_id"], keep="last")
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="last")

    logger.info("Rows after deduplication: %d", len(df))

    # Clean numeric fields
    for col in ["price_mad", "year", "mileage_km", "doors_count", "photos_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean fiscal_power_cv into integer & ceiling flag
    if "fiscal_power_cv" in df.columns:
        parsed = df["fiscal_power_cv"].apply(parse_fiscal_power)
        df["fiscal_power_int"] = [p[0] for p in parsed]
        df["fiscal_power_is_bucket_ceiling"] = [p[1] for p in parsed]
    else:
        df["fiscal_power_int"] = np.nan
        df["fiscal_power_is_bucket_ceiling"] = 0

    # Clean dates
    if "date_scraped" not in df.columns or df["date_scraped"].isna().all():
        df["date_scraped"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["date_scraped"] = df["date_scraped"].fillna(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Outlier Detection (Cahier des Charges Phase 1 & strategy doc)
    # Target: price_mad, bounds: 10,000 MAD to 3,500,000 MAD
    # Year: 1980 to 2027
    # Mileage: 0 to 600,000 km
    is_outlier = (
        df["price_mad"].isna()
        | (df["price_mad"] < 10000)
        | (df["price_mad"] > 3500000)
        | df["year"].isna()
        | (df["year"] < 1980)
        | (df["year"] > 2027)
        | df["mileage_km"].isna()
        | (df["mileage_km"] < 0)
        | (df["mileage_km"] > 600000)
    )

    if "is_outlier" in df.columns:
        df["is_outlier"] = df["is_outlier"].fillna(False).astype(bool) | is_outlier
    else:
        df["is_outlier"] = is_outlier

    outlier_count = df["is_outlier"].sum()
    logger.info("Outliers flagged: %d / %d (%.1f%%)", outlier_count, len(df), (outlier_count / len(df)) * 100)

    # Assign / Preserve repost_group_id
    df["repost_group_id"] = df.apply(generate_repost_group_id, axis=1)

    # Sort deterministically by date_scraped
    df = df.sort_values("date_scraped").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Clean & Standardize Raw Car Data")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw data files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save cleaned data")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 1 Cleaning from %s ...", raw_dir)
    df_raw = load_raw_datasets(raw_dir)
    df_cleaned = clean_dataframe(df_raw)

    out_parquet = output_dir / "cleaned_cars.parquet"
    out_csv = output_dir / "cleaned_cars.csv"

    df_cleaned.to_parquet(out_parquet, index=False, engine="pyarrow")
    df_cleaned.to_csv(out_csv, index=False, encoding="utf-8")

    logger.info("Cleaned dataset successfully saved:")
    logger.info("  -> Parquet: %s (%d rows)", out_parquet, len(df_cleaned))
    logger.info("  -> CSV:     %s (%d rows)", out_csv, len(df_cleaned))


if __name__ == "__main__":
    main()
