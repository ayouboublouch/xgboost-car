#!/usr/bin/env python3
"""
Autohouse.ma ML Pipeline - Phase 1: Data Ingestion, Cleaning & Cross-Source Deduplication
----------------------------------------------------------------------------------------
Implements:
- Raw dataset aggregation (merges all data/raw/*.parquet and *.csv with used_car_training_combined.csv)
- Within-source deduplication by listing_id and url
- Conversion of fiscal_power_cv into integer fiscal_power_int and ceiling flag
- Outlier detection (is_outlier) on price, year, and mileage
- Cross-source duplicate matching: matches listings sharing (brand, model, year, mileage +-2000 km, price +-5%)
  and assigns a shared cross_source_match_id
- Assigns repost_group_id for identical listings over time
- Constructs unified leakage_group_id combining cross-source matches and repost groups to prevent train/test leakage
- Parquet & CSV export to data/processed/cleaned_cars.parquet / .csv
"""

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("clean")


class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression."""

    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def parse_fiscal_power(val: Any) -> Tuple[float, int]:
    """Extract integer fiscal power and detect if it is a bucket ceiling (e.g. '40 CV et plus')."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return np.nan, 0

    s = str(val).lower().strip()
    is_ceiling = 1 if ("plus" in s or ">" in s or "+" in s or ("40" in s and "plus" in s)) else 0

    m = re.search(r"(\d+)", s)
    if m:
        val_int = int(m.group(1))
        if val_int > 50:
            return np.nan, 0
        return float(val_int), is_ceiling
    return np.nan, 0


def generate_repost_group_id(row: pd.Series) -> str:
    """Generate or preserve repost_group_id."""
    existing = row.get("repost_group_id")
    if pd.notna(existing) and str(existing).strip() != "" and str(existing).lower() != "nan":
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
    return "repost_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def compute_cross_source_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect duplicate listings across sources sharing:
    - brand (case-insensitive)
    - model (case-insensitive)
    - year (exact)
    - mileage within 2,000 km
    - price within 5%
    Assigns shared cross_source_match_id and computes unified leakage_group_id.
    """
    df = df.copy()
    n = len(df)
    logger.info("Computing cross-source matches across %d listings ...", n)

    # Initialize cross_source_match_id if not present
    if "cross_source_match_id" not in df.columns:
        df["cross_source_match_id"] = np.nan

    csm_uf = UnionFind()
    leakage_uf = UnionFind()

    # Pre-group by clean brand, model, year to optimize matching complexity
    temp_brand = df["brand"].fillna("").astype(str).str.lower().str.strip()
    temp_model = df["model"].fillna("").astype(str).str.lower().str.strip()
    temp_year = df["year"].fillna(-1).astype(float)
    temp_mileage = df["mileage_km"].values
    temp_price = df["price_mad"].values
    temp_source = df["source"].fillna("").astype(str).values
    row_ids = df["listing_id"].astype(str).values

    # Group candidate indices
    groups: Dict[Tuple[str, str, float], List[int]] = {}
    for idx in range(n):
        b = temp_brand[idx]
        m = temp_model[idx]
        y = temp_year[idx]
        if b and m and y > 1980:
            key = (b, m, y)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

    matches_found = 0

    # Within each (brand, model, year) bucket, find matching pairs
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        for i_pos in range(len(indices)):
            i = indices[i_pos]
            km_i = temp_mileage[i]
            p_i = temp_price[i]
            src_i = temp_source[i]

            if pd.isna(km_i) or pd.isna(p_i) or p_i <= 0:
                continue

            for j_pos in range(i_pos + 1, len(indices)):
                j = indices[j_pos]
                km_j = temp_mileage[j]
                p_j = temp_price[j]
                src_j = temp_source[j]

                if pd.isna(km_j) or pd.isna(p_j) or p_j <= 0:
                    continue

                # Cross-source match criteria:
                # Mileage within 2,000 km AND Price within 5%
                km_diff = abs(km_i - km_j)
                price_rel_diff = abs(p_i - p_j) / max(p_i, p_j)

                if km_diff <= 2000.0 and price_rel_diff <= 0.05:
                    id_i = row_ids[i]
                    id_j = row_ids[j]
                    csm_uf.union(id_i, id_j)
                    leakage_uf.union(id_i, id_j)
                    matches_found += 1

    logger.info("Found %d cross-source / multi-platform pairwise matches.", matches_found)

    # Assign cross_source_match_id from clusters
    csm_ids = []
    for idx in range(n):
        l_id = row_ids[idx]
        existing = df["cross_source_match_id"].iloc[idx]
        if pd.notna(existing) and str(existing).strip() != "" and str(existing).lower() != "nan":
            csm_ids.append(str(existing))
            # Also link to leakage group
            leakage_uf.union(l_id, str(existing))
        else:
            root = csm_uf.find(l_id)
            if root != l_id:
                csm_val = f"csm_{hashlib.md5(root.encode('utf-8')).hexdigest()[:10]}"
                csm_ids.append(csm_val)
            else:
                csm_ids.append(np.nan)

    df["cross_source_match_id"] = csm_ids

    # Unify with repost_group_id into leakage_group_id
    repost_ids = df["repost_group_id"].values
    for idx in range(n):
        l_id = row_ids[idx]
        r_id = str(repost_ids[idx])
        leakage_uf.union(l_id, r_id)
        c_id = df["cross_source_match_id"].iloc[idx]
        if pd.notna(c_id):
            leakage_uf.union(l_id, str(c_id))

    leakage_groups = []
    for idx in range(n):
        l_id = row_ids[idx]
        root = leakage_uf.find(l_id)
        cluster_id = f"leak_{hashlib.md5(root.encode('utf-8')).hexdigest()[:10]}"
        leakage_groups.append(cluster_id)

    df["leakage_group_id"] = leakage_groups

    n_unique_csm = df["cross_source_match_id"].dropna().nunique()
    n_unique_leak = df["leakage_group_id"].nunique()
    logger.info("Unique cross-source match clusters: %d | Unique leakage clusters: %d", n_unique_csm, n_unique_leak)
    return df


def load_raw_datasets(raw_dir: Path) -> pd.DataFrame:
    """Scan and merge all raw parquet and csv datasets in data/raw, guaranteeing seed dataset inclusion."""
    frames = []

    # Guarantee inclusion of the verified baseline seed dataset (656 records)
    seed_file = raw_dir / "used_car_training_combined.csv"
    if seed_file.exists():
        try:
            df_seed = pd.read_csv(seed_file, low_memory=False)
            if not df_seed.empty:
                logger.info("Loaded %d verified baseline records from %s", len(df_seed), seed_file.name)
                frames.append(df_seed)
        except Exception as e:
            logger.warning("Could not load baseline seed file %s: %s", seed_file.name, e)

    parquet_files = sorted(list(raw_dir.glob("*.parquet")))
    for pf in parquet_files:
        try:
            df_p = pd.read_parquet(pf)
            if not df_p.empty:
                logger.info("Loaded %d rows from %s", len(df_p), pf.name)
                frames.append(df_p)
        except Exception as e:
            logger.warning("Could not read %s: %s", pf.name, e)

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    for cf in csv_files:
        if cf.name == "used_car_training_combined.csv":
            continue
        try:
            df_c = pd.read_csv(cf, low_memory=False)
            if not df_c.empty:
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
    """Execute validation, outlier marking, type casting, deduplication, and cross-source matching."""
    # Deduplicate primarily on (source, listing_id) or url
    if "listing_id" in df.columns and "source" in df.columns:
        df["listing_id"] = df["listing_id"].astype(str)
        df = df.drop_duplicates(subset=["source", "listing_id"], keep="last")
    elif "listing_id" in df.columns:
        df["listing_id"] = df["listing_id"].astype(str)
        df = df.drop_duplicates(subset=["listing_id"], keep="last")

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="last")

    logger.info("Rows after within-source deduplication: %d", len(df))

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
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    if "date_scraped" not in df.columns or df["date_scraped"].isna().all():
        df["date_scraped"] = now_str
    else:
        df["date_scraped"] = df["date_scraped"].fillna(now_str)

    # Outlier Detection (Cahier des Charges bounds)
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

    # Detect Cross-Source Matches & Construct unified leakage_group_id
    df = compute_cross_source_matches(df)

    # Sort deterministically by date_scraped
    df = df.sort_values("date_scraped").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Multi-Source Clean & Cross-Source Matching")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw data files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save cleaned data")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Multi-Source Cleaning from %s ...", raw_dir)
    df_raw = load_raw_datasets(raw_dir)
    df_cleaned = clean_dataframe(df_raw)

    out_parquet = output_dir / "cleaned_cars.parquet"
    out_csv = output_dir / "cleaned_cars.csv"

    try:
        df_cleaned.to_parquet(out_parquet, index=False, engine="pyarrow")
        logger.info("  -> Parquet: %s (%d rows)", out_parquet, len(df_cleaned))
    except Exception as e:
        logger.warning("Could not save parquet format (pyarrow missing): %s", e)

    df_cleaned.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("  -> CSV:     %s (%d rows)", out_csv, len(df_cleaned))


if __name__ == "__main__":
    main()
