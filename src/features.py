#!/usr/bin/env python3
"""
Autohouse.ma ML Pipeline - Phase 2: Feature Engineering & MNAR Handling
----------------------------------------------------------------------
Implements:
- Filtering non-outliers with valid target and core features
- Missingness indicators ({col}_is_missing) for MNAR columns including mileage_is_missing and fiscal_power_is_missing
- Grouped median imputation for fiscal_power_cv, fiscal_power_int, mileage_km, doors_count
- Age & wear feature engineering (vehicle_age, km_per_year)
- Trim keyword classification into hierarchical tiers: Tier 3, Tier 2, Tier 1 via title_raw, description_raw, trim
- Intact categorical string preservation for native CatBoost handling (no LabelEncoder)
- High-cardinality rare model bucketing into '<brand>_other'
- Parquet & CSV export to data/processed/features_cars.parquet / .csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("features")

# Trim tier classification keyword dictionaries
TIER_3_KEYWORDS = [
    "pack m", "m pack", "m-sport", "msport", "m sport",
    "s line", "s-line", "sline",
    "amg", "amg line", "amg-line",
    "gtd", "gti",
    "gt line", "gt-line", "gtline",
    "r line", "r-line", "rline",
    "full options", "full option", "toutes options", "toute option", "tout options", "tout option", "toutes les options",
    "cupra", "rs", "st-line", "stline",
    "black edition", "vignale"
]

TIER_2_KEYWORDS = [
    "luxe", "luxury",
    "exclusive",
    "prestige",
    "confort", "confortline",
    "allure",
    "intens",
    "business",
    "titanium",
    "dynamique",
    "shine",
    "life", "style", "active", "zen", "feel", "edition"
]


def classify_trim_tier(row: pd.Series) -> str:
    """
    Classify vehicle trim into luxury tiers based on keyword detection in
    title_raw, description_raw, and trim:
    - Tier 3: High performance, luxury trim, sport packages, or full options.
    - Tier 2: Mid-tier premium, executive, comfort editions.
    - Tier 1: Standard / base configurations.
    """
    title = str(row.get("title_raw", "") or "").lower()
    desc = str(row.get("description_raw", "") or "").lower()
    trim = str(row.get("trim", "") or "").lower()
    combined_text = f"{title} {desc} {trim}"

    for kw in TIER_3_KEYWORDS:
        if kw in combined_text:
            return "Tier 3"

    for kw in TIER_2_KEYWORDS:
        if kw in combined_text:
            return "Tier 2"

    return "Tier 1"


def lookup_impute(
    frame: pd.DataFrame, col: str, keys: Tuple[str, ...] = ("brand", "model")
) -> pd.Series:
    """
    Impute missing numeric values by brand+model group median,
    falling back to brand median, then global median.
    """
    valid_data = frame.dropna(subset=[col])
    lookup = valid_data.groupby(list(keys))[col].median()
    brand_lookup = valid_data.groupby("brand")[col].median() if "brand" in frame.columns else pd.Series(dtype=float)
    global_median = frame[col].median()
    if pd.isna(global_median):
        global_median = 0.0

    def fill(row):
        if pd.notna(row[col]):
            return float(row[col])
        key = tuple(row[k] for k in keys)
        val = lookup.get(key)
        if pd.notna(val):
            return float(val)
        brand = row.get("brand")
        if brand and brand in brand_lookup and pd.notna(brand_lookup[brand]):
            return float(brand_lookup[brand])
        return float(global_median)

    return frame.apply(fill, axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase 2 feature engineering transformations conforming to Cahier des Charges."""
    # 1. Filter out outliers and rows without valid target or year
    initial_len = len(df)
    valid_mask = (
        (~df["is_outlier"])
        & df["price_mad"].notna()
        & (df["price_mad"] > 0)
        & df["year"].notna()
    )
    df = df[valid_mask].copy()
    logger.info("Filtered non-viable rows: %d -> %d rows", initial_len, len(df))

    # 2. Add Missingness Indicators (MNAR - Missing Not At Random) before any imputation
    # Explicit binary indicator columns for mileage and fiscal power
    df["mileage_is_missing"] = df["mileage_km"].isna().astype(int)

    if "fiscal_power_int" in df.columns:
        df["fiscal_power_is_missing"] = df["fiscal_power_int"].isna().astype(int)
    elif "fiscal_power_cv" in df.columns:
        df["fiscal_power_is_missing"] = (
            df["fiscal_power_cv"].isna()
            | (df["fiscal_power_cv"].astype(str).str.strip() == "")
            | (df["fiscal_power_cv"].astype(str).str.lower() == "nan")
        ).astype(int)
    else:
        df["fiscal_power_is_missing"] = 1

    other_mnar_cols = [
        "customs_status",
        "trim",
        "condition",
        "owners_count",
        "transmission",
        "doors_count",
    ]
    for col in other_mnar_cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
        else:
            df[f"{col}_is_missing"] = 1

    # 3. Explicit "Inconnu" categories for structurally missing attributes (never guessed)
    for col in ["customs_status", "condition", "owners_count"]:
        if col in df.columns:
            df[col] = df[col].fillna("Inconnu").astype(str)

    # 4. Impute missing fiscal_power_cv and fiscal_power_int using median grouped by (brand, model)
    if "fiscal_power_int" in df.columns and "brand" in df.columns and "model" in df.columns:
        df["fiscal_power_int"] = lookup_impute(df, "fiscal_power_int")
    if "fiscal_power_cv" in df.columns:
        df["fiscal_power_cv"] = df["fiscal_power_cv"].where(
            df["fiscal_power_cv"].notna()
            & (df["fiscal_power_cv"].astype(str).str.strip() != "")
            & (df["fiscal_power_cv"].astype(str).str.lower() != "nan"),
            df["fiscal_power_int"].apply(lambda v: f"{int(round(v))} CV" if pd.notna(v) else "Inconnu")
            if "fiscal_power_int" in df.columns else "Inconnu",
        )

    # Impute missing mileage_km and doors_count using median grouped by (brand, model)
    if "mileage_km" in df.columns and "brand" in df.columns and "model" in df.columns:
        df["mileage_km"] = lookup_impute(df, "mileage_km")
    if "doors_count" in df.columns and "brand" in df.columns and "model" in df.columns:
        df["doors_count"] = lookup_impute(df, "doors_count")

    # 5. Age & Wear feature engineering
    df["vehicle_age"] = 2026 - df["year"]
    df["km_per_year"] = df["mileage_km"] / (df["vehicle_age"] + 0.5)

    # 6. Trim & Luxury Tiering via keyword detection in title_raw, description_raw, and trim
    df["trim_tier"] = df.apply(classify_trim_tier, axis=1)
    tier_dist = df["trim_tier"].value_counts().to_dict()
    logger.info("Trim tier distribution: %s", tier_dist)

    # 7. Categorical Strings: keep intact for native CatBoost handling (NO raw LabelEncoder)
    cat_cols = [
        "fuel_type",
        "transmission",
        "city",
        "brand",
        "model",
        "seller_type",
        "trim_tier",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Inconnu").astype(str).str.strip()
            df[c] = df[c].replace({"": "Inconnu", "nan": "Inconnu", "None": "Inconnu", "<NA>": "Inconnu"})

    # Neutralize known scraper artifacts on seller_type (seller_type_reliable)
    if "seller_type" in df.columns and "source" in df.columns:
        df["seller_type_reliable"] = np.where(
            df["source"].astype(str).str.lower() == "avito",
            df["seller_type"],
            "Inconnu",
        )
    elif "seller_type" in df.columns:
        df["seller_type_reliable"] = df["seller_type"]
    else:
        df["seller_type_reliable"] = "Inconnu"

    # 8. Rare model bucketing (per Cahier des Charges: limit cardinality relative to sample size)
    if "model" in df.columns and "brand" in df.columns:
        model_counts = df["model"].value_counts()
        rare_models = model_counts[model_counts < 3].index
        df["model"] = df["model"].where(
            ~df["model"].isin(rare_models), df["brand"] + "_other"
        )
        logger.info("Bucketed %d rare models into '<brand>_other'", len(rare_models))

    if "url" not in df.columns:
        df["url"] = None

    logger.info("Feature engineering completed. Total columns: %d", len(df.columns))
    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Feature Engineering & MNAR Handling")
    parser.add_argument("--input-file", type=str, default="data/processed/cleaned_cars.parquet", help="Cleaned input path")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save feature dataset")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        # Fallback to CSV if parquet not found
        csv_path = input_path.with_suffix(".csv")
        if csv_path.exists():
            input_path = csv_path
        else:
            raise FileNotFoundError(f"Neither {input_path} nor {csv_path} exist.")

    logger.info("Loading cleaned dataset from %s ...", input_path)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    df_featured = engineer_features(df)

    # Harden all object columns for Parquet / PyArrow schema compatibility
    for col in df_featured.columns:
        if df_featured[col].dtype == "object":
            df_featured[col] = df_featured[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" and str(x).lower() not in ("nan", "none", "<na>") else None
            )

    out_parquet = output_dir / "features_cars.parquet"
    out_csv = output_dir / "features_cars.csv"

    try:
        df_featured.to_parquet(out_parquet, index=False, engine="pyarrow")
        logger.info("  -> Parquet: %s (%d rows)", out_parquet, len(df_featured))
    except Exception as e:
        logger.warning("Could not save parquet format (pyarrow missing): %s", e)

    df_featured.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("  -> CSV:     %s (%d rows)", out_csv, len(df_featured))


if __name__ == "__main__":
    main()
