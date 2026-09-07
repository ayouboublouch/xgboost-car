#!/usr/bin/env python3
"""
Autohouse.ma ML Pipeline - Phase 2: Feature Engineering & MNAR Handling
----------------------------------------------------------------------
Implements:
- Filtering non-outliers with valid target and core features
- Missingness indicators ({col}_is_missing) for MNAR columns
- Explicit "Inconnu" category assignment for customs_status, condition, owners_count
- Trim keyword classification into hierarchical tiers: top, mid, base, inconnu
- Neutralization of seller_type bias across scrapers (seller_type_reliable)
- Lookup imputation for deterministic attributes (doors_count, fiscal_power_int)
- High-cardinality rare model bucketing into '<brand>_other'
- Parquet & CSV export to data/processed/features_cars.parquet / .csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("features")

# Trim tier classification dictionaries
TOP_KEYWORDS = [
    "gt", "r-line", "rline", "sport", "amg", "m-sport", "msport", "s-line", "sline",
    "black edition", "restyl", "gti", "rs", "st-line", "stline", "titanium", "exclusive",
    "pack m", "line", "luxury", "prestige", "cupra", "fr", "vignale"
]
MID_KEYWORDS = [
    "confort", "confortline", "life", "trend", "style", "active", "dynamique",
    "intens", "zen", "business", "allure", "feel", "shine", "edition"
]


def classify_trim_tier(val: Any) -> str:
    """Classify vehicle trim into categorical tiers: top, mid, base, inconnu."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return "inconnu"
    v = str(val).lower()
    if any(k in v for k in TOP_KEYWORDS):
        return "top"
    if any(k in v for k in MID_KEYWORDS):
        return "mid"
    return "base"


def lookup_impute(
    frame: pd.DataFrame, col: str, keys: Tuple[str, ...] = ("brand", "model")
) -> pd.Series:
    """Impute deterministic missing numeric values by brand+model group median, falling back to global median."""
    lookup = frame.dropna(subset=[col]).groupby(list(keys))[col].median()
    global_median = frame[col].median()
    if pd.isna(global_median):
        global_median = 0.0

    def fill(row):
        if pd.notna(row[col]):
            return float(row[col])
        key = tuple(row[k] for k in keys)
        return float(lookup.get(key, global_median))

    return frame.apply(fill, axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase 2 feature engineering transformations."""
    # 1. Filter out outliers and rows without target or fundamental features
    initial_len = len(df)
    valid_mask = (
        (~df["is_outlier"])
        & df["price_mad"].notna()
        & (df["price_mad"] > 0)
        & df["year"].notna()
        & df["mileage_km"].notna()
    )
    df = df[valid_mask].copy()
    logger.info("Filtered non-viable rows: %d -> %d rows", initial_len, len(df))

    # 2. Add Missingness Indicators (MNAR - Missing Not At Random) before any imputation
    mnar_cols = [
        "customs_status",
        "trim",
        "condition",
        "owners_count",
        "transmission",
        "doors_count",
        "fiscal_power_int",
    ]
    for col in mnar_cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
        else:
            df[f"{col}_is_missing"] = 1

    # 3. Explicit "Inconnu" categories for structurally missing attributes (never guessed)
    for col in ["customs_status", "condition", "owners_count"]:
        if col in df.columns:
            df[col] = df[col].fillna("Inconnu").astype(str)

    # 4. Trim tier bucketing
    if "trim" in df.columns:
        df["trim_tier"] = df["trim"].apply(classify_trim_tier)
    else:
        df["trim_tier"] = "inconnu"

    # 5. Neutralize known scraper artifacts on seller_type (Moteur/Wandaloo over-classification)
    if "seller_type" in df.columns and "source" in df.columns:
        df["seller_type_reliable"] = np.where(
            df["source"].astype(str).str.lower() == "avito",
            df["seller_type"].fillna("Inconnu"),
            "Inconnu",
        )
    elif "seller_type" in df.columns:
        df["seller_type_reliable"] = df["seller_type"].fillna("Inconnu")
    else:
        df["seller_type_reliable"] = "Inconnu"

    # 6. Lookup median imputation for doors_count and fiscal_power_int
    if "doors_count" in df.columns and "brand" in df.columns and "model" in df.columns:
        df["doors_count"] = lookup_impute(df, "doors_count")
    if "fiscal_power_int" in df.columns and "brand" in df.columns and "model" in df.columns:
        df["fiscal_power_int"] = lookup_impute(df, "fiscal_power_int")

    # 7. Transmission fallback
    if "transmission" in df.columns:
        df["transmission"] = df["transmission"].fillna("Inconnu").astype(str)

    # 8. Rare model bucketing (per Cahier des Charges: limit cardinality relative to sample size)
    if "model" in df.columns and "brand" in df.columns:
        model_counts = df["model"].value_counts()
        rare_models = model_counts[model_counts < 3].index
        df["model"] = df["model"].where(
            ~df["model"].isin(rare_models), df["brand"] + "_other"
        )
        logger.info("Bucketed %d rare models into '<brand>_other'", len(rare_models))

    # Clean text columns
    text_cols = ["brand", "fuel_type", "city"]
    for tc in text_cols:
        if tc in df.columns:
            df[tc] = df[tc].fillna("Inconnu").astype(str).str.strip()

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

    out_parquet = output_dir / "features_cars.parquet"
    out_csv = output_dir / "features_cars.csv"

    df_featured.to_parquet(out_parquet, index=False, engine="pyarrow")
    df_featured.to_csv(out_csv, index=False, encoding="utf-8")

    logger.info("Feature dataset successfully saved:")
    logger.info("  -> Parquet: %s (%d rows)", out_parquet, len(df_featured))
    logger.info("  -> CSV:     %s (%d rows)", out_csv, len(df_featured))


if __name__ == "__main__":
    main()
