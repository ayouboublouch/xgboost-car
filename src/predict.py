#!/usr/bin/env python3
"""
Autohouse.ma - Production Inference & Market Deal Rating Engine
----------------------------------------------------------------
Applies trained CatBoost pricing model (models/catboost_model.cbm) to car listings,
calculates fair market value, price deviation percentage, and assigns market deal ratings:
  - 🔥 Great Deal (Underpriced): actual price is >= 12% below predicted market value
  - ⚠️ Overpriced: actual price is >= 15% above predicted market value
  - ✅ Fair Market Value: within -12% to +15% of market value

Usage:
    python src/predict.py
    python src/predict.py --input data/processed/features_cars.parquet --output models/car_price_predictions.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure stdout and stderr handle UTF-8 and emojis safely on all platforms (Windows cp1252 fix)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict")

NUMERIC_FEATURES = [
    "year",
    "vehicle_age",
    "mileage_km",
    "km_per_year",
    "fiscal_power_int",
    "doors_count",
]
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "city",
    "seller_type",
    "trim_tier",
    "customs_status",
    "condition",
    "owners_count",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OUTPUT_COLUMNS = [
    "listing_id",
    "brand",
    "model",
    "year",
    "mileage_km",
    "fuel_type",
    "transmission",
    "city",
    "price_mad",
    "predicted_price_mad",
    "deviation_percentage_%",
    "market_deal_rating",
    "seller_phone",
]


def load_model(model_path: Path) -> CatBoostRegressor:
    """Load native CatBoost model from .cbm file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run 'python src/train.py' first.")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    logger.info("Loaded native CatBoost model from %s", model_path)
    return model


def generate_market_predictions(
    df: pd.DataFrame,
    model: CatBoostRegressor,
) -> pd.DataFrame:
    """Generate price estimations, deviation percentages, and deal ratings."""
    df_eval = df.copy()

    # Determine exact feature set expected by model
    if hasattr(model, "feature_names_") and model.feature_names_:
        model_features = list(model.feature_names_)
    else:
        missing_indicators = sorted([c for c in df_eval.columns if c.endswith("_is_missing")])
        model_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + missing_indicators

    # Ensure required feature columns exist
    for col in model_features:
        if col not in df_eval.columns:
            if col in CATEGORICAL_FEATURES:
                df_eval[col] = "Inconnu"
            else:
                df_eval[col] = 0.0

    X = df_eval[model_features].copy()
    cat_cols_in_x = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    cat_indices = [X.columns.get_loc(c) for c in cat_cols_in_x]

    for c in cat_cols_in_x:
        X[c] = X[c].fillna("Inconnu").astype(str)

    pool = Pool(X, cat_features=cat_indices)
    raw_predictions = model.predict(pool)
    predicted_prices = np.round(raw_predictions, 2)

    df_eval["predicted_price_mad"] = predicted_prices

    # Compute deviation: (actual - predicted) / predicted * 100
    # Safe division avoiding divide-by-zero
    pred_safe = np.where(df_eval["predicted_price_mad"] <= 0, 1.0, df_eval["predicted_price_mad"])
    df_eval["deviation_percentage_%"] = np.round(
        ((df_eval["price_mad"] - df_eval["predicted_price_mad"]) / pred_safe) * 100, 2
    )

    # Market deal rating assignment
    df_eval["market_deal_rating"] = np.where(
        df_eval["deviation_percentage_%"] <= -12.0,
        "🔥 Great Deal (Underpriced)",
        np.where(
            df_eval["deviation_percentage_%"] >= 15.0,
            "⚠️ Overpriced",
            "✅ Fair Market Value",
        ),
    )

    # Clean phone numbers
    if "seller_phone" in df_eval.columns:
        def _fmt_phone(p):
            if pd.isna(p) or p is None:
                return None
            s = str(p).replace(".0", "").strip()
            return s if s and s.lower() not in ("nan", "none", "<na>") else None
        df_eval["seller_phone"] = df_eval["seller_phone"].apply(_fmt_phone)

    # Select standard output schema
    for col in OUTPUT_COLUMNS:
        if col not in df_eval.columns:
            df_eval[col] = None

    return df_eval[OUTPUT_COLUMNS].copy()


def run_inference(
    input_path: str = "data/processed/features_cars.parquet",
    model_path: str = "models/catboost_model.cbm",
    output_path: str = "models/car_price_predictions.csv",
) -> pd.DataFrame:
    """Run full inference pipeline and save predictions CSV."""
    input_p = Path(input_path)
    model_p = Path(model_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # Load input dataset
    if input_p.suffix == ".parquet":
        df = pd.read_parquet(input_p)
    elif input_p.suffix == ".csv":
        df = pd.read_csv(input_p, low_memory=False)
    else:
        # Fallback to features parquet or features csv
        if Path("data/processed/features_cars.parquet").exists():
            df = pd.read_parquet("data/processed/features_cars.parquet")
        else:
            df = pd.read_csv("data/processed/features_cars.csv", low_memory=False)

    logger.info("Loaded %d records for price prediction from %s", len(df), input_p)

    model = load_model(model_p)
    predictions_df = generate_market_predictions(df, model)

    # Save to primary output path
    predictions_df.to_csv(output_p, index=False, encoding="utf-8")
    logger.info("Successfully exported %d market valuation predictions to %s", len(predictions_df), output_p)

    # Also keep synced copy in data/processed/
    processed_p = Path("data/processed/car_price_predictions.csv")
    if processed_p.resolve() != output_p.resolve():
        predictions_df.to_csv(processed_p, index=False, encoding="utf-8")
        logger.info("Synced predictions copy saved to %s", processed_p)

    # Summary analytics
    deal_counts = predictions_df["market_deal_rating"].value_counts()
    print("\n" + "=" * 80)
    print("AUTOHOUSE.MA CAR VALUATION & DEAL RATING REPORT")
    print("=" * 80)
    for rating, count in deal_counts.items():
        pct = (count / len(predictions_df)) * 100
        print(f"  {rating:<30}: {count:5d} cars ({pct:5.1f}%)")
    print("=" * 80)
    print("PREVIEW (First 15 scored listings):")
    print("=" * 80)
    print(predictions_df[["listing_id", "brand", "model", "year", "price_mad", "predicted_price_mad", "deviation_percentage_%", "market_deal_rating", "seller_phone"]].head(15).to_string(index=False))
    print("=" * 80 + "\n")

    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Autohouse.ma Price Prediction & Deal Rating CLI")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/features_cars.parquet",
        help="Input features dataset (.parquet or .csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/catboost_model.cbm",
        help="Trained CatBoost model path (default: models/catboost_model.cbm)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/car_price_predictions.csv",
        help="Output predictions CSV path (default: models/car_price_predictions.csv)",
    )
    args = parser.parse_args()

    run_inference(input_path=args.input, model_path=args.model, output_path=args.output)


if __name__ == "__main__":
    main()
