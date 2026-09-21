#!/usr/bin/env python3
"""
Autohouse.ma - Production Inference & Champion Blended Ensemble Deal Rating Engine
----------------------------------------------------------------------------------
Applies Champion Blended Ensemble (CatBoost 50% + XGBoost 30% + LightGBM 20%)
to car listings, calculates fair market value, price deviation percentage, and assigns
market deal ratings:
  - 🔥 Great Deal (Underpriced): actual price is >= 12% below predicted market value
  - ⚠️ Overpriced: actual price is >= 15% above predicted market value
  - ✅ Fair Market Value: within -12% to +15% of market value

Guarantees 14-column output format with url preserved in column position 2:
['listing_id', 'url', 'brand', 'model', 'year', 'mileage_km', 'fuel_type',
 'transmission', 'city', 'price_mad', 'predicted_price_mad', 'deviation_percentage_%',
 'market_deal_rating', 'seller_phone']

Usage:
    python src/predict.py
    python src/predict.py --input data/processed/features_cars.parquet --models-dir models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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

# Strict 14-column layout positioning url as the second column
OUTPUT_COLUMNS = [
    "listing_id",
    "url",
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


def load_ensemble_models(models_dir: Path) -> Dict[str, Any]:
    """Load CatBoost, XGBoost, and LightGBM models from disk."""
    models: Dict[str, Any] = {}

    cb_path = models_dir / "catboost_model.cbm"
    if cb_path.exists():
        try:
            cb_model = CatBoostRegressor()
            cb_model.load_model(str(cb_path))
            models["catboost"] = cb_model
            logger.info("Loaded native CatBoost model from %s", cb_path)
        except Exception as e:
            logger.warning("Could not load CatBoost model: %s", e)

    xgb_path = models_dir / "xgboost_model.json"
    if xgb_path.exists() and HAS_XGBOOST:
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(xgb_path))
            models["xgboost"] = xgb_model
            logger.info("Loaded native XGBoost model from %s", xgb_path)
        except Exception as e:
            logger.warning("Could not load XGBoost model: %s", e)

    lgb_path = models_dir / "lightgbm_model.txt"
    if lgb_path.exists() and HAS_LIGHTGBM:
        try:
            lgb_booster = lgb.Booster(model_file=str(lgb_path))
            models["lightgbm"] = lgb_booster
            logger.info("Loaded native LightGBM model from %s", lgb_path)
        except Exception as e:
            logger.warning("Could not load LightGBM model: %s", e)

    if not models:
        raise FileNotFoundError(
            f"No trained model artifacts found in {models_dir}. Run 'python src/train.py' first."
        )

    return models


def generate_market_predictions(
    df: pd.DataFrame,
    models: Dict[str, Any],
) -> pd.DataFrame:
    """Generate price estimations using Champion Blended Ensemble, compute deviations and deal ratings."""
    df_eval = df.copy()

    # Determine exact feature set expected by primary CatBoost model
    cb_model = models.get("catboost")
    if cb_model is not None and hasattr(cb_model, "feature_names_") and cb_model.feature_names_:
        model_features = list(cb_model.feature_names_)
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

    # 1. CatBoost Predictions (strings for categorical)
    p_cb = None
    if cb_model is not None:
        X_cb = X.copy()
        for c in cat_cols_in_x:
            X_cb[c] = X_cb[c].fillna("Inconnu").astype(str)
        pool = Pool(X_cb, cat_features=cat_indices)
        p_cb = cb_model.predict(pool)

    # 2. Prepare Categorical Encoded Dataset for XGBoost & LightGBM
    cat_cats_path = Path("models/categorical_categories.json")
    saved_cats = {}
    if cat_cats_path.exists():
        try:
            import json
            with open(cat_cats_path, "r", encoding="utf-8") as f:
                saved_cats = json.load(f)
        except Exception:
            saved_cats = {}

    X_enc = X.copy()
    for c in cat_cols_in_x:
        if c in saved_cats:
            valid_cats = saved_cats[c]
            s = X_enc[c].fillna("Inconnu").astype(str)
            fallback_val = "Inconnu" if "Inconnu" in valid_cats else valid_cats[0]
            s = s.where(s.isin(valid_cats), fallback_val)
            X_enc[c] = pd.Categorical(s, categories=valid_cats)
        else:
            X_enc[c] = X_enc[c].fillna("Inconnu").astype("category")

    # 3. XGBoost Predictions
    p_xgb = None
    xgb_model = models.get("xgboost")
    if xgb_model is not None:
        try:
            p_xgb = xgb_model.predict(X_enc)
        except Exception as e:
            logger.warning("XGBoost prediction notice (%s); using fallback.", e)

    # 4. LightGBM Predictions
    p_lgb = None
    lgb_model = models.get("lightgbm")
    if lgb_model is not None:
        try:
            p_lgb = lgb_model.predict(X_enc)
        except Exception as e:
            logger.warning("LightGBM prediction notice (%s); using fallback.", e)

    # 5. Champion Blended Ensemble: 50% CatBoost + 30% XGBoost + 20% LightGBM
    if p_cb is not None and p_xgb is not None and p_lgb is not None:
        p_ensemble = 0.50 * p_cb + 0.30 * p_xgb + 0.20 * p_lgb
    elif p_cb is not None and p_xgb is not None:
        p_ensemble = 0.60 * p_cb + 0.40 * p_xgb
    elif p_cb is not None and p_lgb is not None:
        p_ensemble = 0.70 * p_cb + 0.30 * p_lgb
    elif p_cb is not None:
        p_ensemble = p_cb
    elif p_xgb is not None:
        p_ensemble = p_xgb
    else:
        p_ensemble = p_lgb

    predicted_prices = np.round(p_ensemble, 2)
    df_eval["predicted_price_mad"] = predicted_prices

    # Compute deviation: (actual - predicted) / predicted * 100
    pred_safe = np.where(df_eval["predicted_price_mad"] <= 0, 1.0, df_eval["predicted_price_mad"])
    df_eval["deviation_percentage_%"] = np.round(
        ((df_eval["price_mad"] - df_eval["predicted_price_mad"]) / pred_safe) * 100, 2
    )

    # Market deal rating assignment with emojis
    df_eval["market_deal_rating"] = np.where(
        df_eval["deviation_percentage_%"] <= -12.0,
        "🔥 Great Deal (Underpriced)",
        np.where(
            df_eval["deviation_percentage_%"] >= 15.0,
            "⚠️ Overpriced",
            "✅ Fair Market Value",
        ),
    )

    # Clean and preserve listing URL
    if "url" not in df_eval.columns:
        df_eval["url"] = None
    else:
        df_eval["url"] = df_eval["url"].astype(str).str.strip().str.replace(r"[\r\n\t]+", "", regex=True)
        df_eval.loc[df_eval["url"].isin(["nan", "None", "", "<NA>"]), "url"] = None

    # Clean phone numbers
    if "seller_phone" in df_eval.columns:
        def _fmt_phone(p):
            if pd.isna(p) or p is None:
                return None
            s = str(p).replace(".0", "").strip()
            return s if s and s.lower() not in ("nan", "none", "<na>") else None
        df_eval["seller_phone"] = df_eval["seller_phone"].apply(_fmt_phone)

    # Enforce strict 14-column output schema
    for col in OUTPUT_COLUMNS:
        if col not in df_eval.columns:
            df_eval[col] = None

    return df_eval[OUTPUT_COLUMNS].copy()


def run_inference(
    input_path: str = "data/processed/features_cars.parquet",
    models_dir: str = "models",
    output_path: str = "models/car_price_predictions.csv",
) -> pd.DataFrame:
    """Run full Champion Blended Ensemble inference pipeline and save predictions CSV."""
    input_p = Path(input_path)
    models_p = Path(models_dir)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # Load input dataset
    if input_p.suffix == ".parquet" and input_p.exists():
        df = pd.read_parquet(input_p)
    elif input_p.suffix == ".csv" and input_p.exists():
        df = pd.read_csv(input_p, low_memory=False)
    else:
        if Path("data/processed/features_cars.parquet").exists():
            df = pd.read_parquet("data/processed/features_cars.parquet")
        elif Path("data/processed/features_cars.csv").exists():
            df = pd.read_csv("data/processed/features_cars.csv", low_memory=False)
        else:
            raise FileNotFoundError(f"Feature dataset not found at {input_p}")

    logger.info("Loaded %d records for price prediction from %s", len(df), input_p)

    models = load_ensemble_models(models_p)
    predictions_df = generate_market_predictions(df, models)

    # Save to primary output path and notebook-compatible path
    predictions_df.to_csv(output_p, index=False, encoding="utf-8")
    notebook_csv = models_p / "car_price_predictions_all.csv"
    predictions_df.to_csv(notebook_csv, index=False, encoding="utf-8")
    logger.info("Successfully exported %d market valuation predictions to %s", len(predictions_df), output_p)

    # Also keep synced copy in data/processed/
    processed_p = Path("data/processed/car_price_predictions.csv")
    if processed_p.resolve() != output_p.resolve():
        predictions_df.to_csv(processed_p, index=False, encoding="utf-8")
        logger.info("Synced predictions copy saved to %s", processed_p)

    # Summary analytics
    deal_counts = predictions_df["market_deal_rating"].value_counts()
    print("\n" + "=" * 80)
    print("🏆 AUTOHOUSE.MA BLENDED ENSEMBLE VALUATION & DEAL RATING REPORT")
    print("=" * 80)
    for rating, count in deal_counts.items():
        pct = (count / len(predictions_df)) * 100
        print(f"  {rating:<30}: {count:5d} cars ({pct:5.1f}%)")
    print("=" * 80)
    print("PREVIEW (First 10 scored listings with listing URL):")
    print("=" * 80)
    print(predictions_df[["listing_id", "url", "brand", "model", "year", "price_mad", "predicted_price_mad", "deviation_percentage_%", "market_deal_rating"]].head(10).to_string(index=False))
    print("=" * 80 + "\n")

    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Autohouse.ma Champion Blended Ensemble Inference CLI")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/features_cars.parquet",
        help="Input features dataset (.parquet or .csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models (default: models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/car_price_predictions.csv",
        help="Output predictions CSV path (default: models/car_price_predictions.csv)",
    )
    args = parser.parse_args()

    run_inference(input_path=args.input, models_dir=args.models_dir, output_path=args.output)


if __name__ == "__main__":
    main()
