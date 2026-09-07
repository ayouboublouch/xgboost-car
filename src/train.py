#!/usr/bin/env python3
"""
Autohouse.ma ML Pipeline - Phases 3 & 4: Chronological Splitting, Model Training & Evaluation
-------------------------------------------------------------------------------------------
Implements:
- Chronological 70/15/15 train/val/test split preserving repost_group_id integrity
- Benchmark model training:
  * Baseline median (by brand + model + year)
  * Ridge Regression
  * RandomForestRegressor
  * LightGBMRegressor
  * CatBoostRegressor (selected production architecture)
- Comprehensive evaluation metrics: MAE (MAD), MAPE (%), R2, within +-10%, within +-15%
- Native CatBoost format export: models/catboost_model.cbm (NO pickle)
- Model comparison table export: models/model_comparison.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostRegressor, Pool

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")

NUMERIC_FEATURES = ["year", "mileage_km", "fiscal_power_int", "doors_count"]
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "city",
    "customs_status",
    "condition",
    "owners_count",
    "trim_tier",
    "seller_type_reliable",
]
TARGET = "price_mad"


def chronological_grouped_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (70/15/15) while keeping all listings in the same
    cross_source_match_id and repost_group_id together on the same side of the partition
    (strictly preventing data leakage across platforms and over time).
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date_scraped"], errors="coerce")

    # Determine grouping column
    group_col = "leakage_group_id"
    if group_col not in df.columns:
        if "cross_source_match_id" in df.columns and "repost_group_id" in df.columns:
            # Fallback unification
            df[group_col] = df["cross_source_match_id"].fillna(df["repost_group_id"]).astype(str)
        elif "repost_group_id" in df.columns:
            df[group_col] = df["repost_group_id"].astype(str)
        else:
            df[group_col] = df.index.astype(str)

    # Group-level earliest date
    group_dates = (
        df.groupby(group_col)["_date"]
        .min()
        .reset_index()
        .sort_values("_date")
        .reset_index(drop=True)
    )

    n_groups = len(group_dates)
    i_train = int(n_groups * train_pct)
    i_val = int(n_groups * (train_pct + val_pct))

    train_groups = set(group_dates.iloc[:i_train][group_col])
    val_groups = set(group_dates.iloc[i_train:i_val][group_col])
    test_groups = set(group_dates.iloc[i_val:][group_col])

    df_train = df[df[group_col].isin(train_groups)].sort_values("_date").reset_index(drop=True)
    df_val = df[df[group_col].isin(val_groups)].sort_values("_date").reset_index(drop=True)
    df_test = df[df[group_col].isin(test_groups)].sort_values("_date").reset_index(drop=True)

    df_train = df_train.drop(columns=["_date"])
    df_val = df_val.drop(columns=["_date"])
    df_test = df_test.drop(columns=["_date"])

    logger.info(
        "Zero-Leakage Chronological Grouped Split: Train=%d, Val=%d, Test=%d (Total=%d rows across %d clusters)",
        len(df_train),
        len(df_val),
        len(df_test),
        len(df),
        n_groups,
    )
    return df_train, df_val, df_test


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Compute business and statistical acceptance KPIs."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    # Avoid zero division
    valid_idx = y_true_arr > 0
    y_true_arr = y_true_arr[valid_idx]
    y_pred_arr = y_pred_arr[valid_idx]

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100
    r2 = r2_score(y_true_arr, y_pred_arr)

    abs_rel_err = np.abs((y_pred_arr - y_true_arr) / y_true_arr)
    within_10 = float((abs_rel_err <= 0.10).mean() * 100)
    within_15 = float((abs_rel_err <= 0.15).mean() * 100)

    logger.info(
        "[%-20s] MAE: %10,.0f MAD | MAPE: %5.1f%% | R2: %5.3f | +-10%%: %4.1f%% | +-15%%: %4.1f%%",
        model_name,
        mae,
        mape,
        r2,
        within_10,
        within_15,
    )

    return {
        "model": model_name,
        "MAE_MAD": round(mae, 2),
        "MAPE_%": round(mape, 2),
        "R2": round(r2, 4),
        "within_10_%": round(within_10, 2),
        "within_15_%": round(within_15, 2),
    }


def train_baseline_median(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
    """Predict median price by brand, model, and year, falling back to brand median and global median."""
    global_median = df_train[TARGET].median()
    group_medians = df_train.groupby(["brand", "model", "year"])[TARGET].median()
    brand_medians = df_train.groupby("brand")[TARGET].median()

    def predict_row(row):
        key = (row["brand"], row["model"], row["year"])
        if key in group_medians:
            return group_medians[key]
        if row["brand"] in brand_medians:
            return brand_medians[row["brand"]]
        return global_median

    return df_test.apply(predict_row, axis=1).values


def build_sklearn_pipeline(model_instance) -> Pipeline:
    """Standard preprocessor pipeline for classical Scikit-Learn models."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model_instance)])


def main():
    parser = argparse.ArgumentParser(description="Phases 3 & 4: Train & Evaluate Price Model")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/features_cars.parquet",
        help="Feature dataset path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models and reports",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        csv_path = input_path.with_suffix(".csv")
        if csv_path.exists():
            input_path = csv_path
        else:
            raise FileNotFoundError(f"Feature dataset not found at {input_path}")

    logger.info("Loading feature dataset from %s ...", input_path)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    # Missing indicators dynamically identified
    missing_indicators = [c for c in df.columns if c.endswith("_is_missing")]
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + missing_indicators

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col in NUMERIC_FEATURES:
                df[col] = np.nan
            else:
                df[col] = "Inconnu"

    # Chronological Split
    df_train, df_val, df_test = chronological_grouped_split(df, train_pct=0.70, val_pct=0.15)

    X_train, y_train = df_train[feature_cols], df_train[TARGET]
    X_val, y_val = df_val[feature_cols], df_val[TARGET]
    X_test, y_test = df_test[feature_cols], df_test[TARGET]

    results: List[Dict[str, Any]] = []

    # 1. Baseline Median
    pred_baseline = train_baseline_median(df_train, df_test)
    results.append(evaluate_predictions(y_test, pred_baseline, "Baseline Median"))

    # 2. Ridge Regression
    try:
        ridge_pipe = build_sklearn_pipeline(Ridge(alpha=100.0))
        ridge_pipe.fit(X_train, y_train)
        pred_ridge = ridge_pipe.predict(X_test)
        results.append(evaluate_predictions(y_test, pred_ridge, "Ridge Regression"))
    except Exception as e:
        logger.warning("Ridge training failed: %s", e)

    # 3. RandomForest Regressor
    try:
        rf_pipe = build_sklearn_pipeline(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        )
        rf_pipe.fit(X_train, y_train)
        pred_rf = rf_pipe.predict(X_test)
        results.append(evaluate_predictions(y_test, pred_rf, "RandomForest"))
    except Exception as e:
        logger.warning("RandomForest training failed: %s", e)

    # 4. LightGBM Regressor (if installed)
    if HAS_LIGHTGBM:
        try:
            X_train_lgb = X_train.copy()
            X_val_lgb = X_val.copy()
            X_test_lgb = X_test.copy()
            for c in CATEGORICAL_FEATURES:
                X_train_lgb[c] = X_train_lgb[c].astype("category")
                X_val_lgb[c] = X_val_lgb[c].astype("category")
                X_test_lgb[c] = X_test_lgb[c].astype("category")
                # Align categories
                cats = X_train_lgb[c].cat.categories
                X_val_lgb[c] = X_val_lgb[c].cat.set_categories(cats)
                X_test_lgb[c] = X_test_lgb[c].cat.set_categories(cats)

            lgb_model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
            )
            lgb_model.fit(
                X_train_lgb,
                y_train,
                eval_set=[(X_val_lgb, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            pred_lgb = lgb_model.predict(X_test_lgb)
            results.append(evaluate_predictions(y_test, pred_lgb, "LightGBM"))
        except Exception as e:
            logger.warning("LightGBM training failed: %s", e)

    # 5. CatBoost Regressor (Selected Architecture)
    logger.info("Training production CatBoost Regressor...")
    cat_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    X_test_cb = X_test.copy()

    for c in CATEGORICAL_FEATURES:
        X_train_cb[c] = X_train_cb[c].fillna("Inconnu").astype(str)
        X_val_cb[c] = X_val_cb[c].fillna("Inconnu").astype(str)
        X_test_cb[c] = X_test_cb[c].fillna("Inconnu").astype(str)

    train_pool = Pool(X_train_cb, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val_cb, y_val, cat_features=cat_indices)
    test_pool = Pool(X_test_cb, cat_features=cat_indices)

    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )

    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    pred_cb = cb_model.predict(test_pool)
    cb_metrics = evaluate_predictions(y_test, pred_cb, "CatBoost (Selected)")
    results.append(cb_metrics)

    # Export comparison table
    results_df = pd.DataFrame(results).sort_values("MAE_MAD").reset_index(drop=True)
    comparison_path = output_dir / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    logger.info("Model comparison results saved to %s", comparison_path)
    print("\n" + results_df.to_string(index=False) + "\n")

    # Export best model exclusively in native .cbm format (NO pickle)
    cbm_model_path = output_dir / "catboost_model.cbm"
    cb_model.save_model(str(cbm_model_path))
    logger.info("Exported native CatBoost model to %s (format: .cbm, no pickle)", cbm_model_path)


if __name__ == "__main__":
    main()
