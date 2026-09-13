#!/usr/bin/env python3
"""
Autohouse.ma ML Pipeline - Phases 3 & 4: Chronological Splitting, Model Training & Evaluation
-------------------------------------------------------------------------------------------
Implements:
- Strict Chronological 70/15/15 train/val/test split preserving repost_group_id integrity
- Benchmark model training:
  * Baseline median (by brand + model + year)
  * Ridge Regression
  * RandomForestRegressor
  * LightGBMRegressor (optional)
  * CatBoostRegressor (selected production architecture with loss_function='MAE')
- Comprehensive evaluation metrics: MAE (MAD), MAPE (%), R2, within +-10%, within +-15% on BOTH Validation and Test
- Explicit verification of test MAPE target threshold (< 20%)
- Native CatBoost format export: models/catboost_model.cbm (NO pickle)
- Model comparison table export: models/model_comparison.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Guarantee required directories exist at script initialization
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

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
TARGET = "price_mad"


def chronological_grouped_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (70/15/15) while ensuring all listings with the same
    repost_group_id (and unified leakage_group_id) reside strictly in the same split
    to prevent data leakage across platforms and over time.
    """
    df = df.copy()
    # Sort chronologically by date_posted / date_scraped
    if "date_posted" in df.columns:
        date_series = pd.to_datetime(df["date_posted"], errors="coerce").fillna(
            pd.to_datetime(df.get("date_scraped", pd.Series(dtype=object)), errors="coerce")
        )
    else:
        date_series = pd.to_datetime(df.get("date_scraped", pd.Series(dtype=object)), errors="coerce")

    df["_date"] = date_series.fillna(pd.Timestamp("2026-09-01"))

    # Determine grouping column for zero-leakage enforcement
    group_col = "leakage_group_id" if "leakage_group_id" in df.columns else "repost_group_id"
    if group_col not in df.columns:
        df[group_col] = [f"group_{i}" for i in range(len(df))]

    # Calculate earliest date and cluster size for each group
    group_stats = (
        df.groupby(group_col)
        .agg(min_date=("_date", "min"), size=(group_col, "count"))
        .reset_index()
        .sort_values("min_date")
        .reset_index(drop=True)
    )

    total_rows = len(df)
    train_cutoff = total_rows * train_pct
    val_cutoff = total_rows * (train_pct + val_pct)

    train_groups = set()
    val_groups = set()
    test_groups = set()

    cum_rows = 0
    for _, row in group_stats.iterrows():
        g_id = row[group_col]
        g_size = row["size"]
        if cum_rows < train_cutoff:
            train_groups.add(g_id)
        elif cum_rows < val_cutoff:
            val_groups.add(g_id)
        else:
            test_groups.add(g_id)
        cum_rows += g_size

    df_train = df[df[group_col].isin(train_groups)].sort_values("_date").reset_index(drop=True)
    df_val = df[df[group_col].isin(val_groups)].sort_values("_date").reset_index(drop=True)
    df_test = df[df[group_col].isin(test_groups)].sort_values("_date").reset_index(drop=True)

    df_train = df_train.drop(columns=["_date"])
    df_val = df_val.drop(columns=["_date"])
    df_test = df_test.drop(columns=["_date"])

    # Zero-leakage verification on repost_group_id
    if "repost_group_id" in df.columns:
        train_reposts = set(df_train["repost_group_id"].dropna())
        val_reposts = set(df_val["repost_group_id"].dropna())
        test_reposts = set(df_test["repost_group_id"].dropna())

        assert len(train_reposts.intersection(val_reposts)) == 0, "Leakage detected: repost_group_id overlap between Train and Val!"
        assert len(train_reposts.intersection(test_reposts)) == 0, "Leakage detected: repost_group_id overlap between Train and Test!"
        assert len(val_reposts.intersection(test_reposts)) == 0, "Leakage detected: repost_group_id overlap between Val and Test!"
        logger.info("Zero-leakage verified: 0 repost_group_id overlap across partitions.")

    logger.info(
        "Chronological Grouped Split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%) [Total=%d rows across %d clusters]",
        len(df_train), (len(df_train) / total_rows) * 100,
        len(df_val), (len(df_val) / total_rows) * 100,
        len(df_test), (len(df_test) / total_rows) * 100,
        total_rows, len(group_stats),
    )
    return df_train, df_val, df_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute business and statistical acceptance KPIs."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    # Filter out invalid target values for clean metric calculation
    valid_idx = y_true_arr > 0
    y_true_arr = y_true_arr[valid_idx]
    y_pred_arr = y_pred_arr[valid_idx]

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100)
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    abs_rel_err = np.abs((y_pred_arr - y_true_arr) / y_true_arr)
    within_10 = float((abs_rel_err <= 0.10).mean() * 100)
    within_15 = float((abs_rel_err <= 0.15).mean() * 100)

    return {
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "within_10": within_10,
        "within_15": within_15,
    }


def train_baseline_median(df_train: pd.DataFrame, df_target: pd.DataFrame) -> np.ndarray:
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

    return df_target.apply(predict_row, axis=1).values


def build_sklearn_pipeline(model_instance, numeric_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """Standard preprocessor pipeline for Scikit-Learn models."""
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "gpu", "auto"],
        help="Device to use for model training (default: auto)",
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
            logger.info("Feature dataset not found at %s. Triggering automatic clean & features pipeline...", input_path)
            try:
                from src.clean import load_raw_datasets, clean_dataframe
                from src.features import engineer_features
            except ImportError:
                from clean import load_raw_datasets, clean_dataframe
                from features import engineer_features
            raw_dir = Path("data/raw")
            df_raw = load_raw_datasets(raw_dir)
            df_cleaned = clean_dataframe(df_raw)
            df_features = engineer_features(df_cleaned)
            input_path.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_parquet(input_path, index=False, engine="pyarrow")
            df_features.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("Generated %d baseline feature rows.", len(df_features))

    logger.info("Loading feature dataset from %s ...", input_path)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    # Discover MNAR missing indicator columns dynamically
    missing_indicators = sorted([c for c in df.columns if c.endswith("_is_missing")])
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + missing_indicators

    # Ensure all feature columns exist in dataset
    for col in feature_cols:
        if col not in df.columns:
            if col in NUMERIC_FEATURES or col.endswith("_is_missing"):
                df[col] = 0.0
            else:
                df[col] = "Inconnu"

    # Chronological Split (70/15/15) preserving repost groups
    df_train, df_val, df_test = chronological_grouped_split(df, train_pct=0.70, val_pct=0.15)

    X_train, y_train = df_train[feature_cols], df_train[TARGET]
    X_val, y_val = df_val[feature_cols], df_val[TARGET]
    X_test, y_test = df_test[feature_cols], df_test[TARGET]

    results: List[Dict[str, Any]] = []

    def record_benchmark(model_name: str, y_pred_val: np.ndarray, y_pred_test: np.ndarray):
        val_m = compute_metrics(y_val, y_pred_val)
        test_m = compute_metrics(y_test, y_pred_test)

        target_met = test_m["MAPE"] < 20.0
        status_str = "PASSED" if target_met else "NOT MET"

        logger.info(
            "[%-20s] VAL  -> MAE: %9.0f MAD | MAPE: %5.1f%% | R2: %6.3f",
            model_name, val_m["MAE"], val_m["MAPE"], val_m["R2"]
        )
        logger.info(
            "[%-20s] TEST -> MAE: %9.0f MAD | MAPE: %5.1f%% | R2: %6.3f | +-10%%: %4.1f%% | +-15%%: %4.1f%% | Target (<20%%): %s",
            model_name, test_m["MAE"], test_m["MAPE"], test_m["R2"], test_m["within_10"], test_m["within_15"], status_str
        )

        results.append({
            "model": model_name,
            "MAE_MAD": round(test_m["MAE"], 2),
            "MAPE_%": round(test_m["MAPE"], 2),
            "R2": round(test_m["R2"], 4),
            "val_MAE_MAD": round(val_m["MAE"], 2),
            "val_MAPE_%": round(val_m["MAPE"], 2),
            "val_R2": round(val_m["R2"], 4),
            "within_10_%": round(test_m["within_10"], 2),
            "within_15_%": round(test_m["within_15"], 2),
            "meets_target_<20%": target_met,
        })

    # 1. Baseline Median
    pred_base_val = train_baseline_median(df_train, df_val)
    pred_base_test = train_baseline_median(df_train, df_test)
    record_benchmark("Baseline Median", pred_base_val, pred_base_test)

    # 2. Ridge Regression
    try:
        ridge_pipe = build_sklearn_pipeline(Ridge(alpha=100.0), NUMERIC_FEATURES + missing_indicators, CATEGORICAL_FEATURES)
        ridge_pipe.fit(X_train, y_train)
        pred_ridge_val = ridge_pipe.predict(X_val)
        pred_ridge_test = ridge_pipe.predict(X_test)
        record_benchmark("Ridge Regression", pred_ridge_val, pred_ridge_test)
    except Exception as e:
        logger.warning("Ridge training failed: %s", e)

    # 3. Baseline RandomForest Regressor
    try:
        rf_pipe = build_sklearn_pipeline(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            NUMERIC_FEATURES + missing_indicators,
            CATEGORICAL_FEATURES,
        )
        rf_pipe.fit(X_train, y_train)
        pred_rf_val = rf_pipe.predict(X_val)
        pred_rf_test = rf_pipe.predict(X_test)
        record_benchmark("RandomForest", pred_rf_val, pred_rf_test)
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
            pred_lgb_val = lgb_model.predict(X_val_lgb)
            pred_lgb_test = lgb_model.predict(X_test_lgb)
            record_benchmark("LightGBM", pred_lgb_val, pred_lgb_test)
        except Exception as e:
            logger.warning("LightGBM training failed: %s", e)

    # 5. Production CatBoost Regressor (Strict loss_function='MAE')
    logger.info("Training production CatBoost Regressor with loss_function='MAE'...")
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

    is_ci = bool(os.environ.get("CI"))
    force_cpu = (args.device.lower() == "cpu") or is_ci

    cb_task_type = "CPU"
    cb_thread_count = 2

    if not force_cpu and args.device.lower() in ("cuda", "gpu"):
        try:
            test_cb = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
            test_cb.fit(np.array([[1.0]]), np.array([1.0]))
            cb_task_type = "GPU"
            cb_thread_count = None
            logger.info("CatBoost GPU acceleration enabled.")
        except Exception as e:
            logger.warning("GPU acceleration unavailable (%s); falling back to CPU.", e)
            cb_task_type = "CPU"
            cb_thread_count = 2
    else:
        logger.info(
            "Configuring CatBoost for task_type=%s, thread_count=%s (CI / CPU memory safety).",
            cb_task_type,
            cb_thread_count,
        )

    cb_kwargs: Dict[str, Any] = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 50,
        "task_type": cb_task_type,
    }
    if cb_thread_count is not None:
        cb_kwargs["thread_count"] = cb_thread_count

    cb_model = CatBoostRegressor(**cb_kwargs)
    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    pred_cb_val = cb_model.predict(val_pool)
    pred_cb_test = cb_model.predict(test_pool)
    record_benchmark("CatBoost (Selected)", pred_cb_val, pred_cb_test)

    # Export comparison table to models/model_comparison.csv
    results_df = pd.DataFrame(results).sort_values("MAE_MAD").reset_index(drop=True)
    comparison_path = output_dir / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    logger.info("Model comparison results saved to %s", comparison_path)
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT (Autohouse.ma Cahier des Charges)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

    # Check and summarize target threshold status for CatBoost
    cb_row = results_df[results_df["model"] == "CatBoost (Selected)"].iloc[0]
    if cb_row["meets_target_<20%"]:
        logger.info(">>> Production CatBoost model PASSED the target threshold with Test MAPE: %.2f%% (< 20%%)", cb_row["MAPE_%"])
    else:
        logger.warning(">>> Production CatBoost model Test MAPE is %.2f%% (target is < 20%%). Continued scraping will reduce error.", cb_row["MAPE_%"])

    # Export production model strictly in native .cbm format (NO pickle)
    cbm_model_path = output_dir / "catboost_model.cbm"
    cb_model.save_model(str(cbm_model_path))
    logger.info("Exported native CatBoost model to %s (format: .cbm, no pickle)", cbm_model_path)


if __name__ == "__main__":
    main()
