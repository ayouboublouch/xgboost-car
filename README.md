# Autohouse.ma - Used Car Price Estimation & Cloud MLOps Pipeline

[![Scrape Avito](https://github.com/ayouboublouch/xgboost-car/actions/workflows/scrape_avito.yml/badge.svg)](https://github.com/ayouboublouch/xgboost-car/actions/workflows/scrape_avito.yml)
[![Train Model](https://github.com/ayouboublouch/xgboost-car/actions/workflows/train_model.yml/badge.svg)](https://github.com/ayouboublouch/xgboost-car/actions/workflows/train_model.yml)

Automated, cloud-based data ingestion and machine learning pipeline for estimating used car prices on the Moroccan market, conforming to the **Autohouse.ma Cahier des Charges (v1.0)**.

**100% Cloud Execution**: All scraping, data processing, model benchmarking, and artifact updates run autonomously on **GitHub Actions runners** (`ubuntu-latest`).

---

## 1. Directory Structure

```text
xgboost-car/
├── .github/
│   └── workflows/
│       ├── scrape_avito.yml        # Scheduled & manual Avito scraper workflow
│       └── train_model.yml         # ML pipeline: clean, features, train & commit
├── scrapers/
│   ├── avito_scraper.py            # Dedicated Avito.ma scraper (A-Z x 2022-2026)
│   └── requirements_scraper.txt    # Scraper dependencies (curl_cffi, bs4, pandas, etc.)
├── src/
│   ├── clean.py                    # Phase 1: Ingestion, schema validation, outlier detection
│   ├── features.py                 # Phase 2: Feature engineering & MNAR imputation
│   └── train.py                    # Phases 3 & 4: Chronological split, CatBoost training
├── data/
│   ├── raw/                        # Raw Parquet & CSV scraped batches + baseline dataset
│   └── processed/                  # Cleaned & feature-engineered datasets
├── models/
│   ├── catboost_model.cbm          # Best production model (native CatBoost format, NO pickle)
│   └── model_comparison.csv       # Comparative benchmarking table (MAE, MAPE, R2)
├── requirements.txt                # ML pipeline dependencies (catboost, lightgbm, scikit-learn)
└── README.md
```

---

## 2. Scraping Architecture (`scrapers/avito_scraper.py`)

- **Iteration Matrix**:
  - Letters: `A` to `Z` (26 queries)
  - Years: `2022`, `2023`, `2024`, `2025`, `2026` (5 years)
  - Total combinations: 130 search slices
- **Extraction Engine**:
  1. Primary: Next.js `__NEXT_DATA__` JSON parsing for resilient, structured parameter extraction.
  2. Fallback: BeautifulSoup DOM parsing for standard HTML elements.
- **Anti-Bot & Rate Limiting**:
  - `curl_cffi` with Chrome 124 TLS/JA3 impersonation to bypass Cloudflare and anti-bot barriers on GitHub Actions.
  - Randomized delays between 1.0 and 3.0 seconds per request.
- **Schema Alignment**:
  Extracts all 24 required fields:
  `listing_id`, `url`, `source`, `date_posted`, `date_scraped`, `title_raw`, `brand`, `model`, `trim`, `year`, `mileage_km`, `fuel_type`, `transmission`, `fiscal_power_cv`, `customs_status`, `condition`, `owners_count`, `doors_count`, `seller_type`, `city`, `region`, `price_mad`, `photos_count`, `description_raw`.
- **Output**: Automatically commits daily batches to `data/raw/avito_YYYY-MM-DD.parquet` and `.csv`.

---

## 3. Machine Learning Pipeline (`src/`)

### Phase 1: Data Cleaning & Validation (`src/clean.py`)
- Aggregates all raw batches in `data/raw/` with existing `used_car_training_combined.csv`.
- Deduplicates on `listing_id` and canonical `url`.
- Cleans and parses `fiscal_power_cv` into integer `fiscal_power_int` and ceiling flag `fiscal_power_is_bucket_ceiling`.
- Flags outliers (`price_mad` outside [10k, 3.5M] MAD, invalid years/mileage).
- Computes `repost_group_id` across listings to prevent duplicate cars leaking across train/test sets.

### Phase 2: Missing Data (MNAR) & Feature Engineering (`src/features.py`)
- **Missingness Indicators**: Generates `{col}_is_missing` for `customs_status`, `trim`, `condition`, `owners_count`, `transmission`, `doors_count`, `fiscal_power_int`.
- **Explicit Unknowns**: Structurally missing fields (`customs_status`, `condition`, `owners_count`) mapped to `"Inconnu"`.
- **Trim Tiering**: Keyword classification into `top`, `mid`, `base`, and `inconnu`.
- **Seller Type Neutralization**: Preserves Avito seller reliability and neutralizes scrapers with known bias into `seller_type_reliable`.
- **Lookup Imputation**: Brand + model median imputation for deterministic attributes (`doors_count`, `fiscal_power_int`).
- **Cardinality Management**: Buckets rare vehicle models (< 3 occurrences) into `<brand>_other`.

### Phases 3 & 4: Modeling & Validation (`src/train.py`)
- **Strict Chronological Split (70/15/15)**: Splits purely by `date_scraped` while guaranteeing all rows of a `repost_group_id` remain in the same set (zero leakage).
- **Comparative Benchmarking**:
  - Baseline Median (Brand + Model + Year)
  - Ridge Regression
  - RandomForestRegressor
  - LightGBMRegressor
  - **CatBoostRegressor (Selected Production Model)**
- **Model Packaging**: Serialized exclusively in **native CatBoost format** (`models/catboost_model.cbm`, **NO pickle**).

---

## 4. GitHub Actions Automation

### Workflow 1: Scheduled Avito Scraper (`.github/workflows/scrape_avito.yml`)
- **Schedule**: Twice daily at `02:00` and `14:00` UTC (`0 2,14 * * *`).
- **Manual Trigger**: Via `workflow_dispatch` with optional dry-run and page limit settings.
- Automatically commits new scraped data to `data/raw/` using `github-actions[bot]`.

### Workflow 2: Model Training & Evaluation (`.github/workflows/train_model.yml`)
- **Automated Triggers**:
  - Runs upon completion of `scrape_avito.yml`.
  - Runs on push to `data/raw/**` or `src/**`.
  - Manual execution via `workflow_dispatch`.
- Automatically tests, cleans, trains, and commits updated `models/catboost_model.cbm` and `models/model_comparison.csv` back to `main`.

---

## 5. Setup & Operational Guide

### GitHub Repository Permissions (One-Time Setup)
To allow GitHub Actions to commit scraped datasets and trained models back to `main`:
1. Navigate to your repository on GitHub: `https://github.com/ayouboublouch/xgboost-car`.
2. Go to **Settings** > **Actions** > **General**.
3. Under **Workflow permissions**, select **Read and write permissions**.
4. Check **Allow GitHub Actions to create and approve pull requests**.
5. Click **Save**.

### How to Trigger Workflows Manually

#### Run Scraper:
1. Go to the **Actions** tab.
2. Select **Scheduled Avito Scraper** in the left sidebar.
3. Click **Run workflow** > Select branch `main` > Click **Run workflow**.

#### Run Model Training:
1. Go to the **Actions** tab.
2. Select **Train & Evaluate Price Model** in the left sidebar.
3. Click **Run workflow** > Select branch `main` > Click **Run workflow**.
