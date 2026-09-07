# Autohouse.ma - Multi-Source Moroccan Car Scraper & Cloud MLOps Pipeline

[![Multi-Source Scraper](https://github.com/ayouboublouch/xgboost-car/actions/workflows/scrape_multi_source.yml/badge.svg)](https://github.com/ayouboublouch/xgboost-car/actions/workflows/scrape_multi_source.yml)
[![Train Model](https://github.com/ayouboublouch/xgboost-car/actions/workflows/train_model.yml/badge.svg)](https://github.com/ayouboublouch/xgboost-car/actions/workflows/train_model.yml)

Automated, cloud-based data ingestion and machine learning pipeline for Moroccan used car platforms, strictly conforming to the **Autohouse.ma Cahier des Charges (v1.0)**.

**100% Cloud Execution**: All scraping, cross-source deduplication, feature engineering, and model training run autonomously on **GitHub Actions runners** (`ubuntu-latest`).

---

## 1. Directory Structure

```text
xgboost-car/
├── .github/
│   └── workflows/
│       ├── scrape_multi_source.yml # Parallel matrix workflow (5 sources concurrently)
│       ├── scrape_avito.yml        # Dedicated Avito scheduled scraper
│       └── train_model.yml         # Cloud ML pipeline (clean, features, train & commit)
├── scrapers/
│   ├── base.py                     # Abstract BaseScraper enforcing the 24 schema fields
│   ├── avito_scraper.py            # Avito provider (Next.js __NEXT_DATA__ extractor)
│   ├── moteur_scraper.py           # Moteur.ma provider
│   ├── wandaloo_scraper.py         # Wandaloo.com provider
│   ├── kifal_scraper.py            # Kifal-Auto.ma provider
│   ├── siaracash_scraper.py        # SiaraCash.ma provider
│   ├── run_all.py                  # CLI runner (--source all/provider --max-pages N)
│   └── requirements_scraper.txt    # Scraper dependencies (curl_cffi, bs4, pandas, pyarrow)
├── src/
│   ├── clean.py                    # Phase 1: Cross-source matching, deduplication, outlier filtering
│   ├── features.py                 # Phase 2: Feature engineering & MNAR imputation
│   └── train.py                    # Phases 3 & 4: Zero-leakage chronological split, CatBoost training
├── data/
│   ├── raw/                        # Raw Parquet & CSV scraped batches + baseline dataset
│   └── processed/                  # Cleaned & feature-engineered datasets
├── models/
│   ├── catboost_model.cbm          # Best production model (native CatBoost format, NO pickle)
│   └── model_comparison.csv       # Comparative benchmarking table (MAE, MAPE, R2)
├── requirements.txt                # ML pipeline dependencies (catboost, lightgbm, scikit-learn)
├── .gitignore
└── README.md
```

---

## 2. Multi-Source Scraping Providers (`scrapers/`)

All scrapers inherit from `BaseScraper` in `scrapers/base.py`, standardizing data into the **24 fields required by the Cahier des Charges**:
`listing_id`, `url`, `source`, `date_posted`, `date_scraped`, `title_raw`, `brand`, `model`, `trim`, `year`, `mileage_km`, `fuel_type`, `transmission`, `fiscal_power_cv`, `customs_status`, `condition`, `owners_count`, `doors_count`, `seller_type`, `city`, `region`, `price_mad`, `photos_count`, `description_raw`.

| Provider | Target Website | Strategy & Key Features |
|---|---|---|
| **Avito** | [Avito.ma](https://www.avito.ma) | Next.js `__NEXT_DATA__` JSON extraction + DOM fallback; A-Z x 2022-2026 matrix; Chrome 124 TLS impersonation via `curl_cffi`. |
| **Moteur** | [Moteur.ma](https://www.moteur.ma) | Catalog scraper; handles seller_type reliability caution. |
| **Wandaloo** | [Wandaloo.com](https://www.wandaloo.com) | Used car section `/occasion/` crawler; parses brand/model/specs. |
| **Kifal** | [Kifal-Auto.ma](https://kifal-auto.ma) | Certified inspected vehicles; high specification completeness. |
| **SiaraCash** | [SiaraCash.ma](https://siaracash.ma) | Marketplace listing cards crawler. |

### CLI Runner Usage
```bash
# Scrape all providers
python scrapers/run_all.py --source all --max-pages 10

# Scrape a specific provider
python scrapers/run_all.py --source moteur --max-pages 15
python scrapers/run_all.py --source wandaloo --max-pages 15
```

---

## 3. Data Cleaning, Cross-Source Matching & Leakage Prevention

### Phase 1: Cross-Source Deduplication (`src/clean.py`)
- **Within-Source Deduplication**: Eliminates duplicate `(source, listing_id)` and URLs.
- **Cross-Source Duplicate Matching**:
  - Matches listings across different platforms sharing:
    - Same Brand & Model (normalized)
    - Same Model Year
    - Mileage within $\pm 2,000$ km
    - Price within $\pm 5\%$
  - Assigns a shared `cross_source_match_id` linking multi-platform postings of the same vehicle.
- **Repost Tracking**: Assigns `repost_group_id` for identical listings over time.
- **Unified Leakage Prevention**: Merges `cross_source_match_id` and `repost_group_id` into a connected-component `leakage_group_id`.

### Phase 2: Missing Data (MNAR) & Features (`src/features.py`)
- Missingness indicators (`{col}_is_missing`) for `customs_status`, `trim`, `condition`, `owners_count`, etc.
- Explicit `"Inconnu"` handling (never imputed arbitrarily).
- Trim tiering (`top`, `mid`, `base`, `inconnu`).
- Seller type reliability neutralization (`seller_type_reliable`).
- Brand+model lookup imputation for deterministic attributes (`doors_count`, `fiscal_power_int`).

### Phases 3 & 4: Zero-Leakage Modeling (`src/train.py`)
- **Strict Chronological 70/15/15 Split**: Partitions data ordered by `date_scraped` while guaranteeing that entire `leakage_group_id` clusters stay together on the same side of the split (train, val, or test).
- **Production Architecture**: CatBoost Regressor evaluated against baseline models on MAE (MAD), MAPE (%), R², and % within $\pm 10\%$ and $\pm 15\%$.
- **Native Export**: Serialized strictly as `models/catboost_model.cbm` (**NO pickle**).

---

## 4. Parallel GitHub Actions Execution

### Parallel Multi-Source Workflow (`.github/workflows/scrape_multi_source.yml`)
- Executes across 5 parallel runners using a strategy matrix:
  ```yaml
  strategy:
    fail-fast: false
    matrix:
      source: [avito, moteur, wandaloo, kifal, siaracash]
  ```
- Uses a rebase-and-push loop (`git pull --rebase origin main`) to prevent git collision conflicts when multiple runners complete concurrently.

---

## 5. Instructions to Deploy & Run

### Step 1: Push Local Updates to GitHub
From your terminal:
```powershell
cd C:\Users\PC\.gemini\antigravity-ide\scratch\xgboost-car
git push origin main
```

### Step 2: Enable Workflow Permissions in GitHub
1. Go to repository **Settings** > **Actions** > **General**.
2. Under **Workflow permissions**, choose **Read and write permissions**.
3. Check **Allow GitHub Actions to create and approve pull requests**.
4. Click **Save**.

### Step 3: Trigger Parallel Scraping
1. Open the **Actions** tab on GitHub.
2. Select **Parallel Multi-Source Car Scraper** > Click **Run workflow**.
3. All 5 runners will launch concurrently in the cloud, scraping and committing their respective data batches back to `data/raw/`!
