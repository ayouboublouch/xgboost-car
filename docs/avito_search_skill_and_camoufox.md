# Technical Evaluation: Avito Search Skill vs. Moroccan Avito (avito.ma) & Camoufox Bypassing

This document details the architectural evaluation of the `avito-search-skill`, explains the domain and structural divergence between Russian `avito.ru` and Moroccan `avito.ma`, and outlines our local Camoufox ingestion strategy.

---

## 1. Domain & Platform Incompatibility: `avito-search-skill` vs `avito.ma`

Community agent skills such as `avito-search-skill` are developed specifically for **Avito Russia (`avito.ru`)**. They cannot be used for Moroccan car price modeling on **Avito Maroc (`avito.ma`)** due to fundamental incompatibilities:

| Metric / Dimension | Avito Russia (`avito.ru`) | Avito Maroc (`avito.ma`) | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **Domain & Backend** | `www.avito.ru` (Russian microservices API) | `www.avito.ma` (EMPG / Frontier Car Group platform) | **Incompatible** (Separate domains, endpoints & auth) |
| **Currency** | Russian Ruble (₽ / RUB) | Moroccan Dirham (DH / MAD) | **Incompatible** (Currency conversion & range scale mismatch) |
| **Language & Taxonomy** | Russian Cyrillic (e.g. `Москва`, `Лада`) | French & Moroccan Arabic (Darija) (e.g. `Casablanca`, `Dacia`) | **Incompatible** (Fails string extraction & normalization) |
| **Phone Number Formats** | Russian 11-digit (`+7 9XX XXX-XX-XX`) | Moroccan 10-digit (`05`, `06`, `07` / `+212`) | **Incompatible** (Phone regex & hash deduplication failure) |
| **Anti-Bot Defenses** | Avito RU proprietary IP challenge | Cloudflare Bot Management & Turnstile | **Incompatible** (Requires different bypass mechanisms) |

**Conclusion**: The `avito-search-skill` must NOT be integrated into the Moroccan automotive pipeline.

---

## 2. Cloudflare Bot Protection & GitHub Actions Limitations

`avito.ma` enforces Cloudflare Turnstile and Bot Management. Requests originating from public cloud datacenter IP ranges (including GitHub Actions `ubuntu-latest` runners in Azure / AWS) receive HTTP 403 Forbidden or interactive Turnstile challenges.

To maintain a green, zero-failure CI/CD pipeline:
1. The GitHub Actions matrix workflow (`.github/workflows/scrape_multi_source.yml`) prioritizes unblocked, high-yield Moroccan platforms:
   - **Moteur.ma** (Primary, ~90+ listings per page, full details & WhatsApp contact links)
   - **Wandaloo.com** (Secondary, unblocked catalog pagination)
2. `continue-on-error: true` is configured on the scraping step so transient Cloudflare challenges on Avito, Kifal, or SiaraCash never block CI execution or model training.

---

## 3. Local Residential Ingestion via Camoufox

For harvesting Moroccan Avito listings without datacenter IP blocks, we provide an optional standalone local crawler:
`scrapers/camofox_avito.py`

### What is Camoufox?
[Camoufox](https://github.com/daijro/camoufox) is a stealth browser engine based on Firefox designed to bypass modern anti-bot systems like Cloudflare Turnstile. It spoofs:
- Hardware fingerprints (canvas, audio, WebGL)
- TLS / JA3 / JA4 fingerprints
- Human-like mouse movements and interaction latencies

### Setup & Execution (Local Developer Machine Only)
`camoufox` is deliberately excluded from `requirements.txt` and GitHub Actions dependencies to guarantee **zero dependency bloat** on CI runners.

To run locally on a residential IP:
```bash
# 1. Install Camoufox into your local environment
pip install camoufox

# 2. Download the stealth browser binary
camoufox fetch

# 3. Run the local Avito harvester
python scrapers/camofox_avito.py --max-pages 5 --output-dir data/raw
```

Harvested records are formatted directly into the 24 standardized Cahier des Charges columns and saved to `data/raw/avito_YYYY-MM-DD.csv` and `.parquet`, ready for merging by `src/clean.py`.
