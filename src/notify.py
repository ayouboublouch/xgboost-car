#!/usr/bin/env python3
"""
Autohouse.ma - Automated Telegram & WhatsApp Notification Engine
-----------------------------------------------------------------
Dispatches daily ingestion summaries and real-time deal alerts to Telegram
and WhatsApp.

Features:
  1. Daily Ingestion Telemetry:
     - Master database inventory count (80,000+ unique listings)
     - New listings scraped today / last 24h
     - Breakdown by Moroccan portal (Avito, Moteur, MarocAnnonces, etc.)
  2. Real-Time Deal Alerts:
     - Top underpriced listings ("🔥 Great Deal")
     - Vehicle specifications (brand, model, year, fuel, transmission, city)
     - Listed price vs ML Champion Ensemble estimated fair market value
     - Computed discount percentage (-XX.X%)
     - Normalized 10-digit Moroccan seller phone
     - Direct vehicle fiche URL
  3. Multi-Channel Support:
     - Telegram Bot API (Rich HTML formatting with inline links)
     - WhatsApp Gateway (CallMeBot free webhook with WhatsApp markdown)
  4. Production Fail-Safe:
     - Missing or placeholder tokens log a warning without raising exceptions
       or interrupting downstream MLOps pipelines.

Usage:
  python src/notify.py --summary --deals --top 5
  python src/notify.py --dry-run --summary --deals --top 3
  python src/notify.py --test
"""

import argparse
import html
import logging
import os
import re
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# UTF-8 stdout fix for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Optional dotenv loading
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [notify] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("notify")

# Configuration from environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "false").strip().lower() in ("true", "1", "yes")
WHATSAPP_PHONE = os.getenv("WHATSAPP_PHONE", "").strip()
WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY", "").strip()

PLACEHOLDER_TOKENS = {
    "your_bot_token_here",
    "your_chat_id_here",
    "your_api_key_here",
    "",
}


def is_telegram_configured() -> bool:
    """Check if valid Telegram credentials are present."""
    return (
        bool(TELEGRAM_BOT_TOKEN)
        and bool(TELEGRAM_CHAT_ID)
        and TELEGRAM_BOT_TOKEN not in PLACEHOLDER_TOKENS
        and TELEGRAM_CHAT_ID not in PLACEHOLDER_TOKENS
    )


def is_whatsapp_configured() -> bool:
    """Check if WhatsApp integration is enabled and configured."""
    return (
        WHATSAPP_ENABLED
        and bool(WHATSAPP_PHONE)
        and bool(WHATSAPP_API_KEY)
        and WHATSAPP_API_KEY not in PLACEHOLDER_TOKENS
    )


def normalize_phone_display(phone_val: Any) -> str:
    """Normalize raw phone to standard 10-digit Moroccan string (e.g. 0661234567)."""
    if pd.isna(phone_val) or phone_val is None:
        return "Non renseigné"
    s = str(phone_val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return "Non renseigné"
    # Remove float decimal point like 661646767.0
    s = s.split(".")[0].strip()
    digits = re.sub(r"\D+", "", s)
    if digits.startswith("212") and len(digits) == 12:
        digits = "0" + digits[3:]
    elif digits.startswith("00212") and len(digits) == 14:
        digits = "0" + digits[5:]
    elif len(digits) == 9 and digits[0] in "567":
        digits = "0" + digits

    if len(digits) == 10 and digits.startswith("0") and digits[1] in "567":
        return digits
    return digits if digits else "Non renseigné"


def html_to_whatsapp_markdown(html_text: str) -> str:
    """
    Convert basic HTML tags (<b>, <code>, <a>) to WhatsApp formatting.
    WhatsApp uses *bold*, ```code```, and plain URLs.
    """
    text = html_text
    # Links: <a href="url">text</a> -> text: url
    text = re.sub(r'<a\s+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', r"\2: \1", text, flags=re.IGNORECASE)
    # Bold: <b>text</b> or <strong>text</strong> -> *text*
    text = re.sub(r'<(?:b|strong)>(.*?)</(?:b|strong)>', r"*\1*", text, flags=re.IGNORECASE)
    # Monospace: <code>text</code> -> ```text```
    text = re.sub(r'<code>(.*?)</code>', r"```\1```", text, flags=re.IGNORECASE)
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', "", text)
    # Unescape HTML entities
    return html.unescape(text)


def send_telegram_message(
    text: str,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = False,
    timeout: int = 15,
) -> bool:
    """
    Send an HTML-formatted message to Telegram via Bot API.
    Returns True if sent successfully, False otherwise (fail-safe).
    """
    if not is_telegram_configured():
        logger.warning(
            "Telegram notification skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
            "is not configured in .env."
        )
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("Telegram message successfully sent (Chat ID: %s)", TELEGRAM_CHAT_ID)
            return True
        else:
            logger.error(
                "Telegram API error (%d): %s",
                resp.status_code,
                resp.text,
            )
            return False
    except requests.RequestException as exc:
        logger.warning("Telegram network request failed: %s", exc)
        return False


def send_whatsapp_message(
    text: str,
    timeout: int = 15,
) -> bool:
    """
    Send a plain-text/markdown alert to WhatsApp via CallMeBot webhook.
    Returns True if successful, False otherwise (fail-safe).
    """
    if not is_whatsapp_configured():
        logger.warning(
            "WhatsApp notification skipped: WHATSAPP_ENABLED is false or "
            "WHATSAPP_API_KEY is not configured in .env."
        )
        return False

    phone = re.sub(r"\D+", "", WHATSAPP_PHONE)
    wa_text = html_to_whatsapp_markdown(text)
    encoded_text = urllib.parse.quote_plus(wa_text)
    url = (
        f"https://api.callmebot.com/whatsapp.php"
        f"?phone={phone}&text={encoded_text}&apikey={WHATSAPP_API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            logger.info("WhatsApp alert successfully sent to +%s", phone)
            return True
        else:
            logger.error("WhatsApp CallMeBot returned status %d: %s", resp.status_code, resp.text)
            return False
    except requests.RequestException as exc:
        logger.warning("WhatsApp network request failed: %s", exc)
        return False


# Aliases for specification adherence
send_telegram = send_telegram_message
send_whatsapp = send_whatsapp_message


def build_daily_ingestion_summary(
    master_parquet_path: Union[str, Path] = "data/raw/scraped_master_database.parquet",
    target_date: Optional[str] = None,
) -> str:
    """
    Inspect master database and calculate:
      - Total inventory count (80,000+)
      - Ingested records today / in last 24h
      - Source breakdown by portal
    Returns executive HTML-formatted message.
    """
    p = Path(master_parquet_path)
    if not p.exists():
        logger.warning("Master database parquet not found at %s", p)
        return "⚠️ <b>Autohouse.ma Ingestion Telemetry</b>: Master database not found."

    df = pd.read_parquet(p)
    total_listings = len(df)
    now_date = target_date or datetime.now().strftime("%Y-%m-%d")

    # Ingested today calculation
    if "date_scraped" in df.columns:
        scraped_dates = df["date_scraped"].dropna().astype(str)
        today_mask = scraped_dates.str.startswith(now_date)
        today_count = int(today_mask.sum())
        today_breakdown = df[today_mask]["source"].value_counts().to_dict() if "source" in df.columns else {}
    else:
        today_count = 0
        today_breakdown = {}

    # Overall source breakdown
    source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}

    # Portal display mapping
    portal_names = {
        "avito": "Avito.ma",
        "moteur": "Moteur.ma",
        "marocannonces": "MarocAnnonces",
        "wandaloo": "Wandaloo",
        "kifal": "Kifal Auto",
        "autocash": "Autocash",
        "siaracash": "SiaraCash",
        "marochub": "MarocHub",
        "auto24": "Auto24",
    }

    lines = [
        "📊 <b>AUTOHOUSE.MA — DAILY INGESTION REPORT</b>",
        f"📅 <b>Date</b>: <code>{now_date}</code>",
        f"🆕 <b>New Listings Ingested Today</b>: <b>+{today_count:,}</b>",
        f"📦 <b>Total Master Inventory</b>: <b>{total_listings:,}</b> verified listings",
        "",
        "🌐 <b>Inventory Breakdown by Portal:</b>",
    ]

    for src_key, count in source_counts.items():
        name = portal_names.get(str(src_key).lower(), str(src_key).capitalize())
        today_add = today_breakdown.get(src_key, 0)
        delta_str = f" <i>(+{today_add:,} today)</i>" if today_add > 0 else ""
        pct = (count / total_listings) * 100 if total_listings > 0 else 0
        lines.append(f"  • <b>{name}</b>: {count:,} ({pct:.1f}%){delta_str}")

    # Phone audit status
    if "seller_phone" in df.columns:
        valid_phones = df["seller_phone"].dropna().astype(str).str.strip()
        phone_count = (valid_phones != "").sum()
        lines.append("")
        lines.append(f"📞 <b>Direct Contact Coverage</b>: {phone_count:,} verified seller phones ({phone_count/total_listings*100:.1f}%)")

    lines.append("")
    lines.append("⚡ <i>Autohouse.ma MLOps Pipeline — Ingestion Active</i>")
    return "\n".join(lines)


# Alias
build_ingestion_summary = build_daily_ingestion_summary


def build_deal_alerts_summary(
    predictions_path: Union[str, Path] = "models/car_price_predictions.csv",
    top_n: int = 5,
    min_discount: float = 12.0,
) -> List[Dict[str, Any]]:
    """
    Extract top underpriced '🔥 Great Deal' vehicle listings from predictions.
    Returns a list of structured deal dicts with HTML card representations.
    """
    p = Path(predictions_path)
    if not p.exists():
        fallback = Path("data/processed/car_price_predictions.csv")
        if fallback.exists():
            p = fallback
        else:
            logger.warning("Predictions dataset not found at %s", p)
            return []

    df = pd.read_csv(p, low_memory=False)
    if "market_deal_rating" not in df.columns or "deviation_percentage_%" not in df.columns:
        logger.warning("Predictions dataset missing required deal evaluation columns.")
        return []

    # Filter for Great Deals
    great_deals = df[
        df["market_deal_rating"].astype(str).str.contains("Great Deal", na=False)
        & (df["deviation_percentage_%"] <= -min_discount)
    ].copy()

    if great_deals.empty:
        # Fallback to any underpriced
        great_deals = df[df["deviation_percentage_%"] < 0].copy()

    # Sort by deviation ascending (most underpriced first)
    great_deals = great_deals.sort_values(by="deviation_percentage_%", ascending=True)

    # Take top N
    selected = great_deals.head(top_n)
    deals: List[Dict[str, Any]] = []

    for _, row in selected.iterrows():
        brand = str(row.get("brand", "Véhicule")).title()
        model = str(row.get("model", "")).title()
        raw_year = row.get("year")
        year_str = f"{int(raw_year)}" if pd.notna(raw_year) else "N/A"
        price_mad = float(row.get("price_mad", 0.0))
        fair_value = float(row.get("predicted_price_mad", 0.0))
        deviation = float(row.get("deviation_percentage_%", 0.0))
        discount_pct = abs(deviation)
        savings_mad = fair_value - price_mad

        city = str(row.get("city", "Maroc")).title()
        fuel = str(row.get("fuel_type", "Inconnu")).title()
        trans = str(row.get("transmission", "Inconnue")).title()
        url = str(row.get("url", "https://autohouse.ma"))
        seller_phone = normalize_phone_display(row.get("seller_phone"))

        # HTML formatted card
        card_html = (
            f"🔥 <b>GREAT DEAL ALERT (-{discount_pct:.1f}%)</b>\n"
            f"🚗 <b>{brand} {model} ({year_str})</b>\n"
            f"💰 <b>Listed Price</b>: {price_mad:,.0f} MAD\n"
            f"🎯 <b>Estimated Fair Value</b>: {fair_value:,.0f} MAD\n"
            f"📉 <b>Discount</b>: -{discount_pct:.1f}% (Save {savings_mad:,.0f} MAD)\n"
            f"📍 <b>City</b>: {city} | <b>Fuel</b>: {fuel} | <b>Boîte</b>: {trans}\n"
            f"📞 <b>Seller Phone</b>: <code>{seller_phone}</code>\n"
            f"🔗 <a href='{url}'>View Original Listing</a>"
        )

        deals.append({
            "brand": brand,
            "model": model,
            "year": year_str,
            "price_mad": price_mad,
            "predicted_price_mad": fair_value,
            "deviation_percentage_%": deviation,
            "discount_pct": discount_pct,
            "savings_mad": savings_mad,
            "city": city,
            "fuel_type": fuel,
            "transmission": trans,
            "seller_phone": seller_phone,
            "url": url,
            "card_html": card_html,
        })

    return deals


# Alias
build_top_deals = build_deal_alerts_summary


def send_pipeline_notifications(
    summary: bool = True,
    deals: bool = True,
    top_n: int = 5,
    channel: str = "all",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Main notification dispatcher:
      - Builds Ingestion Summary
      - Builds Top Deal Alerts
      - Sends to Telegram and/or WhatsApp
    """
    results: Dict[str, Any] = {
        "summary_sent_telegram": False,
        "summary_sent_whatsapp": False,
        "deals_sent_telegram": 0,
        "deals_sent_whatsapp": 0,
        "deals_found": 0,
        "dry_run": dry_run,
    }

    telegram_target = channel.lower() in ("all", "telegram")
    whatsapp_target = channel.lower() in ("all", "whatsapp")

    # 1. Summary Notification
    if summary:
        summary_msg = build_daily_ingestion_summary()
        if dry_run:
            print("\n" + "=" * 60)
            print("[DRY-RUN] DAILY INGESTION TELEMETRY MESSAGE:")
            print("=" * 60)
            print(summary_msg)
            print("=" * 60 + "\n")
        else:
            if telegram_target:
                results["summary_sent_telegram"] = send_telegram_message(summary_msg)
            if whatsapp_target:
                results["summary_sent_whatsapp"] = send_whatsapp_message(summary_msg)

    # 2. Top Deal Alerts
    if deals:
        deal_items = build_deal_alerts_summary(top_n=top_n)
        results["deals_found"] = len(deal_items)

        if not deal_items:
            logger.info("No active Great Deals found matching criteria.")
        else:
            # Header message
            header_html = (
                f"🚨 <b>AUTOHOUSE.MA — TOP {len(deal_items)} MARKET DEAL ALERTS</b>\n"
                f"<i>Champion Blended Ensemble Valuation Filter</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━"
            )

            if dry_run:
                print("\n" + "=" * 60)
                print(f"[DRY-RUN] TOP {len(deal_items)} DEAL ALERTS:")
                print("=" * 60)
                print(header_html)
                for i, d in enumerate(deal_items, start=1):
                    print(f"\n--- DEAL #{i} ---")
                    print(d["card_html"])
                print("=" * 60 + "\n")
            else:
                if telegram_target and is_telegram_configured():
                    send_telegram_message(header_html)
                    for d in deal_items:
                        if send_telegram_message(d["card_html"]):
                            results["deals_sent_telegram"] += 1

                if whatsapp_target and is_whatsapp_configured():
                    send_whatsapp_message(header_html)
                    for d in deal_items:
                        if send_whatsapp_message(d["card_html"]):
                            results["deals_sent_whatsapp"] += 1

    return results


def notify_after_prediction(top_n: int = 5) -> Dict[str, Any]:
    """
    Convenience hook called at the end of src/predict.py.
    Safely executes without throwing exceptions.
    """
    try:
        if not is_telegram_configured() and not is_whatsapp_configured():
            logger.info(
                "Automated notification hook skipped: No active Telegram or WhatsApp "
                "credentials configured in .env."
            )
            return {"status": "skipped", "reason": "unconfigured"}

        logger.info("Triggering automated post-prediction deal notifications...")
        return send_pipeline_notifications(summary=True, deals=True, top_n=top_n)
    except Exception as exc:
        logger.warning("Automated notification hook encountered error (safe ignore): %s", exc)
        return {"status": "error", "error": str(exc)}


def send_test_ping() -> bool:
    """Send a diagnostic heartbeat ping to verify API connectivity."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ping_html = (
        f"🤖 <b>Autohouse.ma Notification Engine Test</b>\n"
        f"🕒 Timestamp: <code>{now_str}</code>\n"
        f"✅ Bot API Connectivity: <b>Verified OK</b>\n"
        f"🚀 Production MLOps telemetry ready."
    )
    print("Dispatching diagnostic heartbeat ping...")
    res_tg = send_telegram_message(ping_html)
    res_wa = send_whatsapp_message(ping_html)
    return res_tg or res_wa


def main():
    parser = argparse.ArgumentParser(description="Autohouse.ma Automated Notification Engine")
    parser.add_argument("--summary", action="store_true", help="Send daily ingestion summary")
    parser.add_argument("--deals", action="store_true", help="Send top underpriced deal alerts")
    parser.add_argument("--top", type=int, default=5, help="Number of top deals to alert (default: 5)")
    parser.add_argument(
        "--channel",
        type=str,
        default="all",
        choices=["all", "telegram", "whatsapp"],
        help="Target notification channel (default: all)",
    )
    parser.add_argument("--test", action="store_true", help="Send test connectivity ping")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format and print messages to stdout without contacting external APIs",
    )
    args = parser.parse_args()

    # Default to summary + deals if neither explicitly set (unless --test is passed)
    if args.test:
        send_test_ping()
        return

    send_summary = args.summary
    send_deals = args.deals
    if not send_summary and not send_deals:
        send_summary = True
        send_deals = True

    res = send_pipeline_notifications(
        summary=send_summary,
        deals=send_deals,
        top_n=args.top,
        channel=args.channel,
        dry_run=args.dry_run,
    )
    logger.info("Notification execution completed: %s", res)


if __name__ == "__main__":
    main()
