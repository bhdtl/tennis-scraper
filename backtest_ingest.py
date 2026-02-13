# -*- coding: utf-8 -*-
"""
🎾 NEURAL SCOUT - HISTORICAL DATA INGESTION
Status: PRODUCTION (Environment Logic synced with Scraper V85)
"""

import os
import io
import sys
import hashlib
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from supabase import create_client, Client

# --- 0. LOAD DOTENV (LOCAL DEV SUPPORT) ---
try:
    from dotenv import load_dotenv
    load_dotenv() 
except ImportError:
    pass 

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_History")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere History Ingest (Synced Environment)...")

# --- SECRETS MANAGEMENT (1:1 VOM SCRAPER) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Wir erlauben Service Role ODER normalen Key
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Check: Wir brauchen Supabase zwingend. Groq ist hier optional (da wir nur Excel parsen),
# aber um die Struktur gleich zu halten, loggen wir es nur als Warnung oder Fehler, je nach Wunsch.
# Da das Script aktuell keine AI nutzt, machen wir Groq optional, aber Supabase PFLICHT.

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen! Prüfe GitHub/Groq Secrets.")
    log("   -> Stelle sicher, dass SUPABASE_URL und SUPABASE_KEY in der .yml gesetzt sind.")
    sys.exit(1)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log("✅ Supabase Verbindung hergestellt.")
except Exception as e:
    log(f"❌ Supabase Init Error: {e}")
    sys.exit(1)

# --- DATA SOURCES ---
DATA_URLS = [
    "http://www.tennis-data.co.uk/2023/2023.xlsx",
    "http://www.tennis-data.co.uk/2024/2024.xlsx",
    "http://www.tennis-data.co.uk/2023w/2023.xlsx",
    "http://www.tennis-data.co.uk/2024w/2024.xlsx",
    "http://www.tennis-data.co.uk/2025/2025.xlsx",
    "http://www.tennis-data.co.uk/2025w/2024.xlsx"
]

# =================================================================
# 2. NETWORK & SESSION UTILS
# =================================================================
def get_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# =================================================================
# 3. CORE LOGIC
# =================================================================

def normalize_name(name):
    if not isinstance(name, str): return "Unknown"
    name = name.strip()
    parts = name.split()
    if len(parts) > 1 and len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name

def generate_canonical_id(date_str, p1, p2):
    n1 = normalize_name(p1).lower()
    n2 = normalize_name(p2).lower()
    players = sorted([n1, n2])
    raw = f"{date_str}_{players[0]}_{players[1]}".encode('utf-8')
    md5_hash = hashlib.md5(raw).hexdigest()
    return f"{md5_hash[:8]}-{md5_hash[8:12]}-{md5_hash[12:16]}-{md5_hash[16:20]}-{md5_hash[20:]}"

def process_and_upload(session, url):
    filename = url.split("/")[-1]
    folder = url.split("/")[-2]
    label = f"[{folder}/{filename}]"
    
    log(f"📥 {label} Download startet...")
    
    try:
        response = session.get(url, timeout=45)
        if response.status_code == 404:
            log(f"⚠️ {label} Datei nicht gefunden. Überspringe.")
            return

        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))
        total_rows = len(df)
        log(f"   📊 {label} Analysiere {total_rows} Matches...")

        records_to_upsert = []
        
        for _, row in df.iterrows():
            try:
                if pd.isna(row.get('Winner')) or pd.isna(row.get('Loser')): continue
                
                raw_date = row.get('Date')
                if pd.isna(raw_date): continue
                
                if isinstance(raw_date, datetime):
                    date_str = raw_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(raw_date)[:10]

                p1_name = normalize_name(row['Winner'])
                p2_name = normalize_name(row['Loser'])
                
                match_uuid = generate_canonical_id(date_str, p1_name, p2_name)
                
                o1 = row.get('B365W'); o2 = row.get('B365L')
                if pd.isna(o1) or pd.isna(o2):
                    o1 = row.get('AvgW'); o2 = row.get('AvgL')
                
                if pd.isna(o1) or pd.isna(o2): continue 
                
                score = str(row.get('Score', '')) if not pd.isna(row.get('Score')) else ""
                surface = str(row.get('Surface', 'Hard'))
                simulated_time = f"{date_str}T12:00:00Z"

                match_obj = {
                    "id": match_uuid,
                    "player1_name": p1_name,
                    "player2_name": p2_name,
                    "tournament": str(row.get('Tournament', 'Unknown')),
                    "match_time": simulated_time,
                    "created_at": simulated_time, 
                    "odds1": float(o1),
                    "odds2": float(o2),
                    "opening_odds1": float(o1),
                    "opening_odds2": float(o2),
                    "actual_winner_name": p1_name, 
                    "score": score,
                    "bookmaker": "Historical (B365)",
                    "ai_analysis_text": f"[HISTORICAL] Surface: {surface} | Rank: {row.get('WRank','-')} vs {row.get('LRank','-')}"
                }
                records_to_upsert.append(match_obj)
            except Exception: continue

        if not records_to_upsert:
            log(f"   ⚠️ {label} Keine validen Daten gefunden.")
            return

        CHUNK_SIZE = 200 
        log(f"   🚀 {label} Starte Upsert für {len(records_to_upsert)} Matches...")
        
        for i in range(0, len(records_to_upsert), CHUNK_SIZE):
            chunk = records_to_upsert[i:i + CHUNK_SIZE]
            try:
                supabase.table("market_odds").upsert(chunk, on_conflict='id').execute()
                time.sleep(0.05) 
            except Exception as e:
                log(f"      ⚠️ Chunk Error (Batch {i}): {e}")
                
        log(f"   ✅ {label} Erfolgreich importiert.")

    except Exception as e:
        log(f"❌ {label} Kritischer Fehler: {e}")

if __name__ == "__main__":
    log("🏁 Starte Historical Data Ingest Pipeline...")
    session = get_session()
    for u in DATA_URLS:
        process_and_upload(session, u)
    log("✅ Alle Jobs erledigt.")
