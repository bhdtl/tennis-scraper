# -*- coding: utf-8 -*-
"""
🎾 NEURAL SCOUT - HISTORICAL DATA INGESTION (ETL ENGINE)
Architecture: Robust Environment Loading, Canonical IDs, Bulk Upsert
Status: PRODUCTION READY
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
    load_dotenv() # Liest .env Datei ein, falls vorhanden
except ImportError:
    pass # In GitHub Actions ist dotenv oft nicht nötig/installiert, da Secrets injected werden

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

log("🔌 Initialisiere History Ingest (V4.0 - ROBUST BOOTLOADER)...")

# --- SECRETS MANAGEMENT ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Wir prüfen beide Keys: Service Role (bevorzugt für Writes) oder Anon Key
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen!")
    log("   -> Stelle sicher, dass SUPABASE_URL und SUPABASE_KEY in der .env oder den Secrets gesetzt sind.")
    sys.exit(1)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log("✅ Supabase Verbindung hergestellt.")
except Exception as e:
    log(f"❌ Supabase Init Error: {e}")
    sys.exit(1)

# --- DATA SOURCES ---
DATA_URLS = [
    # ATP
    "http://www.tennis-data.co.uk/2023/2023.xlsx",
    "http://www.tennis-data.co.uk/2024/2024.xlsx",
    # WTA
    "http://www.tennis-data.co.uk/2023w/2023.xlsx",
    "http://www.tennis-data.co.uk/2024w/2024.xlsx"
]

# =================================================================
# 2. NETWORK & SESSION UTILS
# =================================================================
def get_session():
    """Erstellt eine Session mit Retry-Logik (wie ein Browser der neu lädt)"""
    session = requests.Session()
    retry = Retry(
        total=5, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# =================================================================
# 3. CORE LOGIC
# =================================================================

def normalize_name(name):
    """
    Standardisiert Namen: 'Sinner J.' -> 'Sinner'
    """
    if not isinstance(name, str): return "Unknown"
    name = name.strip()
    parts = name.split()
    # Entferne Initialen am Ende (z.B. "Alcaraz C.")
    if len(parts) > 1 and len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name

def generate_canonical_id(date_str, p1, p2):
    """
    IDEMPOTENCY KEY:
    Sortiert Namen alphabetisch, damit Sinner vs Alcaraz == Alcaraz vs Sinner.
    Verhindert Duplikate in der DB.
    """
    n1 = normalize_name(p1).lower()
    n2 = normalize_name(p2).lower()
    
    # Alphabetische Sortierung für Konsistenz
    players = sorted([n1, n2])
    
    raw = f"{date_str}_{players[0]}_{players[1]}".encode('utf-8')
    md5_hash = hashlib.md5(raw).hexdigest()
    
    # Fake UUID Format für Postgres
    return f"{md5_hash[:8]}-{md5_hash[8:12]}-{md5_hash[12:16]}-{md5_hash[16:20]}-{md5_hash[20:]}"

def process_and_upload(session, url):
    filename = url.split("/")[-1]
    folder = url.split("/")[-2] # z.B. 2024 oder 2024w
    label = f"[{folder}/{filename}]"
    
    log(f"📥 {label} Download startet...")
    
    try:
        response = session.get(url, timeout=45) # Timeout erhöht für große Files
        if response.status_code == 404:
            log(f"⚠️ {label} Datei nicht gefunden. Überspringe.")
            return

        response.raise_for_status()
        
        # Excel in Memory laden
        df = pd.read_excel(io.BytesIO(response.content))
        total_rows = len(df)
        log(f"   📊 {label} Analysiere {total_rows} Matches...")

        records_to_upsert = []
        
        for _, row in df.iterrows():
            try:
                # Validierung: Winner/Loser müssen existieren
                if pd.isna(row.get('Winner')) or pd.isna(row.get('Loser')): continue
                
                # Datum Validierung
                raw_date = row.get('Date')
                if pd.isna(raw_date): continue
                
                if isinstance(raw_date, datetime):
                    date_str = raw_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(raw_date)[:10]

                # Namen
                p1_name = normalize_name(row['Winner'])
                p2_name = normalize_name(row['Loser'])
                
                # CANONICAL ID (Der Schutz gegen Duplikate)
                match_uuid = generate_canonical_id(date_str, p1_name, p2_name)
                
                # Odds Check (B365 > Avg)
                o1 = row.get('B365W')
                o2 = row.get('B365L')
                
                if pd.isna(o1) or pd.isna(o2):
                    o1 = row.get('AvgW')
                    o2 = row.get('AvgL')
                
                # Wenn keine Odds da sind, ist das Match für uns wertlos
                if pd.isna(o1) or pd.isna(o2): continue 
                
                # Score & Surface
                score = str(row.get('Score', '')) if not pd.isna(row.get('Score')) else ""
                surface = str(row.get('Surface', 'Hard'))
                
                # TIME TRAVEL: Setze created_at auf das tatsächliche Spieldatum!
                simulated_time = f"{date_str}T12:00:00Z"

                match_obj = {
                    "id": match_uuid,
                    "player1_name": p1_name,
                    "player2_name": p2_name,
                    "tournament": str(row.get('Tournament', 'Unknown')),
                    "match_time": simulated_time,
                    "created_at": simulated_time, # WICHTIG für Sortierung
                    "odds1": float(o1),
                    "odds2": float(o2),
                    "opening_odds1": float(o1),
                    "opening_odds2": float(o2),
                    "actual_winner_name": p1_name, # Winner Spalte aus Excel
                    "score": score,
                    "bookmaker": "Historical (B365)",
                    "ai_analysis_text": f"[HISTORICAL] Surface: {surface} | Rank: {row.get('WRank','-')} vs {row.get('LRank','-')}"
                }
                
                records_to_upsert.append(match_obj)
                
            except Exception:
                continue

        # BATCH UPSERT (Chunks um DB nicht zu überlasten)
        if not records_to_upsert:
            log(f"   ⚠️ {label} Keine validen Daten gefunden.")
            return

        CHUNK_SIZE = 200 
        log(f"   🚀 {label} Starte Upsert für {len(records_to_upsert)} Matches...")
        
        for i in range(0, len(records_to_upsert), CHUNK_SIZE):
            chunk = records_to_upsert[i:i + CHUNK_SIZE]
            try:
                # upsert: Überschreibt existierende IDs -> Keine Duplikate!
                supabase.table("market_odds").upsert(chunk, on_conflict='id').execute()
                # Kurze Atempause für die API
                time.sleep(0.05) 
            except Exception as e:
                log(f"      ⚠️ Chunk Error (Batch {i}): {e}")
                
        log(f"   ✅ {label} Erfolgreich importiert.")

    except Exception as e:
        log(f"❌ {label} Kritischer Fehler: {e}")

# =================================================================
# 4. MAIN ENTRY POINT
# =================================================================
if __name__ == "__main__":
    log("🏁 Starte Historical Data Ingest Pipeline...")
    session = get_session()
    
    for url in DATA_URLS:
        process_and_upload(session, url)
        
    log("✅ Alle Jobs erledigt. Datenbank aktualisiert.")
