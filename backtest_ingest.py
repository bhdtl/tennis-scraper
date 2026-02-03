# -*- coding: utf-8 -*-
"""
🎾 NEURAL SCOUT - HISTORICAL DATA INGESTION ENGINE (V3 - CANONICAL & ROBUST)
Source: tennis-data.co.uk
Target: Supabase 'market_odds'
Features: Canonical IDs, Requests Retry, Time-Travel Timestamping
"""

import os
import io
import sys
import requests
import pandas as pd
import hashlib
import time
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from supabase import create_client, Client

# --- CONFIG ---
DATA_URLS = [
    # ATP
    "http://www.tennis-data.co.uk/2023/2023.xlsx",
    "http://www.tennis-data.co.uk/2024/2024.xlsx",
    "http://www.tennis-data.co.uk/2025/2025.xlsx", # Falls verfügbar
    # WTA
    "http://www.tennis-data.co.uk/2023w/2023.xlsx",
    "http://www.tennis-data.co.uk/2024w/2024.xlsx",
    "http://www.tennis-data.co.uk/2025w/2025.xlsx"
]

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ CRITICAL: Supabase Secrets missing.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- NETWORK SESSION WITH RETRY ---
def get_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# --- CORE LOGIC ---

def normalize_name(name):
    """
    Standardisiert Namen: 'Sinner J.' -> 'Sinner'
    """
    if not isinstance(name, str): return "Unknown"
    name = name.strip()
    # Entferne Initialen am Ende wenn vorhanden (Space + 1-2 Buchstaben + Punkt optional)
    parts = name.split()
    if len(parts) > 1 and len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name

def generate_canonical_id(date_str, p1, p2):
    """
    Erzeugt eine ID, die unabhängig von der Reihenfolge der Spieler ist.
    Sinner vs Alcaraz = Alcaraz vs Sinner.
    """
    # 1. Namen normalisieren
    n1 = normalize_name(p1).lower()
    n2 = normalize_name(p2).lower()
    
    # 2. Alphabetisch sortieren (DAS IST DER FIX)
    players = sorted([n1, n2])
    
    # 3. Hash erzeugen
    raw = f"{date_str}_{players[0]}_{players[1]}".encode('utf-8')
    md5_hash = hashlib.md5(raw).hexdigest()
    
    # 4. Als UUID formatieren
    return f"{md5_hash[:8]}-{md5_hash[8:12]}-{md5_hash[12:16]}-{md5_hash[16:20]}-{md5_hash[20:]}"

def process_and_upload(session, url):
    filename = url.split("/")[-1]
    folder = url.split("/")[-2]
    label = f"[{folder}/{filename}]"
    
    print(f"📥 {label} Downloading...")
    
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 404:
            print(f"⚠️ {label} Not found (yet). Skipping.")
            return

        response.raise_for_status()
        
        # Load Excel
        df = pd.read_excel(io.BytesIO(response.content))
        total_rows = len(df)
        print(f"   📊 {label} Parsing {total_rows} matches...")

        records_to_upsert = []
        
        for _, row in df.iterrows():
            try:
                # Validierung
                if pd.isna(row.get('Winner')) or pd.isna(row.get('Loser')): continue
                
                # Datum
                raw_date = row.get('Date')
                if pd.isna(raw_date): continue
                
                if isinstance(raw_date, datetime):
                    date_str = raw_date.strftime("%Y-%m-%d")
                else:
                    # Fallback für Strings
                    date_str = str(raw_date)[:10]

                # Namen
                p1_name = normalize_name(row['Winner'])
                p2_name = normalize_name(row['Loser'])
                
                # CANONICAL ID
                match_uuid = generate_canonical_id(date_str, p1_name, p2_name)
                
                # Odds (Priorität: Bet365 -> Avg -> Pinnacle)
                o1 = row.get('B365W')
                o2 = row.get('B365L')
                
                if pd.isna(o1) or pd.isna(o2):
                    o1 = row.get('AvgW')
                    o2 = row.get('AvgL')
                
                if pd.isna(o1) or pd.isna(o2): continue # Skip if no odds
                
                # Score & Surface
                score = str(row.get('Score', '')) if not pd.isna(row.get('Score')) else ""
                surface = str(row.get('Surface', 'Hard'))
                
                # Timestamp Simulation
                # Wir setzen created_at auf das Match-Datum, damit die Sortierung stimmt
                simulated_time = f"{date_str}T12:00:00Z"

                match_obj = {
                    "id": match_uuid,
                    "player1_name": p1_name,
                    "player2_name": p2_name,
                    "tournament": str(row.get('Tournament', 'Unknown')),
                    "match_time": simulated_time,
                    "created_at": simulated_time, # TIME-TRAVEL FIX
                    "odds1": float(o1),
                    "odds2": float(o2),
                    "opening_odds1": float(o1), # Historisch = Closing ist unser Opening
                    "opening_odds2": float(o2),
                    "actual_winner_name": p1_name, # Winner Spalte aus Excel
                    "score": score,
                    "bookmaker": "Historical (B365)",
                    "ai_analysis_text": f"[HISTORICAL] {surface} | Rank: {row.get('WRank','-')} vs {row.get('LRank','-')}"
                }
                
                records_to_upsert.append(match_obj)
                
            except Exception:
                continue

        # BATCH UPSERT
        if not records_to_upsert:
            print(f"   ⚠️ {label} No valid records found.")
            return

        CHUNK_SIZE = 200 # Erhöht für Speed
        print(f"   🚀 {label} Upserting {len(records_to_upsert)} matches...")
        
        for i in range(0, len(records_to_upsert), CHUNK_SIZE):
            chunk = records_to_upsert[i:i + CHUNK_SIZE]
            try:
                # upsert with ignore_duplicates=False (wir wollen überschreiben/updaten)
                supabase.table("market_odds").upsert(chunk, on_conflict='id').execute()
                # Kleiner Sleep um API Rate Limits zu schonen
                time.sleep(0.1) 
            except Exception as e:
                print(f"      ⚠️ Chunk Error: {e}")
                
        print(f"   ✅ {label} Done.")

    except Exception as e:
        print(f"❌ {label} Critical Error: {e}")

if __name__ == "__main__":
    print("🏁 Starting Robust Historical Ingest...")
    sess = get_session()
    for u in DATA_URLS:
        process_and_upload(sess, u)
    print("✅ All jobs finished.")
