# -*- coding: utf-8 -*-
"""
🎾 NEURAL SCOUT - HISTORICAL DATA INGESTION ENGINE (V2 - IDEMPOTENT)
Source: tennis-data.co.uk
Target: Supabase 'market_odds'
Feature: Deterministic IDs prevents duplicates automatically.
"""

import os
import io
import sys
import requests
import pandas as pd
import hashlib
from datetime import datetime
from supabase import create_client, Client

# --- CONFIG ---
DATA_URLS = [
    "http://www.tennis-data.co.uk/2023/2023.xlsx",
    "http://www.tennis-data.co.uk/2024/2024.xlsx",
    "http://www.tennis-data.co.uk/2023w/2023.xlsx",
    "http://www.tennis-data.co.uk/2024w/2024.xlsx"
]

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ CRITICAL: Supabase Secrets missing.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_deterministic_id(date_str, p1, p2):
    """
    Erstellt einen einzigartigen Fingerabdruck für das Match.
    Gleiches Spiel = Gleiche ID. Verhindert Duplikate zu 100%.
    """
    raw = f"{date_str}_{p1}_{p2}".lower().encode('utf-8')
    return hashlib.md5(raw).hexdigest() # Gibt eine UUID-ähnliche ID zurück

def normalize_name(name):
    if not isinstance(name, str): return "Unknown"
    # "Sinner J." -> "Sinner"
    parts = name.strip().split()
    if len(parts) > 1 and len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name.strip()

def process_and_upload(url):
    print(f"📥 Downloading: {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))
        
        records_to_upsert = []
        
        print(f"   📊 Processing {len(df)} matches...")

        for _, row in df.iterrows():
            try:
                if pd.isna(row['Winner']) or pd.isna(row['Loser']): continue
                
                # Namen normalisieren
                p1_name = normalize_name(row['Winner'])
                p2_name = normalize_name(row['Loser'])
                
                # Datum formatieren
                raw_date = row['Date']
                if isinstance(raw_date, datetime):
                    date_str = raw_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(raw_date)[:10] # Fallback
                
                # Deterministic ID generieren (Der Duplikat-Killer)
                # Wir nutzen ein Prefix 'hist_', damit man sie leicht erkennen kann
                match_uuid = generate_deterministic_id(date_str, p1_name, p2_name)
                # Supabase nutzt UUIDs, MD5 ist ein 32-char hex string. 
                # Wir formatieren es als UUID: 8-4-4-4-12
                fake_uuid = f"{match_uuid[:8]}-{match_uuid[8:12]}-{match_uuid[12:16]}-{match_uuid[16:20]}-{match_uuid[20:]}"

                # Odds Check
                o1 = row.get('B365W') if not pd.isna(row.get('B365W')) else row.get('AvgW')
                o2 = row.get('B365L') if not pd.isna(row.get('B365L')) else row.get('AvgL')
                
                if pd.isna(o1) or pd.isna(o2): continue
                
                match_obj = {
                    "id": fake_uuid, # WICHTIG: Wir setzen die ID selbst!
                    "player1_name": p1_name,
                    "player2_name": p2_name,
                    "tournament": str(row.get('Tournament', 'Unknown')),
                    "match_time": f"{date_str}T12:00:00Z",
                    "created_at": datetime.now().isoformat(),
                    "odds1": float(o1),
                    "odds2": float(o2),
                    "opening_odds1": float(o1),
                    "opening_odds2": float(o2),
                    "actual_winner_name": p1_name, 
                    "score": str(row.get('Score', '')),
                    "bookmaker": "Historical (B365)",
                    "ai_analysis_text": f"[HISTORICAL] Surface: {row.get('Surface','Hard')}"
                }
                
                records_to_upsert.append(match_obj)
                
            except Exception:
                continue

        # Batch Upsert
        CHUNK_SIZE = 100
        print(f"   🚀 Upserting {len(records_to_upsert)} matches...")
        
        for i in range(0, len(records_to_upsert), CHUNK_SIZE):
            chunk = records_to_upsert[i:i + CHUNK_SIZE]
            try:
                # upsert statt insert! 
                # on_conflict='id' sorgt dafür, dass existierende IDs überschrieben werden.
                supabase.table("market_odds").upsert(chunk, on_conflict='id').execute()
                print(f"      ✅ Chunk {i // CHUNK_SIZE + 1} synced.")
            except Exception as e:
                print(f"      ⚠️ Chunk Error: {e}")
                
    except Exception as main_e:
        print(f"❌ Failed: {main_e}")

if __name__ == "__main__":
    print("🏁 Starting Idempotent Backfill...")
    for url in DATA_URLS:
        process_and_upload(url)
    print("✅ Complete.")
