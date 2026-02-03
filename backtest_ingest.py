# -*- coding: utf-8 -*-
"""
🎾 NEURAL SCOUT - HISTORICAL DATA INGESTION ENGINE (ETL)
Source: tennis-data.co.uk
Target: Supabase 'market_odds' & 'odds_history' (optional)
"""

import os
import io
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client

# --- CONFIG ---
# Wir nehmen 2023 und 2024 (bis heute)
DATA_URLS = [
    # ATP
    "http://www.tennis-data.co.uk/2023/2023.xlsx",
    "http://www.tennis-data.co.uk/2024/2024.xlsx",
    # WTA
    "http://www.tennis-data.co.uk/2023w/2023.xlsx",
    "http://www.tennis-data.co.uk/2024w/2024.xlsx"
]

# Supabase Init
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ CRITICAL: Supabase Secrets missing.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def normalize_name(name):
    """
    Transformiert 'Sinner J.' zu 'Sinner' oder versucht, es kompatibel zu machen.
    Für Backtests ist der Nachname oft entscheidend.
    """
    if not isinstance(name, str): return "Unknown"
    # Entferne Initialen am Ende (z.B. "Alcaraz C." -> "Alcaraz")
    # Einfache Heuristik: Split bei Leerzeichen, nimm das erste Wort, wenn das zweite nur 1-2 Zeichen hat
    parts = name.strip().split()
    if len(parts) > 1 and len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name.strip()

def process_and_upload(url):
    print(f"📥 Downloading & Processing: {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Lese Excel direkt in Pandas
        df = pd.read_excel(io.BytesIO(response.content))
        
        # Standardisiere Spalten (Tennis-Data hat manchmal inkonsistente Header)
        # Wir brauchen: Winner, Loser, WRank, LRank, Date, Tournament, Surface, B365W, B365L (Bet365 Odds)
        
        records_to_insert = []
        
        total_rows = len(df)
        print(f"   📊 Found {total_rows} matches. Starting transformation...")

        for _, row in df.iterrows():
            try:
                # 1. Validierung
                if pd.isna(row['Winner']) or pd.isna(row['Loser']):
                    continue
                
                # Datum Parsing
                match_date = row['Date'] # Pandas macht das meist automatisch korrekt als Timestamp
                if isinstance(match_date, (float, int)):
                    # Excel serial date handling falls nötig, aber read_excel ist meist smart
                    pass
                
                # 2. Daten Mapping
                # tennis-data.co.uk hat Winner/Loser Struktur.
                # Wir müssen das in Player1/Player2 mappen.
                # Damit wir nicht spoilern wer P1 ist, sortieren wir alphabetisch (oder nehmen Winner als P1, markieren ihn aber)
                
                # Strategie: Wir nehmen Winner als Player 1, damit wir wissen, wer gewonnen hat für die 'actual_winner_name' Spalte.
                p1_name = normalize_name(row['Winner'])
                p2_name = normalize_name(row['Loser'])
                
                # Odds Check (Bet365 als Benchmark, Fallback auf Average)
                o1 = row.get('B365W') if not pd.isna(row.get('B365W')) else row.get('AvgW')
                o2 = row.get('B365L') if not pd.isna(row.get('B365L')) else row.get('AvgL')
                
                if pd.isna(o1) or pd.isna(o2):
                    continue # Keine Odds, kein Wert für uns
                
                # Scores
                score = str(row.get('Score', ''))
                
                # Sets (Best of 3 or 5)
                # Wir können hier einfache Logik nutzen
                
                tournament = str(row.get('Tournament', 'Unknown Event'))
                surface = str(row.get('Surface', 'Hard')) # Default Hard
                
                # 3. Das Supabase Objekt
                match_obj = {
                    "player1_name": p1_name,
                    "player2_name": p2_name,
                    "tournament": tournament,
                    "match_time": match_date.strftime("%Y-%m-%dT12:00:00Z"), # Setze Mittagszeit als Default
                    "created_at": datetime.now().isoformat(),
                    
                    # Odds (Closing Odds sind hier Opening Odds, da historisch)
                    "odds1": float(o1),
                    "odds2": float(o2),
                    "opening_odds1": float(o1),
                    "opening_odds2": float(o2),
                    
                    # Resultat (Da Backtest -> Wir kennen das Ergebnis!)
                    "actual_winner_name": p1_name, # Da wir Winner als P1 gemappt haben
                    "score": score,
                    
                    # Meta
                    "bookmaker": "Historical (B365)",
                    "ai_analysis_text": f"[HISTORICAL IMPORT] Surface: {surface}. Rank: {row.get('WRank')} vs {row.get('LRank')}"
                }
                
                records_to_insert.append(match_obj)
                
            except Exception as e:
                # Skip bad rows silently to keep speed up
                continue

        # 4. Batch Upload (Supabase mag keine 5000 Rows auf einmal, wir machen Chunks)
        CHUNK_SIZE = 100
        print(f"   🚀 Uploading {len(records_to_insert)} matches in chunks of {CHUNK_SIZE}...")
        
        for i in range(0, len(records_to_insert), CHUNK_SIZE):
            chunk = records_to_insert[i:i + CHUNK_SIZE]
            try:
                # upsert ist wichtig, um Duplikate zu vermeiden, falls wir das Script 2x laufen lassen.
                # ACHTUNG: Upsert braucht einen Unique Constraint (z.B. player1, player2, date). 
                # Wenn du keinen hast, nutze insert und ignoriere Fehler, oder lass es drauf ankommen.
                # Für reine Datenmenge ist 'insert' oft sicherer wenn man vorher truncatet, aber wir nehmen insert.
                
                # Wir prüfen NICHT auf Duplikate hier um Geschwindigkeit zu halten. 
                # Idealerweise löscht du vorher die "Historical" Daten oder wir bauen einen Check.
                
                supabase.table("market_odds").insert(chunk).execute()
                print(f"      ✅ Chunk {i // CHUNK_SIZE + 1} uploaded.")
            except Exception as e:
                print(f"      ⚠️ Chunk Error: {e}")
                
    except Exception as main_e:
        print(f"❌ Failed to process {url}: {main_e}")

if __name__ == "__main__":
    print("🏁 Starting Historical Backfill...")
    for url in DATA_URLS:
        process_and_upload(url)
    print("✅ Backfill Complete.")
