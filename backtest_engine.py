# -*- coding: utf-8 -*-

import asyncio
import os
import pandas as pd
import io
import requests
import math
import logging
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional
from supabase import create_client, Client

# =================================================================
# 1. CONFIG & SETUP
# =================================================================
logging.basicConfig(level=logging.INFO, format='[BACKTEST] %(message)s')
logger = logging.getLogger("NeuralScout_Backtest")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("❌ Secrets fehlen! SUPABASE_URL/KEY benötigt.")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# UPDATED SOURCES (User Verified)
DATA_SOURCES = [
    "http://www.tennis-data.co.uk/2024/2024.xlsx",     # ATP 2024
    "http://www.tennis-data.co.uk/2024w/2024.xlsx",    # WTA 2024
    "http://www.tennis-data.co.uk/2025/2025.xlsx",     # ATP 2025
    "http://www.tennis-data.co.uk/2025w/2025.xlsx"     # WTA 2025
]

TOURNAMENT_SPEED_MAP = {
    "Stuttgart": 9.0, "Brussels": 8.5, "Halle": 8.0, "Brisbane": 8.0, 
    "Basel": 7.8, "Queens Club": 7.7, "Mallorca": 7.5, "Chengdu": 7.5,
    "Dallas": 7.4, "Vienna": 7.4, "'s-Hertogenbosch": 7.4, "Cincinnati": 7.3,
    "Dubai": 7.2, "Paris": 7.2, "Adelaide": 7.2, "Wimbledon": 7.1,
    "Shanghai": 7.0, "Hong Kong": 6.8, "Washington": 6.8, "Stockholm": 6.8,
    "Australian Open": 6.8, "US Open": 6.8, "Miami": 6.8, "Tokyo": 6.8,
    "Montreal": 6.7, "Toronto": 6.7, "Winston-Salem": 6.7, "Doha": 6.7,
    "Indian Wells": 5.2, "Acapulco": 5.5, "Beijing": 5.6, "Delray Beach": 5.8,
    "Madrid": 5.9,
    "Roland Garros": 4.5, "Rome": 4.5, "Monte Carlo": 3.5, "Barcelona": 3.2,
    "Rio de Janeiro": 3.3, "Buenos Aires": 3.0, "Hamburg": 4.0, "Munich": 4.0,
    "Estoril": 3.5, "Gstaad": 6.0,
    "Kitzbuhel": 5.0, "Umag": 3.5, "Bastad": 3.5, "Marrakech": 4.5
}

SURFACE_DEFAULTS = {
    "Hard": 6.5, "Clay": 3.5, "Grass": 8.5, "Carpet": 9.0
}

# =================================================================
# 2. INTERNAL ELO ENGINE
# =================================================================
class TimeMachineElo:
    def __init__(self):
        self.ratings = {} 
        self.k_factor = 32

    def get_elo(self, player):
        return self.ratings.get(player, 1500)

    def update(self, winner, loser):
        r_w = self.get_elo(winner)
        r_l = self.get_elo(loser)
        expected_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        new_w = r_w + self.k_factor * (1 - expected_w)
        new_l = r_l + self.k_factor * (0 - expected_w)
        self.ratings[winner] = new_w
        self.ratings[loser] = new_l
        return r_w, r_l

elo_engine = TimeMachineElo()

# =================================================================
# 3. MATH CORE
# =================================================================
def calculate_kelly_stake(fair_prob: float, market_odds: float) -> float:
    if market_odds <= 1.0 or fair_prob <= 0: return 0.0
    b = market_odds - 1
    p = fair_prob
    q = 1 - p
    kelly = (b * p - q) / b
    safe_kelly = kelly * 0.25 
    if safe_kelly <= 0: return 0.0
    raw_units = safe_kelly / 0.02 
    if market_odds > 4.0: raw_units = min(raw_units, 0.5)
    elif market_odds > 2.5: raw_units = min(raw_units, 1.25)
    elif market_odds < 2.0: raw_units = min(raw_units, 3.0)
    else: raw_units = min(raw_units, 2.0)
    return round(raw_units * 4) / 4

def calculate_historical_fair_odds(elo1, elo2, surface, bsi, m_odds1, m_odds2):
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    prob_alpha = prob_elo 
    prob_market = 0.5
    if m_odds1 > 1 and m_odds2 > 1:
        marg = (1/m_odds1) + (1/m_odds2)
        prob_market = (1/m_odds1) / marg
    final_prob = (prob_alpha * 0.70) + (prob_market * 0.30)
    if final_prob > 0.60: final_prob = min(final_prob * 1.05, 0.94)
    elif final_prob < 0.40: final_prob = max(final_prob * 0.95, 0.06)
    return final_prob

# =================================================================
# 4. PROCESSING PIPELINE
# =================================================================
def clean_name(name):
    if not isinstance(name, str): return "Unknown"
    return name.strip().lower()

def match_player_db(csv_name, db_players):
    if not isinstance(csv_name, str): return None
    parts = csv_name.split()
    last_name = parts[0].lower()
    candidates = [p for p in db_players if p['last_name'].lower() == last_name]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        if len(parts) > 1:
            initial = parts[1][0].lower()
            for cand in candidates:
                if cand['first_name'].lower().startswith(initial):
                    return cand
    return None

def get_bsi(tournament_name, surface):
    for k, v in TOURNAMENT_SPEED_MAP.items():
        if k.lower() in str(tournament_name).lower():
            return v
    return SURFACE_DEFAULTS.get(surface, 5.0)

async def run_backtest():
    logger.info("⏳ Lade Player Database...")
    db_res = supabase.table("players").select("id, first_name, last_name").execute()
    db_players = db_res.data
    logger.info(f"✅ {len(db_players)} Spieler in DB geladen.")

    records_to_insert = []
    
    # Headers to mimic browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    for url in DATA_SOURCES:
        logger.info(f"📥 Downloade: {url}")
        try:
            r = requests.get(url, headers=headers)
            
            if r.status_code == 404:
                logger.warning(f"⚠️ Datei nicht gefunden (404): {url} - Skipping.")
                continue
            
            if "html" in r.headers.get("Content-Type", "").lower():
                logger.warning(f"⚠️ Warnung: Server lieferte HTML statt Excel für {url}. Skipping.")
                continue

            try:
                # Read Excel
                df = pd.read_excel(io.BytesIO(r.content), engine='openpyxl')
            except zipfile.BadZipFile:
                logger.error(f"❌ Corrupt File (BadZipFile) von {url}. Skipping.")
                continue
            except Exception as e:
                logger.error(f"❌ Excel Parsing Error bei {url}: {e}")
                continue
            
            # Normalize Columns
            df.columns = [str(c).strip() for c in df.columns]
            
            logger.info(f"   Processing {len(df)} rows...")
            
            for _, row in df.iterrows():
                try:
                    if pd.isna(row.get('Winner')) or pd.isna(row.get('Loser')): continue
                    
                    winner_raw = str(row['Winner'])
                    loser_raw = str(row['Loser'])
                    
                    p1_obj = match_player_db(winner_raw, db_players)
                    p2_obj = match_player_db(loser_raw, db_players)
                    
                    # Update Time Machine ELO regardless of DB match
                    elo1_pre, elo2_pre = elo_engine.update(clean_name(winner_raw), clean_name(loser_raw))
                    
                    if not p1_obj or not p2_obj: continue
                    
                    tournament = str(row.get('Tournament', 'Unknown'))
                    surface = str(row.get('Surface', 'Hard'))
                    date_obj = row.get('Date')
                    
                    # Robust Odds Fetching
                    w_odds = row.get('B365W') or row.get('PSW') or row.get('AvgW')
                    l_odds = row.get('B365L') or row.get('PSL') or row.get('AvgL')
                    
                    try:
                        w_odds = float(w_odds)
                        l_odds = float(l_odds)
                    except: continue 
                    
                    bsi = get_bsi(tournament, surface)
                    
                    fair_prob_p1 = calculate_historical_fair_odds(elo1_pre, elo2_pre, surface, bsi, w_odds, l_odds)
                    fair_odds_p1 = round(1 / fair_prob_p1, 2)
                    fair_odds_p2 = round(1 / (1 - fair_prob_p1), 2)
                    
                    edge_p1 = (1/fair_odds_p1) - (1/w_odds)
                    edge_p2 = (1/fair_odds_p2) - (1/l_odds)
                    
                    ai_text = f"BACKTEST [BSI {bsi}]: "
                    
                    if w_odds > fair_odds_p1 and edge_p1 > -0.05:
                        ai_text += f"Value on Winner ({p1_obj['last_name']}). Elo {int(elo1_pre)} vs {int(elo2_pre)}."
                    elif l_odds > fair_odds_p2 and edge_p2 > -0.05:
                        ai_text += f"Value on Loser ({p2_obj['last_name']}). Elo {int(elo1_pre)} vs {int(elo2_pre)}."
                    else:
                        ai_text += "No Value found."

                    record = {
                        "player1_name": p1_obj['last_name'],
                        "player2_name": p2_obj['last_name'],
                        "tournament": tournament,
                        "odds1": w_odds,
                        "odds2": l_odds,
                        "ai_fair_odds1": fair_odds_p1,
                        "ai_fair_odds2": fair_odds_p2,
                        "ai_analysis_text": ai_text,
                        "created_at": date_obj.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(date_obj) else datetime.now().isoformat(),
                        "match_time": date_obj.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(date_obj) else datetime.now().isoformat(),
                        "actual_winner_name": p1_obj['last_name']
                    }
                    
                    records_to_insert.append(record)
                    
                    if len(records_to_insert) >= 100:
                        logger.info(f"💾 Upserting Batch of {len(records_to_insert)}...")
                        # SOTA FIX: Using UPSERT instead of INSERT to handle duplicates (409 Conflict)
                        # We try to match on player names + match_time if a unique constraint exists, 
                        # otherwise upsert might fail if no Primary Key is provided. 
                        # Ideally 'on_conflict' columns should be specified if they form a unique constraint.
                        # Assuming 'player1_name, player2_name, match_time' might be unique?
                        # Safer fallback: Just try upsert without params if table has PK, or ignore_duplicates.
                        supabase.table("market_odds").upsert(records_to_insert, on_conflict="player1_name,player2_name,match_time", ignore_duplicates=True).execute()
                        records_to_insert = []
                        
                except Exception as e:
                    # logger.warning(f"Skipped row: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")

    if records_to_insert:
        logger.info(f"💾 Upserting Final Batch of {len(records_to_insert)}...")
        supabase.table("market_odds").upsert(records_to_insert, on_conflict="player1_name,player2_name,match_time", ignore_duplicates=True).execute()

    logger.info("🏁 Backtest Complete.")

if __name__ == "__main__":
    asyncio.run(run_backtest())
