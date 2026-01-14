# -*- coding: utf-8 -*-

import asyncio
import os
import pandas as pd
import io
import requests
import math
import logging
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

# URLs from tennis-data.co.uk
DATA_SOURCES = [
    # 2024
    "http://www.tennis-data.co.uk/2024/2024.xlsx",     # ATP 2024
    "http://www.tennis-data.co.uk/2024w/2024f.xlsx",   # WTA 2024
    # 2025 (YTD)
    "http://www.tennis-data.co.uk/2025/2025.xlsx",     # ATP 2025
    "http://www.tennis-data.co.uk/2025w/2025f.xlsx"    # WTA 2025
]

# --- PDF DATA: BSI SPEED MAPPING (Normalized to 1-10 Scale) ---
# [span_0](start_span)[span_1](start_span)Source: Your PDF[span_0](end_span)[span_1](end_span). Logic: (Percent - 60) / 2 roughly maps 60-80% to 0-10.
# Stuttgart (77.9%) -> ~9.0 (Fast)
# Indian Wells (70.3%) -> ~5.0 (Medium)
# Monte Carlo (67.1%) -> ~3.5 (Slow)
TOURNAMENT_SPEED_MAP = {
    # Fast / Grass / Indoor
    "Stuttgart": 9.0, "Brussels": 8.5, "Halle": 8.0, "Brisbane": 8.0, 
    "Basel": 7.8, "Queens Club": 7.7, "Mallorca": 7.5, "Chengdu": 7.5,
    "Dallas": 7.4, "Vienna": 7.4, "'s-Hertogenbosch": 7.4, "Cincinnati": 7.3,
    "Dubai": 7.2, "Paris": 7.2, "Adelaide": 7.2, "Wimbledon": 7.1,
    "Shanghai": 7.0, "Hong Kong": 6.8, "Washington": 6.8, "Stockholm": 6.8,
    "Australian Open": 6.8, "US Open": 6.8, "Miami": 6.8, "Tokyo": 6.8,
    "Montreal": 6.7, "Toronto": 6.7, "Winston-Salem": 6.7, "Doha": 6.7,
    
    # Medium / Clay-ish Hard
    "Indian Wells": 5.2, "Acapulco": 5.5, "Beijing": 5.6, "Delray Beach": 5.8,
    "Madrid": 5.9, # Fast Clay due to altitude
    
    # Slow / Clay
    "Roland Garros": 4.5, "Rome": 4.5, "Monte Carlo": 3.5, "Barcelona": 3.2,
    "Rio de Janeiro": 3.3, "Buenos Aires": 3.0, "Hamburg": 4.0, "Munich": 4.0,
    "Estoril": 3.5, "Gstaad": 6.0, # High Altitude Clay
    "Kitzbuhel": 5.0, "Umag": 3.5, "Bastad": 3.5, "Marrakech": 4.5
}

# Default Surface Speeds if Tournament not in list
SURFACE_DEFAULTS = {
    "Hard": 6.5, "Clay": 3.5, "Grass": 8.5, "Carpet": 9.0
}

# =================================================================
# 2. INTERNAL ELO ENGINE (THE TIME MACHINE)
# =================================================================
class TimeMachineElo:
    def __init__(self):
        self.ratings = {} # {player_name: 1500}
        self.k_factor = 32

    def get_elo(self, player):
        return self.ratings.get(player, 1500)

    def update(self, winner, loser):
        r_w = self.get_elo(winner)
        r_l = self.get_elo(loser)
        
        expected_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        
        # Calculate new ratings
        new_w = r_w + self.k_factor * (1 - expected_w)
        new_l = r_l + self.k_factor * (0 - expected_w)
        
        self.ratings[winner] = new_w
        self.ratings[loser] = new_l
        return r_w, r_l # Return PRE-MATCH Elo for the record

elo_engine = TimeMachineElo()

# =================================================================
# 3. CORE CALCULATION LOGIC (IMPORTED FROM SCANNER)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_kelly_stake(fair_prob: float, market_odds: float) -> float:
    if market_odds <= 1.0 or fair_prob <= 0: return 0.0
    b = market_odds - 1
    p = fair_prob
    q = 1 - p
    kelly = (b * p - q) / b
    safe_kelly = kelly * 0.25 # Quarter Kelly
    if safe_kelly <= 0: return 0.0
    raw_units = safe_kelly / 0.02 # 1 Unit = 2%
    
    # Logic from scanner
    if market_odds > 4.0: raw_units = min(raw_units, 0.5)
    elif market_odds > 2.5: raw_units = min(raw_units, 1.25)
    elif market_odds < 2.0: raw_units = min(raw_units, 3.0)
    else: raw_units = min(raw_units, 2.0)
    
    return round(raw_units * 4) / 4

def calculate_historical_fair_odds(elo1, elo2, surface, bsi, m_odds1, m_odds2):
    # Simplified Physics Engine for Backtesting (Without live Gemini calls)
    
    # 1. Elo Probability
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    # 2. BSI/Surface Modifier (Mocking skills)
    # We assume higher ELO players adapt better, but we add randomness or skill-mock
    # In a real backtest, we would fetch player_skills from DB. 
    # For speed, we assume Elo encapsulates skill.
    
    prob_alpha = prob_elo 
    
    # 3. Market Wisdom (The "Wisdom of Crowds" weight)
    # In the live scanner, we use 0.30 weight for market.
    prob_market = 0.5
    if m_odds1 > 1 and m_odds2 > 1:
        marg = (1/m_odds1) + (1/m_odds2)
        prob_market = (1/m_odds1) / marg
        
    final_prob = (prob_alpha * 0.70) + (prob_market * 0.30)
    
    # Compression (same as scanner)
    if final_prob > 0.60: final_prob = min(final_prob * 1.05, 0.94)
    elif final_prob < 0.40: final_prob = max(final_prob * 0.95, 0.06)
    
    return final_prob

# =================================================================
# 4. PROCESSING PIPELINE
# =================================================================
def clean_name(name):
    if not isinstance(name, str): return "Unknown"
    # Convert "Nadal R." to "Rafael Nadal" style matching is hard with just "Nadal R."
    # Tennis-data.co.uk uses "Winner", "Loser" columns usually just Last Name or "Last Name F."
    return name.strip().lower()

def match_player_db(csv_name, db_players):
    # csv_name is often "Sinner J." or "Alcaraz C."
    parts = csv_name.split()
    last_name = parts[0].lower() # Usually "Sinner"
    
    candidates = [p for p in db_players if p['last_name'].lower() == last_name]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Check initial if available
        if len(parts) > 1:
            initial = parts[1][0].lower()
            for cand in candidates:
                if cand['first_name'].lower().startswith(initial):
                    return cand
    return None

def get_bsi(tournament_name, surface):
    # Try direct map
    for k, v in TOURNAMENT_SPEED_MAP.items():
        if k.lower() in tournament_name.lower():
            return v
    # Fallback to surface
    return SURFACE_DEFAULTS.get(surface, 5.0)

async def run_backtest():
    logger.info("⏳ Lade Player Database...")
    db_res = supabase.table("players").select("id, first_name, last_name").execute()
    db_players = db_res.data
    logger.info(f"✅ {len(db_players)} Spieler in DB geladen.")

    records_to_insert = []
    
    for url in DATA_SOURCES:
        logger.info(f"📥 Downloade: {url}")
        try:
            r = requests.get(url)
            r.raise_for_status()
            
            # Read Excel
            df = pd.read_excel(io.BytesIO(r.content))
            
            # Normalize Columns (Winner, Loser, Date, Tournament, Surface, B365W, B365L)
            # Standardizing
            df.columns = [c.strip() for c in df.columns]
            
            logger.info(f"   Processing {len(df)} rows...")
            
            for _, row in df.iterrows():
                try:
                    # Required fields
                    if pd.isna(row['Winner']) or pd.isna(row['Loser']): continue
                    
                    winner_raw = str(row['Winner'])
                    loser_raw = str(row['Loser'])
                    
                    # Match with DB
                    p1_obj = match_player_db(winner_raw, db_players) # Winner becomes P1 for simulation? Or random?
                    p2_obj = match_player_db(loser_raw, db_players)
                    
                    # Update ELO regardless of DB match (to keep ecosystem accurate)
                    # We normalize name for ELO engine strictly by raw string to persist state
                    elo1_pre, elo2_pre = elo_engine.update(clean_name(winner_raw), clean_name(loser_raw))
                    
                    if not p1_obj or not p2_obj:
                        continue # Skip if players not in our SaaS DB
                    
                    # Data Prep
                    tournament = str(row.get('Tournament', 'Unknown'))
                    surface = str(row.get('Surface', 'Hard'))
                    date_obj = row.get('Date')
                    
                    # Odds (Prefer Bet365, fallbacks exist)
                    w_odds = row.get('B365W') or row.get('PSW') or row.get('AvgW')
                    l_odds = row.get('B365L') or row.get('PSL') or row.get('AvgL')
                    
                    if pd.isna(w_odds) or pd.isna(l_odds): continue
                    
                    bsi = get_bsi(tournament, surface)
                    
                    # SIMULATION: 
                    # We treat this as P1 (Winner) vs P2 (Loser). 
                    # Since we know P1 won, 'actual_winner_name' = p1_obj['last_name']
                    
                    # Calculate Fair Odds
                    # Note: We pass elo1_pre (Winner's ELO before match) and elo2_pre
                    fair_prob_p1 = calculate_historical_fair_odds(elo1_pre, elo2_pre, surface, bsi, w_odds, l_odds)
                    fair_odds_p1 = round(1 / fair_prob_p1, 2)
                    fair_odds_p2 = round(1 / (1 - fair_prob_p1), 2)
                    
                    # Calculate Units
                    # Did the AI find value on the Winner?
                    units = 0
                    pick_name = ""
                    
                    edge_p1 = (1/fair_odds_p1) - (1/w_odds)
                    edge_p2 = (1/fair_odds_p2) - (1/l_odds)
                    
                    ai_text = f"BACKTEST [BSI {bsi}]: "
                    
                    if w_odds > fair_odds_p1 and edge_p1 > -0.05:
                        units = calculate_kelly_stake(fair_prob_p1, w_odds)
                        pick_name = p1_obj['last_name']
                        ai_text += f"Value on Winner ({p1_obj['last_name']}). Elo {int(elo1_pre)} vs {int(elo2_pre)}."
                    elif l_odds > fair_odds_p2 and edge_p2 > -0.05:
                        units = calculate_kelly_stake(1-fair_prob_p1, l_odds)
                        pick_name = p2_obj['last_name']
                        ai_text += f"Value on Loser ({p2_obj['last_name']}). Elo {int(elo1_pre)} vs {int(elo2_pre)}."
                    else:
                        ai_text += "No Value found."

                    # Prepare DB Record
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
                        "actual_winner_name": p1_obj['last_name'] # Since P1 is always Winner column in CSV
                    }
                    
                    records_to_insert.append(record)
                    
                    if len(records_to_insert) >= 100:
                        logger.info(f"💾 Inserting Batch of {len(records_to_insert)}...")
                        supabase.table("market_odds").insert(records_to_insert).execute()
                        records_to_insert = []
                        
                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")

    # Final Batch
    if records_to_insert:
        logger.info(f"💾 Inserting Final Batch of {len(records_to_insert)}...")
        supabase.table("market_odds").insert(records_to_insert).execute()

    logger.info("🏁 Backtest Complete.")

if __name__ == "__main__":
    asyncio.run(run_backtest())
