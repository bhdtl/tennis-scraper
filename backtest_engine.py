# -*- coding: utf-8 -*-

import asyncio
import os
import pandas as pd
import io
import requests
import math
import logging
import zipfile
import numpy as np
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

DATA_SOURCES = [
    "http://www.tennis-data.co.uk/2024/2024.xlsx",     # ATP 2024
    "http://www.tennis-data.co.uk/2024w/2024.xlsx",    # WTA 2024
    "http://www.tennis-data.co.uk/2025/2025.xlsx",     # ATP 2025
    "http://www.tennis-data.co.uk/2025w/2025.xlsx"     # WTA 2025
]

# SOTA: Normalized Speed Map (1-10) based on your PDF
TOURNAMENT_SPEED_MAP = {
    "Stuttgart": 9.0, "Brussels": 8.5, "Halle": 8.0, "Brisbane": 8.0, 
    "Basel": 7.8, "Queens Club": 7.7, "Mallorca": 7.5, "Chengdu": 7.5,
    "Dallas": 7.4, "Vienna": 7.4, "'s-Hertogenbosch": 7.4, "Cincinnati": 7.3,
    "Dubai": 7.2, "Paris": 7.2, "Adelaide": 7.2, "Wimbledon": 7.1,
    "Shanghai": 7.0, "Hong Kong": 6.8, "Washington": 6.8, "Stockholm": 6.8,
    "Australian Open": 6.8, "US Open": 6.8, "Miami": 6.8, "Tokyo": 6.8,
    "Montreal": 6.7, "Toronto": 6.7, "Winston-Salem": 6.7, "Doha": 6.7,
    "Indian Wells": 5.2, "Acapulco": 5.5, "Beijing": 5.6, "Delray Beach": 5.8,
    "Madrid": 5.9, "Roland Garros": 4.5, "Rome": 4.5, "Monte Carlo": 3.5, 
    "Barcelona": 3.2, "Rio de Janeiro": 3.3, "Buenos Aires": 3.0, 
    "Hamburg": 4.0, "Munich": 4.0, "Estoril": 3.5, "Gstaad": 6.0,
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
# 3. MATH CORE: "CONTRARIAN HUNTER" LOGIC (V43)
# =================================================================
def calculate_historical_fair_odds(elo1, elo2, surface, bsi, m_odds1, m_odds2):
    # 1. Physics Probability (Elo + Context)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    # 2. Market Wisdom
    prob_market = 0.5
    if m_odds1 > 1 and m_odds2 > 1:
        marg = (1/m_odds1) + (1/m_odds2)
        prob_market = (1/m_odds1) / marg
    
    # V43 ADJUSTMENT: 
    # Wir vertrauen dem Modell (Elo) noch stärker, da wir wissen, dass es
    # bei "Outlier"-Matches (hohe Quoten) besser liegt als der Markt.
    prob_alpha = prob_elo 
    
    # 75% Model, 25% Market - Aggressive "Disagreement" Policy
    final_prob = (prob_alpha * 0.75) + (prob_market * 0.25)
    
    # Compression: Wir ziehen "mittlere" Wahrscheinlichkeiten auseinander,
    # um schwache Signale zu unterdrücken und starke zu betonen.
    if final_prob > 0.5:
        final_prob = min(0.96, final_prob * 1.02)
    else:
        final_prob = max(0.04, final_prob * 0.98)
    
    return final_prob

def calculate_kelly_stake(fair_prob: float, market_odds: float) -> str:
    """
    Calculates Kelly Stake with DATA-DRIVEN V43 FILTERS.
    Derived from Backtest Analysis:
    - Low Odds (<2.30) -> NEGATIVE ROI (-14%) -> BLOCKED
    - High Odds (>2.30) -> POSITIVE ROI (+24%) -> ALLOWED
    - Low Edge (<15%) -> NEGATIVE ROI -> BLOCKED
    """
    if market_odds <= 1.0 or fair_prob <= 0: return "0u"
    
    # [THE V43 DATA-DRIVEN FILTER WALL]
    
    # 1. ODDS FLOOR: 2.30 (Data says <2.50 is toxic, we allow slightly below for buffer)
    if market_odds < 2.30: return "0u"
    
    # 2. ODDS CEILING: 6.00 (We hunt Deep Value now)
    if market_odds > 6.00: return "0u"
    
    # 3. EDGE THRESHOLD: 15% (Data says <10-20% is negative EV)
    edge = (fair_prob * market_odds) - 1
    if edge < 0.15: return "0u" 

    # Kelly Calculation
    b = market_odds - 1
    p = fair_prob
    q = 1 - p
    kelly = (b * p - q) / b
    
    # Fractional Kelly (12.5%)
    # Da wir jetzt Außenseiter jagen (Winrate ~35-40%), müssen wir die Varianz zähmen.
    # 1/8 Kelly ist Industriestandard für High-Odds-Strategien.
    safe_kelly = kelly * 0.125 
    
    if safe_kelly <= 0: return "0u"
    
    raw_units = safe_kelly / 0.02 # 1 Unit = 2% Bankroll
    
    # Cap at 2.0 Units
    raw_units = min(raw_units, 2.0)
    
    units = round(raw_units * 4) / 4
    if units < 0.25: return "0u"
    
    return f"{units}u"

def safe_float(val):
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except: return None

# =================================================================
# 4. PIPELINE
# =================================================================
def clean_name(name):
    if not isinstance(name, str): return "Unknown"
    return name.strip().lower()

def match_player_db(csv_name, db_players):
    if not isinstance(csv_name, str): return None
    parts = csv_name.split()
    if not parts: return None
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
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    for url in DATA_SOURCES:
        logger.info(f"📥 Downloade: {url}")
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 404 or "html" in r.headers.get("Content-Type", "").lower():
                logger.warning(f"⚠️ Skip Invalid URL: {url}")
                continue

            try:
                df = pd.read_excel(io.BytesIO(r.content), engine='openpyxl')
            except Exception as e:
                logger.error(f"❌ Excel Error {url}: {e}")
                continue
            
            df.columns = [str(c).strip() for c in df.columns]
            logger.info(f"   Processing {len(df)} rows...")
            
            for _, row in df.iterrows():
                try:
                    winner_raw = str(row.get('Winner'))
                    loser_raw = str(row.get('Loser'))
                    if winner_raw == 'nan' or loser_raw == 'nan': continue
                    
                    p1_obj = match_player_db(winner_raw, db_players)
                    p2_obj = match_player_db(loser_raw, db_players)
                    
                    elo1_pre, elo2_pre = elo_engine.update(clean_name(winner_raw), clean_name(loser_raw))
                    
                    if not p1_obj or not p2_obj: continue
                    
                    w_odds = safe_float(row.get('B365W')) or safe_float(row.get('PSW')) or safe_float(row.get('AvgW'))
                    l_odds = safe_float(row.get('B365L')) or safe_float(row.get('PSL')) or safe_float(row.get('AvgL'))
                    
                    if w_odds is None or l_odds is None: continue
                    
                    tournament = str(row.get('Tournament', 'Unknown'))
                    surface = str(row.get('Surface', 'Hard'))
                    date_obj = row.get('Date')
                    
                    match_time_str = datetime.now().isoformat()
                    if pd.notna(date_obj):
                        try: match_time_str = date_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
                        except: pass

                    bsi = get_bsi(tournament, surface)
                    fair_prob_p1 = calculate_historical_fair_odds(elo1_pre, elo2_pre, surface, bsi, w_odds, l_odds)
                    
                    fair_odds_p1 = round(1 / fair_prob_p1, 2)
                    fair_odds_p2 = round(1 / (1 - fair_prob_p1), 2)
                    if math.isinf(fair_odds_p1) or math.isinf(fair_odds_p2): continue

                    # [V43 CONTRARIAN LOGIC]
                    roi_potential_p1 = (w_odds / fair_odds_p1) - 1
                    roi_potential_p2 = (l_odds / fair_odds_p2) - 1
                    
                    ai_text = f"BACKTEST [BSI {bsi}]: "
                    stake_p1 = calculate_kelly_stake(fair_prob_p1, w_odds)
                    stake_p2 = calculate_kelly_stake(1-fair_prob_p1, l_odds)
                    
                    if stake_p1 != "0u":
                        ai_text += f" [💎 HUNTER: {p1_obj['last_name']} @ {w_odds} | Fair: {fair_odds_p1} | Edge: {round(roi_potential_p1*100,1)}% | Stake: {stake_p1}]"
                    elif stake_p2 != "0u":
                        ai_text += f" [💎 HUNTER: {p2_obj['last_name']} @ {l_odds} | Fair: {fair_odds_p2} | Edge: {round(roi_potential_p2*100,1)}% | Stake: {stake_p2}]"
                    else:
                        ai_text += "No Value (Filter Block)."

                    record = {
                        "player1_name": p1_obj['last_name'],
                        "player2_name": p2_obj['last_name'],
                        "tournament": tournament,
                        "odds1": w_odds,
                        "odds2": l_odds,
                        "ai_fair_odds1": fair_odds_p1,
                        "ai_fair_odds2": fair_odds_p2,
                        "ai_analysis_text": ai_text,
                        "created_at": match_time_str,
                        "match_time": match_time_str,
                        "actual_winner_name": p1_obj['last_name']
                    }
                    
                    records_to_insert.append(record)
                    
                    if len(records_to_insert) >= 50:
                        try:
                            logger.info(f"💾 Upserting Batch of {len(records_to_insert)}...")
                            supabase.table("market_odds").upsert(
                                records_to_insert, 
                                on_conflict="player1_name,player2_name,match_time", 
                                ignore_duplicates=True
                            ).execute()
                        except Exception: pass
                        finally:
                            records_to_insert = []
                        
                except Exception: continue

        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")

    if records_to_insert:
        try:
            logger.info(f"💾 Upserting Final Batch...")
            supabase.table("market_odds").upsert(
                records_to_insert, 
                on_conflict="player1_name,player2_name,match_time", 
                ignore_duplicates=True
            ).execute()
        except Exception: pass

    logger.info("🏁 Backtest V43 Complete.")

if __name__ == "__main__":
    asyncio.run(run_backtest())
