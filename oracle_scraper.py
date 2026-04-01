# -*- coding: utf-8 -*-

import asyncio
import os
import re
import random
import logging
import sys
import unicodedata
import httpx
import math
import difflib
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Set

from playwright.async_api import async_playwright, Browser
from bs4 import BeautifulSoup
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Oracle_PreWarmer_SOTA")

def log(msg: str):
    logger.info(msg)

log("🔮 Initializing Oracle Pre-Warmer (V155.1 SOTA Parity - High Variance Quant Synergy)...")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
FUNCTION_URL = os.environ.get("SUPABASE_FUNCTION_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") 
API_TENNIS_KEY = os.environ.get("API_TENNIS_KEY") # 🚀 Externe API für historische Ground Truth

if not SUPABASE_URL or not SUPABASE_KEY or not FUNCTION_URL:
    log("❌ CRITICAL: Secrets missing (Need URL, KEY and FUNCTION_URL)!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. HELPERS & SOTA PLAYER MATCHING (TE FORMAT)
# =================================================================
def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    if not raw: return ""
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    clean = re.sub(r'\s*\(\d+\)', '', clean) 
    clean = re.sub(r'\s*\(.*?\)', '', clean) 
    return clean.replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = raw
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<.*?>', '', clean)
    clean = re.sub(r'S\d+.*$', '', clean) 
    clean = re.sub(r'H2H.*$', '', clean)
    clean = re.sub(r'\b(Challenger|Men|Women|Singles|Doubles)\b', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s\d+$', '', clean)
    return clean.strip()

def normalize_db_name(name: str) -> str:
    if not name: return ""
    n = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = n.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def calculate_fuzzy_score(scraped_name: str, db_name: str) -> int:
    s_norm = normalize_text(scraped_name).lower()
    d_norm = normalize_text(db_name).lower()
    if d_norm in s_norm and len(d_norm) > 3: return 100
    s_tokens = set(re.findall(r'\w+', s_norm))
    d_tokens = set(re.findall(r'\w+', d_norm))
    stop_words = {'atp', 'wta', 'open', 'tour', '2025', '2026', 'challenger'}
    s_tokens -= stop_words; d_tokens -= stop_words
    if not s_tokens or not d_tokens: return 0
    common = s_tokens.intersection(d_tokens)
    score = len(common) * 10
    if "indoor" in s_tokens and "indoor" in d_tokens: score += 20
    return score

# -----------------------------------------------------------------
# THE SMART BROTHER-RESOLUTION ENGINE 
# -----------------------------------------------------------------
def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str]) -> Optional[Dict]:
    if not scraped_name_raw or not db_players: return None
    clean_scrape = clean_player_name(scraped_name_raw)
    parts = clean_scrape.split()
    scrape_last = ""
    scrape_initial = ""
    
    if len(parts) >= 2:
        last_token = parts[-1].replace('.', '')
        if len(last_token) == 1 and last_token.isalpha():
            scrape_initial = last_token.lower()
            scrape_last = " ".join(parts[:-1]) 
        else:
            scrape_last = parts[-1]
            scrape_initial = parts[0][0].lower() if parts[0] else ""
    else:
        scrape_last = clean_scrape

    target_last = normalize_db_name(scrape_last)
    candidates = []
    
    for p in db_players:
        db_last_raw = p.get('last_name', '')
        db_last = normalize_db_name(db_last_raw)
        match_score = 0
        
        if db_last == target_last: match_score = 100
        elif target_last in db_last or db_last in target_last: 
            if len(target_last) > 3 and len(db_last) > 3: match_score = 80
            
        if match_score > 0:
            db_first = p.get('first_name', '').lower()
            if scrape_initial and db_first:
                if db_first.startswith(scrape_initial): 
                    match_score += 20 
                else: 
                    match_score -= 50  # FATAL PENALTY FOR WRONG INITIAL
            if match_score > 50: 
                candidates.append((p, match_score))
                
    if not candidates: return None
    
    candidates.sort(key=lambda x: (x[1], x[0]['id'] in report_ids), reverse=True)
    return candidates[0][0]

def parse_draws_locally(html):
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        pending_p1_raw = None
        
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                pending_p1_raw = None
                i += 1
                continue
            
            cols = row.find_all('td')
            if len(cols) < 2: 
                i += 1
                continue
            
            first_cell = row.find('td', class_='first')
            p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
            if not p_cell: 
                i += 1
                continue
                
            p_raw = clean_player_name(p_cell.get_text(strip=True))
            
            if pending_p1_raw:
                p2_raw = p_raw
                if '/' in pending_p1_raw or '/' in p2_raw: 
                    pending_p1_raw = None
                    i += 1
                    continue
                
                row_text = row.get_text(strip=True).lower()
                prev_row_text = rows[i-1].get_text(strip=True).lower()
                is_finished = False
                
                if 'ret' in row_text or 'w.o' in row_text or 'ret' in prev_row_text or 'w.o' in prev_row_text:
                    is_finished = True
                
                p_idx = cols.index(p_cell)
                if p_idx + 1 < len(cols):
                    if cols[p_idx+1].get_text(strip=True).isdigit():
                        is_finished = True

                if not is_finished:
                    found.append({
                        "tour": clean_tournament_name(current_tour),
                        "p1_raw": pending_p1_raw,
                        "p2_raw": p2_raw
                    })
                
                pending_p1_raw = None
            else:
                if first_cell and first_cell.get('rowspan') == '2': 
                    pending_p1_raw = p_raw
                else: 
                    pending_p1_raw = p_raw
            i += 1
            
    return found

def extract_rating(rating_obj: Any, default: float = 5.0) -> float:
    if not rating_obj: return default
    if isinstance(rating_obj, (int, float)): return float(rating_obj)
    if isinstance(rating_obj, dict) and 'score' in rating_obj: return float(rating_obj['score'])
    return default

# 🚀 BSI Bucket Resolution
def get_bsi_bucket(surface: str, bsi: float) -> str:
    surf = str(surface).lower()
    b = float(bsi) if bsi else 5.0
    if 'indoor' in surf or 'carpet' in surf: return 'indoor'
    if 'grass' in surf: return 'grass'
    if 'clay' in surf or 'sand' in surf:
        return 'fast_clay' if b > 4.0 else 'slow_clay'
    if b > 7.0: return 'fast_hard'
    if b < 5.0: return 'slow_hard'
    return 'medium_hard'


# =================================================================
# 🚀 API-TENNIS.COM AUTO-DISCOVERY ENGINE
# =================================================================
async def build_api_tennis_player_map() -> Dict[str, str]:
    if not API_TENNIS_KEY:
        log("⚠️ No API_TENNIS_KEY found. Skipping auto-discovery.")
        return {}
        
    log("🌍 Auto-Discovering Player Keys via API-Tennis Standings...")
    player_map = {}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for tour in ['ATP', 'WTA']:
            try:
                url = f"https://api.api-tennis.com/tennis/?method=get_standings&event_type={tour}&APIkey={API_TENNIS_KEY}"
                res = await client.get(url)
                data = res.json()
                
                if data.get("success") == 1 and data.get("result"):
                    for p in data["result"]:
                        name_raw = p.get("player", "")
                        p_key = p.get("player_key")
                        if name_raw and p_key:
                            last_name = name_raw.split(' ')[-1].lower().strip()
                            player_map[last_name] = str(p_key)
            except Exception as e:
                log(f"    ⚠️ Failed to fetch {tour} standings for ID mapping: {e}")
                
    log(f"✅ Discovered {len(player_map)} player keys from global standings.")
    return player_map


async def fetch_macro_surface_winrate(player_key: str, last_name: str, surface: str) -> str:
    target_surf = surface.lower()
    is_clay = 'clay' in target_surf or 'sand' in target_surf
    is_grass = 'grass' in target_surf
    
    if API_TENNIS_KEY and player_key:
        try:
            url = f"https://api.api-tennis.com/tennis/?method=get_players&player_key={player_key}&APIkey={API_TENNIS_KEY}"
            async with httpx.AsyncClient() as client:
                res = await client.get(url, timeout=10.0)
                data = res.json()
                
                if data.get("success") == 1 and data.get("result"):
                    stats = data["result"][0].get("stats", [])
                    surf_won, surf_lost = 0, 0
                    
                    key_won = 'clay_won' if is_clay else 'grass_won' if is_grass else 'hard_won'
                    key_lost = 'clay_lost' if is_clay else 'grass_lost' if is_grass else 'hard_lost'

                    for s in stats:
                        if s.get("type") == "singles": 
                            try:
                                w = s.get(key_won)
                                l = s.get(key_lost)
                                if w: surf_won += int(w)
                                if l: surf_lost += int(l)
                            except: pass
                    
                    total = surf_won + surf_lost
                    if total >= 10: 
                        win_rate = (surf_won / total) * 100
                        return f"{win_rate:.1f}% ({surf_won}W - {surf_lost}L via Global API)"
        except Exception as e:
            log(f"    ⚠️ API-Tennis fetch failed for {last_name}: {e}")

    try:
        res = supabase.table('market_odds').select('player1_name, player2_name, actual_winner_name, tournament, ai_analysis_text').or_(f"player1_name.ilike.%{last_name}%,player2_name.ilike.%{last_name}%").neq('actual_winner_name', 'None').limit(300).execute()
        matches = res.data or []
        
        wins, losses = 0, 0
        for m in matches:
            m_surf_text = f"{m.get('tournament', '')} {m.get('ai_analysis_text', '')}".lower()
            m_is_clay = 'clay' in m_surf_text or 'roland' in m_surf_text
            m_is_grass = 'grass' in m_surf_text or 'wimbledon' in m_surf_text
            
            match_fits = False
            if is_clay and m_is_clay: match_fits = True
            elif is_grass and m_is_grass: match_fits = True
            elif not is_clay and not is_grass and not m_is_clay and not m_is_grass: match_fits = True
            
            if match_fits:
                is_win = last_name.lower() in str(m.get('actual_winner_name', '')).lower()
                if is_win: wins += 1
                else: losses += 1
                
        total = wins + losses
        if total > 0:
            win_rate = (wins / total) * 100
            return f"{win_rate:.1f}% ({wins}W - {losses}L via Database)"
    except Exception as e:
        log(f"    ⚠️ DB Fallback calc failed: {e}")
        
    return "Unknown (Insufficient Data)"

# =================================================================
# 3. SOTA MARKOV CHAIN ENGINE (Local Physics)
# =================================================================
class MarkovChainEngine:
    @staticmethod
    def run_simulation(s1: Dict, s2: Dict, formA: float, formB: float, 
                       bsi: float, styleA: str, styleB: str, 
                       iterations: int = 1500) -> Dict[str, Any]:
        
        def get_serve_prob(serve_skill, power_skill):
            return 0.50 + (((serve_skill * 0.7) + (power_skill * 0.3)) / 100.0) * 0.25
            
        def get_return_def(speed_skill, backhand_skill, forehand_skill):
            return (((speed_skill * 0.4) + (backhand_skill * 0.3) + (forehand_skill * 0.3)) / 100.0)

        base_serve_win_A = get_serve_prob(s1.get('serve', 50), s1.get('power', 50))
        base_serve_win_B = get_serve_prob(s2.get('serve', 50), s2.get('power', 50))

        return_def_A = get_return_def(s1.get('speed', 50), s1.get('backhand', 50), s1.get('forehand', 50))
        return_def_B = get_return_def(s2.get('speed', 50), s2.get('backhand', 50), s2.get('forehand', 50))

        p_A_wins_on_serve = base_serve_win_A - (return_def_B * 0.12)
        p_B_wins_on_serve = base_serve_win_B - (return_def_A * 0.12)

        overall_A = s1.get('overall_rating', 50)
        overall_B = s2.get('overall_rating', 50)
        overall_gap_delta = (overall_A - overall_B) * 0.0035

        p_A_wins_on_serve += overall_gap_delta
        p_B_wins_on_serve -= overall_gap_delta

        bsi_modifier_A = (bsi - 6.5) * 0.015
        bsi_modifier_B = bsi_modifier_A

        safe_style_A = (styleA or "").lower()
        safe_style_B = (styleB or "").lower()

        if "big server" in safe_style_A or "first-strike" in safe_style_A:
            bsi_modifier_A *= (2.5 if bsi < 6.0 else 1.5)
        if "big server" in safe_style_B or "first-strike" in safe_style_B:
            bsi_modifier_B *= (2.5 if bsi < 6.0 else 1.5)

        if ("counter puncher" in safe_style_A or "grinder" in safe_style_A) and bsi < 6.0:
            return_def_A *= 1.20
            p_B_wins_on_serve -= 0.03
        if ("counter puncher" in safe_style_B or "grinder" in safe_style_B) and bsi < 6.0:
            return_def_B *= 1.20
            p_A_wins_on_serve -= 0.03

        p_A_wins_on_serve += bsi_modifier_A
        p_B_wins_on_serve += bsi_modifier_B

        p_A_wins_on_serve += ((formA - 5) * 0.008)
        p_B_wins_on_serve += ((formB - 5) * 0.008)

        p_A_wins_on_serve = max(0.40, min(0.92, p_A_wins_on_serve))
        p_B_wins_on_serve = max(0.40, min(0.92, p_B_wins_on_serve))

        def simulate_game(prob_serve_win):
            pts_srv, pts_ret = 0, 0
            while True:
                if random.random() < prob_serve_win: pts_srv += 1
                else: pts_ret += 1
                if pts_srv >= 4 and pts_srv - pts_ret >= 2: return True
                if pts_ret >= 4 and pts_ret - pts_srv >= 2: return False

        def simulate_tiebreak(prob_A, prob_B):
            pts_A, pts_B = 0, 0
            serves_A = True
            pts_played = 0
            while True:
                if serves_A:
                    if random.random() < prob_A: pts_A += 1
                    else: pts_B += 1
                else:
                    if random.random() < prob_B: pts_B += 1
                    else: pts_A += 1
                pts_played += 1
                if pts_played % 2 == 1: serves_A = not serves_A

                if pts_A >= 7 and pts_A - pts_B >= 2: return True
                if pts_B >= 7 and pts_B - pts_A >= 2: return False

        def simulate_set():
            games_A, games_B = 0, 0
            serves_A = True
            while True:
                if serves_A:
                    if simulate_game(p_A_wins_on_serve): games_A += 1
                    else: games_B += 1
                else:
                    if simulate_game(p_B_wins_on_serve): games_B += 1
                    else: games_A += 1
                serves_A = not serves_A

                if games_A == 6 and games_B == 6: return simulate_tiebreak(p_A_wins_on_serve, p_B_wins_on_serve)
                if games_A >= 6 and games_A - games_B >= 2: return True
                if games_B >= 6 and games_B - games_A >= 2: return False

        match_wins_A, match_wins_B = 0, 0
        for _ in range(iterations):
            sets_A, sets_B = 0, 0
            while sets_A < 2 and sets_B < 2:
                if simulate_set(): sets_A += 1
                else: sets_B += 1
            if sets_A == 2: match_wins_A += 1
            else: match_wins_B += 1

        prob_A = (match_wins_A / iterations) * 100
        prob_B = (match_wins_B / iterations) * 100

        return {
            "probA": round(prob_A, 1),
            "probB": round(prob_B, 1),
            "scoreA": overall_A,
            "scoreB": overall_B
        }

async def call_edge_function(payload: dict):
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(FUNCTION_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log(f"Edge Function Call Failed: {e}")
            return None

# =================================================================
# 🚀 TOURNAMENT SYNERGY AGENT (HIGH VARIANCE QUANT MODE)
# =================================================================
async def generate_synergy_for_player(player_name, tournament_name, surface, bsi, notes, strengths, weaknesses, play_style, roi, total_matches, macro_win_rate):
    prompt = f"""
    You are an elite, data-driven Quantitative Tennis Analyst for a major betting syndicate.
    Your objective is to find true "Horses for Courses" Alpha by analyzing how perfectly a player fits a SPECIFIC court.
    
    Avoid generic hype. Avoid being artificially harsh. Be mathematically precise.

    PLAYER PROFILE:
    - Name: {player_name}
    - Playstyle: {play_style}
    - Strengths: {strengths}
    - Weaknesses: {weaknesses}
    - Overall Macro Win-Rate on {surface}: {macro_win_rate}

    COURT PROFILE:
    - Tournament: {tournament_name}
    - Surface: {surface}
    - BSI Speed Rating (1-10): {bsi} (1 = extremely slow clay, 10 = lightning fast indoor hard)
    - Court Notes: {notes}
    
    HISTORICAL PERFORMANCE ON THIS EXACT SPEED BUCKET:
    - ROI on this specific BSI range: {roi:.1f}% (over {total_matches} tracked betting matches)

    CRITICAL ANALYTICAL LOGIC TO FOLLOW:
    1. Read the Court Notes very carefully. Is this FAST clay (like Houston/Madrid/Altitude) or SLOW clay? Is it fast indoor hard or slow outdoor hard?
    2. Fast Clay/Altitude strictly rewards aggressive baseliners, big servers, and flatter hitters (e.g. Shelton, Etcheverry, Gomez). DO NOT penalize them just because it's "Clay". If the court is fast/altitude, aggressive styles get a massive BOOST.
    3. Slow Clay punishes bad rally tolerance. If a player lacks patience, and the BSI is low, penalize them heavily.
    4. Ground your rating in the 'Macro Win-Rate'. If a player wins >55% on this surface historically, they are objectively good on it. Do not rate them below 6.0 unless this specific BSI entirely destroys their game.

    INSTRUCTIONS FOR HIGH VARIANCE SCORING:
    1. Calculate a highly precise 'synergy_score' from 1.00 to 10.00 using TWO decimal places (e.g., 7.84, 6.31, 9.12) to ensure maximum variance and individuality.
       - 5.00 is a dead-neutral fit.
       - 6.00 - 7.50 means a strong, advantageous fit where strengths align with the court.
       - 8.00 - 10.00 is reserved ONLY for exceptional harmony (e.g., a massive server on extreme altitude/fast courts with a proven high ROI).
       - 1.00 - 4.99 means their game is actively hindered by the physics.
       - Weight the final score: 90% based on your tactical playstyle physics analysis, and 10% based on the Historical ROI. Do not cluster scores around 8.0! Be extremely specific based on the exact matchup of style vs. BSI.
    2. Provide a 1-3 word 'verdict' (e.g. "Lethal Edge", "Solid Fit", "Vulnerable", "Exposed by Speed").
    3. Write 3 highly technical 'tactical_bullets' explaining WHY. Connect their specific strokes to the Court Notes and the BSI rating.
    4. RETURN ONLY VALID JSON. English language.

    JSON FORMAT:
    {{
      "synergy_score": 7.84,
      "verdict": "Strong Advantage",
      "tactical_bullets": ["Reason 1", "Reason 2", "Reason 3"]
    }}
    """
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json'},
            json={
                "model": "meta-llama/llama-3.3-70b-instruct",
                "messages": [{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": prompt}],
                "temperature": 0.35, # 🚀 SOTA FIX: Leicht erhöht für mehr Score-Dispersion und Varianz
                "response_format": {"type": "json_object"}
            }
        )
        data = response.json()
        content = data['choices'][0]['message']['content']
        return json.loads(content)

async def execute_synergy_analysis():
    log("🚀 Starting Pre-Calculation Synergy Pipeline...")
    if not OPENROUTER_API_KEY:
        log("⚠️ OPENROUTER_API_KEY missing, skipping Synergy Pipeline.")
        return
        
    api_player_map = await build_api_tennis_player_map()

    # 1. Hole alle aktuellen Oracle Draws (Turniere & Spieler)
    draws_res = supabase.table('tournament_oracle_draws').select('tournament_name, player_a_name, player_b_name').execute()
    draws = draws_res.data if draws_res else []
    
    tournaments = {}
    for draw in draws:
        t_name = draw.get('tournament_name', '').strip()
        if not t_name: continue
        if t_name not in tournaments: tournaments[t_name] = set()
        if draw.get('player_a_name'): tournaments[t_name].add(draw['player_a_name'].strip())
        if draw.get('player_b_name'): tournaments[t_name].add(draw['player_b_name'].strip())

    # 2. Hole Turnier-Details (BSI, Notes)
    tour_details = supabase.table('tournaments').select('name, surface, bsi_rating, notes').execute().data
    tour_map = {t['name'].strip().lower(): t for t in (tour_details or [])}

    # 3. Hole Spieler & Scout-Reports
    players = supabase.table('players').select('id, last_name, play_style, first_name').execute().data
    reports = supabase.table('scouting_reports').select('player_id, strengths, weaknesses').execute().data
    
    report_map = {r['player_id']: r for r in (reports or [])}
    player_map = {p['last_name'].strip().lower(): p for p in (players or [])}

    # 4. Hole bereits berechnete Synergies
    existing = supabase.table('tournament_court_synergy').select('tournament_name, player_name').execute().data
    existing_set = {f"{e['tournament_name']}_{e['player_name']}" for e in (existing or [])}

    for t_name, player_set in tournaments.items():
        t_info = tour_map.get(t_name.lower())
        if not t_info: continue
        
        t_surface = t_info.get('surface', 'Hard')
        t_bsi = t_info.get('bsi_rating', 5.0)
        t_bucket = get_bsi_bucket(t_surface, t_bsi)

        for p_name in player_set:
            if f"{t_name}_{p_name}" in existing_set:
                continue # Schon berechnet

            last_name_part = p_name.split(' ')[-1].lower()
            p_info = player_map.get(last_name_part) 
            if not p_info: continue
            
            p_report = report_map.get(p_info['id'])
            
            api_key_discovered = api_player_map.get(last_name_part)
            macro_win_rate = await fetch_macro_surface_winrate(api_key_discovered, last_name_part, t_surface)

            # 🚀 ROI BERECHNUNG
            roi, total_matches = 0.0, 0
            try:
                res = supabase.table('market_odds').select('player1_name, player2_name, odds1, odds2, actual_winner_name, tournament, ai_analysis_text').or_(f"player1_name.ilike.%{last_name_part}%,player2_name.ilike.%{last_name_part}%").neq('actual_winner_name', 'None').limit(300).execute()
                matches = res.data or []
                
                wins, losses, profit = 0, 0, 0
                for m in matches:
                    m_tour = str(m.get('tournament', '')).lower().strip()
                    m_info = tour_map.get(m_tour)
                    
                    m_surf = m_info['surface'] if m_info else 'Hard'
                    m_bsi = m_info.get('bsi_rating') if m_info else None
                    
                    if not m_info:
                        text_search = f"{m.get('tournament', '')} {m.get('ai_analysis_text', '')}".lower()
                        if 'clay' in text_search or 'roland garros' in text_search: m_surf = 'Clay'
                        elif 'grass' in text_search or 'wimbledon' in text_search: m_surf = 'Grass'
                        elif 'indoor' in text_search or 'carpet' in text_search: m_surf = 'Indoor Hard'
                    
                    if m_bsi is None:
                        if m_surf == 'Clay': m_bsi = 3.5
                        elif m_surf == 'Grass': m_bsi = 8.0
                        elif m_surf == 'Indoor Hard': m_bsi = 8.0
                        else: m_bsi = 6.0
                        
                    m_bucket = get_bsi_bucket(m_surf, m_bsi)
                    if m_bucket == t_bucket:
                        is_p1 = last_name_part in str(m.get('player1_name', '')).lower()
                        my_odds = float(m.get('odds1') or 0) if is_p1 else float(m.get('odds2') or 0)
                        if my_odds > 1.01:
                            is_win = last_name_part in str(m.get('actual_winner_name', '')).lower()
                            if is_win:
                                wins += 1
                                profit += (my_odds - 1.0)
                            else:
                                losses += 1
                                profit -= 1.0
                
                total_matches = wins + losses
                roi = (profit / total_matches * 100) if total_matches > 0 else 0.0
            except Exception as e:
                log(f"    ⚠️ ROI Calc error for {p_name}: {e}")
            
            log(f"🤖 Quant Synergy Analysis for {p_name} @ {t_name} | Form: {macro_win_rate} | BSI ROI: {roi:.1f}%")
            try:
                ai_data = await generate_synergy_for_player(
                    p_name, t_name, 
                    t_surface, 
                    t_bsi, 
                    t_info.get('notes', ''),
                    p_report['strengths'] if p_report and p_report.get('strengths') else 'Solid baseline game',
                    p_report['weaknesses'] if p_report and p_report.get('weaknesses') else 'Struggles with heavy spin',
                    p_info.get('play_style') or 'All-Rounder',
                    roi,
                    total_matches,
                    macro_win_rate
                )
                
                # In Supabase abspeichern
                supabase.table('tournament_court_synergy').upsert({
                    'tournament_name': t_name,
                    'player_name': p_name,
                    'surface': t_surface,
                    'bsi_rating': t_bsi,
                    'synergy_score': ai_data.get('synergy_score', 5.0),
                    'verdict': ai_data.get('verdict', 'Neutral'),
                    'tactical_bullets': ai_data.get('tactical_bullets', [])
                }, on_conflict="tournament_name,player_name").execute()
                log(f"  ✅ Saved Synergy Matrix for {p_name}!")
                await asyncio.sleep(1) # Schutz gegen API-Rate-Limits
            except Exception as e:
                log(f"  ❌ Failed Synergy Generation for {p_name}: {e}")

# =================================================================
# 4. ORACLE SCRAPER & PIPELINE
# =================================================================
async def scrape_tennis_oracle_for_date(browser: Browser, target_date: datetime, db_data: Dict):
    page = await browser.new_page()

    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        log(f"📡 Scanning draws for: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        html = await page.content()
        
        matches = parse_draws_locally(html)
        
        db_players = db_data['players']
        db_tournaments = db_data['tournaments']
        db_skills = db_data['skills']
        db_reports = db_data['reports']
        
        report_ids = set(db_reports.keys())
        
        inserted_matches = 0
        
        for m in matches:
            s_low = m['tour'].lower().strip()
            best_tour = None
            best_score = 0
            for t in db_tournaments:
                score = calculate_fuzzy_score(s_low, t['name'])
                if score > best_score:
                    best_score = score
                    best_tour = t
            
            if best_score < 20 or not best_tour: continue
                
            p1_obj = find_player_smart(m['p1_raw'], db_players, report_ids)
            p2_obj = find_player_smart(m['p2_raw'], db_players, report_ids)
            
            if p1_obj and p2_obj and p1_obj['id'] != p2_obj['id']:
                
                f1_init = f"{p1_obj.get('first_name', '')[0]}." if p1_obj.get('first_name') else ""
                n1 = f"{f1_init} {p1_obj.get('last_name', '')}".strip()
                
                f2_init = f"{p2_obj.get('first_name', '')[0]}." if p2_obj.get('first_name') else ""
                n2 = f"{f2_init} {p2_obj.get('last_name', '')}".strip()

                tour_name = best_tour['name']
                tour_surface = best_tour['surface']
                bsi_rating = best_tour.get('bsi_rating', 5.0)

                s1_data = db_skills.get(p1_obj['id'], {})
                s2_data = db_skills.get(p2_obj['id'], {})
                f1_score = extract_rating(p1_obj.get('form_rating'))
                f2_score = extract_rating(p2_obj.get('form_rating'))
                
                mc_results = MarkovChainEngine.run_simulation(
                    s1=s1_data, s2=s2_data, 
                    formA=f1_score, formB=f2_score,
                    bsi=bsi_rating, 
                    styleA=p1_obj.get('play_style', ''), styleB=p2_obj.get('play_style', '')
                )
                
                log(f"   🧠 Sending {n1} vs {n2} to AI Edge Function (MC Context: {mc_results['probA']}%)...")
                
                payload = {
                    "playerAId": p1_obj['id'],
                    "playerBId": p2_obj['id'],
                    "surface": tour_surface,
                    "bsi": bsi_rating,
                    "location": tour_name,
                    "skillsA": s1_data,
                    "skillsB": s2_data,
                    "reportA": db_reports.get(p1_obj['id']),
                    "reportB": db_reports.get(p2_obj['id']),
                    "mc_prob_a": mc_results['probA'],
                    "mc_prob_b": mc_results['probB'],
                    "language": "en"
                }

                edge_result = await call_edge_function(payload)
                
                if edge_result and "winner_prediction" in edge_result:
                    predicted_winner_db = edge_result["winner_prediction"]
                    if predicted_winner_db.lower() == p1_obj['last_name'].lower():
                        predicted_winner_db = n1
                    elif predicted_winner_db.lower() == p2_obj['last_name'].lower():
                        predicted_winner_db = n2

                    raw_prob = str(edge_result.get("win_probability", "50.0")).replace("%", "")
                    try: win_prob = float(raw_prob)
                    except: win_prob = 50.0

                    db_payload = {
                        "tournament_name": tour_name,
                        "match_date": target_date.strftime('%Y-%m-%d'),
                        "player_a_name": n1,  
                        "player_b_name": n2,  
                        "predicted_winner": predicted_winner_db,
                        "win_probability": win_prob,
                        "surface": tour_surface,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    supabase.table("tournament_oracle_draws").upsert(
                        db_payload, on_conflict="tournament_name,player_a_name,player_b_name"
                    ).execute()
                    
                    inserted_matches += 1
                    log(f"   ✅ AI Result: {predicted_winner_db} wins.")
                    await asyncio.sleep(2)
                else:
                    log(f"   ⚠️ Edge Function failed for {n1} vs {n2}. Skipping.")

        log(f"✅ Synced {inserted_matches} matches via Edge Function.")

    except Exception as e: 
        log(f"⚠️ Scraping Error: {e}")
    finally: 
        await page.close()

async def run_pipeline():
    try:
        log("📥 Loading DB Context (Players)...")
        try:
            res = supabase.table("players").select("id, last_name, first_name, form_rating, surface_ratings").execute()
            players = res.data
        except Exception as e:
            log(f"🚨 CRITICAL: Database fetch failed. Error details: {str(e)}")
            return

        log("📥 Loading DB Context (Tournaments)...")
        tournaments = supabase.table("tournaments").select("id, name, surface, bsi_rating, notes").execute().data
        
        if not players or not tournaments:
            log("❌ Error: Could not load critical DB reference data.")
            return

        for p in players:
            if isinstance(p.get('form_rating'), str):
                try: import json; p['form_rating'] = json.loads(p['form_rating'])
                except: p['form_rating'] = None
            if isinstance(p.get('surface_ratings'), str):
                try: import json; p['surface_ratings'] = json.loads(p['surface_ratings'])
                except: p['surface_ratings'] = None

        skills = []
        try:
            skills = supabase.table("player_skills").select("*").execute().data
        except: log("⚠️ Skipping player_skills due to data error.")

        reports = []
        try:
            reports = supabase.table("scouting_reports").select("player_id, strengths, weaknesses").execute().data
        except: log("⚠️ Skipping scouting_reports due to data error.")

        skills_dict = {s['player_id']: s for s in (skills or [])}
        reports_dict = {r['player_id']: r for r in (reports or [])}
        db_data = {'players': players, 'tournaments': tournaments, 'skills': skills_dict, 'reports': reports_dict}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            for day_offset in range(0, 3): 
                target_date = datetime.now() + timedelta(days=day_offset)
                await scrape_tennis_oracle_for_date(browser, target_date, db_data)
            await browser.close()
        
        await execute_synergy_analysis()

        log("🏁 Oracle Cycle Finished.")
        
    except Exception as e:
        log(f"❌ Pipeline crashed: {e}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
