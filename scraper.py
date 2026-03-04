# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import random
import time
import csv
import io
import difflib 
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Set
import urllib.parse
import numpy as np 

import httpx
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_Architect")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V155.0 - Elite Master-Slave Sync Architecture)...")

# Secrets Load
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not OPENROUTER_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub/OpenRouter Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_NAME = 'meta-llama/llama-3.3-70b-instruct'

# Global Caches & Dynamic Memory
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
METADATA_CACHE: Dict[str, Any] = {} 
WEATHER_CACHE: Dict[str, Any] = {} 
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 
TML_MATCH_CACHE: List[Dict] = [] 

DYNAMIC_WEIGHTS = {
    "ATP": {"SKILL": 0.50, "FORM": 0.35, "SURFACE": 0.15, "MC_VARIANCE": 1.20},
    "WTA": {"SKILL": 0.50, "FORM": 0.35, "SURFACE": 0.15, "MC_VARIANCE": 1.20}
}
SYSTEM_ACCURACY = {"ATP": 65.0, "WTA": 65.0}

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena", "Sydney": "Ken Rosewall Arena",
    "Brisbane": "Pat Rafter Arena", "Adelaide": "Memorial Drive Tennis Centre",
    "Melbourne": "Rod Laver Arena"
}
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

# =================================================================
# 1.5 TENNIS-MY-LIFE (TML) INGESTION ENGINE
# =================================================================
async def fetch_tml_database():
    log("📡 Verbinde mit TennisMyLife API (Downloading ATP Data Lake)...")
    loaded_matches = 0
    async with httpx.AsyncClient() as client:
        try:
            tml_api_url = "https://stats.tennismylife.org/api/data-files"
            res = await client.get(tml_api_url, timeout=15.0)
            if res.status_code == 200:
                files = res.json().get('files', [])
                for f in files:
                    if f['name'] in ['2025.csv', '2026.csv', 'ongoing_tourneys.csv']:
                        csv_res = await client.get(f['url'], timeout=30.0)
                        reader = csv.DictReader(io.StringIO(csv_res.text))
                        for row in reader:
                            TML_MATCH_CACHE.append(row)
                            loaded_matches += 1
                log(f"✅ TML Data Lake aktiv: {loaded_matches} historische/live ATP-Matches geladen.")
        except Exception as e:
            log(f"⚠️ TML API Error (Nutze lokale/Fallback-Daten): {e}")

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def to_float(val: Any, default: float = 0.0) -> float:
    if val is None: 
        return default
    try: 
        return float(val)
    except: 
        return default

def normalize_text(text: str) -> str:
    if not text: 
        return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    if not raw: 
        return ""
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    clean = re.sub(r'\s*\(\d+\)', '', clean) 
    clean = re.sub(r'\s*\(.*?\)', '', clean) 
    return clean.replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: 
        return "Unknown"
    clean = raw
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<.?>', '', clean)
    clean = re.sub(r'S\d+.$', '', clean) 
    clean = re.sub(r'H2H.*$', '', clean)
    clean = re.sub(r'\b(Challenger|Men|Women|Singles|Doubles)\b', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s\d+$', '', clean)
    return clean.strip()

def get_last_name(full_name: str) -> str:
    if not full_name: 
        return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip()
    parts = clean.split()
    if parts: 
        return parts[-1].lower()
    return ""

def ensure_dict(data: Any) -> Dict:
    try:
        if isinstance(data, dict): 
            return data
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict): 
                return data[0]
        return {}
    except: 
        return {}

def normalize_db_name(name: str) -> str:
    if not name: 
        return ""
    n = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = n.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def is_same_player(target_name: str, db_name: str) -> bool:
    t_norm = normalize_db_name(target_name)
    d_norm = normalize_db_name(db_name)
    
    if t_norm == d_norm: 
        return True
        
    t_parts = t_norm.split()
    d_parts = d_norm.split()
    
    if not t_parts or not d_parts: 
        return False
        
    if t_parts[-1] != d_parts[-1]: 
        return False
        
    t_first = t_parts[0] if len(t_parts) > 1 else ""
    d_first = d_parts[0] if len(d_parts) > 1 else ""
    
    if t_first and d_first: 
        return t_first[0] == d_first[0]
        
    return True

# 🚀 SOTA: ROBUST FUZZY MATCH FOR ORACLE
def is_fuzzy_match(name1: str, name2: str) -> bool:
    n1 = normalize_db_name(name1)
    n2 = normalize_db_name(name2)
    
    if n1 in n2 or n2 in n1: 
        return True
    
    w1 = [w for w in n1.split() if len(w) > 2]
    w2 = [w for w in n2.split() if len(w) > 2]
    
    if w1 and w2 and w1[-1] == w2[-1]: 
        return True # Last names match perfectly
        
    return False

def parse_te_name(raw: str):
    clean = re.sub(r'\s*\(\d+\)|\s*\(.*?\)', '', raw).strip()
    parts = clean.split()
    
    if not parts: 
        return "", ""
        
    initial = ""
    last_name_parts = []
    
    for p in parts:
        if (len(p) <= 2 and p.endswith('.')) or (len(p) == 1 and p.isalpha()):
            if not initial: 
                initial = p[0].lower()
        else:
            last_name_parts.append(p)
            
    last_name = " ".join(last_name_parts)
    
    if not initial and len(parts) > 1:
        initial = parts[0][0].lower()
        last_name = " ".join(parts[1:])
        
    return normalize_db_name(last_name), initial

def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str] = None) -> Optional[Dict]:
    if report_ids is None: 
        report_ids = set()
        
    if not scraped_name_raw or len(scraped_name_raw) < 3 or re.search(r'\d', scraped_name_raw): 
        return None 
        
    bad_words = ['satz', 'set', 'game', 'über', 'unter', 'handicap', 'sieger', 'winner', 'tennis', 'live', 'stream', 'stats', 'tv']
    if any(w in scraped_name_raw.lower() for w in bad_words): 
        return None

    clean_scrape = normalize_db_name(clean_player_name(scraped_name_raw))
    scrape_tokens = clean_scrape.split()
    
    if not scrape_tokens: 
        return None

    candidates = []
    has_comma = "," in scraped_name_raw
    
    if has_comma:
        parts = scraped_name_raw.split(',')
        scrape_last_part = normalize_db_name(parts[0])
        scrape_first_part = normalize_db_name(parts[1]) if len(parts) > 1 else ""
    else:
        scrape_last_part = clean_scrape
        scrape_first_part = ""
        if len(scrape_tokens) > 1: 
            scrape_first_part = scrape_tokens[0]
    
    for p in db_players:
        db_last = normalize_db_name(p.get('last_name', ''))
        db_first = normalize_db_name(p.get('first_name', ''))
        db_last_tokens = db_last.split()
        db_first_tokens = db_first.split()
        
        score = 0
        last_matched = False
        
        if has_comma:
            if db_last == scrape_last_part:
                score += 60
                last_matched = True
            elif len(db_last) >= 5 and get_similarity(db_last, scrape_last_part) >= 0.80:
                score += 60
                last_matched = True
            else:
                continue 
                
            if last_matched and db_first and scrape_first_part:
                if db_first == scrape_first_part or scrape_first_part in db_first: 
                    score += 80 
                elif len(db_first) >= 5 and get_similarity(db_first, scrape_first_part) >= 0.80: 
                    score += 50
                else:
                    sf_tokens = scrape_first_part.split()
                    if sf_tokens and len(sf_tokens[0]) > 0 and sf_tokens[0][0] == db_first[0]: 
                        score += 15 
                    elif sf_tokens and len(sf_tokens[0]) > 0 and db_first and sf_tokens[0][0] != db_first[0]: 
                        score -= 100 
        else:
            if db_last_tokens and db_last_tokens[-1] in scrape_tokens:
                score += 60
                last_matched = True
            elif len(db_last) >= 6:
                for t in scrape_tokens:
                    if len(t) >= 6 and get_similarity(db_last, t) >= 0.85:
                        score += 60
                        last_matched = True
                        break
                        
            if last_matched:
                toxic_leftover = False
                for st in scrape_tokens:
                    if len(st) > 3: 
                        explained = False
                        for dt in db_last_tokens + db_first_tokens:
                            if st == dt or get_similarity(st, dt) > 0.80:
                                explained = True
                                break
                        if not explained:
                            toxic_leftover = True
                            break
                
                if toxic_leftover: 
                    score -= 50 
                    
                if db_first and score >= 60:
                    if any(ft in scrape_tokens for ft in db_first_tokens) or (scrape_first_part and scrape_first_part in db_first):
                        score += 80 
                    else:
                        db_f_init = db_first[0]
                        has_contradicting = False
                        has_matching = False
                        for st in scrape_tokens:
                            c_st = st.replace('.', '')
                            if 0 < len(c_st) <= 2: 
                                if c_st[0] == db_f_init: 
                                    has_matching = True
                                else: 
                                    has_contradicting = True
                        
                        if has_matching: 
                            score += 15
                        elif has_contradicting: 
                            score -= 100 
                            
        if score >= 60: 
            candidates.append((p, score))
                
    if not candidates: 
        return None
        
    candidates.sort(key=lambda x: (x[1], x[0]['id'] in report_ids), reverse=True)
    
    if len(candidates) > 1:
        if candidates[0][1] == candidates[1][1]: 
            return "TIE_BREAKER"
            
    return candidates[0][0]

def calculate_fuzzy_score(scraped_name: str, db_name: str) -> int:
    s_norm = normalize_text(scraped_name).lower()
    d_norm = normalize_text(db_name).lower()
    
    if d_norm in s_norm and len(d_norm) > 3: 
        return 100
        
    s_tokens = set(re.findall(r'\w+', s_norm))
    d_tokens = set(re.findall(r'\w+', d_norm))
    stop_words = {'atp', 'wta', 'open', 'tour', '2025', '2026', 'challenger', 'itf', 'world', 'tennis'}
    
    s_tokens -= stop_words
    d_tokens -= stop_words
    
    if not s_tokens or not d_tokens: 
        return 0
        
    common = s_tokens.intersection(d_tokens)
    score = len(common) * 15
    
    if "indoor" in s_tokens and "indoor" in d_tokens: 
        score += 20
    if "canberra" in s_tokens and "canberra" in d_tokens: 
        score += 30
        
    return score

def has_active_signal(text: Optional[str]) -> bool:
    if not text: 
        return False
    if "[" in text and "]" in text:
        if any(icon in text for icon in ["💎", "🛡️", "⚖️", "💰", "🔥", "✨", "📈", "👀"]): 
            return True
    return False

# --- SOTA WEATHER SERVICE ---
async def fetch_weather_data(location_name: str) -> Optional[Dict]:
    if not location_name or location_name == "Unknown": 
        return None
        
    clean_location = re.sub(r'a-zA-Z0-9\s,', '', location_name).strip()
    if len(clean_location) > 50: 
        clean_location = clean_location[:50] 
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    cache_key = f"{clean_location}_{today_str}"
    
    if cache_key in WEATHER_CACHE: 
        return WEATHER_CACHE[cache_key]

    try:
        async with httpx.AsyncClient() as client:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_location}&count=1&language=en&format=json"
            geo_res = await client.get(geo_url)
            geo_data = geo_res.json()
            
            if not geo_data.get('results'): 
                WEATHER_CACHE[cache_key] = None
                return None

            loc = geo_data['results'][0]
            lat, lon = loc['latitude'], loc['longitude']

            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
            w_res = await client.get(w_url)
            w_data = w_res.json()
            
            curr = w_data.get('current', {})
            if not curr: 
                return None

            impact = "Neutral conditions."
            temp = curr.get('temperature_2m', 20)
            hum = curr.get('relative_humidity_2m', 50)
            wind = curr.get('wind_speed_10m', 10)

            if temp > 30: 
                impact = "EXTREME HEAT: Ball flies fast, physically draining."
            elif temp < 12: 
                impact = "COLD: Low bounce, heavy conditions."
                
            if hum > 70: 
                impact += " HIGH HUMIDITY: Air is heavy, ball travels slower."
            if wind > 20: 
                impact += " WINDY: Serve toss difficult, high variance."

            result = {"summary": f"{temp}°C, {hum}% Hum, Wind: {wind} km/h", "impact_note": impact}
            WEATHER_CACHE[cache_key] = result
            return result
    except: 
        return None

# --- MARKET INTEGRITY & ANTI-SPIKE ENGINE ---
def validate_market_integrity(o1: float, o2: float) -> bool:
    if o1 <= 1.01 or o2 <= 1.01: 
        return False 
    if o1 > 200 or o2 > 200: 
        return False 
    implied_prob = (1/o1) + (1/o2)
    if implied_prob < 0.92 or implied_prob > 1.45: 
        return False
    return True

def is_suspicious_movement(old_o1: float, new_o1: float, old_o2: float, new_o2: float) -> bool:
    if old_o1 == 0 or old_o2 == 0: 
        return False 
        
    if abs(new_o1 - old_o2) < 0.15 and abs(new_o2 - old_o1) < 0.15: 
        return False
        
    change_p1 = abs(new_o1 - old_o1) / old_o1
    change_p2 = abs(new_o2 - old_o2) / old_o2
    
    if change_p1 > 0.60 or change_p2 > 0.60:
        if old_o1 < 1.10 or old_o2 < 1.10: 
            return False
        return True
        
    return False

# =================================================================
# 3. SOTA MOMENTUM V3 ENGINE (xG Model)
# =================================================================
class MomentumV2Engine:  
    @staticmethod
    def calculate_rating(matches: List[Dict], player_name: str, max_matches: int = 10) -> Dict[str, Any]:
        if not matches: 
            return {"score": 6.5, "text": "Neutral (No Data)", "history_summary": "", "color_hex": "#808080"}
            
        recent_matches = sorted(matches, key=lambda x: str(x.get('created_at', '')), reverse=True)[:max_matches]
        chrono_matches = recent_matches[::-1]

        base_rating = 6.5 
        cumulative_edge = 0.0
        total_weight = 0.0
        history_log = []
        
        for idx, m in enumerate(chrono_matches):
            p1_str = str(m.get('player1_name', ''))
            p2_str = str(m.get('player2_name', ''))
            
            is_p1 = is_same_player(player_name, p1_str)
            is_p2 = is_same_player(player_name, p2_str)
            
            if not is_p1 and not is_p2: 
                continue

            winner = str(m.get('actual_winner_name', ''))
            won = is_same_player(player_name, winner)
            odds = to_float(m.get('odds1') if is_p1 else m.get('odds2'), 1.85)
            
            if odds <= 1.01: 
                odds = 1.85
            
            expected_perf = 1 / odds 
            actual_perf = 0.5 
            score_str = str(m.get('score', '')).lower()
            
            if "ret" in score_str or "w.o" in score_str:
                actual_perf = 0.6 if won else 0.4
            else:
                sets = re.findall(r'(\d+)-(\d+)', score_str)
                if not sets:
                    actual_perf = 0.75 if won else 0.25
                else:
                    player_sets_won = 0
                    opp_sets_won = 0
                    player_games_won = 0
                    opp_games_won = 0
                    
                    for s in sets:
                        try:
                            l, r = int(s[0]), int(s[1])
                            p_games = l if is_p1 else r
                            o_games = r if is_p1 else l
                            player_games_won += p_games
                            opp_games_won += o_games
                            if p_games > o_games: 
                                player_sets_won += 1
                            elif o_games > p_games: 
                                opp_sets_won += 1
                        except: 
                            pass
                    
                    if won:
                        if opp_sets_won == 0: 
                            game_diff = player_games_won - opp_games_won
                            if game_diff >= 8: actual_perf = 1.0      
                            elif game_diff >= 5: actual_perf = 0.9    
                            elif game_diff >= 3: actual_perf = 0.8    
                            else: actual_perf = 0.7                   
                        else: 
                            game_diff = player_games_won - opp_games_won
                            if game_diff >= 4: actual_perf = 0.75     
                            elif game_diff >= 1: actual_perf = 0.65   
                            else: actual_perf = 0.55                  
                    else:
                        if player_sets_won == 1: 
                            game_diff = opp_games_won - player_games_won
                            if game_diff <= 1: actual_perf = 0.45     
                            elif game_diff <= 4: actual_perf = 0.35   
                            else: actual_perf = 0.25                  
                        else: 
                            game_diff = opp_games_won - player_games_won
                            if game_diff <= 3: actual_perf = 0.30     
                            elif game_diff <= 5: actual_perf = 0.20   
                            elif game_diff <= 7: actual_perf = 0.10   
                            else: actual_perf = 0.0                   

            match_edge = actual_perf - expected_perf 
            
            if won: 
                match_edge += 0.40  
            else: 
                match_edge -= 0.20
            
            time_weight = 0.3 + (0.7 * (idx / max(1, len(chrono_matches) - 1)))
            cumulative_edge += (match_edge * time_weight)
            total_weight += time_weight
            history_log.append("W" if won else "L")

        streak_bonus = 0.0
        if len(history_log) >= 3:
            recent_3 = history_log[-3:]
            if recent_3 == ["W", "W", "W"]: 
                streak_bonus = 0.4
            elif recent_3 == ["L", "L", "L"]: 
                streak_bonus = -0.4
            if len(history_log) >= 5:
                recent_5 = history_log[-5:]
                if recent_5.count("W") == 5: 
                    streak_bonus = 0.8
                elif recent_5.count("L") == 5: 
                    streak_bonus = -0.8

        avg_edge = (cumulative_edge / total_weight) if total_weight > 0 else 0.0
        final_rating = base_rating + (avg_edge * 10.0) + streak_bonus
        final_rating = max(1.0, min(10.0, final_rating))
        
        desc = "Average"
        color_hex = "#F0C808" 
        
        if final_rating >= 8.5: 
            desc, color_hex = "🔥 ELITE", "#FF00FF" 
        elif final_rating >= 7.2: 
            desc, color_hex = "📈 Strong", "#3366FF" 
        elif final_rating >= 6.0: 
            desc, color_hex = "Solid", "#00B25B" 
        elif final_rating >= 4.5: 
            desc, color_hex = "⚠️ Vulnerable", "#FF9933" 
        else: 
            desc, color_hex = "❄️ Cold", "#CC0000" 

        return {"score": round(final_rating, 2), "text": desc, "color_hex": color_hex, "history_summary": "".join(history_log[-5:])}

# =================================================================
# 4. SURFACE INTELLIGENCE ENGINE
# =================================================================
class SurfaceIntelligence:
    @staticmethod
    def normalize_surface_key(raw_surface: str) -> str:
        if not raw_surface: 
            return "unknown"
        s = raw_surface.lower()
        if "grass" in s: 
            return "grass"
        if "clay" in s or "sand" in s: 
            return "clay"
        if "hard" in s or "carpet" in s or "acrylic" in s or "indoor" in s: 
            return "hard"
        return "unknown"

    @staticmethod
    def clean_name_for_matching(name: str) -> str:
        if not name: 
            return ""
        n = name.lower()
        n = re.sub(r'\b(atp|wta|ch|challenger|tour|masters|1000|500|250|open|championships|intl|international|men|women|singles)\b', '', n)
        n = re.sub(r'\b(202[0-9])\b', '', n)
        n = re.sub(r'a-z0-9', '', n)
        return n.strip()

    @staticmethod
    def get_matches_by_surface(all_matches: List[Dict], target_surface: str) -> List[Dict]:
        filtered = []
        target = SurfaceIntelligence.normalize_surface_key(target_surface)
        
        for m in all_matches:
            tour_name = str(m.get('tournament', '')).lower()
            ai_text = str(m.get('ai_analysis_text', '')).lower()
            found_surface = "unknown"
            
            match_hist = re.search(r'surface:\s*(hard|clay|grass)', ai_text)
            if match_hist: 
                found_surface = match_hist.group(1)
            elif "hard court" in ai_text or "hard surface" in ai_text: 
                found_surface = "hard"
            elif "red clay" in ai_text or "clay court" in ai_text: 
                found_surface = "clay"
            elif "grass court" in ai_text: 
                found_surface = "grass"
            elif "clay" in tour_name or "roland garros" in tour_name: 
                found_surface = "clay"
            elif "grass" in tour_name or "wimbledon" in tour_name: 
                found_surface = "grass"
            elif "hard" in tour_name or "us open" in tour_name or "australian open" in tour_name: 
                found_surface = "hard"
            else:
                for db_key, db_surf in GLOBAL_SURFACE_MAP.items():
                    if db_key in tour_name or tour_name in db_key:
                        if len(db_key) > 3:
                            found_surface = db_surf
                            break
                            
            if SurfaceIntelligence.normalize_surface_key(found_surface) == target:
                filtered.append(m)
                
        return filtered

    @staticmethod
    def compute_player_surface_profile(matches: List[Dict], player_name: str) -> Dict[str, Any]:
        profile = {}
        surfaces_data = {
            "hard": SurfaceIntelligence.get_matches_by_surface(matches, "hard"),
            "clay": SurfaceIntelligence.get_matches_by_surface(matches, "clay"),
            "grass": SurfaceIntelligence.get_matches_by_surface(matches, "grass")
        }
        
        for surf, surf_matches in surfaces_data.items():
            n_surf = len(surf_matches)
            
            if n_surf == 0:
                profile[surf] = {"rating": 3.5, "color": "#808080", "matches_tracked": 0, "text": "No Experience"}
                continue
                
            wins = 0
            for m in surf_matches:
                winner = str(m.get('actual_winner_name', ""))
                if is_same_player(player_name, winner): 
                    wins += 1
                    
            win_rate = wins / n_surf
            vol_score = min(1.0, n_surf / 30.0) * 1.95
            win_score = win_rate * 4.55
            final_rating = max(1.0, min(10.0, 3.5 + vol_score + win_score))
            
            desc, color_hex = "Average", "#F0C808" 
            if final_rating >= 8.5: 
                desc, color_hex = "🔥 SPECIALIST", "#FF00FF" 
            elif final_rating >= 7.5: 
                desc, color_hex = "📈 Strong", "#3366FF" 
            elif final_rating >= 6.5: 
                color_hex = "#00B25B" 
            elif final_rating >= 5.5: 
                desc, color_hex = "Solid", "#99CC33" 
            elif final_rating <= 4.5: 
                desc, color_hex = "⚠️ Vulnerable", "#CC0000" 
            elif final_rating < 5.5: 
                desc, color_hex = "❄️ Weakness", "#FF9933" 

            profile[surf] = {"rating": round(final_rating, 2), "color": color_hex, "matches_tracked": n_surf, "text": desc}
            
        profile['_v95_mastery_applied'] = True
        return profile

# =================================================================
# 5. SOTA MARKOV CHAIN ENGINE
# =================================================================
class MarkovChainEngine:
    @staticmethod
    def run_simulation(s1: Dict, s2: Dict, formA: float, formB: float, 
                       bsi: float, styleA: str, styleB: str, iterations: int = 2500) -> Dict[str, Any]:
                       
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

        overall_gap_delta = (s1.get('overall_rating', 50) - s2.get('overall_rating', 50)) * 0.0035
        p_A_wins_on_serve += overall_gap_delta
        p_B_wins_on_serve -= overall_gap_delta

        bsi_modifier_A = (bsi - 6.5) * 0.015
        bsi_modifier_B = bsi_modifier_A

        safe_style_A, safe_style_B = (styleA or "").lower(), (styleB or "").lower()

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
                if random.random() < prob_serve_win: 
                    pts_srv += 1
                else: 
                    pts_ret += 1
                if pts_srv >= 4 and pts_srv - pts_ret >= 2: 
                    return True
                if pts_ret >= 4 and pts_ret - pts_srv >= 2: 
                    return False

        def simulate_tiebreak(prob_A, prob_B):
            pts_A, pts_B, serves_A, pts_played = 0, 0, True, 0
            while True:
                if serves_A:
                    if random.random() < prob_A: 
                        pts_A += 1
                    else: 
                        pts_B += 1
                else:
                    if random.random() < prob_B: 
                        pts_B += 1
                    else: 
                        pts_A += 1
                pts_played += 1
                if pts_played % 2 == 1: 
                    serves_A = not serves_A
                if pts_A >= 7 and pts_A - pts_B >= 2: 
                    return True
                if pts_B >= 7 and pts_B - pts_A >= 2: 
                    return False

        def simulate_set():
            games_A, games_B, serves_A = 0, 0, True
            while True:
                if serves_A:
                    if simulate_game(p_A_wins_on_serve): 
                        games_A += 1
                    else: 
                        games_B += 1
                else:
                    if simulate_game(p_B_wins_on_serve): 
                        games_B += 1
                    else: 
                        games_A += 1
                serves_A = not serves_A
                
                if games_A == 6 and games_B == 6: 
                    return simulate_tiebreak(p_A_wins_on_serve, p_B_wins_on_serve)
                if games_A >= 6 and games_A - games_B >= 2: 
                    return True
                if games_B >= 6 and games_B - games_A >= 2: 
                    return False

        match_wins_A, match_wins_B = 0, 0
        for _ in range(iterations):
            sets_A, sets_B = 0, 0
            while sets_A < 2 and sets_B < 2:
                if simulate_set(): 
                    sets_A += 1
                else: 
                    sets_B += 1
            if sets_A == 2: 
                match_wins_A += 1
            else: 
                match_wins_B += 1

        return {
            "probA": round((match_wins_A / iterations) * 100, 1),
            "probB": round((match_wins_B / iterations) * 100, 1),
            "scoreA": s1.get('overall_rating', 50),
            "scoreB": s2.get('overall_rating', 50)
        }

# =================================================================
# 5.5 SOTA: SELF-LEARNING NEURAL OPTIMIZER
# =================================================================
class NeuralOptimizer:
    @staticmethod
    def optimize_ai_weights(matches_history: List[Dict], current_weights: Dict) -> Dict:
        log("🧠 Starte Neural Weight Optimization (Backpropagation Simulation)...")
        best_weights, best_brier_score = current_weights, float('inf') 
        
        for w_skill in np.arange(0.30, 0.70, 0.05):
            for w_form in np.arange(0.20, 0.50, 0.05):
                w_surf = 1.0 - (w_skill + w_form)
                if w_surf < 0.05 or w_surf > 0.30: 
                    continue 
                    
                current_brier_score, valid_matches = 0.0, 0
                for m in matches_history:
                    baseA = (m['skillA']/10) * w_skill + m['formA'] * w_form + m['surfA'] * w_surf
                    baseB = (m['skillB']/10) * w_skill + m['formB'] * w_form + m['surfB'] * w_surf
                    prob_a = 1 / (1 + math.exp(-(baseA - baseB)))
                    actual_result = 1.0 if m['winner_is_A'] else 0.0
                    current_brier_score += (prob_a - actual_result)**2
                    valid_matches += 1
                    
                if valid_matches > 0:
                    avg_brier = current_brier_score / valid_matches
                    if avg_brier < best_brier_score:
                        best_brier_score = avg_brier
                        best_weights = {
                            "SKILL": round(float(w_skill), 2), "FORM": round(float(w_form), 2), 
                            "SURFACE": round(float(w_surf), 2), "MC_VARIANCE": current_weights.get("MC_VARIANCE", 1.20)
                        }
        log(f"✅ Neues Gehirn-Setup gefunden! Brier Score: {round(best_brier_score, 4)} -> {best_weights}")
        return best_weights

    @staticmethod
    def trigger_learning_cycle(players: List[Dict], all_skills: Dict):
        log("🔄 Initiating Weekly Self-Learning Routine...")
        for tour in ["ATP", "WTA"]:
            tour_players = [p['id'] for p in players if p.get('tour') == tour]
            if not tour_players: 
                continue
                
            recent_res = supabase.table("market_odds").select("*").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(200).execute()
            recent_matches, history_data, correct_predictions, total_predictions = recent_res.data or [], [], 0, 0
            
            for m in recent_matches:
                p1_name, p2_name, winner = m.get('player1_name', ''), m.get('player2_name', ''), m.get('actual_winner_name', '')
                fair1, fair2 = to_float(m.get('ai_fair_odds1'), 0), to_float(m.get('ai_fair_odds2'), 0)
                
                if fair1 > 0 and fair2 > 0:
                    total_predictions += 1
                    if (fair1 < fair2 and p1_name.lower() in winner.lower()) or (fair2 < fair1 and p2_name.lower() in winner.lower()): 
                        correct_predictions += 1
                        
                p1_obj = next((p for p in players if p.get('last_name') == p1_name and p['id'] in tour_players), None)
                p2_obj = next((p for p in players if p.get('last_name') == p2_name and p['id'] in tour_players), None)
                
                if p1_obj and p2_obj:
                    history_data.append({
                        "skillA": all_skills.get(p1_obj['id'], {}).get('overall_rating', 50), "formA": 5.5, "surfA": 5.5,
                        "skillB": all_skills.get(p2_obj['id'], {}).get('overall_rating', 50), "formB": 5.5, "surfB": 5.5,
                        "winner_is_A": p1_name.lower() in winner.lower()
                    })
                    
            if total_predictions > 0:
                SYSTEM_ACCURACY[tour] = round((correct_predictions / total_predictions) * 100, 1)
                log(f"🎯 {tour} System Accuracy Rating: {SYSTEM_ACCURACY[tour]}% ({correct_predictions}/{total_predictions})")
                
            if len(history_data) >= 20:
                DYNAMIC_WEIGHTS[tour] = NeuralOptimizer.optimize_ai_weights(history_data, DYNAMIC_WEIGHTS[tour])
                try:
                    supabase.table("ai_system_weights").upsert({"tour": tour, "weight_skill": DYNAMIC_WEIGHTS[tour]["SKILL"], "weight_form": DYNAMIC_WEIGHTS[tour]["FORM"], "weight_surface": DYNAMIC_WEIGHTS[tour]["SURFACE"], "mc_variance": DYNAMIC_WEIGHTS[tour]["MC_VARIANCE"], "last_optimized": datetime.now(timezone.utc).isoformat()}).execute()
                    log(f"💾 {tour} Gewichte erfolgreich in Supabase gesichert.")
                except Exception as e: 
                    log(f"❌ Fehler beim Speichern der AI-Gewichte: {e}")

# =================================================================
# 6. OPENROUTER AI ENGINE (SOTA)
# =================================================================
async def call_openrouter(prompt: str, model: str = MODEL_NAME, temp: float = 0.1) -> Optional[str]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://neuralscout.com", "X-Title": "NeuralScout"}
    payload = {"model": model, "messages": [{"role": "system", "content": "You are a data extraction AI. Return ONLY valid JSON."}, {"role": "user", "content": prompt}], "temperature": temp, "response_format": {"type": "json_object"}}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=45.0)
            if response.status_code != 200: 
                log(f"⚠️ OpenRouter API Error: {response.status_code} - {response.text}")
                return None
            return response.json()['choices'][0]['message']['content']
        except Exception as e: 
            log(f"⚠️ OpenRouter Exception: {e}")
            return None

# =================================================================
# 7. DATA FETCHING & ORACLE (MASTER-SLAVE ARCHITECTURE)
# =================================================================

async def fetch_te_master_schedule(browser: Browser) -> List[Dict]:
    log("🕒 [TE MASTER] Erstelle Master-Schedule aus TennisExplorer (Source of Truth)...")
    schedule = []
    context = await browser.new_context()
    await context.add_cookies([{"name": "tz", "value": "1", "domain": ".tennisexplorer.com", "path": "/"}])
    
    # We scan yesterday, today, tomorrow, and the day after to ensure full timezone coverage
    for day_offset in range(-1, 4):
        t_date = datetime.now(timezone.utc) + timedelta(days=day_offset)
        page = await context.new_page()
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={t_date.year}&month={t_date.month}&day={t_date.day}"
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            soup = BeautifulSoup(await page.content(), 'html.parser')
            table = soup.find('table', class_='result')
            
            if not table: 
                continue
            
            current_tour = "Unknown"
            for row in table.find_all('tr'):
                # Extract Tournament Name Header
                if 'head' in row.get('class', []):
                    tour_name_td = row.find('td', class_='t-name')
                    if tour_name_td: 
                        current_tour = clean_tournament_name(tour_name_td.get_text(strip=True))
                    continue
                
                # Extract Time
                time_td = row.find('td', class_='time')
                if not time_td: 
                    continue
                    
                time_str = time_td.get_text(strip=True)
                if ":" not in time_str: 
                    continue
                
                # Extract Players
                links = [l for l in row.find_all('a') if 'player/' in l.get('href', '')]
                if len(links) >= 2:
                    p1_raw = links[0].get_text(strip=True)
                    p2_raw = links[1].get_text(strip=True)
                    p1_last, _ = parse_te_name(p1_raw)
                    p2_last, _ = parse_te_name(p2_raw)
                    
                    if p1_last and p2_last:
                        h, m = time_str.split(':')
                        dt = t_date.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
                        dt = dt - timedelta(hours=1) # Strip CET +1 to get UTC
                        iso_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        
                        schedule.append({
                            'p1': p1_last, 'p1_raw': p1_raw,
                            'p2': p2_last, 'p2_raw': p2_raw,
                            'time': iso_time, 'tour': current_tour
                        })
        except Exception as e: 
            pass
        finally: 
            await page.close()
            
    await context.close()
    log(f"✅ [TE MASTER] {len(schedule)} offizielle Matches im System-Speicher hinterlegt.")
    return schedule

async def fetch_1win_raw_lines(browser: Browser) -> List[str]:
    log("🚀 [1WIN GHOST] Extrahiere rohen DOM-Text (Slave Feed)...")
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", 
        viewport={"width": 1920, "height": 1080}, 
        java_script_enabled=True
    )
    await context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); window.navigator.chrome = { runtime: {} };")
    page = await context.new_page()
    all_raw_text_blocks = [] 

    try:
        await page.goto("https://1win.io/betting/prematch/tennis-33", wait_until="networkidle", timeout=60000)
        
        if "Just a moment" in await page.title() or "Cloudflare" in await page.title():
            log("🛑 WARNUNG: Cloudflare Challenge aktiv! Warte 5 Sekunden...")
            await asyncio.sleep(5)
            
        await page.mouse.move(960, 540)
        await asyncio.sleep(1.5)
        
        for scroll_step in range(80): 
            try:
                await page.evaluate("""
                    let els = Array.from(document.querySelectorAll('div, button, span'));
                    for(let e of els) {
                        if(e.clientHeight > 0) {
                            let text = e.innerText ? e.innerText.trim() : '';
                            let textLow = text.toLowerCase();
                            if (text.length < 80 && (textLow.includes('atp') || textLow.includes('wta') || textLow.includes('itf') || textLow.includes('challenger'))) {
                                if (window.getComputedStyle(e).cursor === 'pointer' || e.querySelector('svg')) {
                                    try { e.click(); } catch(err){}
                                }
                            }
                            if (textLow === 'more' || textLow === 'show more' || textLow === 'anzeigen' || textLow === 'alle') {
                                try { e.click(); } catch(err){}
                            }
                        }
                    }
                """)
                all_raw_text_blocks.append(await page.evaluate("document.body.innerText"))
                await page.mouse.wheel(delta_x=0, delta_y=500)
                await asyncio.sleep(0.5) 
            except: 
                continue
                
    except Exception as e: 
        log(f"⚠️ [1WIN GHOST] Timeout/Fehler: {e}")
    finally: 
        await context.close()

    unified_lines = []
    for block in all_raw_text_blocks:
        for line in [l.strip() for l in block.split('\n') if l.strip()]:
            if not unified_lines or unified_lines[-1] != line:
                unified_lines.append(line)
                
    log(f"✅ [1WIN GHOST] {len(unified_lines)} Zeilen rohen Text extrahiert.")
    return unified_lines

def find_odds_in_lines(p1_name: str, p2_name: str, lines: List[str]) -> tuple:
    p1_norm = normalize_db_name(p1_name)
    p2_norm = normalize_db_name(p2_name)
    
    def name_in_text(name, text):
        if name in text: return True
        parts = [w for w in name.split() if len(w) > 2]
        if parts and parts[-1] in text: return True
        return False

    for i in range(len(lines)):
        line_norm = normalize_db_name(lines[i])
        
        if name_in_text(p1_norm, line_norm) or name_in_text(p2_norm, line_norm):
            window = lines[max(0, i-6):min(len(lines), i+15)]
            window_text = normalize_db_name(" ".join(window).lower())
            
            if name_in_text(p1_norm, window_text) and name_in_text(p2_norm, window_text):
                floats = []
                for w_line in window:
                    matches_val = re.findall(r'\b\d+\.\d{1,3}\b', w_line.replace(',', '.'))
                    for m_val in matches_val:
                        try:
                            v = float(m_val)
                            if 1.01 < v <= 250.0: floats.append(v)
                        except: pass
                if len(floats) >= 2:
                    return floats[0], floats[1]
                    
    return None, None

async def scrape_oracle_metadata(browser: Browser, target_date: datetime):
    date_str = target_date.strftime('%Y-%m-%d')
    page = await browser.new_page()
    metadata = {}
    try:
        await page.goto(f"https://de.tennistemple.com/matches/{date_str}", wait_until="domcontentloaded", timeout=20000)
        soup = BeautifulSoup(await page.content(), 'html.parser')
        current_tournament = "Unknown"
        for element in soup.find_all(['h2', 'a']):
            text, href = element.get_text(strip=True), element.get('href', '')
            if '/turnier/' in href or '/tournament/' in href: 
                current_tournament = text
                continue
            if '/spieler/' in href or '/player/' in href:
                norm_name = normalize_db_name(text)
                if norm_name and current_tournament != "Unknown": 
                    metadata[norm_name] = current_tournament
    except: 
        pass
    finally: 
        await page.close()
    return metadata

async def fetch_player_history_extended(player_last_name: str, limit: int = 80) -> List[Dict]:
    try:
        return supabase.table("market_odds").select("player1_name, player2_name, odds1, odds2, actual_winner_name, score, created_at, tournament, ai_analysis_text").or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(limit).execute().data or []
    except: 
        return []

async def fetch_tennisexplorer_stats(browser: Browser, relative_url: str, surface: str) -> float:
    if not relative_url: return 0.5
    cache_key = f"{relative_url}_{surface}"
    if cache_key in SURFACE_STATS_CACHE: return SURFACE_STATS_CACHE[cache_key]
    if not relative_url.startswith("/"): relative_url = f"/{relative_url}"
    page = await browser.new_page()
    try:
        await page.goto(f"https://www.tennisexplorer.com{relative_url}?annual=all&t={int(time.time())}", timeout=15000, wait_until="domcontentloaded")
        soup = BeautifulSoup(await page.content(), 'html.parser')
        target_header = "Clay" if "clay" in surface.lower() else "Grass" if "grass" in surface.lower() else "Indoors" if "indoor" in surface.lower() else "Hard"
        
        for table in soup.find_all('table', class_='result'):
            headers = [h.get_text(strip=True) for h in table.find_all('th')]
            if "Summary" in headers and target_header in headers:
                col_idx = headers.index(target_header)
                for row in table.find_all('tr'):
                    if "Summary" in row.get_text():
                        cols = row.find_all(['td', 'th'])
                        if len(cols) > col_idx and "/" in cols[col_idx].get_text(strip=True):
                            w, l = map(int, cols[col_idx].get_text(strip=True).split('/'))
                            if (w + l) > 0:
                                SURFACE_STATS_CACHE[cache_key] = w / (w + l)
                                return w / (w + l)
                break
    except: 
        pass
    finally: 
        await page.close()
    return 0.5

# L8 SOTA: THE RELIABLE RESULTS ENGINE
async def update_past_results(browser: Browser, players: List[Dict]):
    log("🏆 The Auditor: Checking Real-Time Results & Scores via TE...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending or not isinstance(pending, list): 
        return
    safe_to_check = list(pending)

    for day_off in range(0, 3): 
        t_date = datetime.now() - timedelta(days=day_off)
        page = await browser.new_page()
        try:
            await page.goto(f"https://www.tennisexplorer.com/results/?type=all&year={t_date.year}&month={t_date.month}&day={t_date.day}", wait_until="domcontentloaded", timeout=60000)
            soup = BeautifulSoup(await page.content(), 'html.parser')
            table = soup.find('table', class_='result')
            
            if not table: 
                continue
            
            def extract_row_data(r_row):
                cells = r_row.find_all('td')
                p_idx = next((idx for idx, c in enumerate(cells) if c.find('a') and 'time' not in c.get('class', [])), -1)
                
                if p_idx != -1 and p_idx + 1 < len(cells) and cells[p_idx + 1].get_text(strip=True).isdigit():
                    scores = []
                    for k in range(1, 6):
                        if p_idx + 1 + k >= len(cells): break
                        raw_txt = re.sub(r'<[^>]+>', '', "".join(str(c).strip() if isinstance(c, str) else c.get_text(strip=True) for c in cells[p_idx + 1 + k].children if c.name != 'sup')).strip()
                        if raw_txt.isdigit(): 
                            scores.append(raw_txt)
                        else: 
                            break
                    return int(cells[p_idx + 1].get_text(strip=True)), scores
                return -1, []

            def match_player_db_te(db_full_name, te_last, te_init):
                db_n = normalize_db_name(db_full_name)
                db_last, db_first = db_n.split()[-1] if db_n.split() else "", db_n.split()[0] if len(db_n.split()) > 1 else ""
                
                if te_last == db_last or set(db_n.split()).intersection(set(te_last.split())) or (len(te_last) >= 5 and te_last in db_n) or (len(db_last) >= 5 and db_last in te_last):
                    if te_init and db_first: 
                        return db_first.startswith(te_init)
                    return True
                return False

            rows, i, pending_p1_raw, pending_p1_row = table.find_all('tr'), 0, None, None
            
            while i < len(rows):
                row = rows[i]
                
                if 'head' in row.get('class', []): 
                    pending_p1_raw = None
                    i += 1
                    continue
                    
                cols = row.find_all('td')
                
                if len(cols) < 2: 
                    i += 1
                    continue
                    
                p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
                
                if not p_cell: 
                    i += 1
                    continue
                    
                p_raw = clean_player_name(p_cell.get_text(strip=True))
                
                if pending_p1_raw:
                    if '/' in pending_p1_raw or '/' in p_raw: 
                        pending_p1_raw = None
                        i += 1
                        continue
                        
                    te_last1, te_init1 = parse_te_name(pending_p1_raw)
                    te_last2, te_init2 = parse_te_name(p_raw)
                    matched_pm, is_reversed = None, False
                    
                    for pm in list(safe_to_check):
                        if match_player_db_te(pm['player1_name'], te_last1, te_init1) and match_player_db_te(pm['player2_name'], te_last2, te_init2):
                            matched_pm = pm
                            break
                        elif match_player_db_te(pm['player1_name'], te_last2, te_init2) and match_player_db_te(pm['player2_name'], te_last1, te_init1):
                            matched_pm, is_reversed = pm, True
                            break

                    if matched_pm:
                        s1, scores1 = extract_row_data(pending_p1_row)
                        s2, scores2 = extract_row_data(row)
                        winner, final_score = None, ""
                        is_ret = "ret." in pending_p1_row.get_text().lower() or "ret." in row.get_text().lower() or "w.o." in pending_p1_row.get_text().lower() or "w.o." in row.get_text().lower()
                        
                        if s1 != -1 and s2 != -1:
                            winner = matched_pm['player2_name' if is_reversed else 'player1_name'] if s1 > s2 else matched_pm['player1_name' if is_reversed else 'player2_name']
                            score_parts = [f"{scores2[k]}-{scores1[k]}" if is_reversed else f"{scores1[k]}-{scores2[k]}" for k in range(min(len(scores1), len(scores2)))]
                            if score_parts: 
                                final_score = " ".join(score_parts) + (" ret." if is_ret else "")

                        if winner:
                            log(f"      🔍 AUDITOR FOUND: {final_score} -> Winner: {winner}")
                            supabase.table("market_odds").update({"actual_winner_name": winner, "score": final_score}).eq("id", matched_pm['id']).execute()

                            p1_id = next((p['id'] for p in players if p.get('last_name') == matched_pm['player1_name']), None)
                            p2_id = next((p['id'] for p in players if p.get('last_name') == matched_pm['player2_name']), None)
                            
                            if p1_id and p2_id:
                                try:
                                    db_skills = supabase.table('player_skills').select('*').in_('player_id', [p1_id, p2_id]).execute().data or []
                                    p1_skills_db = next((s for s in db_skills if s.get('player_id') == p1_id), None)
                                    p2_skills_db = next((s for s in db_skills if s.get('player_id') == p2_id), None)
                                    
                                    if p1_skills_db:
                                        new_s1 = LiveSkillEngine.calculate_new_skills(p1_skills_db, to_float(matched_pm.get('odds1', 1.85)), winner == matched_pm['player1_name'], final_score, True)
                                        if new_s1: 
                                            supabase.table('player_skills').update({**new_s1, 'updated_at': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}).eq('player_id', p1_id).execute()
                                    if p2_skills_db:
                                        new_s2 = LiveSkillEngine.calculate_new_skills(p2_skills_db, to_float(matched_pm.get('odds2', 1.85)), winner == matched_pm['player2_name'], final_score, False)
                                        if new_s2: 
                                            supabase.table('player_skills').update({**new_s2, 'updated_at': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}).eq('player_id', p2_id).execute()
                                except Exception as se: 
                                    log(f"⚠️ Live Skill Update Error: {se}")

                            for p_name_hook in [matched_pm['player1_name'], matched_pm['player2_name']]:
                                p_hist = await fetch_player_history_extended(p_name_hook, limit=80)
                                supabase.table('players').update({'surface_ratings': SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name_hook), 'form_rating': MomentumV2Engine.calculate_rating(p_hist[:20], p_name_hook)}).ilike('last_name', f"%{p_name_hook}%").execute()
                                
                        safe_to_check = [x for x in safe_to_check if x['id'] != matched_pm['id']]
                        
                    pending_p1_raw = None
                else:
                    first_cell = row.find('td', class_='first')
                    if first_cell and first_cell.get('rowspan') == '2': 
                        pending_p1_raw, pending_p1_row = p_raw, row
                    else: 
                        pending_p1_raw, pending_p1_row = p_raw, row
                i += 1
        except: 
            pass
        finally: 
            await page.close()

async def get_advanced_load_analysis(matches: List[Dict]) -> str:
    try:
        if not matches[:5]: 
            return "Fresh (No recent data)"
            
        now_ts = datetime.now().timestamp()
        fatigue_score, details = 0.0, []
        
        try: 
            hours_since_last = (now_ts - datetime.fromisoformat(matches[0]['created_at'].replace('Z', '+00:00')).timestamp()) / 3600
        except: 
            return "Unknown"
            
        if hours_since_last < 24: 
            fatigue_score += 50
            details.append("Back-to-back match")
        elif hours_since_last < 48: 
            fatigue_score += 25
            details.append("Short rest")
        elif hours_since_last > 336: 
            return "Rusty (2+ weeks break)"
            
        if hours_since_last < 72 and matches[0].get('score'):
            score_str = str(matches[0]['score']).lower()
            if 'ret' in score_str or 'wo' in score_str: 
                fatigue_score *= 0.5
            else:
                sets = len(re.findall(r'(\d+)-(\d+)', score_str))
                tiebreaks = len(re.findall(r'7-6|6-7', score_str))
                total_games = sum(int(s[0]) + int(s[1]) for s in re.findall(r'(\d+)-(\d+)', score_str) if s[0].isdigit() and s[1].isdigit())
                
                if sets >= 3: 
                    fatigue_score += 20
                    details.append("Last match 3+ sets")
                if total_games > 30: 
                    fatigue_score += 15
                    details.append("Marathon match (>30 games)")
                if tiebreaks > 0: 
                    fatigue_score += 5 * tiebreaks
                    details.append(f"{tiebreaks} Tiebreaks played")
                    
        matches_in_week, sets_in_week = 0, 0
        for m in matches[:5]:
            try:
                if (now_ts - datetime.fromisoformat(m['created_at'].replace('Z', '+00:00')).timestamp()) < (7 * 24 * 3600):
                    matches_in_week += 1
                    if m.get('score'): 
                        sets_in_week += len(re.findall(r'\d+-\d+', str(m['score'])))
            except: 
                pass
                
        if matches_in_week >= 4: 
            fatigue_score += 20
            details.append(f"Busy week ({matches_in_week} matches)")
        if sets_in_week > 10: 
            fatigue_score += 15
            details.append(f"Heavy leg load ({sets_in_week} sets in 7 days)")
            
        status = "CRITICAL FATIGUE" if fatigue_score > 75 else "Heavy Legs" if fatigue_score > 50 else "In Rhythm (Active)" if fatigue_score > 30 else "Fresh"
        return f"{status} [{', '.join(details)}]" if details else status
    except: 
        return "Unknown"

async def fetch_elo_ratings(browser: Browser):
    log("📊 Lade Elo Ratings...")
    for tour, url in {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}.items():
        page = await browser.new_page()
        try:
            await page.goto(f"{url}?t={int(time.time())}", wait_until="domcontentloaded", timeout=60000)
            table = BeautifulSoup(await page.content(), 'html.parser').find('table', {'id': 'reportable'})
            if table:
                for row in table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) > 4:
                        name = normalize_text(cols[0].get_text(strip=True)).lower()
                        ELO_CACHE[tour][name.split()[-1] if " " in name else name] = {'Hard': to_float(cols[3].get_text(strip=True), 1500), 'Clay': to_float(cols[4].get_text(strip=True), 1500), 'Grass': to_float(cols[5].get_text(strip=True), 1500)}
        except: 
            pass
        finally: 
            await page.close()

async def get_db_data():
    try:
        weights_res = supabase.table("ai_system_weights").select("*").execute()
        if weights_res.data:
            for w in weights_res.data:
                tour = w.get("tour", "ATP")
                DYNAMIC_WEIGHTS[tour] = {"SKILL": to_float(w.get("weight_skill"), 0.50), "FORM": to_float(w.get("weight_form"), 0.35), "SURFACE": to_float(w.get("weight_surface"), 0.15), "MC_VARIANCE": to_float(w.get("mc_variance"), 1.20)}

        players = supabase.table("players").select("*").execute().data or []
        skills = supabase.table("player_skills").select("*").execute().data or []
        reports = supabase.table("scouting_reports").select("*").execute().data or []
        tournaments = supabase.table("tournaments").select("*").execute().data or []
        
        for t in tournaments:
            t_name, t_loc, t_surf = clean_tournament_name(t.get('name', '')), t.get('location', ''), t.get('surface', 'Unknown')
            if t_name and t_surf: GLOBAL_SURFACE_MAP[t_name.lower()] = t_surf
            if t_loc and t_surf:
                for part in t_loc.split(','):
                    if part.strip().lower() and len(part.strip()) > 2: 
                        GLOBAL_SURFACE_MAP[part.strip().lower()] = t_surf
                            
        clean_skills = {s['player_id']: {k: to_float(s.get(k, 50)) for k in ['serve', 'power', 'forehand', 'backhand', 'volley', 'speed', 'stamina', 'mental', 'overall_rating']} for s in skills if isinstance(s, dict) and s.get('player_id')}
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# 8. MATH CORE
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float: 
    return 1 / (1 + math.exp(-sensitivity * diff))
    
def normal_cdf_prob(elo_diff: float, sigma: float = 280.0) -> float: 
    return 0.5 * (1 + math.erf(elo_diff / (sigma * math.sqrt(2))))

def calculate_value_metrics(fair_prob: float, market_odds: float) -> Dict[str, Any]:
    if market_odds <= 1.01 or fair_prob <= 0: 
        return {"type": "NONE", "edge_percent": 0.0, "is_value": False}
    edge_percent = round(((fair_prob * min(market_odds, 100.0)) - 1) * 100, 1)
    if edge_percent <= 0.5: 
        return {"type": "NONE", "edge_percent": edge_percent, "is_value": False}
    return {"type": "🔥 HIGH VALUE" if edge_percent >= 15.0 else "✨ GOOD VALUE" if edge_percent >= 8.0 else "📈 THIN VALUE" if edge_percent >= 2.0 else "👀 WATCH", "edge_percent": edge_percent, "is_value": True}

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, mc_prob_a, market_odds1, market_odds2):
    tour, n1, n2 = "ATP", get_last_name(p1_name), get_last_name(p2_name)
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo_diff_model = ELO_CACHE.get(tour, {}).get(n1, {}).get(elo_surf, 1500) - ELO_CACHE.get(tour, {}).get(n2, {}).get(elo_surf, 1500)
    
    if market_odds1 > 0 and market_odds2 > 0:
        implied_p1 = (1/market_odds1) / ((1/market_odds1) + (1/market_odds2))
        try: 
            elo_diff_market = -400 * math.log10(1/implied_p1 - 1) if 0.01 < implied_p1 < 0.99 else elo_diff_model
        except: 
            elo_diff_market = elo_diff_model
        elo_diff_final = (elo_diff_model * 0.70) + (elo_diff_market * 0.30)
    else: 
        elo_diff_final = elo_diff_model
        
    prob_elo = normal_cdf_prob(elo_diff_final, sigma=280.0)
    prob_alpha = (prob_elo * 0.35) + (max(0.01, min(0.99, mc_prob_a / 100.0)) * 0.65)
    
    prob_market = (1/market_odds1) / ((1/market_odds1) + (1/market_odds2)) if market_odds1 > 1 and market_odds2 > 1 else 0.5
    return (prob_alpha * 0.35) + (prob_market * 0.65)

def recalculate_fair_odds_with_new_market(old_fair_odds1: float, old_market_odds1: float, old_market_odds2: float, new_market_odds1: float, new_market_odds2: float) -> float:
    try:
        old_prob_market = (1/old_market_odds1) / ((1/old_market_odds1) + (1/old_market_odds2)) if old_market_odds1 > 1 and old_market_odds2 > 1 else 0.5
        if old_fair_odds1 <= 1.01: 
            return 0.5
        new_prob_market = (1/new_market_odds1) / ((1/new_market_odds1) + (1/new_market_odds2)) if new_market_odds1 > 1 and new_market_odds2 > 1 else 0.5
        new_final_prob = ((((1 / old_fair_odds1) - (old_prob_market * 0.40)) / 0.60) * 0.60) + (new_prob_market * 0.40)
        return (new_final_prob * 0.15) + ((1/new_market_odds1) * 0.85) if new_market_odds1 < 1.10 else new_final_prob
    except: 
        return 0.5

async def find_best_court_match_smart(tour, db_tours, p1, p2, p1_country="Unknown", p2_country="Unknown", match_date: datetime = None): 
    s_low = clean_tournament_name(tour).lower().strip()
    best_match, best_score = None, 0
    for t in db_tours:
        score = calculate_fuzzy_score(s_low, t['name'])
        if score > best_score: 
            best_score, best_match = score, t
            
    if best_match and best_score >= 20: 
        loc = best_match.get('location', '')
        return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', ''), loc.split(',')[0] if loc else best_match['name'], best_match['name']
        
    return 'Hard Court Outdoor', 6.5, 'Fallback', tour.split()[0], tour

def format_skills(s: Dict) -> str:
    if not s: 
        return "No granular skill data."
    return f"Serve: {s.get('serve', 50)}, FH: {s.get('forehand', 50)}, BH: {s.get('backhand', 50)}, Volley: {s.get('volley', 50)}, Speed: {s.get('speed', 50)}, Stamina: {s.get('stamina', 50)}, Power: {s.get('power', 50)}, Mental: {s.get('mental', 50)}, OVR: {s.get('overall_rating', 50)}"

async def analyze_match_with_ai(tour_name, p1, p2, s1, s2, report1, report2, surface, bsi, notes, form1_data, form2_data, weather_data, p1_surface_profile, p2_surface_profile, mc_results):
    fatigueA = await get_advanced_load_analysis(await fetch_player_history_extended(p1['last_name'], 10))
    fatigueB = await get_advanced_load_analysis(await fetch_player_history_extended(p2['last_name'], 10))
    
    weather_str = f"WEATHER: {weather_data['summary']}. IMPACT: {weather_data['impact_note']}" if weather_data else "Weather: Neutral/No Data."
    current_surf_key = SurfaceIntelligence.normalize_surface_key(surface)
    
    p1_s_rating = p1_surface_profile.get(current_surf_key, {}).get('rating', 5.0)
    p2_s_rating = p2_surface_profile.get(current_surf_key, {}).get('rating', 5.0)
    
    scoutA = f"Strengths: {report1.get('strengths', 'Unknown')}. Weakness: {report1.get('weaknesses', 'Unknown')}." if report1 else "No scouting report available for Player A."
    scoutB = f"Strengths: {report2.get('strengths', 'Unknown')}. Weakness: {report2.get('weaknesses', 'Unknown')}." if report2 else "No scouting report available for Player B."
    validCourtNotes = notes if notes else "No specific court physics or bounce data provided."
    
    aWins = mc_results['probA'] > mc_results['probB']
    predictedMCWinner = p1['last_name'] if aWins else p2['last_name']
    predictedMCLoser = p2['last_name'] if aWins else p1['last_name']
    finalProb_val = float(mc_results['probA']) if aWins else float(mc_results['probB'])
    
    sys_acc = SYSTEM_ACCURACY.get("WTA" if "WTA" in tour_name.upper() else "ATP", 65.0)
    
    convictionDirective = ""
    if finalProb_val >= 65.0:
        convictionDirective = f"*** CONVICTION DIRECTIVE (CRITICAL) ***\nThe mathematical model gives {predictedMCWinner} a massive {finalProb_val:.1f}% probability of winning because of a clear baseline talent mismatch. You MUST write this analysis with HIGH CONVICTION. Do not write \"If he can...\". State confidently that {predictedMCWinner}'s overall quality and baseline strengths will overwhelm {predictedMCLoser}. Explain exactly why {predictedMCLoser}'s game will break down. NO HEDGING." 
    elif finalProb_val >= 58.0:
        convictionDirective = f"*** CONVICTION DIRECTIVE ***\nThe mathematical model gives {predictedMCWinner} a clear edge ({finalProb_val:.1f}%). Write confidently about why {predictedMCWinner} is the favorite to win, focusing on the tactical mismatch. Avoid 50/50 language." 
    else:
        convictionDirective = f"*** CONVICTION DIRECTIVE ***\nThe mathematical model sees this as a tight battle ({finalProb_val:.1f}% for {predictedMCWinner}). Write an analysis highlighting the fine margins that will decide this match."

    prompt = f"""You are an elite Senior Tennis Analyst (Style: Gil Gross). 
    *** SYSTEM SELF-REFLECTION (CRITICAL) ***
    Our internal neural network has an active prediction accuracy of {sys_acc}%. 
    *** DATA GROUNDING (SOURCE OF TRUTH) ***
    Player A ({p1['last_name']}): Style: {p1.get('play_style', 'Unknown')} | Form: {form1_data['text']} | Surface Rating: {p1_s_rating}/10 | Granular Skills: {format_skills(s1)} | Scouting: {scoutA} | Fatigue: {fatigueA}
    Player B ({p2['last_name']}): Style: {p2.get('play_style', 'Unknown')} | Form: {form2_data['text']} | Surface Rating: {p2_s_rating}/10 | Granular Skills: {format_skills(s2)} | Scouting: {scoutB} | Fatigue: {fatigueB}
    Match Conditions: Surface: {surface} (BSI: {bsi}) | Notes: {validCourtNotes} | {weather_str}
    *** INTERNAL MATCHUP DATA ***
    Winner: {predictedMCWinner} (Win Probability: {finalProb_val:.1f}%)
    {convictionDirective}
    *** CRITICAL DIRECTIVES (MUST OBEY) ***
    1. NO NUMBERS IN TEXT: Strictly forbidden to use percentages, numerical ratings, or skill points in 'prediction_text' and 'key_factor'.
    2. TACTICAL PROSA: Use Gil Gross style "Matchup Physics".
    3. FACTUAL INTEGRITY: Ground analysis in the provided 'Weaknesses'.
    4. DO NOT EXPLAIN CALCULATIONS: Output strictly the JSON. No introductory chatter.
    OUTPUT JSON:
    {{"winner_prediction": "{predictedMCWinner}", "key_factor": "One sharp tactical sentence focusing on the primary technical mismatch (NO NUMBERS).", "prediction_text": "Deep Gil Gross style analysis (~200 words). Focus on tactical matchup physics, court conditions, and how the scouting report details manifest on court. STRICTLY NO NUMBERS OR PERCENTAGES.", "tactical_bullets": ["Tactic 1 based on report", "Tactic 2 based on report", "Tactic 3 based on report"]}}"""
    
    res = await call_openrouter(prompt)
    if not res: 
        return {'ai_text': f"Analysis unavailable for {p1['last_name']} vs {p2['last_name']}.", 'mc_prob_a': mc_results['probA']}
        
    try:
        data = ensure_dict(json.loads(res.replace("json", "").replace("```", "").strip()))
        return {
            'ai_text': f"🔑 {data.get('key_factor', '')}\n\n📝 {data.get('prediction_text', '')}\n\n🎯 Tactical Keys:\n" + "\n".join([f"- {b}" for b in data.get('tactical_bullets', [])]), 
            'mc_prob_a': mc_results['probA']
        }
    except: 
        return {'ai_text': f"Analysis unavailable for {p1['last_name']} vs {p2['last_name']}.", 'mc_prob_a': mc_results['probA']}

# =================================================================
# PIPELINE EXECUTION (MASTER-SLAVE ARCHITECTURE)
# =================================================================
async def run_pipeline():
    log(f"🚀 Neural Scout V155.0 (MASTER-SLAVE ARCHITECTURE) Starting...")
    
    TARGET_TABLE = "market_odds_demo" # 🔴 SOTA: Demo/Staging Environment für den Test
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: 
                return
            
            await update_past_results(browser, players)
            
            try: 
                NeuralOptimizer.trigger_learning_cycle(players, all_skills)
            except Exception as e: 
                log(f"⚠️ Optimizer Error: {e}")
            
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            
            # 🔴 SOTA PHASE 1: MASTER SCHEDULE (TE)
            master_schedule = await fetch_te_master_schedule(browser)
            
            # 🔴 SOTA PHASE 2: SLAVE FEED (1WIN RAW HTML)
            onewin_lines = await fetch_1win_raw_lines(browser)
            
            log(f"🔍 Starte Synergy Merge: Matche Master-Zeiten mit 1Win-Quoten...")
            db_matched_count = 0
            
            # 🔴 SOTA PHASE 3: THE MERGE
            for match in master_schedule:
                try:
                    await asyncio.sleep(0.1) 
                    p1_name = match['p1']
                    p2_name = match['p2']
                    match_time = match['time']
                    tour_name = match['tour']
                    
                    # Fuzzy Suche im rohen 1Win HTML Text
                    o1, o2 = find_odds_in_lines(p1_name, p2_name, onewin_lines)
                    
                    if not o1 or not o2 or not validate_market_integrity(o1, o2):
                        continue # 1Win bietet dieses Match nicht an oder Quoten kaputt
                        
                    p1_obj = find_player_smart(match['p1_raw'], players, report_ids)
                    p2_obj = find_player_smart(match['p2_raw'], players, report_ids)
                    
                    if p1_obj == "TIE_BREAKER" or p2_obj == "TIE_BREAKER" or not p1_obj or not p2_obj:
                        continue
                        
                    n1, n2 = p1_obj['last_name'], p2_obj['last_name']
                    if n1 == n2: 
                        continue
                        
                    db_matched_count += 1

                    res1 = supabase.table(TARGET_TABLE).select("*").eq("player1_name", n1).eq("player2_name", n2).order("created_at", desc=True).limit(1).execute()
                    existing_match = res1.data[0] if res1.data else None
                    
                    if not existing_match:
                        res2 = supabase.table(TARGET_TABLE).select("*").eq("player1_name", n2).eq("player2_name", n1).order("created_at", desc=True).limit(1).execute()
                        if res2.data:
                            existing_match = res2.data[0]
                            n1, n2, p1_obj, p2_obj, o1, o2 = n2, n1, p2_obj, p1_obj, o2, o1
                    
                    if existing_match and is_suspicious_movement(to_float(existing_match.get('odds1'), 0), o1, to_float(existing_match.get('odds2'), 0), o2): 
                        continue
                        
                    if existing_match and existing_match.get('actual_winner_name'): 
                        continue 
                        
                    db_match_id = existing_match['id'] if existing_match else None

                    hist_fair1, hist_fair2, hist_is_value, hist_pick_player = 0, 0, False, None
                    is_signal_locked = has_active_signal(existing_match.get('ai_analysis_text', '')) if existing_match else False
                    
                    if is_signal_locked:
                        log(f"      🔒 DIAMOND LOCK ACTIVE: {n1} vs {n2}")
                        update_data = {"odds1": o1, "odds2": o2, "match_time": match_time, "is_visible_in_scanner": True}
                        if to_float(existing_match.get('opening_odds1'), 0) <= 1.01 and o1 > 1.01: 
                            update_data["opening_odds1"], update_data["opening_odds2"] = o1, o2
                        try: 
                            supabase.table(TARGET_TABLE).update(update_data).eq("id", db_match_id).execute()
                        except: 
                            pass
                        hist_fair1, hist_fair2 = to_float(existing_match.get('ai_fair_odds1'), 0), to_float(existing_match.get('ai_fair_odds2'), 0)
                        
                    else:
                        cached_ai = {'ai_text': existing_match.get('ai_analysis_text'), 'ai_fair_odds1': existing_match.get('ai_fair_odds1'), 'old_odds1': existing_match.get('odds1', 0), 'old_odds2': existing_match.get('odds2', 0), 'last_update': existing_match.get('created_at')} if existing_match and existing_match.get('ai_analysis_text') else {}
                        
                        surf, bsi, notes, city_for_weather, matched_tour_name = await find_best_court_match_smart(tour_name, all_tournaments, n1, n2, p1_obj.get('country', 'Unknown'), p2_obj.get('country', 'Unknown'), match_date=datetime.now())
                        weather_data = await fetch_weather_data(city_for_weather)
                        s1, s2 = all_skills.get(p1_obj['id'], {}), all_skills.get(p2_obj['id'], {})
                        report1, report2 = next((r for r in all_reports if r.get('player_id') == p1_obj['id']), None), next((r for r in all_reports if r.get('player_id') == p2_obj['id']), None)
                        
                        full_n1, full_n2 = f"{p1_obj.get('first_name', '')} {p1_obj.get('last_name', '')}".strip(), f"{p2_obj.get('first_name', '')} {p2_obj.get('last_name', '')}".strip()
                        
                        # 🔴 WICHTIG: Die Historie liest weiterhin aus der ECHTEN market_odds Tabelle, um korrekte Ratings zu berechnen!
                        p1_history, p2_history = await fetch_player_history_extended(n1, limit=80), await fetch_player_history_extended(n2, limit=80)
                        
                        p1_surface_profile, p2_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p1_history, full_n1), SurfaceIntelligence.compute_player_surface_profile(p2_history, full_n2)
                        p1_form_v2, p2_form_v2 = MomentumV2Engine.calculate_rating(p1_history[:20], full_n1), MomentumV2Engine.calculate_rating(p2_history[:20], full_n2)

                        should_run_ai = True
                        if db_match_id and cached_ai and not (existing_match.get('is_visible_in_scanner') is False or "[BACKGROUND DATA]" in existing_match.get('ai_analysis_text', '')):
                            try: 
                                is_stale = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_ai.get('last_update', '').replace('Z', '+00:00'))) > timedelta(hours=6)
                            except: 
                                is_stale = True
                            if max(abs(cached_ai['old_odds1'] - o1), abs(cached_ai['old_odds2'] - o2)) <= (o1 * 0.05) and not is_stale: 
                                should_run_ai = False
                            
                        if not should_run_ai:
                            new_prob = recalculate_fair_odds_with_new_market(cached_ai['ai_fair_odds1'], cached_ai['old_odds1'], cached_ai['old_odds2'], o1, o2)
                            fair1, fair2 = round(1/new_prob, 2) if new_prob > 0.01 else 99, round(1/(1-new_prob), 2) if new_prob < 0.99 else 99
                            val_p1, val_p2 = calculate_value_metrics(1/fair1, o1), calculate_value_metrics(1/fair2, o2)
                            
                            value_tag = ""
                            if val_p1["is_value"]: 
                                value_tag = f"\n\n[{val_p1['type']}: {n1} @ {o1} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                hist_is_value, hist_pick_player = True, n1
                            elif val_p2["is_value"]: 
                                value_tag = f"\n\n[{val_p2['type']}: {n2} @ {o2} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                hist_is_value, hist_pick_player = True, n2
                                
                            hist_fair1, hist_fair2 = fair1, fair2
                            if db_match_id:
                                try: 
                                    supabase.table(TARGET_TABLE).update({"odds1": o1, "odds2": o2, "ai_fair_odds1": fair1, "ai_fair_odds2": fair2, "ai_analysis_text": re.sub(r'\[VALUE.*?\]', '', cached_ai['ai_text']).strip() + value_tag, "match_time": match_time, "is_visible_in_scanner": True}).eq("id", db_match_id).execute()
                                except: 
                                    pass
                        else:
                            log(f"   🧠 Fresh AI & Markov Chain Sim: {n1} vs {n2} | T: {matched_tour_name}")
                            sim_result = QuantumGamesSimulator.run_simulation(s1, s2, bsi, surf)
                            mc_results = MarkovChainEngine.run_simulation(s1=s1, s2=s2, formA=p1_form_v2['score'], formB=p2_form_v2['score'], bsi=bsi, styleA=p1_obj.get('play_style', ''), styleB=p2_obj.get('play_style', ''), iterations=2500)
                            ai = await analyze_match_with_ai(matched_tour_name, p1_obj, p2_obj, s1, s2, report1, report2, surf, bsi, notes, p1_form_v2, p2_form_v2, weather_data, p1_surface_profile, p2_surface_profile, mc_results)
                            prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai['mc_prob_a'], o1, o2)
                            
                            fair1, fair2 = round(1/prob, 2) if prob > 0.01 else 99, round(1/(1-prob), 2) if prob < 0.99 else 99
                            val_p1, val_p2 = calculate_value_metrics(1/fair1, o1), calculate_value_metrics(1/fair2, o2)
                            
                            value_tag = ""
                            if val_p1["is_value"]: 
                                value_tag = f"\n\n[{val_p1['type']}: {n1} @ {o1} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                hist_is_value, hist_pick_player = True, n1
                            elif val_p2["is_value"]: 
                                value_tag = f"\n\n[{val_p2['type']}: {n2} @ {o2} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                hist_is_value, hist_pick_player = True, n2
                            
                            data = {"player1_name": n1, "player2_name": n2, "tournament": matched_tour_name, "ai_fair_odds1": fair1, "ai_fair_odds2": fair2, "ai_analysis_text": f"{ai['ai_text']} {value_tag}\n[🎲 SIM: {sim_result['predicted_line']} Games]", "games_prediction": sim_result, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "match_time": match_time, "odds1": o1, "odds2": o2, "is_visible_in_scanner": True}
                            hist_fair1, hist_fair2 = fair1, fair2
                            
                            if db_match_id: 
                                try: 
                                    supabase.table(TARGET_TABLE).update(data).eq("id", db_match_id).execute()
                                except: 
                                    pass
                            else:
                                if to_float(existing_match.get('opening_odds1') if existing_match else 0, 0) <= 1.01 and o1 > 1.01: 
                                    data["opening_odds1"], data["opening_odds2"] = o1, o2
                                try:
                                    res_ins = supabase.table(TARGET_TABLE).insert(data).execute()
                                    if res_ins.data: 
                                        db_match_id = res_ins.data[0]['id']
                                    log(f"💾 Saved: {n1} vs {n2} - Odds: {o1} | {o2}")
                                except: 
                                    pass

                    if db_match_id and (not existing_match or is_signal_locked or hist_is_value or round(to_float(existing_match.get('odds1'), 0), 3) != round(o1, 3) or round(to_float(existing_match.get('odds2'), 0), 3) != round(o2, 3)):
                        try: 
                            supabase.table("odds_history").insert({"match_id": db_match_id, "odds1": o1, "odds2": o2, "fair_odds1": hist_fair1, "fair_odds2": hist_fair2, "is_hunter_pick": (hist_is_value or is_signal_locked), "pick_player_name": "LOCKED" if is_signal_locked else hist_pick_player, "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}).execute()
                        except: 
                            pass
                except Exception as e: 
                    log(f"⚠️ Match Error bei {match['p1']} vs {match['p2']}: {e}")
                    
            log(f"📊 SUMMARY: {db_matched_count} Matches durch Synergy-Merge prozessiert.")
        finally: 
            await browser.close()
            
    try: 
        FantasySettlementEngine.run_settlement()
    except Exception as e: 
        log(f"⚠️ FANTASY ENGINE ERROR: {e}")
        
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
