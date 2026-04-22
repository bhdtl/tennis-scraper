# -- coding: utf-8 --

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
import numpy as np # SOTA: Required for Neural Grid Search

import httpx
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# 🚀 SOTA FIX: Schalldämpfer für externe Bibliotheken (verhindert den "1000 HTTP Requests" Spam)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("postgrest").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("NeuralScout_Architect")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V205.8 - VARIANCE PENALTY & RISK MANAGEMENT EDITION)...")

# Secrets Load
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
API_TENNIS_KEY = os.environ.get("API_TENNIS_KEY") # 🚀 SOTA API KEY

if not OPENROUTER_API_KEY or not SUPABASE_URL or not SUPABASE_KEY or not API_TENNIS_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub/OpenRouter/API_TENNIS Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# SOTA Model Selection
MODEL_NAME = 'meta-llama/llama-3.3-70b-instruct'

# Global Caches & Dynamic Memory
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
METADATA_CACHE: Dict[str, Any] = {} 
WEATHER_CACHE: Dict[str, Any] = {} 
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 
TML_MATCH_CACHE: List[Dict] = [] 

# 🚀 SOTA: TRUE POPULATION STATISTICS (Z-SCORE BASELINES EXTRACTED FROM DB)
SKILL_MEANS = {'serve': 77.6, 'forehand': 78.7, 'backhand': 76.4, 'volley': 72.4, 'speed': 79.4, 'stamina': 77.4, 'power': 75.5, 'mental': 76.8, 'overall_rating': 76.6}
SKILL_STDS = {'serve': 8.3, 'forehand': 7.2, 'backhand': 8.1, 'volley': 8.8, 'speed': 6.8, 'stamina': 6.8, 'power': 8.8, 'mental': 7.0, 'overall_rating': 6.6}

# SELF LEARNING STATE MEMORY
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
# 1.1 TENNIS DATA API CLIENT (THE NEW SOTA ENGINE)
# =================================================================
class TennisDataAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.api-tennis.com/tennis/"
        # 🚀 SOTA: Wir covern jetzt ATP, WTA, Challenger UND ITF
        self.valid_tours = [
            "Atp Singles", 
            "Wta Singles", 
            "Challenger Men Singles",
            "Challenger Women Singles",  
            "Itf Men Singles",            
            "Itf Women Singles"          
        ]

    async def get_fixtures(self, date_str: str) -> List[Dict]:
        """Holt saubere JSON-Fixtures für ein bestimmtes Datum."""
        log(f"📡 [API] Fetching Fixtures for {date_str}...")
        url = f"{self.base_url}?method=get_fixtures&APIkey={self.api_key}&date_start={date_str}&date_stop={date_str}&timezone=UTC"
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, timeout=25.0)
                data = res.json()
                if data.get('success') == 1 and data.get('result'):
                    fixtures = [m for m in data['result'] if m.get('event_type_type') in self.valid_tours]
                    log(f"✅ [API] Found {len(fixtures)} relevant fixtures for {date_str}.")
                    return fixtures
            except Exception as e:
                log(f"❌ [API] Fixture Request Failed: {e}")
        return []

    async def get_odds(self, match_key: str) -> Dict:
        """Holt die Pre-Match-Quoten für eine bestimmte Match-ID."""
        url = f"{self.base_url}?method=get_odds&APIkey={self.api_key}&match_key={match_key}"
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, timeout=15.0)
                data = res.json()
                if data.get('success') == 1 and data.get('result'):
                    return data['result'].get(str(match_key), {})
            except Exception as e:
                pass
        return {}

    async def get_player_stats(self, player_key: str) -> Dict:
        """🚀 SOTA: Holt tiefe historische API-Daten (W/L auf Belägen, Rank etc.) eines Spielers."""
        url = f"{self.base_url}?method=get_players&APIkey={self.api_key}&player_key={player_key}"
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, timeout=15.0)
                data = res.json()
                if data.get('success') == 1 and data.get('result'):
                    return data['result'][0] if isinstance(data['result'], list) and len(data['result']) > 0 else {}
            except Exception as e:
                pass
        return {}

    async def get_h2h(self, p1_key: str, p2_key: str) -> Dict:
        """🚀 JUICE REEL FEATURE: Holt offizielle H2H Daten für zwei Spieler."""
        url = f"{self.base_url}?method=get_H2H&APIkey={self.api_key}&first_player_key={p1_key}&second_player_key={p2_key}"
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, timeout=15.0)
                data = res.json()
                if data.get('success') == 1 and data.get('result'):
                    return data['result']
            except:
                pass
        return {}

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

# 🚀 SOTA ARCHITECT: The Ultimate Name-Parsing Gatekeeper
def is_same_player(target_name: str, db_name: str) -> bool:
    t_norm = normalize_db_name(target_name)
    d_norm = normalize_db_name(db_name)
    if t_norm == d_norm: return True
    
    t_parts = t_norm.split()
    d_parts = d_norm.split()
    if not t_parts or not d_parts: return False
    
    if t_parts[-1] != d_parts[-1]: return False
    
    t_first = t_parts[0] if len(t_parts) > 1 else ""
    d_first = d_parts[0] if len(d_parts) > 1 else ""
    
    if t_first and d_first:
        return t_first[0] == d_first[0]
        
    return True

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str] = None) -> Optional[Dict]:
    if report_ids is None: report_ids = set()
    
    if not scraped_name_raw or len(scraped_name_raw) < 3 or re.search(r'\d', scraped_name_raw): return None 
    
    bad_words = ['satz', 'set', 'game', 'über', 'unter', 'handicap', 'sieger', 'winner', 'tennis', 'live', 'stream', 'stats', 'tv']
    if any(w in scraped_name_raw.lower() for w in bad_words): return None

    clean_scrape = normalize_db_name(clean_player_name(scraped_name_raw))
    scrape_tokens = clean_scrape.split()
    if not scrape_tokens: return None

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
        top_score = candidates[0][1]
        second_score = candidates[1][1]
        
        if top_score == second_score:
            p1_n = f"{candidates[0][0].get('first_name')} {candidates[0][0].get('last_name')}"
            p2_n = f"{candidates[1][0].get('first_name')} {candidates[1][0].get('last_name')}"
            log(f"🚨 TIE-BREAKER ALARM: '{scraped_name_raw}' ist mehrdeutig zwischen {p1_n} und {p2_n}. Match wird sicherheitshalber ignoriert!")
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

            result = {
                "summary": f"{temp}°C, {hum}% Hum, Wind: {wind} km/h", 
                "impact_note": impact
            }
            WEATHER_CACHE[cache_key] = result
            return result
            
    except Exception as e:
        return None

# --- MARKET INTEGRITY & ANTI-SPIKE ENGINE ---
def validate_market_integrity(o1: float, o2: float) -> bool:
    if o1 <= 1.01 or o2 <= 1.01: 
        return False 
    if o1 > 200 or o2 > 200: 
        return False 
    implied_prob = (1/o1) + (1/o2)
    if implied_prob < 0.92: 
        return False 
    if implied_prob > 1.45: 
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
# 3. SOTA MOMENTUM V3 ENGINE (🔥 HYBRID PARSER EDITION)
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
            
            if not is_p1 and not is_p2: continue

            winner = str(m.get('actual_winner_name', ''))
            won = is_same_player(player_name, winner)
            
            odds = to_float(m.get('odds1') if is_p1 else m.get('odds2'), 1.85)
            if odds <= 1.01: odds = 1.85
            
            expected_perf = 1 / odds 
            actual_perf = 0.5 
            
            score_str = str(m.get('score', '')).lower().replace(":", "-").strip()
            score_str = re.sub(r'\.\d+', '', score_str)
            
            if "ret" in score_str or "w.o" in score_str:
                actual_perf = 0.6 if won else 0.4
            else:
                sets = re.findall(r'\b(\d+)\s*-\s*(\d+)\b', score_str)
                
                if not sets:
                    actual_perf = 0.75 if won else 0.25
                elif len(sets) == 1 and (int(sets[0][0]) + int(sets[0][1]) <= 5):
                    l, r = int(sets[0][0]), int(sets[0][1])
                    p_sets = l if is_p1 else r
                    o_sets = r if is_p1 else l
                    
                    if p_sets >= o_sets + 2 or (p_sets == 2 and o_sets == 0):
                        actual_perf = 0.85  
                    elif p_sets > o_sets:
                        actual_perf = 0.65  
                    elif o_sets >= p_sets + 2 or (o_sets == 2 and p_sets == 0):
                        actual_perf = 0.15  
                    elif o_sets > p_sets:
                        actual_perf = 0.35  
                    else:
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
                            
                            if p_games > o_games: player_sets_won += 1
                            elif o_games > p_games: opp_sets_won += 1
                        except: pass
                    
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
            if recent_3 == ["W", "W", "W"]: streak_bonus = 0.4
            elif recent_3 == ["L", "L", "L"]: streak_bonus = -0.4
            if len(history_log) >= 5:
                recent_5 = history_log[-5:]
                if recent_5.count("W") == 5: streak_bonus = 0.8
                elif recent_5.count("L") == 5: streak_bonus = -0.8

        avg_edge = (cumulative_edge / total_weight) if total_weight > 0 else 0.0
        
        final_rating = base_rating + (avg_edge * 10.0) + streak_bonus
        final_rating = max(1.0, min(10.0, final_rating))
        
        desc = "Average"
        color_hex = "#F0C808" 
        
        if final_rating >= 8.5: 
            desc = "🔥 ELITE"
            color_hex = "#FF00FF" 
        elif final_rating >= 7.2: 
            desc = "📈 Strong"
            color_hex = "#3366FF" 
        elif final_rating >= 6.0: 
            desc = "Solid"
            color_hex = "#00B25B" 
        elif final_rating >= 4.5: 
            desc = "⚠️ Vulnerable"
            color_hex = "#FF9933" 
        else: 
            desc = "❄️ Cold"
            color_hex = "#CC0000" 

        return {
            "score": round(final_rating, 2),
            "text": desc,
            "color_hex": color_hex,
            "history_summary": "".join(history_log[-5:])
        }

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
    def extract_api_surface_stats(api_stats: List[Dict], surface_key: str) -> tuple:
        if not api_stats or not isinstance(api_stats, list):
            return 0, 0
            
        total_wins = 0
        total_losses = 0
        
        search_keys = [f"{surface_key}_won", f"surface_{surface_key}_won", f"matches_won_{surface_key}", f"wins_{surface_key}"]
        loss_keys = [f"{surface_key}_lost", f"surface_{surface_key}_lost", f"matches_lost_{surface_key}", f"losses_{surface_key}"]
        
        for season in api_stats:
            if isinstance(season, dict):
                
                if str(season.get("type", "")).lower() == "doubles":
                    continue

                def search_dict(d: dict):
                    nonlocal total_wins, total_losses
                    for k, v in d.items():
                        if isinstance(v, dict):
                            search_dict(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    search_dict(item)
                        else:
                            k_lower = str(k).lower()
                            if any(sk in k_lower for sk in search_keys):
                                try: total_wins += int(v)
                                except: pass
                            if any(lk in k_lower for lk in loss_keys):
                                try: total_losses += int(v)
                                except: pass
                                
                search_dict(season)
                
        return total_wins, total_losses

    @staticmethod
    def compute_player_surface_profile(matches: List[Dict], player_name: str, api_stats: Dict = None) -> Dict[str, Any]:
        profile = {}
        
        surfaces_data = {
            "hard": SurfaceIntelligence.get_matches_by_surface(matches, "hard"),
            "clay": SurfaceIntelligence.get_matches_by_surface(matches, "clay"),
            "grass": SurfaceIntelligence.get_matches_by_surface(matches, "grass")
        }
        
        for surf, surf_matches in surfaces_data.items():
            n_surf_db = len(surf_matches)
            wins_db = 0
            for m in surf_matches:
                winner = str(m.get('actual_winner_name', ""))
                if is_same_player(player_name, winner):
                    wins_db += 1
            
            api_won, api_lost = SurfaceIntelligence.extract_api_surface_stats(api_stats, surf)
            api_total = api_won + api_lost
            
            n_surf = n_surf_db
            wins = wins_db
            
            if api_total > n_surf_db and api_total > 0:
                n_surf = api_total
                wins = api_won

            if n_surf == 0:
                profile[surf] = {
                    "rating": 5.0,  
                    "color": "#808080",
                    "matches_tracked": 0,
                    "text": "No Data"
                }
                continue
                
            win_rate = wins / n_surf
            final_rating = max(1.0, min(10.0, win_rate * 10.0))
            
            if final_rating >= 8.5: 
                desc = "🔥 SPECIALIST"
                color_hex = "#FF00FF" 
            elif final_rating >= 7.0: 
                desc = "📈 Strong"
                color_hex = "#3366FF" 
            elif final_rating >= 5.5: 
                desc = "Solid"
                color_hex = "#00B25B" 
            elif final_rating >= 4.0: 
                desc = "Average"
                color_hex = "#F0C808" 
            else: 
                desc = "❄️ Weakness"
                color_hex = "#CC0000" 

            profile[surf] = {
                "rating": round(final_rating, 2),
                "color": color_hex,
                "matches_tracked": n_surf,
                "text": desc,
                "win_rate": f"{round(win_rate * 100, 1)}%"
            }
            
        profile['_v95_mastery_applied'] = True
        return profile

# =================================================================
# 5. SOTA MARKOV CHAIN ENGINE
# =================================================================
class MarkovChainEngine:
    @staticmethod
    def run_simulation(s1: Dict, s2: Dict, formA: float, formB: float, 
                       bsi: float, styleA: str, styleB: str, 
                       iterations: int = 2500) -> Dict[str, Any]:
        
        def get_z_score(val, mean, std):
            return (val - mean) / std if std else 0

        def get_serve_prob(serve_skill, power_skill):
            z_serve = get_z_score(serve_skill, SKILL_MEANS['serve'], SKILL_STDS['serve'])
            z_power = get_z_score(power_skill, SKILL_MEANS['power'], SKILL_STDS['power'])
            prob = 0.64 + (z_serve * 0.04) + (z_power * 0.02)
            return max(0.40, min(0.95, prob))
            
        def get_return_def(speed_skill, backhand_skill, forehand_skill):
            z_speed = get_z_score(speed_skill, SKILL_MEANS['speed'], SKILL_STDS['speed'])
            z_bh = get_z_score(backhand_skill, SKILL_MEANS['backhand'], SKILL_STDS['backhand'])
            z_fh = get_z_score(forehand_skill, SKILL_MEANS['forehand'], SKILL_STDS['forehand'])
            return (z_speed * 0.02) + (z_bh * 0.015) + (z_fh * 0.015)

        base_serve_win_A = get_serve_prob(s1.get('serve', 50), s1.get('power', 50))
        base_serve_win_B = get_serve_prob(s2.get('serve', 50), s2.get('power', 50))

        return_def_A = get_return_def(s1.get('speed', 50), s1.get('backhand', 50), s1.get('forehand', 50))
        return_def_B = get_return_def(s2.get('speed', 50), s2.get('backhand', 50), s2.get('forehand', 50))

        p_A_wins_on_serve = base_serve_win_A - return_def_B
        p_B_wins_on_serve = base_serve_win_B - return_def_A

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
            return_def_A += 0.02
            p_B_wins_on_serve -= 0.03
        if ("counter puncher" in safe_style_B or "grinder" in safe_style_B) and bsi < 6.0:
            return_def_B += 0.02
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

                if games_A == 6 and games_B == 6: 
                    tb_win = simulate_tiebreak(p_A_wins_on_serve, p_B_wins_on_serve)
                    return tb_win, games_A + (1 if tb_win else 0), games_B + (0 if tb_win else 1)
                if games_A >= 6 and games_A - games_B >= 2: return True, games_A, games_B
                if games_B >= 6 and games_B - games_A >= 2: return False, games_A, games_B

        match_wins_A, match_wins_B = 0, 0
        scores_log = {"2:0": 0, "2:1": 0, "0:2": 0, "1:2": 0}
        total_game_diff_A = 0

        for _ in range(iterations):
            sets_A, sets_B = 0, 0
            games_A_match, games_B_match = 0, 0
            
            while sets_A < 2 and sets_B < 2:
                a_won_set, gA, gB = simulate_set()
                games_A_match += gA
                games_B_match += gB
                if a_won_set: sets_A += 1
                else: sets_B += 1
                
            if sets_A == 2: 
                match_wins_A += 1
                if sets_B == 0: scores_log["2:0"] += 1
                else: scores_log["2:1"] += 1
            else: 
                match_wins_B += 1
                if sets_A == 0: scores_log["0:2"] += 1
                else: scores_log["1:2"] += 1
                
            total_game_diff_A += (games_A_match - games_B_match)

        prob_A = (match_wins_A / iterations) * 100
        prob_B = (match_wins_B / iterations) * 100
        
        set_betting_probs = {
            "2:0": round((scores_log["2:0"] / iterations) * 100, 1),
            "2:1": round((scores_log["2:1"] / iterations) * 100, 1),
            "0:2": round((scores_log["0:2"] / iterations) * 100, 1),
            "1:2": round((scores_log["1:2"] / iterations) * 100, 1)
        }
        
        avg_handicap_A = total_game_diff_A / iterations

        return {
            "probA": round(prob_A, 1),
            "probB": round(prob_B, 1),
            "scoreA": overall_A,
            "scoreB": overall_B,
            "set_betting_probs": set_betting_probs,
            "projected_handicap_A": round(avg_handicap_A, 1)
        }

# =================================================================
# 5.5 SOTA: SELF-LEARNING NEURAL OPTIMIZER
# =================================================================
class NeuralOptimizer:
    @staticmethod
    def optimize_ai_weights(matches_history: List[Dict], current_weights: Dict) -> Dict:
        best_weights = current_weights
        best_brier_score = float('inf') 
        
        for w_skill in np.arange(0.30, 0.70, 0.05):
            for w_form in np.arange(0.20, 0.50, 0.05):
                w_surf = 1.0 - (w_skill + w_form)
                if w_surf < 0.05 or w_surf > 0.30: continue 
                
                current_brier_score = 0.0
                valid_matches = 0
                
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
                            "SKILL": round(float(w_skill), 2), 
                            "FORM": round(float(w_form), 2), 
                            "SURFACE": round(float(w_surf), 2),
                            "MC_VARIANCE": current_weights.get("MC_VARIANCE", 1.20)
                        }

        log(f"✅ Neues Gehirn-Setup gefunden! Brier Score: {round(best_brier_score, 4)} -> {best_weights}")
        return best_weights

    @staticmethod
    def trigger_learning_cycle(players: List[Dict], all_skills: Dict):
        for tour in ["ATP", "WTA"]:
            tour_players = [p['id'] for p in players if p.get('tour') == tour]
            if not tour_players: continue
            
            recent_res = supabase.table("market_odds").select("*").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(200).execute()
            recent_matches = recent_res.data or []
            
            history_data = []
            correct_predictions = 0
            total_predictions = 0
            
            for m in recent_matches:
                p1_name = m.get('player1_name', '')
                p2_name = m.get('player2_name', '')
                winner = m.get('actual_winner_name', '')
                
                fair1 = to_float(m.get('ai_fair_odds1'), 0)
                fair2 = to_float(m.get('ai_fair_odds2'), 0)
                if fair1 > 0 and fair2 > 0:
                    total_predictions += 1
                    if (fair1 < fair2 and is_same_player(p1_name, winner)) or (fair2 < fair1 and is_same_player(p2_name, winner)):
                        correct_predictions += 1

                p1_obj = next((p for p in players if is_same_player(p.get('last_name', ''), p1_name) and p['id'] in tour_players), None)
                p2_obj = next((p for p in players if is_same_player(p.get('last_name', ''), p2_name) and p['id'] in tour_players), None)
                
                if p1_obj and p2_obj:
                    s1 = all_skills.get(p1_obj['id'], {}).get('overall_rating', 50)
                    s2 = all_skills.get(p2_obj['id'], {}).get('overall_rating', 50)
                    history_data.append({
                        "skillA": s1, "formA": 5.5, "surfA": 5.5,
                        "skillB": s2, "formB": 5.5, "surfB": 5.5,
                        "winner_is_A": is_same_player(p1_name, winner)
                    })

            if total_predictions > 0:
                acc = (correct_predictions / total_predictions) * 100
                SYSTEM_ACCURACY[tour] = round(acc, 1)

            if len(history_data) >= 20:
                new_weights = NeuralOptimizer.optimize_ai_weights(history_data, DYNAMIC_WEIGHTS[tour])
                DYNAMIC_WEIGHTS[tour] = new_weights
                
                try:
                    supabase.table("ai_system_weights").upsert({
                        "tour": tour,
                        "weight_skill": new_weights["SKILL"],
                        "weight_form": new_weights["FORM"],
                        "weight_surface": new_weights["SURFACE"],
                        "mc_variance": new_weights["MC_VARIANCE"],
                        "last_optimized": datetime.now(timezone.utc).isoformat()
                    }).execute()
                except Exception as e:
                    pass

# =================================================================
# 6. OPENROUTER AI ENGINE (SOTA)
# =================================================================
async def call_openrouter(prompt: str, model: str = MODEL_NAME, temp: float = 0.1) -> Optional[str]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neuralscout.com",
        "X-Title": "NeuralScout"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a data extraction AI. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp, 
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=45.0)
            if response.status_code != 200: 
                return None
            return response.json()['choices'][0]['message']['content']
        except Exception as e: 
            return None

# =================================================================
# 7. DATA FETCHING & ORACLE (API INTEGRATED)
# =================================================================
async def fetch_player_history_extended(player_last_name: str, limit: int = 200) -> List[Dict]:
    try:
        res = supabase.table("market_odds").select("player1_name, player2_name, odds1, odds2, actual_winner_name, score, created_at, tournament, ai_analysis_text").or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(limit).execute()
        return res.data or []
    except: 
        return []

async def update_past_results_api(api: TennisDataAPI, players: List[Dict]):
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending or not isinstance(pending, list): 
        return
    safe_to_check = list(pending)

    for day_off in range(0, 3): 
        t_date = (datetime.now(timezone.utc) - timedelta(days=day_off)).strftime('%Y-%m-%d')
        fixtures = await api.get_fixtures(t_date)
        
        for fix in fixtures:
            if fix.get("event_status") != "Finished": continue
            
            p1_api = fix.get("event_first_player", "")
            p2_api = fix.get("event_second_player", "")
            if not p1_api or not p2_api: continue
            
            matched_pm = None
            
            for pm in list(safe_to_check):
                if pm.get('api_match_key') and str(pm['api_match_key']) == str(fix.get('event_key')):
                    matched_pm = pm
                    break

                if (is_same_player(pm['player1_name'], p1_api) and is_same_player(pm['player2_name'], p2_api)) or \
                   (is_same_player(pm['player1_name'], p2_api) and is_same_player(pm['player2_name'], p1_api)):
                    matched_pm = pm
                    break

            if matched_pm:
                api_winner = fix.get("event_winner")
                winner = None
                
                if api_winner == "First Player":
                    winner = matched_pm['player1_name'] if is_same_player(matched_pm['player1_name'], p1_api) else matched_pm['player2_name']
                elif api_winner == "Second Player":
                    winner = matched_pm['player1_name'] if is_same_player(matched_pm['player1_name'], p2_api) else matched_pm['player2_name']
                    
                scores_array = fix.get("scores", [])
                if scores_array and isinstance(scores_array, list):
                    constructed_score = []
                    try:
                        scores_array = sorted(scores_array, key=lambda x: int(x.get("score_set", 0)))
                    except: pass
                    
                    for s in scores_array:
                        sf = s.get("score_first")
                        ss = s.get("score_second")
                        if sf is not None and ss is not None and str(sf).strip() != "" and str(ss).strip() != "":
                            constructed_score.append(f"{sf}-{ss}")
                    
                    if constructed_score:
                        final_score = " ".join(constructed_score)
                    else:
                        final_score = str(fix.get("event_final_result", ""))
                else:
                    final_score = str(fix.get("event_final_result", ""))
                
                if winner:
                    supabase.table("market_odds").update({
                        "actual_winner_name": winner,
                        "score": final_score
                    }).eq("id", matched_pm['id']).execute()

                    p1_name = matched_pm['player1_name']
                    p2_name = matched_pm['player2_name']
                    
                    p1_id = next((p['id'] for p in players if is_same_player(p1_name, p.get('last_name', ''))), None)
                    p2_id = next((p['id'] for p in players if is_same_player(p2_name, p.get('last_name', ''))), None)
                    
                    db_player_ids = [pid for pid in [p1_id, p2_id] if pid]
                    if db_player_ids:
                        try:
                            skills_res = supabase.table('player_skills').select('*').in_('player_id', db_player_ids).execute()
                            db_skills = skills_res.data or []
                            
                            if p1_id:
                                p1_skills_db = next((s for s in db_skills if s.get('player_id') == p1_id), None)
                                odds1 = to_float(matched_pm.get('odds1', 1.85))
                                if p1_skills_db:
                                    new_s1 = LiveSkillEngine.calculate_new_skills(p1_skills_db, odds1, is_same_player(winner, p1_name), final_score, is_player1=True)
                                    if new_s1:
                                        new_s1['updated_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                        supabase.table('player_skills').update(new_s1).eq('player_id', p1_id).execute()
                                        
                            if p2_id:
                                p2_skills_db = next((s for s in db_skills if s.get('player_id') == p2_id), None)
                                odds2 = to_float(matched_pm.get('odds2', 1.85))
                                if p2_skills_db:
                                    new_s2 = LiveSkillEngine.calculate_new_skills(p2_skills_db, odds2, is_same_player(winner, p2_name), final_score, is_player1=False)
                                    if new_s2:
                                        new_s2['updated_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                        supabase.table('player_skills').update(new_s2).eq('player_id', p2_id).execute()
                        except Exception as se: pass

                    for p_name_hook in [matched_pm['player1_name'], matched_pm['player2_name']]:
                        p_exists = next((p for p in players if is_same_player(p_name_hook, p.get('last_name', ''))), None)
                        if not p_exists: continue
                        
                        p_hist = await fetch_player_history_extended(p_name_hook, limit=200)
                        
                        p_key = None
                        if is_same_player(p_name_hook, p1_api):
                            p_key = fix.get('first_player_key')
                        elif is_same_player(p_name_hook, p2_api):
                            p_key = fix.get('second_player_key')

                        api_stats = []
                        if p_key:
                            raw_api_stats = await api.get_player_stats(str(p_key))
                            api_stats = raw_api_stats.get('stats', [])
                        
                        p_profile = SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name_hook, api_stats)
                        p_form = MomentumV2Engine.calculate_rating(p_hist[:20], p_name_hook)
                        
                        target_p_id = p1_id if is_same_player(p_name_hook, matched_pm['player1_name']) else p2_id
                        if target_p_id:
                            supabase.table('players').update({
                                'surface_ratings': p_profile,
                                'form_rating': p_form 
                            }).eq('id', target_p_id).execute()

                safe_to_check = [x for x in safe_to_check if x['id'] != matched_pm['id']]

def get_total_games(score_str: str) -> int:
    if not isinstance(score_str, str): return 0
    clean_score = re.sub(r'\.\d+', '', score_str.lower().replace(":", "-").strip())
    if "ret" in clean_score or "w.o" in clean_score: return 0
    sets = re.findall(r'\b(\d+)\s*-\s*(\d+)\b', clean_score)
    try:
        return sum(int(s[0]) + int(s[1]) for s in sets)
    except:
        return 0

def calculate_empirical_ou(history1: List[Dict], history2: List[Dict]) -> Dict[str, float]:
    games_list = []
    for h in history1[:15] + history2[:15]:
        score_str = str(h.get('score', ''))
        if score_str and len(score_str) > 2:
            tg = get_total_games(score_str)
            if tg > 12: 
                games_list.append(tg)
    
    if not games_list:
        return {"over_20_5": 0.5, "over_21_5": 0.5, "over_22_5": 0.5, "over_23_5": 0.5}
        
    total = len(games_list)
    return {
        "over_20_5": sum(1 for x in games_list if x > 20.5) / total,
        "over_21_5": sum(1 for x in games_list if x > 21.5) / total,
        "over_22_5": sum(1 for x in games_list if x > 22.5) / total,
        "over_23_5": sum(1 for x in games_list if x > 23.5) / total
    }

async def get_advanced_load_analysis(matches: List[Dict]) -> str:
    try:
        recent_matches = matches[:5]
        if not recent_matches: 
            return "Fresh (No recent data)"
            
        now_ts = datetime.now().timestamp()
        fatigue_score = 0.0
        details = []
        last_match = recent_matches[0]
        
        try:
            lm_time = datetime.fromisoformat(last_match['created_at'].replace('Z', '+00:00')).timestamp()
            hours_since_last = (now_ts - lm_time) / 3600
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
            
        if hours_since_last < 72 and last_match.get('score'):
            score_str = str(last_match['score']).lower().replace(":", "-")
            
            score_str = re.sub(r'\.\d+', '', score_str)
            
            if 'ret' in score_str or 'wo' in score_str: 
                fatigue_score *= 0.5
            else:
                sets = len(re.findall(r'\b(\d+)\s*-\s*(\d+)\b', score_str))
                tiebreaks = len(re.findall(r'7-6|6-7', score_str))
                total_games = 0
                for s in re.findall(r'\b(\d+)\s*-\s*(\d+)\b', score_str):
                    try: 
                        total_games += int(s[0]) + int(s[1])
                    except: 
                        pass
                        
                if sets >= 3: 
                    fatigue_score += 20
                    details.append("Last match 3+ sets")
                if total_games > 30: 
                    fatigue_score += 15
                    details.append("Marathon match (>30 games)")
                if tiebreaks > 0: 
                    fatigue_score += 5 * tiebreaks
                    details.append(f"{tiebreaks} Tiebreaks played")
                    
        matches_in_week = 0
        sets_in_week = 0
        
        for m in recent_matches:
            try:
                mt = datetime.fromisoformat(m['created_at'].replace('Z', '+00:00')).timestamp()
                if (now_ts - mt) < (7 * 24 * 3600):
                    matches_in_week += 1
                    if m.get('score'): 
                        clean_score = re.sub(r'\.\d+', '', str(m['score']).lower().replace(":", "-"))
                        sets_in_week += len(re.findall(r'\b\d+\s*-\s*\d+\b', clean_score))
            except: 
                pass
                
        if matches_in_week >= 4: 
            fatigue_score += 20
            details.append(f"Busy week ({matches_in_week} matches)")
        if sets_in_week > 10: 
            fatigue_score += 15
            details.append(f"Heavy leg load ({sets_in_week} sets in 7 days)")
            
        status = "Fresh"
        if fatigue_score > 75: 
            status = "CRITICAL FATIGUE"
        elif fatigue_score > 50: 
            status = "Heavy Legs"
        elif fatigue_score > 30: 
            status = "In Rhythm (Active)"
            
        if details: 
            return f"{status} [{', '.join(details)}]"
            
        return status
    except: 
        return "Unknown"

def fetch_all_rows(table_name: str) -> List[Dict]:
    data = []
    offset = 0
    limit = 1000
    while True:
        try:
            res = supabase.table(table_name).select("*").range(offset, offset + limit - 1).execute()
            chunk = res.data or []
            data.extend(chunk)
            if len(chunk) < limit:
                break
            offset += limit
        except Exception as e:
            log(f"⚠️ Pagination error on {table_name}: {e}")
            break
    return data

def initialize_g_elo_system(historical_matches: List[Dict], tournaments: List[Dict]):
    log("🧠 Initialisiere True Surface-Specific G-Elo System...")
    global ELO_CACHE
    ELO_CACHE = {"ATP": {}, "WTA": {}}
    
    valid_matches = [m for m in historical_matches if m.get('actual_winner_name') and m.get('score')]
    valid_matches.sort(key=lambda x: str(x.get('created_at', '')))
    
    for m in valid_matches:
        tour_raw = str(m.get('tournament', '')).lower()
        surf = "Hard"
        if "clay" in tour_raw or "roland garros" in tour_raw: surf = "Clay"
        elif "grass" in tour_raw or "wimbledon" in tour_raw: surf = "Grass"
        else:
            for db_key, db_surf in GLOBAL_SURFACE_MAP.items():
                if db_key in tour_raw or tour_raw in db_key:
                    if len(db_key) > 3:
                        if "clay" in db_surf.lower(): surf = "Clay"
                        elif "grass" in db_surf.lower(): surf = "Grass"
                        break
        
        tour = "WTA" if "wta" in tour_raw else "ATP"
        
        p1 = get_last_name(m.get('player1_name', ''))
        p2 = get_last_name(m.get('player2_name', ''))
        winner = get_last_name(m.get('actual_winner_name', ''))
        
        if not p1 or not p2 or not winner: continue
        
        p1_won = (p1 == winner)
        
        score_str = re.sub(r'\.\d+', '', str(m['score']).lower().replace(":", "-"))
        sets = re.findall(r'\b(\d+)\s*-\s*(\d+)\b', score_str)
        g1, g2 = 0, 0
        for s in sets:
            try:
                g1 += int(s[0])
                g2 += int(s[1])
            except: pass
        game_diff = abs(g1 - g2)
        
        K = 25
        margin_multiplier = math.log(game_diff + 1.5) if game_diff else 1.0
        
        if p1 not in ELO_CACHE[tour]: ELO_CACHE[tour][p1] = {'Hard': 1500.0, 'Clay': 1500.0, 'Grass': 1500.0}
        if p2 not in ELO_CACHE[tour]: ELO_CACHE[tour][p2] = {'Hard': 1500.0, 'Clay': 1500.0, 'Grass': 1500.0}
        
        elo1 = ELO_CACHE[tour][p1][surf]
        elo2 = ELO_CACHE[tour][p2][surf]
        
        exp1 = 1 / (1 + math.pow(10, (elo2 - elo1) / 400))
        
        shift = K * margin_multiplier * ((1 if p1_won else 0) - exp1)
        
        ELO_CACHE[tour][p1][surf] += shift
        ELO_CACHE[tour][p2][surf] -= shift
    
    log(f"✅ G-Elo System kalibriert: ATP ({len(ELO_CACHE['ATP'])} Spieler), WTA ({len(ELO_CACHE['WTA'])} Spieler)")

async def get_db_data():
    try:
        weights_res = supabase.table("ai_system_weights").select("*").execute()
        if weights_res.data:
            for w in weights_res.data:
                tour = w.get("tour", "ATP")
                DYNAMIC_WEIGHTS[tour] = {
                    "SKILL": to_float(w.get("weight_skill"), 0.50),
                    "FORM": to_float(w.get("weight_form"), 0.35),
                    "SURFACE": to_float(w.get("weight_surface"), 0.15),
                    "MC_VARIANCE": to_float(w.get("mc_variance"), 1.20)
                }

        players = fetch_all_rows("players")
        skills = fetch_all_rows("player_skills")
        reports = fetch_all_rows("scouting_reports")
        tournaments = fetch_all_rows("tournaments")
        
        if tournaments:
            for t in tournaments:
                t_name = clean_tournament_name(t.get('name', ''))
                t_loc = t.get('location', '')
                t_surf = t.get('surface', 'Unknown')
                if t_name and t_surf: 
                    GLOBAL_SURFACE_MAP[t_name.lower()] = t_surf
                if t_loc and t_surf:
                    for part in t_loc.split(','):
                        part = part.strip().lower()
                        if part and len(part) > 2: 
                            GLOBAL_SURFACE_MAP[part] = t_surf
                            
        historical_odds = fetch_all_rows("market_odds")
        initialize_g_elo_system(historical_odds, tournaments)
                            
        clean_skills = {}
        if skills:
            for entry in skills:
                if not isinstance(entry, dict): 
                    continue
                pid = entry.get('player_id')
                if pid:
                    clean_skills[pid] = {
                        'serve': to_float(entry.get('serve')), 
                        'power': to_float(entry.get('power')),
                        'forehand': to_float(entry.get('forehand')), 
                        'backhand': to_float(entry.get('backhand')),
                        'volley': to_float(entry.get('volley')),
                        'speed': to_float(entry.get('speed')), 
                        'stamina': to_float(entry.get('stamina')),
                        'mental': to_float(entry.get('mental')),
                        'overall_rating': to_float(entry.get('overall_rating', 50))
                    }
                    
        return players or [], clean_skills, reports or [], tournaments or []
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# 8. MATH CORE (🚀 SOTA: KELLY CRITERION & VARIANCE PENALTY INJECTED)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def normal_cdf_prob(elo_diff: float, sigma: float = 280.0) -> float:
    z = elo_diff / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def calculate_value_metrics(fair_prob: float, market_odds: float) -> Dict[str, Any]:
    if market_odds <= 1.01 or fair_prob <= 0: 
        return {"type": "NONE", "edge_percent": 0.0, "is_value": False, "kelly_stake": 0.0}
        
    market_odds = min(market_odds, 100.0)
    edge = (fair_prob * market_odds) - 1
    edge_percent = round(edge * 100, 1)
    
    # 🚀 SOTA: With the Z-Score fix, a 1.5% edge is a true edge.
    if edge_percent <= 1.5: 
        return {"type": "NONE", "edge_percent": edge_percent, "is_value": False, "kelly_stake": 0.0}

    # 🚀 SOTA: Risk-Adjusted Fractional Kelly Criterion
    b = market_odds - 1
    kelly_fraction = edge / b
    base_stake = (kelly_fraction / 4) * 100 
    
    # VARIANCE PENALTY: High odds mean massive variance. We protect the bankroll.
    if market_odds >= 5.0:
        optimal_stake = min(0.5, base_stake * 0.3)
    elif market_odds >= 3.0:
        optimal_stake = min(1.5, base_stake * 0.4)
    elif market_odds >= 2.0:
        optimal_stake = min(2.5, base_stake * 0.6)
    else:
        optimal_stake = min(5.0, base_stake)

    optimal_stake = max(0.1, round(optimal_stake, 1))

    label = "VALUE"
    if edge_percent >= 15.0: 
        label = "🔥 HIGH VALUE" 
    elif edge_percent >= 8.0: 
        label = "✨ GOOD VALUE" 
    elif edge_percent >= 2.0: 
        label = "📈 THIN VALUE" 
    else: 
        label = "👀 WATCH"

    return {"type": label, "edge_percent": edge_percent, "is_value": True, "kelly_stake": optimal_stake}

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, mc_prob_a, market_odds1, market_odds2, pinnacle_prob_a=None):
    n1 = get_last_name(p1_name)
    n2 = get_last_name(p2_name)
    tour = "ATP"
    
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo1 = p1_stats.get(elo_surf, 1500)
    elo2 = p2_stats.get(elo_surf, 1500)
    
    elo_diff_model = elo1 - elo2
    
    if market_odds1 > 0 and market_odds2 > 0:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        implied_p1 = inv1 / (inv1 + inv2)
        if 0.01 < implied_p1 < 0.99:
            try: 
                elo_diff_market = -400 * math.log10(1/implied_p1 - 1)
            except: 
                elo_diff_market = elo_diff_model
        else: 
            elo_diff_market = elo_diff_model 
        elo_diff_final = (elo_diff_model * 0.70) + (elo_diff_market * 0.30)
    else: 
        elo_diff_final = elo_diff_model
        
    prob_elo = normal_cdf_prob(elo_diff_final, sigma=280.0)
    
    mc_prob_a = max(0.01, min(0.99, mc_prob_a / 100.0))
    prob_alpha = (prob_elo * 0.35) + (mc_prob_a * 0.65)
    
    prob_market = 0.5
    if pinnacle_prob_a is not None:
        prob_market = pinnacle_prob_a
        model_trust_factor = 0.25 
    elif market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
        model_trust_factor = 0.35 
    else:
        model_trust_factor = 0.5
        
    final_prob = (prob_alpha * model_trust_factor) + (prob_market * (1 - model_trust_factor))
    return final_prob

def recalculate_fair_odds_with_new_market(old_fair_odds1: float, old_market_odds1: float, old_market_odds2: float, new_market_odds1: float, new_market_odds2: float) -> float:
    try:
        old_prob_market = 0.5
        if old_market_odds1 > 1 and old_market_odds2 > 1:
            inv1 = 1/old_market_odds1
            inv2 = 1/old_market_odds2
            old_prob_market = inv1 / (inv1 + inv2)
            
        if old_fair_odds1 <= 1.01: 
            return 0.5
            
        old_final_prob = 1 / old_fair_odds1
        alpha_part = old_final_prob - (old_prob_market * 0.40)
        prob_alpha = alpha_part / 0.60
        
        new_prob_market = 0.5
        if new_market_odds1 > 1 and new_market_odds2 > 1:
            inv1 = 1/new_market_odds1
            inv2 = 1/new_market_odds2
            new_prob_market = inv1 / (inv1 + inv2)
            
        new_final_prob = (prob_alpha * 0.60) + (new_prob_market * 0.40)
        
        if new_market_odds1 < 1.10:
             mkt_prob1 = 1/new_market_odds1
             new_final_prob = (new_final_prob * 0.15) + (mkt_prob1 * 0.85)
             
        return new_final_prob
    except: 
        return 0.5

# =================================================================
# 9. PIPELINE UTILS
# =================================================================
async def resolve_ambiguous_tournament(p1, p2, scraped_name, p1_country, p2_country):
    if scraped_name in TOURNAMENT_LOC_CACHE: 
        return TOURNAMENT_LOC_CACHE[scraped_name]
        
    prompt = f"TASK: Identify tournament location. MATCH: {p1} ({p1_country}) vs {p2} ({p2_country}). SOURCE: '{scraped_name}'. OUTPUT JSON: {{ 'city': 'Name', 'surface': 'Hard/Clay/Grass', 'indoor': true/false }}"
    res = await call_openrouter(prompt)
    
    if res:
        try: 
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            
            surface_type = data.get('surface', 'Hard')
            if data.get('indoor'): 
                surface_type += " Indoor"
            else: 
                surface_type += " Outdoor"
                
            est_bsi = 6.5
            if 'clay' in surface_type.lower(): 
                est_bsi = 3.5
            elif 'grass' in surface_type.lower(): 
                est_bsi = 8.0
            elif 'indoor' in surface_type.lower(): 
                est_bsi = 7.5
                
            city = data.get('city', 'Unknown')
            if "plantation" in city.lower() and p1_country == "USA": 
                city = "Winston-Salem"
                surface_type = "Hard Indoor"
                est_bsi = 7.5
                
            simulated_db_entry = {"city": city, "surface_guessed": surface_type, "bsi_estimate": est_bsi, "note": f"AI/Oracle: {city}"}
            TOURNAMENT_LOC_CACHE[scraped_name] = simulated_db_entry
            return simulated_db_entry
        except: 
            pass
            
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2, p1_country="Unknown", p2_country="Unknown", match_date: datetime = None): 
    s_low = clean_tournament_name(tour).lower().strip()
    
    best_match = None
    best_score = 0
    
    for t in db_tours:
        score = calculate_fuzzy_score(s_low, t['name'])
        if score > best_score: 
            best_score = score
            best_match = t
            
    if best_match and best_score >= 20: 
        loc = best_match.get('location', '')
        city_for_weather = loc.split(',')[0] if loc else best_match['name']
        return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', ''), city_for_weather, best_match['name']
    
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour, p1_country, p2_country)
    ai_loc = ensure_dict(ai_loc)
    
    if ai_loc and ai_loc.get('city'):
        surf = ai_loc.get('surface_guessed', 'Hard Court Outdoor')
        bsi = ai_loc.get('bsi_estimate', 6.5)
        city = ai_loc.get('city', 'Unknown')
        return surf, bsi, "AI Guess", city, tour
        
    return 'Hard Court Outdoor', 6.5, 'Fallback', tour.split()[0], tour

def format_skills(s: Dict) -> str:
    if not s: 
        return "No granular skill data."
    return f"Serve: {s.get('serve', 50)}, FH: {s.get('forehand', 50)}, BH: {s.get('backhand', 50)}, Volley: {s.get('volley', 50)}, Speed: {s.get('speed', 50)}, Stamina: {s.get('stamina', 50)}, Power: {s.get('power', 50)}, Mental: {s.get('mental', 50)}, OVR: {s.get('overall_rating', 50)}"

async def analyze_match_with_ai(tour_name, p1, p2, s1, s2, report1, report2, surface, bsi, notes, form1_data, form2_data, weather_data, p1_surface_profile, p2_surface_profile, mc_results, h2h_record):
    fatigueA = await get_advanced_load_analysis(await fetch_player_history_extended(p1['last_name'], 10))
    fatigueB = await get_advanced_load_analysis(await fetch_player_history_extended(p2['last_name'], 10))
    
    if weather_data:
        weather_str = f"WEATHER: {weather_data['summary']}. IMPACT: {weather_data['impact_note']}"
    else:
        weather_str = "Weather: Neutral/No Data."
        
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
    finalProb = f"{finalProb_val:.1f}%"

    tour = "WTA" if "WTA" in tour_name.upper() else "ATP"
    sys_acc = SYSTEM_ACCURACY.get(tour, 65.0)

    convictionDirective = ""
    if finalProb_val >= 65.0:
        convictionDirective = f"*** CONVICTION DIRECTIVE (CRITICAL) \nThe mathematical model gives {predictedMCWinner} a massive {finalProb_val:.1f}% probability of winning because of a clear baseline talent mismatch. You MUST write this analysis with HIGH CONVICTION. Do not write \"If he can...\". State confidently that {predictedMCWinner}'s overall quality and baseline strengths will overwhelm {predictedMCLoser}. Explain exactly why {predictedMCLoser}'s game will break down. NO HEDGING."
    elif finalProb_val >= 58.0:
        convictionDirective = f" CONVICTION DIRECTIVE \nThe mathematical model gives {predictedMCWinner} a clear edge ({finalProb_val:.1f}%). Write confidently about why {predictedMCWinner} is the favorite to win, focusing on the tactical mismatch. Avoid 50/50 language."
    else:
        convictionDirective = f" CONVICTION DIRECTIVE ***\nThe mathematical model sees this as a tight battle ({finalProb_val:.1f}% for {predictedMCWinner}). Write an analysis highlighting the fine margins that will decide this match."

    prompt = f"""
    You are an elite Senior Tennis Analyst (Style: Gil Gross). 
    Your analysis must be grounded EXCLUSIVELY in the provided technical data and scouting reports.
    
    *** SYSTEM SELF-REFLECTION (CRITICAL) ***
    Our internal neural network has an active prediction accuracy of {sys_acc}%. 
    If this accuracy is below 65%, you MUST be more conservative in your tone and acknowledge potential variance. If it is above 70%, be highly assertive about the data trends.
    
    *** DATA GROUNDING (SOURCE OF TRUTH) ***
    Head-to-Head (H2H): {h2h_record}

    Player A ({p1['last_name']}):
    - Style: {p1.get('play_style', 'Unknown')}
    - Form / Momentum: {form1_data['text']}
    - Surface Rating ({current_surf_key}): {p1_s_rating}/10
    - Granular Skills: {format_skills(s1)}
    - Scouting Report: {scoutA}
    - Fatigue: {fatigueA}
    
    Player B ({p2['last_name']}):
    - Style: {p2.get('play_style', 'Unknown')}
    - Form / Momentum: {form2_data['text']}
    - Surface Rating ({current_surf_key}): {p2_s_rating}/10
    - Granular Skills: {format_skills(s2)}
    - Scouting Report: {scoutB}
    - Fatigue: {fatigueB}
    
    Match Conditions:
    - Surface: {surface} (BSI: {bsi})
    - Court Notes: {validCourtNotes}
    - {weather_str}

    *** INTERNAL MATCHUP DATA (FOR LOGIC ONLY, DO NOT OUTPUT THESE NUMBERS) ***
    Winner: {predictedMCWinner} (Internal Win Probability: {finalProb})
    
    {convictionDirective}
    
    *** TACTICAL PLAYSTYLE MATCHUP (CRITICAL) ***
    Explicitly analyze how {p1['last_name']}'s style ({p1.get('play_style', 'Unknown')}) matches up against {p2['last_name']}'s style ({p2.get('play_style', 'Unknown')}). Does the Grinder break down the Aggressive Baseliner? Does the Counterpuncher neutralize the Big Server on this specific {surface}? Base this strictly on their granular skills and scouting reports.

    *** CRITICAL DIRECTIVES (MUST OBEY) ***
    1. NO NUMBERS IN TEXT: Strictly forbidden to use percentages (%), numerical ratings (e.g., 8/10), or skill points in 'prediction_text' and 'key_factor'.
    2. TACTICAL PROSA: Use Gil Gross style "Matchup Physics". Explain how the specific "Court Notes" (bounce height, court speed) amplify a player's strengths or expose their weaknesses.
    3. FACTUAL INTEGRITY: If the Scouting Report says a player has poor movement, NEVER describe them as "athletic". Ground your analysis in the provided 'Weaknesses' and the H2H stats.
    4. PATTERN ANALYSIS: Explain HOW {predictedMCWinner}'s specific skills interact with {predictedMCLoser}'s specific weaknesses under these exact court conditions. Emphasize the playstyle clash (e.g., Counterpuncher vs Aggressive Baseliner).
    5. DO NOT EXPLAIN CALCULATIONS: Output strictly the JSON. No introductory chatter.
    
    OUTPUT JSON:
    {{
        "winner_prediction": "{predictedMCWinner}",
        "key_factor": "One sharp tactical sentence focusing on the primary playstyle mismatch (NO NUMBERS).",
        "prediction_text": "Deep Gil Gross style analysis (~200 words). Focus on tactical matchup physics, the clash of their specific playstyles, court conditions, and how the scouting report details manifest on court. STRICTLY NO NUMBERS OR PERCENTAGES.",
        "tactical_bullets": ["Tactic 1 based on playstyle/report", "Tactic 2 based on surface physics", "Tactic 3 based on specific weakness"]
    }}
    """
    
    res = await call_openrouter(prompt)
    default_text = f"Analysis unavailable for {p1['last_name']} vs {p2['last_name']}."
    
    if not res: 
        return {'ai_text': default_text, 'mc_prob_a': mc_results['probA']}
        
    try:
        cleaned = res.replace("json", "").replace("```", "").strip()
        data = ensure_dict(json.loads(cleaned))
        
        bullets = "\n".join([f"- {b}" for b in data.get('tactical_bullets', [])])
        formatted_text = f"🔑 {data.get('key_factor', '')}\n\n📝 {data.get('prediction_text', '')}\n\n🎯 Tactical Keys:\n{bullets}"
        
        return {
            'ai_text': formatted_text.strip(),
            'mc_prob_a': mc_results['probA']
        }
    except: 
        return {'ai_text': default_text, 'mc_prob_a': mc_results['probA']}

# =================================================================
# 10. QUANTUM GAMES SIMULATOR (OVER/UNDER) 
# =================================================================
class QuantumGamesSimulator:
    @staticmethod
    def derive_hold_probability(server_skills: Dict, returner_skills: Dict, bsi: float, surface: str) -> float:
        p_hold = 67.0 
        p_hold += (server_skills.get('serve', 50) - 50) * 0.35 
        p_hold += (server_skills.get('power', 50) - 50) * 0.10
        p_hold -= (returner_skills.get('speed', 50) - 50) * 0.15 
        p_hold -= (returner_skills.get('mental', 50) - 50) * 0.08
        p_hold += (bsi - 6.0) * 1.4 
        return max(52.0, min(94.0, p_hold)) / 100.0

    @staticmethod
    def simulate_set(p1_prob: float, p2_prob: float) -> tuple[int, int]:
        g1, g2 = 0, 0
        while True:
            if random.random() < p1_prob: 
                g1 += 1
            else: 
                g2 += 1 
                
            if g1 >= 6 and g1 - g2 >= 2: 
                return (1, g1 + g2)
            if g2 >= 6 and g2 - g1 >= 2: 
                return (2, g1 + g2)
                
            if g1 == 6 and g2 == 6:
                if random.random() < 0.5 + (p1_prob - p2_prob): 
                    return (1, 13)
                else: 
                    return (2, 13)
            
            if random.random() < p2_prob: 
                g2 += 1
            else: 
                g1 += 1 
                
            if g1 >= 6 and g1 - g2 >= 2: 
                return (1, g1 + g2)
            if g2 >= 6 and g2 - g1 >= 2: 
                return (2, g1 + g2)
                
            if g1 == 6 and g2 == 6:
                if random.random() < 0.5 + (p1_prob - p2_prob): 
                    return (1, 13)
                else: 
                    return (2, 13)

    @staticmethod
    def run_simulation(p1_skills: Dict, p2_skills: Dict, bsi: float, surface: str, actual_ou_line: float = None, iterations: int = 1000, empirical_ou: Dict[str, float] = None) -> Dict[str, Any]:
        p1_hold_prob = QuantumGamesSimulator.derive_hold_probability(p1_skills, p2_skills, bsi, surface)
        p2_hold_prob = QuantumGamesSimulator.derive_hold_probability(p2_skills, p1_skills, bsi, surface)
        
        total_games_log = []
        
        for _ in range(iterations):
            winner_s1, games_s1 = QuantumGamesSimulator.simulate_set(p1_hold_prob, p2_hold_prob)
            
            p1_hold_s2 = p1_hold_prob + (0.02 if winner_s1 == 1 else -0.01)
            p2_hold_s2 = p2_hold_prob + (0.02 if winner_s1 == 2 else -0.01)
            
            winner_s2, games_s2 = QuantumGamesSimulator.simulate_set(p1_hold_s2, p2_hold_s2)
            total = games_s1 + games_s2
            
            if winner_s1 != winner_s2:
                winner_s3, games_s3 = QuantumGamesSimulator.simulate_set(p1_hold_prob, p2_hold_prob)
                total += games_s3
                
            total_games_log.append(total)
            
        total_games_log.sort()
        
        median = total_games_log[len(total_games_log)//2]
        
        probs = {}
        derivative_edge = None

        if empirical_ou is None:
            empirical_ou = {"over_20_5": 0.5, "over_21_5": 0.5, "over_22_5": 0.5, "over_23_5": 0.5}

        if actual_ou_line:
            mc_prob = sum(1 for x in total_games_log if x > actual_ou_line) / iterations
            
            emp_val = 0.5
            if actual_ou_line <= 21.0: emp_val = empirical_ou.get("over_20_5", 0.5)
            elif actual_ou_line <= 22.0: emp_val = empirical_ou.get("over_21_5", 0.5)
            elif actual_ou_line <= 23.0: emp_val = empirical_ou.get("over_22_5", 0.5)
            else: emp_val = empirical_ou.get("over_23_5", 0.5)
            
            blended_prob = (mc_prob * 0.65) + (emp_val * 0.35)
            probs[f"over_{actual_ou_line}"] = blended_prob
            
            if blended_prob > 0.60:
                derivative_edge = f"🔥 MASSIVE OVER EDGE: Model & History project {round(sum(total_games_log) / len(total_games_log), 1)} games against bookmaker line {actual_ou_line}."
            elif blended_prob < 0.40:
                derivative_edge = f"🔥 MASSIVE UNDER EDGE: Model & History project {round(sum(total_games_log) / len(total_games_log), 1)} games against bookmaker line {actual_ou_line}."
                
        else:
            probs = {
                "over_20_5": (sum(1 for x in total_games_log if x > 20.5) / iterations) * 0.65 + (empirical_ou.get("over_20_5", 0.5) * 0.35),
                "over_21_5": (sum(1 for x in total_games_log if x > 21.5) / iterations) * 0.65 + (empirical_ou.get("over_21_5", 0.5) * 0.35),
                "over_22_5": (sum(1 for x in total_games_log if x > 22.5) / iterations) * 0.65 + (empirical_ou.get("over_22_5", 0.5) * 0.35),
                "over_23_5": (sum(1 for x in total_games_log if x > 23.5) / iterations) * 0.65 + (empirical_ou.get("over_23_5", 0.5) * 0.35)
            }
            
        return {
            "predicted_line": round(sum(total_games_log) / len(total_games_log), 1),
            "median_games": median,
            "probabilities": probs,
            "derivative_edge": derivative_edge,
            "sim_details": {
                "p1_est_hold_pct": round(p1_hold_prob * 100, 1), 
                "p2_est_hold_pct": round(p2_hold_prob * 100, 1)
            }
        }

# =================================================================
# 11. LIVE SKILL ENGINE (🔥 HYBRID PARSER EDITION)
# =================================================================
class LiveSkillEngine:
    @staticmethod
    def calculate_new_skills(current_skills: Dict[str, Any], odds: float, is_winner: bool, score: str, is_player1: bool) -> Dict[str, float]:
        if not current_skills: return {}

        base_shift = 0.0
        
        if is_winner:
            if odds <= 1.30: base_shift = 0.1
            elif odds <= 1.701: base_shift = 0.2
            elif odds <= 1.850: base_shift = 0.3
            elif odds <= 2.2501: base_shift = 0.4
            else: base_shift = 0.6
        else:
            if odds <= 1.30: base_shift = -0.8
            elif odds <= 1.701: base_shift = -0.4
            elif odds <= 1.850: base_shift = -0.3
            elif odds <= 2.2501: base_shift = -0.2
            else: base_shift = -0.1

        skill_fields = ['serve', 'forehand', 'backhand', 'volley', 'speed', 'stamina', 'power', 'mental']
        new_skills = {}
        for k in skill_fields:
            if k in current_skills and current_skills[k] is not None:
                new_skills[k] = float(current_skills[k]) + base_shift

        if not new_skills: return {}

        score_lower = str(score).lower().replace(":", "-").strip()
        
        score_lower = re.sub(r'\.\d+', '', score_lower)

        is_clean_sweep = False
        is_grind_match = False
        lost_tiebreak = False

        if "ret" in score_lower or "w.o" in score_lower:
            pass
        else:
            sets = re.findall(r'\b(\d+)\s*-\s*(\d+)\b', score_lower)
            if len(sets) == 1 and (int(sets[0][0]) + int(sets[0][1]) <= 5):
                l, r = int(sets[0][0]), int(sets[0][1])
                p_sets = l if is_player1 else r
                o_sets = r if is_player1 else l
                
                if p_sets >= o_sets + 2 or (p_sets == 2 and o_sets == 0): is_clean_sweep = True
                if (p_sets == 2 and o_sets == 1) or (p_sets == 1 and o_sets == 2): is_grind_match = True
            else:
                if len(sets) == 2 and not "ret." in score_lower and not "w.o." in score_lower:
                    is_clean_sweep = True
                elif len(sets) >= 3:
                    is_grind_match = True

                for s in sets:
                    try:
                        l, r = int(s[0]), int(s[1])
                        if is_player1 and l < r and r == 7: lost_tiebreak = True
                        if not is_player1 and r < l and l == 7: lost_tiebreak = True
                    except: pass

        if is_winner and is_clean_sweep:
            for skill in ['power', 'serve', 'forehand', 'backhand', 'volley', 'speed']:
                if skill in new_skills: new_skills[skill] += 0.2

        if is_winner and is_grind_match:
            if 'mental' in new_skills: new_skills['mental'] += 0.3
            if 'stamina' in new_skills: new_skills['stamina'] += 0.3

        if not is_winner and lost_tiebreak:
             if 'mental' in new_skills: new_skills['mental'] -= 0.2

        if not is_winner and not is_clean_sweep and is_grind_match:
             if 'mental' in new_skills: new_skills['mental'] -= 0.2

        if not is_winner and "ret" in score_lower:
             if 'stamina' in new_skills: new_skills['stamina'] -= 0.5
             if 'speed' in new_skills: new_skills['speed'] -= 0.5

        new_overall = sum(new_skills[k] for k in skill_fields if k in new_skills) / len([k for k in skill_fields if k in new_skills])
        new_skills['overall_rating'] = new_overall

        for k, v in new_skills.items():
            new_skills[k] = max(1.0, min(99.0, round(v, 2)))

        return new_skills

# =================================================================
# PIPELINE EXECUTION (SOTA API EDITION)
# =================================================================
async def run_pipeline():
    log(f"🚀 Neural Scout V205.8 (VARIANCE PENALTY & RISK MANAGEMENT) Starting...")
    
    api = TennisDataAPI(API_TENNIS_KEY)

    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: 
        return
        
    await update_past_results_api(api, players)
        
    try:
        NeuralOptimizer.trigger_learning_cycle(players, all_skills)
    except Exception as opt_err:
        log(f"⚠️ Self-Learning Cycle Exception: {opt_err}")
        
    report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
        
    matches = []
    
    for day_offset in range(0, 4):
        target_date = (datetime.now(timezone.utc) + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        fixtures = await api.get_fixtures(target_date)
        
        for fix in fixtures:
            status = fix.get("event_status", "")
            if status in ["Finished", "Cancelled", "Retired", "Walkover"]: 
                continue
            
            p1_raw = fix.get("event_first_player")
            p2_raw = fix.get("event_second_player")
            match_key = fix.get("event_key")
            tour_name = fix.get("tournament_name", "Unknown")
            e_date = fix.get("event_date")
            e_time = fix.get("event_time", "00:00")
            
            if not p1_raw or not p2_raw or not match_key: continue
            
            odds_data = await api.get_odds(match_key)
            if not odds_data: continue
            
            home_away = odds_data.get("Home/Away", {})
            if not home_away: continue
            
            home_odds_dict = home_away.get("Home", {})
            away_odds_dict = home_away.get("Away", {})
            
            bookmaker_odds = {}
            all_bookies = set(list(home_odds_dict.keys()) + list(away_odds_dict.keys()))
            for bookie in all_bookies:
                bookmaker_odds[bookie] = {
                    "odds1": to_float(home_odds_dict.get(bookie, 0)),
                    "odds2": to_float(away_odds_dict.get(bookie, 0))
                }
            
            o1 = to_float(home_odds_dict.get("bet365") or home_odds_dict.get("1xbet") or next(iter(home_odds_dict.values()), 0))
            o2 = to_float(away_odds_dict.get("bet365") or away_odds_dict.get("1xbet") or next(iter(away_odds_dict.values()), 0))
            
            if o1 <= 1.01 or o2 <= 1.01: continue
            
            set_betting_api = odds_data.get("Set Betting", {})
            bookie_set_odds = {}
            if set_betting_api:
                for score_key, bookies in set_betting_api.items():
                    main_line = to_float(bookies.get("bet365") or bookies.get("1xbet") or next(iter(bookies.values()), 0))
                    bookie_set_odds[score_key] = main_line

            actual_ou_line = None
            ou_dict = odds_data.get("Over/Under", {})
            if ou_dict:
                best_diff = 999.0
                for line_str, line_odds in ou_dict.items():
                    try:
                        line_val = float(line_str)
                        over_odd = to_float(line_odds.get("bet365") or next(iter(line_odds.values()), 0))
                        diff_to_even = abs(over_odd - 1.90)
                        if diff_to_even < best_diff and over_odd > 1.01:
                            best_diff = diff_to_even
                            actual_ou_line = line_val
                    except:
                        pass

            try:
                dt = datetime.strptime(f"{e_date} {e_time}", "%Y-%m-%d %H:%M")
                iso_time = dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except:
                iso_time = f"{e_date}T00:00:00Z"
                
            matches.append({
                "p1_raw": p1_raw, 
                "p2_raw": p2_raw,
                "p1_api_key": fix.get("first_player_key"),
                "p2_api_key": fix.get("second_player_key"),
                "api_match_key": match_key, 
                "tour": tour_name,
                "time": iso_time, 
                "odds1": o1, 
                "odds2": o2,
                "bookmaker_odds": bookmaker_odds,
                "bookie_set_odds": bookie_set_odds,
                "actual_ou_line": actual_ou_line
            })
            
    if not matches:
        log("❌ Keine relevanten API-Matches mit Quoten gefunden. Beende Zyklus.")
        return
            
    log(f"🔍 Starte Neural Enrichment für {len(matches)} API-Matches...")
    db_matched_count = 0
    seen_match_keys = set()
    
    for m in matches:
        try:
            m_key = tuple(sorted([m['p1_raw'], m['p2_raw']]))
            if m_key in seen_match_keys: continue
            seen_match_keys.add(m_key)

            p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
            p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
            
            if p1_obj == "TIE_BREAKER" or p2_obj == "TIE_BREAKER":
                continue

            if not p1_obj and not p2_obj: 
                continue
                
            both_players_found = (p1_obj is not None) and (p2_obj is not None)
            
            n1_last = p1_obj['last_name'] if p1_obj else normalize_db_name(get_last_name(m['p1_raw']))
            n2_last = p2_obj['last_name'] if p2_obj else normalize_db_name(get_last_name(m['p2_raw']))
            
            full_n1 = f"{p1_obj.get('first_name', '')} {p1_obj.get('last_name', '')}".strip() if p1_obj else clean_player_name(m['p1_raw'])
            full_n2 = f"{p2_obj.get('first_name', '')} {p2_obj.get('last_name', '')}".strip() if p2_obj else clean_player_name(m['p2_raw'])
            
            if n1_last == n2_last: 
                continue
                
            db_matched_count += 1
                
            if not validate_market_integrity(m['odds1'], m['odds2']):
                continue 

            res1 = supabase.table("market_odds").select("*").eq("api_match_key", m['api_match_key']).execute()
            existing_match = res1.data[0] if res1.data else None

            if not existing_match:
                res_fb1 = supabase.table("market_odds").select("*").ilike("player1_name", f"%{n1_last}%").ilike("player2_name", f"%{n2_last}%").is_("actual_winner_name", "null").order("created_at", desc=True).limit(1).execute()
                existing_match = res_fb1.data[0] if res_fb1.data else None
                
            if not existing_match:
                res_fb2 = supabase.table("market_odds").select("*").ilike("player1_name", f"%{n2_last}%").ilike("player2_name", f"%{n1_last}%").is_("actual_winner_name", "null").order("created_at", desc=True).limit(1).execute()
                existing_match = res_fb2.data[0] if res_fb2.data else None
                
                if existing_match:
                    full_n1, full_n2 = full_n2, full_n1
                    p1_obj, p2_obj = p2_obj, p1_obj
                    m['odds1'], m['odds2'] = m['odds2'], m['odds1']
                    
                    swapped_bookies = {}
                    for b_name, b_odds in m['bookmaker_odds'].items():
                        swapped_bookies[b_name] = {"odds1": b_odds["odds2"], "odds2": b_odds["odds1"]}
                    m['bookmaker_odds'] = swapped_bookies
                    
                    swapped_sets = {}
                    if "2:0" in m.get('bookie_set_odds', {}): swapped_sets["0:2"] = m['bookie_set_odds']["2:0"]
                    if "0:2" in m.get('bookie_set_odds', {}): swapped_sets["2:0"] = m['bookie_set_odds']["0:2"]
                    if "2:1" in m.get('bookie_set_odds', {}): swapped_sets["1:2"] = m['bookie_set_odds']["2:1"]
                    if "1:2" in m.get('bookie_set_odds', {}): swapped_sets["2:1"] = m['bookie_set_odds']["1:2"]
                    m['bookie_set_odds'] = swapped_sets
            
            if existing_match:
                if is_suspicious_movement(to_float(existing_match.get('odds1'), 0), m['odds1'], to_float(existing_match.get('odds2'), 0), m['odds2']):
                    continue

            db_match_id = existing_match['id'] if existing_match else None
            if existing_match and existing_match.get('actual_winner_name'): 
                continue 
                
            final_time_str = m['time'] 

            if not both_players_found:
                c1_country = p1_obj.get('country', 'Unknown') if p1_obj else 'Unknown'
                c2_country = p2_obj.get('country', 'Unknown') if p2_obj else 'Unknown'
                surf, bsi, notes, city_for_weather, matched_tour_name = await find_best_court_match_smart(m['tour'], all_tournaments, full_n1, full_n2, c1_country, c2_country, match_date=datetime.now())

                shadow_data = {
                    "player1_name": full_n1, 
                    "player2_name": full_n2, 
                    "tournament": matched_tour_name,
                    "match_time": final_time_str, 
                    "odds1": m['odds1'], 
                    "odds2": m['odds2'],
                    "bookmaker_odds": m['bookmaker_odds'],
                    "is_visible_in_scanner": False, 
                    "api_match_key": m['api_match_key'],
                    "ai_analysis_text": "[SHADOW TRACKING] Match collected for historical player metrics."
                }
                if not db_match_id:
                    shadow_data["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    shadow_data["opening_odds1"] = m['odds1']
                    shadow_data["opening_odds2"] = m['odds2']
                    try:
                        res_ins = supabase.table("market_odds").insert(shadow_data).execute()
                        if res_ins.data:
                            db_match_id = res_ins.data[0]['id']
                    except: pass
                else:
                    try:
                        supabase.table("market_odds").update(shadow_data).eq("id", db_match_id).execute()
                    except: pass
                continue 

            # --- FULL MATCHUP PATH --->
            hist_fair1, hist_fair2 = 0, 0
            hist_is_value, hist_pick_player = False, None
            is_signal_locked = has_active_signal(existing_match.get('ai_analysis_text', '')) if existing_match else False
            
            pinnacle_prob_a = None
            pncl_data = m['bookmaker_odds'].get("Pncl") or m['bookmaker_odds'].get("Pinnacle")
            if pncl_data and pncl_data.get("odds1") > 1 and pncl_data.get("odds2") > 1:
                inv1 = 1 / pncl_data["odds1"]
                inv2 = 1 / pncl_data["odds2"]
                pinnacle_prob_a = inv1 / (inv1 + inv2)
            
            if is_signal_locked:
                update_data = {
                    "odds1": m['odds1'], 
                    "odds2": m['odds2'], 
                    "is_visible_in_scanner": True, 
                    "bookmaker_odds": m['bookmaker_odds'],
                    "api_match_key": m['api_match_key'] 
                }
                
                if final_time_str and not final_time_str.endswith("T00:00:00Z"):
                    update_data["match_time"] = final_time_str
                
                try:
                    supabase.table("market_odds").update(update_data).eq("id", db_match_id).execute()
                except: pass
                    
                hist_fair1 = to_float(existing_match.get('ai_fair_odds1'), 0)
                hist_fair2 = to_float(existing_match.get('ai_fair_odds2'), 0)
                
            else:
                cached_ai = {}
                if existing_match and existing_match.get('ai_analysis_text'):
                    cached_ai = {
                        'ai_text': existing_match.get('ai_analysis_text'), 
                        'ai_fair_odds1': existing_match.get('ai_fair_odds1'), 
                        'old_odds1': existing_match.get('odds1', 0), 
                        'old_odds2': existing_match.get('odds2', 0), 
                        'last_update': existing_match.get('created_at')
                    }
                
                surf, bsi, notes, city_for_weather, matched_tour_name = await find_best_court_match_smart(m['tour'], all_tournaments, full_n1, full_n2, p1_obj.get('country', 'Unknown'), p2_obj.get('country', 'Unknown'), match_date=datetime.now())
                weather_data = await fetch_weather_data(city_for_weather)

                s1 = all_skills.get(p1_obj['id'], {})
                s2 = all_skills.get(p2_obj['id'], {})
                
                report1 = next((r for r in all_reports if r.get('player_id') == p1_obj['id']), None)
                report2 = next((r for r in all_reports if r.get('player_id') == p2_obj['id']), None)

                p1_history = await fetch_player_history_extended(full_n1, limit=200)
                p2_history = await fetch_player_history_extended(full_n2, limit=200)
                
                p1_form_v2 = MomentumV2Engine.calculate_rating(p1_history[:20], full_n1)
                p2_form_v2 = MomentumV2Engine.calculate_rating(p2_history[:20], full_n2)

                should_run_ai = True
                if db_match_id and cached_ai:
                    odds_diff = max(abs(cached_ai['old_odds1'] - m['odds1']), abs(cached_ai['old_odds2'] - m['odds2']))
                    try: 
                        is_stale = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_ai.get('last_update', '').replace('Z', '+00:00'))) > timedelta(hours=6)
                    except: 
                        is_stale = True
                    if odds_diff <= (m['odds1'] * 0.05) and not is_stale: 
                        should_run_ai = False
                
                if not should_run_ai:
                    new_prob = recalculate_fair_odds_with_new_market(cached_ai['ai_fair_odds1'], cached_ai['old_odds1'], cached_ai['old_odds2'], m['odds1'], m['odds2'])
                    fair1 = round(1/new_prob, 2) if new_prob > 0.01 else 99
                    fair2 = round(1/(1-new_prob), 2) if new_prob < 0.99 else 99
                    
                    val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                    val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                    
                    value_tag = ""
                    if val_p1["is_value"]: 
                        value_tag = f"\n\n[{val_p1['type']}: {full_n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}% | Stake: {val_p1['kelly_stake']}u]"
                        hist_is_value = True
                        hist_pick_player = full_n1
                    elif val_p2["is_value"]: 
                        value_tag = f"\n\n[{val_p2['type']}: {full_n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}% | Stake: {val_p2['kelly_stake']}u]"
                        hist_is_value = True
                        hist_pick_player = full_n2
                        
                    ai_text_final = re.sub(r'\n*\s*\[.*?(Fair|Edge|VALUE|WATCH|NONE).*?\]', '', cached_ai['ai_text']).strip() 
                    ai_text_final += value_tag
                    
                    hist_fair1 = fair1
                    hist_fair2 = fair2
                    
                    if db_match_id:
                        try:
                            supabase.table("market_odds").update({
                                "odds1": m['odds1'], 
                                "odds2": m['odds2'], 
                                "bookmaker_odds": m['bookmaker_odds'],
                                "ai_fair_odds1": fair1, 
                                "ai_fair_odds2": fair2,
                                "ai_analysis_text": ai_text_final,
                                "match_time": final_time_str, 
                                "is_visible_in_scanner": True,
                                "api_match_key": m['api_match_key'] 
                            }).eq("id", db_match_id).execute()
                        except: pass

                else:
                    log(f"  🧠 Fresh AI & Markov Chain Sim: {full_n1} vs {full_n2} | T: {matched_tour_name}")
                    
                    p1_api_name_live = str(m.get('p1_raw', ''))
                    p2_api_name_live = str(m.get('p2_raw', ''))
                    
                    p1_key = None
                    p2_key = None
                    
                    if is_same_player(full_n1, p1_api_name_live): p1_key = m.get('p1_api_key')
                    elif is_same_player(full_n1, p2_api_name_live): p1_key = m.get('p2_api_key')
                    
                    if is_same_player(full_n2, p2_api_name_live): p2_key = m.get('p2_api_key')
                    elif is_same_player(full_n2, p1_api_name_live): p2_key = m.get('p1_api_key')

                    api_stats_p1 = []
                    if p1_key:
                        try:
                            raw_api_stats1 = await api.get_player_stats(str(p1_key))
                            api_stats_p1 = raw_api_stats1.get('stats', [])
                        except: pass
                        
                    api_stats_p2 = []
                    if p2_key:
                        try:
                            raw_api_stats2 = await api.get_player_stats(str(p2_key))
                            api_stats_p2 = raw_api_stats2.get('stats', [])
                        except: pass

                    p1_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p1_history, full_n1, api_stats_p1)
                    p2_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p2_history, full_n2, api_stats_p2)
                    
                    try:
                        target_p1_id = p1_obj.get('id')
                        if target_p1_id:
                            supabase.table('players').update({
                                'form_rating': p1_form_v2,
                                'surface_ratings': p1_surface_profile
                            }).eq('id', target_p1_id).execute()
                        
                        target_p2_id = p2_obj.get('id')
                        if target_p2_id:
                            supabase.table('players').update({
                                'form_rating': p2_form_v2,
                                'surface_ratings': p2_surface_profile
                            }).eq('id', target_p2_id).execute()
                            
                    except Exception as update_err:
                        log(f"⚠️ Failed to update live player ratings: {update_err}")

                    empirical_ou = calculate_empirical_ou(p1_history, p2_history)

                    sim_result = QuantumGamesSimulator.run_simulation(s1, s2, bsi, surf, actual_ou_line=m.get('actual_ou_line'), empirical_ou=empirical_ou)
                    
                    styleA = p1_obj.get('play_style', '')
                    styleB = p2_obj.get('play_style', '')
                    
                    mc_results = MarkovChainEngine.run_simulation(
                        s1=s1, s2=s2,
                        formA=p1_form_v2['score'], formB=p2_form_v2['score'],
                        bsi=bsi, styleA=styleA, styleB=styleB,
                        iterations=2500
                    )
                    
                    h2h_record = "0 - 0"
                    if m.get('p1_api_key') and m.get('p2_api_key'):
                        try:
                            h2h_response = await api.get_h2h(str(m['p1_api_key']), str(m['p2_api_key']))
                            
                            if h2h_response and isinstance(h2h_response, dict):
                                h2h_matches = h2h_response.get("H2H", [])
                                
                                if isinstance(h2h_matches, list) and len(h2h_matches) > 0:
                                    w1, w2 = 0, 0
                                    
                                    for h_match in h2h_matches:
                                        winner = str(h_match.get("event_winner", "")).lower()
                                        if winner:
                                            if winner == "first player":
                                                if is_same_player(full_n1, str(h_match.get("event_first_player", ""))): w1 += 1
                                                else: w2 += 1
                                            elif winner == "second player":
                                                if is_same_player(full_n1, str(h_match.get("event_second_player", ""))): w1 += 1
                                                else: w2 += 1
                                            else:
                                                if is_same_player(full_n1, winner): w1 += 1
                                                else: w2 += 1
                                    
                                    h2h_record = f"{w1} - {w2}"
                        except Exception as e:
                            log(f"⚠️ H2H Error: {e}")

                    sim_result["h2h"] = h2h_record

                    ai = await analyze_match_with_ai(
                        matched_tour_name, p1_obj, p2_obj, s1, s2, report1, report2, surf, bsi, notes, 
                        p1_form_v2, p2_form_v2, weather_data, p1_surface_profile, p2_surface_profile, mc_results, h2h_record
                    )
                    
                    prob = calculate_physics_fair_odds(full_n1, full_n2, s1, s2, bsi, surf, ai['mc_prob_a'], m['odds1'], m['odds2'], pinnacle_prob_a)
                    
                    fair1 = round(1/prob, 2) if prob > 0.01 else 99
                    fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                    
                    p1_set_prob = math.pow(prob, 0.65)
                    p2_set_prob = math.pow(1.0 - prob, 0.65)
                    
                    sim_result['set_probs'] = {
                        "2:0": round((p1_set_prob * p1_set_prob) * 100, 1),
                        "2:1": round((prob - (p1_set_prob * p1_set_prob)) * 100, 1),
                        "0:2": round((p2_set_prob * p2_set_prob) * 100, 1),
                        "1:2": round(((1.0 - prob) - (p2_set_prob * p2_set_prob)) * 100, 1)
                    }
                    
                    sim_result['projected_handicap'] = round((prob - 0.50) * 100 * 0.14, 2)
                    sim_result['bookmaker_set_odds'] = m.get('bookie_set_odds', {})

                    val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                    val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                    
                    value_tag = ""
                    if val_p1["is_value"]: 
                        value_tag = f"\n\n[{val_p1['type']}: {full_n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}% | Stake: {val_p1['kelly_stake']}u]"
                        hist_is_value = True
                        hist_pick_player = full_n1
                    elif val_p2["is_value"]: 
                        value_tag = f"\n\n[{val_p2['type']}: {full_n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}% | Stake: {val_p2['kelly_stake']}u]"
                        hist_is_value = True
                        hist_pick_player = full_n2

                    derivative_note = f"\n[{sim_result['derivative_edge']}]" if sim_result.get('derivative_edge') else ""
                    ai_text_final = f"{ai['ai_text']} {value_tag}{derivative_note}\n[🎲 SIM: {sim_result['predicted_line']} Games]"

                    data = {
                        "player1_name": full_n1, 
                        "player2_name": full_n2, 
                        "tournament": matched_tour_name,
                        "ai_fair_odds1": fair1, 
                        "ai_fair_odds2": fair2,
                        "ai_analysis_text": ai_text_final, 
                        "games_prediction": sim_result, 
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), 
                        "match_time": final_time_str, 
                        "odds1": m['odds1'], 
                        "odds2": m['odds2'],
                        "bookmaker_odds": m['bookmaker_odds'],
                        "is_visible_in_scanner": True,
                        "api_match_key": m['api_match_key']
                    }
                    
                    hist_fair1 = fair1
                    hist_fair2 = fair2
                    
                    if db_match_id: 
                        try:
                            supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                        except: pass
                    else:
                        data["opening_odds1"] = m['odds1']
                        data["opening_odds2"] = m['odds2']
                        try:
                            res_ins = supabase.table("market_odds").insert(data).execute()
                            if res_ins.data: 
                                db_match_id = res_ins.data[0]['id']
                            log(f"💾 Saved New Match: {full_n1} vs {full_n2} - Open: {m['odds1']} | {m['odds2']}")
                        except Exception as ins_err: pass

            if db_match_id:
                should_log_history = False
                if not existing_match:
                    should_log_history = True
                elif is_signal_locked or hist_is_value: 
                    should_log_history = True
                else:
                    try:
                        old_o1 = to_float(existing_match.get('odds1'), 0)
                        old_o2 = to_float(existing_match.get('odds2'), 0)
                        if round(old_o1, 3) != round(m['odds1'], 3) or round(old_o2, 3) != round(m['odds2'], 3):
                            should_log_history = True
                    except: pass
                
                if should_log_history:
                    h_data = {
                        "match_id": db_match_id, 
                        "odds1": m['odds1'], 
                        "odds2": m['odds2'], 
                        "fair_odds1": hist_fair1, 
                        "fair_odds2": hist_fair2, 
                        "is_hunter_pick": (hist_is_value or is_signal_locked),
                        "pick_player_name": "LOCKED" if is_signal_locked else hist_pick_player,
                        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    try: 
                        supabase.table("odds_history").insert(h_data).execute()
                    except: pass

        except Exception as e: 
            log(f"⚠️ Match Error bei Iteration: {e}")
            
    log(f"📊 SUMMARY: {db_matched_count} relevante DB-Matches erfolgreich prozessiert.")
        
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    async def main():
        await run_pipeline()
    
    asyncio.run(main())
