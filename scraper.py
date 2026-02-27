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
import numpy as np # SOTA: Required for Neural Grid Search

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

log("🔌 Initialisiere Neural Scout (V150.3 - SOTA VISIBILITY FIX)...")

# Secrets Load
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub/Groq Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# SOTA Model Selection - Free Tier Optimized
MODEL_NAME = 'llama-3.1-8b-instant'

# Global Caches & Dynamic Memory
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
METADATA_CACHE: Dict[str, Any] = {} 
WEATHER_CACHE: Dict[str, Any] = {} 
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 
TML_MATCH_CACHE: List[Dict] = [] 

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
    n = name.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def parse_te_name(raw: str):
    """Extrahiert aus 'Cerundolo F.' -> Nachname: 'cerundolo', Initiale: 'f'"""
    clean = re.sub(r'\s*\(\d+\)|\s*\(.*?\)', '', raw).strip()
    parts = clean.split()
    if not parts: return "", ""
    
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
                if db_first in scrape_first_part or scrape_first_part in db_first:
                    score += 30
                elif len(db_first) >= 5 and get_similarity(db_first, scrape_first_part) >= 0.80:
                    score += 25
                else:
                    sf_tokens = scrape_first_part.split()
                    if sf_tokens and len(sf_tokens[0]) > 0 and sf_tokens[0][0] == db_first[0]:
                        score += 15
                    # 🚀 SOTA FIX: Brother Penalty!
                    elif sf_tokens and len(sf_tokens[0]) > 0 and db_first and sf_tokens[0][0] != db_first[0]:
                        score -= 100 
        else:
            if all(t in scrape_tokens for t in db_last_tokens):
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
                    if any(ft in scrape_tokens for ft in db_first_tokens):
                        score += 30
                    else:
                        # 🚀 SOTA FIX: Check Initialien zur Brüder-Trennung (z.B. Cerundolo F.)
                        db_f_init = db_first[0]
                        has_contradicting = False
                        has_matching = False
                        for st in scrape_tokens:
                            if len(st) <= 2:
                                c_st = st.replace('.', '')
                                if len(c_st) == 1:
                                    if c_st == db_f_init: has_matching = True
                                    else: has_contradicting = True
                        if has_matching: 
                            score += 20
                        elif has_contradicting: 
                            score -= 100 # Brother Penalty!
                            
        if score >= 60: 
            candidates.append((p, score))
                
    if not candidates: 
        return None
        
    # SOTA FIX: THE ULTIMATE TIE-BREAKER FOR BROTHERS
    candidates.sort(key=lambda x: (x[1], x[0]['id'] in report_ids), reverse=True)
    
    if len(candidates) > 1:
        top_score = candidates[0][1]
        second_score = candidates[1][1]
        
        if top_score == second_score:
            top_has_report = candidates[0][0]['id'] in report_ids
            second_has_report = candidates[1][0]['id'] in report_ids
            
            # Wenn beide exakt gleichauf sind (Gleicher Score, gleicher Report-Status) -> TIE!
            if top_has_report == second_has_report:
                p1_n = f"{candidates[0][0].get('first_name')} {candidates[0][0].get('last_name')}"
                p2_n = f"{candidates[1][0].get('first_name')} {candidates[1][0].get('last_name')}"
                log(f"🚨 TIE-BREAKER ALARM: '{scraped_name_raw}' ist mehrdeutig zwischen {p1_n} und {p2_n}. Match wird sicherheitshalber ignoriert!")
                return None
                
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
# 3. MOMENTUM V2 ENGINE
# =================================================================
class MomentumV2Engine:
    @staticmethod
    def calculate_rating(matches: List[Dict], player_name: str, max_matches: int = 15) -> Dict[str, Any]:
        if not matches: 
            return {"score": 5.0, "text": "Neutral (No Data)", "history_summary": "", "color_hex": "#808080"}

        recent_matches = sorted(matches, key=lambda x: x.get('created_at', ''), reverse=True)[:max_matches]
        chrono_matches = recent_matches[::-1]

        rating = 5.5 
        momentum = 0.0
        history_log = []

        for idx, m in enumerate(chrono_matches):
            p_name_lower = player_name.lower()
            is_p1 = p_name_lower in m['player1_name'].lower()
            winner = m.get('actual_winner_name', "") or ""
            won = p_name_lower in winner.lower()

            odds = m.get('odds1', 1.50) if is_p1 else m.get('odds2', 1.50)
            if not odds or odds <= 1.0: 
                odds = 1.50

            is_recent = idx >= (len(chrono_matches) - 5)
            weight = 1.5 if is_recent else 0.8
            impact = 0.0

            if won:
                if odds < 1.30: 
                    impact = 0.4      
                elif odds <= 2.00: 
                    impact = 0.8   
                else: 
                    impact = 1.8                
                    
                score = str(m.get('score', ''))
                if score and "2-1" not in score and "1-2" not in score: 
                    impact += 0.3
                    
                momentum += 0.2 
                history_log.append("W")
            else:
                if odds < 1.30: 
                    impact = -0.6      
                elif odds < 1.50: 
                    impact = -0.5
                elif odds <= 2.20: 
                    impact = -0.6  
                else: 
                    impact = -0.2                
                    
                score = str(m.get('score', ''))
                if "2-1" in score or "1-2" in score: 
                    momentum = max(0.0, momentum - 0.1)
                else: 
                    momentum = 0.0 
                    
                history_log.append("L")
            
            rating += (impact * weight)

        rating += momentum
        final_rating = max(1.0, min(10.0, rating))
        
        desc = "Average"
        color_hex = "#F0C808" 
        
        if final_rating > 8.5: 
            desc = "🔥 ELITE"
            color_hex = "#FF00FF" 
        elif final_rating > 7.0: 
            desc = "📈 Strong"
            color_hex = "#3366FF" 
        elif final_rating >= 6.0: 
            color_hex = "#00B25B" 
        elif final_rating < 4.0: 
            desc = "❄️ Cold"
            color_hex = "#CC0000" 
        elif final_rating < 5.5: 
            desc = "⚠️ Weak"
            color_hex = "#FF9933" 

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
                profile[surf] = {
                    "rating": 3.5, 
                    "color": "#808080",
                    "matches_tracked": 0,
                    "text": "No Experience"
                }
                continue
                
            wins = 0
            for m in surf_matches:
                winner = m.get('actual_winner_name', "") or ""
                if player_name.lower() in winner.lower(): 
                    wins += 1
                    
            win_rate = wins / n_surf
            vol_score = min(1.0, n_surf / 30.0) * 1.95
            win_score = win_rate * 4.55
            
            final_rating = 3.5 + vol_score + win_score
            final_rating = max(1.0, min(10.0, final_rating))
            
            desc = "Average"
            color_hex = "#F0C808" 
            
            if final_rating >= 8.5: 
                desc = "🔥 SPECIALIST"
                color_hex = "#FF00FF" 
            elif final_rating >= 7.5: 
                desc = "📈 Strong"
                color_hex = "#3366FF" 
            elif final_rating >= 6.5: 
                color_hex = "#00B25B" 
            elif final_rating >= 5.5: 
                desc = "Solid"
                color_hex = "#99CC33" 
            elif final_rating <= 4.5: 
                desc = "⚠️ Vulnerable"
                color_hex = "#CC0000" 
            elif final_rating < 5.5: 
                desc = "❄️ Weakness"
                color_hex = "#FF9933" 

            profile[surf] = {
                "rating": round(final_rating, 2),
                "color": color_hex,
                "matches_tracked": n_surf,
                "text": desc
            }
            
        profile['_v95_mastery_applied'] = True
        return profile

# =================================================================
# 5. MONTE CARLO & TACTICAL EDGE ENGINE (NEW & DYNAMIC)
# =================================================================
class MonteCarloEngine:
    @staticmethod
    def run_simulation(tour: str, skillA: float, formA: float, surfaceA: float, 
                       skillB: float, formB: float, surfaceB: float, 
                       iterations: int = 1000) -> Dict[str, float]:
        
        # SOTA FIX: Dynamic Weights Retrieval
        weights = DYNAMIC_WEIGHTS.get(tour, DYNAMIC_WEIGHTS["ATP"])
        w_skill = weights["SKILL"]
        w_form = weights["FORM"]
        w_surf = weights["SURFACE"]
        variance = weights.get("MC_VARIANCE", 1.20)
        
        baseA = (skillA / 10) * w_skill + formA * w_form + surfaceA * w_surf
        baseB = (skillB / 10) * w_skill + formB * w_form + surfaceB * w_surf
        
        # Fish out of water penalty
        if surfaceA < 4.5:
            baseA -= (4.5 - surfaceA) * 1.5
        if surfaceB < 4.5:
            baseB -= (4.5 - surfaceB) * 1.5
            
        winsA = 0
        winsB = 0
        
        for _ in range(iterations):
            simA = random.gauss(baseA, variance)
            simB = random.gauss(baseB, variance)
            
            if simA > simB:
                winsA += 1
            else:
                winsB += 1
                
        probA = (winsA / iterations) * 100
        probB = (winsB / iterations) * 100
        
        return {
            "probA": round(probA, 1),
            "probB": round(probB, 1),
            "scoreA": baseA,
            "scoreB": baseB
        }

# =================================================================
# 5.5 SOTA: SELF-LEARNING NEURAL OPTIMIZER
# =================================================================
class NeuralOptimizer:
    @staticmethod
    def optimize_ai_weights(matches_history: List[Dict], current_weights: Dict) -> Dict:
        log("🧠 Starte Neural Weight Optimization (Backpropagation Simulation)...")
        
        best_weights = current_weights
        best_brier_score = float('inf') 
        
        # GRID SEARCH: Wir testen verschiedene Gewichtungen auf vergangenen Daten
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
        log("🔄 Initiating Weekly Self-Learning Routine...")
        
        for tour in ["ATP", "WTA"]:
            tour_players = [p['id'] for p in players if p.get('tour') == tour]
            if not tour_players: continue
            
            # Hole die letzten beendeten Matches für diese Tour
            recent_res = supabase.table("market_odds").select("*").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(200).execute()
            recent_matches = recent_res.data or []
            
            history_data = []
            correct_predictions = 0
            total_predictions = 0
            
            for m in recent_matches:
                p1_name = m.get('player1_name', '')
                p2_name = m.get('player2_name', '')
                winner = m.get('actual_winner_name', '')
                
                # Check System Accuracy
                fair1 = to_float(m.get('ai_fair_odds1'), 0)
                fair2 = to_float(m.get('ai_fair_odds2'), 0)
                if fair1 > 0 and fair2 > 0:
                    total_predictions += 1
                    if (fair1 < fair2 and p1_name.lower() in winner.lower()) or (fair2 < fair1 and p2_name.lower() in winner.lower()):
                        correct_predictions += 1

                # Build Grid Search History
                p1_obj = next((p for p in players if p.get('last_name') == p1_name and p['id'] in tour_players), None)
                p2_obj = next((p for p in players if p.get('last_name') == p2_name and p['id'] in tour_players), None)
                
                if p1_obj and p2_obj:
                    s1 = all_skills.get(p1_obj['id'], {}).get('overall_rating', 50)
                    s2 = all_skills.get(p2_obj['id'], {}).get('overall_rating', 50)
                    # Approximation: Use default 5.0 for historical form/surf to keep it lightweight, 
                    # or pull from their current state if no snapshot exists.
                    history_data.append({
                        "skillA": s1, "formA": 5.5, "surfA": 5.5,
                        "skillB": s2, "formB": 5.5, "surfB": 5.5,
                        "winner_is_A": p1_name.lower() in winner.lower()
                    })

            # 1. Update Global Accuracy
            if total_predictions > 0:
                acc = (correct_predictions / total_predictions) * 100
                SYSTEM_ACCURACY[tour] = round(acc, 1)
                log(f"🎯 {tour} System Accuracy Rating: {SYSTEM_ACCURACY[tour]}% ({correct_predictions}/{total_predictions})")

            # 2. Run Grid Search
            if len(history_data) >= 20:
                new_weights = NeuralOptimizer.optimize_ai_weights(history_data, DYNAMIC_WEIGHTS[tour])
                DYNAMIC_WEIGHTS[tour] = new_weights
                
                # 3. Save to Supabase
                try:
                    supabase.table("ai_system_weights").upsert({
                        "tour": tour,
                        "weight_skill": new_weights["SKILL"],
                        "weight_form": new_weights["FORM"],
                        "weight_surface": new_weights["SURFACE"],
                        "mc_variance": new_weights["MC_VARIANCE"],
                        "last_optimized": datetime.now(timezone.utc).isoformat()
                    }).execute()
                    log(f"💾 {tour} Gewichte erfolgreich in Supabase gesichert.")
                except Exception as e:
                    log(f"❌ Fehler beim Speichern der AI-Gewichte: {e}")

# =================================================================
# 6. GROQ ENGINE
# =================================================================
async def call_groq(prompt: str, model: str = MODEL_NAME, temp: float = 0.1) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
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
                log(f"⚠️ Groq API Error: {response.status_code} - {response.text}")
                return None
            return response.json()['choices'][0]['message']['content']
        except Exception as e: 
            log(f"⚠️ Groq Exception: {e}")
            return None

# =================================================================
# 6.5 1WIN SOTA MASTER FEED (ODDS SCRAPER)
# =================================================================
def extract_time_context(lines_slice: List[str]) -> str:
    full_text = " ".join(lines_slice)
    
    # 1. Prio: SOTA Methode für das exakte "Zeit • Datum" Format
    sota_match = re.search(r'(\d{1,2}:\d{2})\s*[•\.\-|]\s*(\d{1,2}\.\d{1,2}\.\d{2,4})', full_text)
    if sota_match:
        return f"{sota_match.group(2)} {sota_match.group(1)}"

    # 2. Prio: HYBRID Fallback für "Heute", "Morgen" oder fehlende Trennzeichen
    date_patterns = [
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b',
        r'\b\d{1,2}[\./]\d{1,2}(?:\.\d{2,4})?\b',
        r'\bToday\b', r'\bTomorrow\b', r'\bHeute\b', r'\bMorgen\b',
        r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b'
    ]
    time_patterns = [r'\b\d{1,2}:\d{2}\b']
    
    found_date = ""
    found_time = "00:00"
    
    for l in lines_slice:
        for dp in date_patterns:
            m = re.search(dp, l, re.IGNORECASE)
            if m: 
                found_date = m.group(0)
        for tp in time_patterns:
            m = re.search(tp, l)
            if m: 
                found_time = m.group(0)
            
    if found_date:
        return f"{found_date} {found_time}"
        
    return found_time

def parse_time_to_iso(raw_time_str: str) -> str:
    if not raw_time_str or raw_time_str == "00:00":
        return f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}T00:00:00Z"
        
    try:
        t_match = re.search(r'(\d{1,2}):(\d{2})', raw_time_str)
        d_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})', raw_time_str)
        
        h, m_min = t_match.groups() if t_match else ("00", "00")
        
        if d_match:
            day, month, year = d_match.groups()
            y = int(year)
            if y < 100: 
                y += 2000 # Korrigiert 2-stellige Jahre (26 -> 2026)
            dt = datetime(y, int(month), int(day), int(h), int(m_min), tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif t_match:
            now = datetime.now(timezone.utc)
            dt = now.replace(hour=int(h), minute=int(m_min), second=0, microsecond=0)
            if dt < now - timedelta(hours=3): 
                dt += timedelta(days=1)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        pass
        
    return f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}T00:00:00Z"

async def fetch_1win_markets_spatial_stream(browser: Browser, db_players: List[Dict]) -> List[Dict]:
    log("🚀 [1WIN GHOST] Starte SOTA Database Surgeon (V144.1)...")
    
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
        java_script_enabled=True
    )
    
    await context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); window.navigator.chrome = { runtime: {} };")
    page = await context.new_page()

    parsed_matches = []
    seen_matches = set()
    all_raw_text_blocks = [] 

    try:
        log("🌍 Navigiere im Stealth-Modus zu 1win...")
        await page.goto("https://1win.io/betting/prematch/tennis-33", wait_until="networkidle", timeout=60000)
        
        page_title = await page.title()
        if "Just a moment" in page_title or "Cloudflare" in page_title:
            log("🛑 WARNUNG: Cloudflare Challenge aktiv! Warte 5 Sekunden...")
            await asyncio.sleep(5)
            
        log("⏳ Aktiviere Safe Smart-Clicker (Accordions & Show More)...")
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
                            
                            if (text.length < 80 && (textLow.includes('atp') || textLow.includes('wta') || textLow.includes('itf') || textLow.includes('challenger') || textLow.includes('utr'))) {
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
                
                text_dump = await page.evaluate("document.body.innerText")
                all_raw_text_blocks.append(text_dump)
                
                await page.mouse.wheel(delta_x=0, delta_y=500)
                await asyncio.sleep(0.5) 
                
            except Exception as scroll_e: 
                continue
                
    except Exception as e: 
        log(f"⚠️ [1WIN GHOST] Timeout/Fehler beim Laden: {e}")
    finally: 
        await context.close()

    log(f"🧩 Erzeuge Stream...")
    unified_lines = []
    
    for block in all_raw_text_blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        for line in lines:
            if not unified_lines or unified_lines[-1] != line:
                unified_lines.append(line)

    log(f"🧠 Starte Dynamic Anchor Extraktion auf {len(unified_lines)} Zeilen...")
    
    current_tour = "Unknown"
    
    for i in range(len(unified_lines)):
        line = unified_lines[i]
        line_norm = normalize_text(line).lower()
        
        if len(line) < 100 and any(kw in line_norm for kw in ['atp', 'wta', 'open', 'masters', 'tour', 'challenger', 'itf', 'utr']):
            if "sieger" not in line_norm and "winner" not in line_norm:
                current_tour = line
        
        if line_norm == "sieger" or line_norm == "winner":
            up_slice = unified_lines[max(0, i-6):i]
            potential_names = []
            
            for ul in up_slice:
                ul_norm = normalize_text(ul).lower()
                
                if len(ul) < 3: continue
                if re.search(r'\d', ul): continue
                if any(kw in ul_norm for kw in ['atp', 'wta', 'itf', 'challenger', 'utr', 'tennis', 'live', 'stream', 'tv', 'satz', 'set']): continue
                if "•" in ul or ":" in ul: continue
                
                potential_names.append(ul)

            if len(potential_names) >= 2:
                p2_raw = potential_names[-1]
                p1_raw = potential_names[-2]
                    
                match_key = tuple(sorted([p1_raw, p2_raw]))
                if match_key in seen_matches:
                    continue
                    
                time_slice = unified_lines[max(0, i-5):i]
                extracted_time = extract_time_context(time_slice)
                
                odds_slice = unified_lines[i+1 : min(i+12, len(unified_lines))]
                floats = []
                for ol in odds_slice:
                    cl = ol.replace(',', '.').strip()
                    matches_val = re.findall(r'\b\d+\.\d{1,3}\b', cl) 
                    for m_val in matches_val:
                        try:
                            val = float(m_val)
                            if 1.0 < val <= 250.0:
                                floats.append(val)
                        except: 
                            pass
                            
                if len(floats) >= 2:
                    o1, o2 = floats[0], floats[1]
                    
                    try:
                        implied = (1/o1) + (1/o2)
                        if 1.01 <= implied <= 1.45: 
                            seen_matches.add(match_key)
                            parsed_matches.append({
                                "p1_raw": p1_raw, 
                                "p2_raw": p2_raw,
                                "tour": clean_tournament_name(current_tour),
                                "time": extracted_time, 
                                "odds1": o1, 
                                "odds2": o2,
                                "handicap_line": None, "handicap_odds1": 0, "handicap_odds2": 0,
                                "over_under_line": None, "over_odds": 0, "under_odds": 0,
                                "actual_winner": None, "score": ""
                            })
                    except:
                        pass

    log(f"✅ [1WIN GHOST] {len(parsed_matches)} saubere DB-Matches extrahiert (Memory-Pool).")
    return parsed_matches

# =================================================================
# 7. DATA FETCHING & ORACLE (TE SETTLEMENT INTEGRATED)
# =================================================================
async def scrape_oracle_metadata(browser: Browser, target_date: datetime):
    date_str = target_date.strftime('%Y-%m-%d')
    url = f"https://de.tennistemple.com/matches/{date_str}"
    page = await browser.new_page()
    metadata = {}
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        current_tournament = "Unknown"
        for element in soup.find_all(['h2', 'a']):
            text = element.get_text(strip=True)
            href = element.get('href', '')
            if ('/turnier/' in href or '/tournament/' in href):
                current_tournament = text
                continue
            if ('/spieler/' in href or '/player/' in href):
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
        res = supabase.table("market_odds").select("player1_name, player2_name, odds1, odds2, actual_winner_name, score, created_at, tournament, ai_analysis_text").or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(limit).execute()
        return res.data or []
    except: 
        return []

async def fetch_tennisexplorer_stats(browser: Browser, relative_url: str, surface: str) -> float:
    if not relative_url: 
        return 0.5
    cache_key = f"{relative_url}_{surface}"
    if cache_key in SURFACE_STATS_CACHE: 
        return SURFACE_STATS_CACHE[cache_key]
        
    if not relative_url.startswith("/"): 
        relative_url = f"/{relative_url}"
        
    url = f"https://www.tennisexplorer.com{relative_url}?annual=all&t={int(time.time())}"
    page = await browser.new_page()
    
    try:
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        target_header = "Hard"
        
        if "clay" in surface.lower(): 
            target_header = "Clay"
        elif "grass" in surface.lower(): 
            target_header = "Grass"
        elif "indoor" in surface.lower(): 
            target_header = "Indoors"
            
        tables = soup.find_all('table', class_='result')
        total_matches = 0
        total_wins = 0
        
        for table in tables:
            headers = [h.get_text(strip=True) for h in table.find_all('th')]
            if "Summary" in headers and target_header in headers:
                try:
                    col_idx = headers.index(target_header)
                    for row in table.find_all('tr'):
                        if "Summary" in row.get_text():
                            cols = row.find_all(['td', 'th'])
                            if len(cols) > col_idx:
                                stats_text = cols[col_idx].get_text(strip=True)
                                if "/" in stats_text:
                                    w, l = map(int, stats_text.split('/'))
                                    total_matches = w + l
                                    total_wins = w
                                    break
                except: 
                    pass
                break
                
        if total_matches > 0:
            rate = total_wins / total_matches
            SURFACE_STATS_CACHE[cache_key] = rate
            return rate
            
    except: 
        pass
    finally: 
        await page.close()
        
    return 0.5

# L8 SOTA: THE RELIABLE RESULTS ENGINE (TE INTEGRATION)
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
            url = f"https://www.tennisexplorer.com/results/?type=all&year={t_date.year}&month={t_date.month}&day={t_date.day}"
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            soup = BeautifulSoup(await page.content(), 'html.parser')
            table = soup.find('table', class_='result')
            if not table: continue
            
            def extract_row_data(r_row):
                cells = r_row.find_all('td')
                p_idx = -1
                for idx, c in enumerate(cells):
                    if c.find('a') and 'time' not in c.get('class', []):
                        p_idx = idx
                        break
                
                if p_idx != -1 and p_idx + 1 < len(cells):
                    sets_cell = cells[p_idx + 1]
                    sets_val = sets_cell.get_text(strip=True)
                    
                    if sets_val.isdigit():
                        scores = []
                        for k in range(1, 6):
                            if p_idx + 1 + k >= len(cells): break
                            sc_cell = cells[p_idx + 1 + k]
                            
                            raw_txt = ""
                            for child in sc_cell.children:
                                if child.name == 'sup': continue
                                raw_txt += str(child).strip() if isinstance(child, str) else child.get_text(strip=True)
                            
                            raw_txt = re.sub(r'<[^>]+>', '', raw_txt).strip()
                            
                            if raw_txt.isdigit():
                                scores.append(raw_txt)
                            else:
                                break
                        return int(sets_val), scores
                return -1, []

            # -------------------------------------------------------------
            # 🚀 SOTA FIX: HELPER FÜR EXAKTEN DB -> TE ABGLEICH (Brüder & Substrings)
            # -------------------------------------------------------------
            def match_player_db_te(db_full_name, te_last, te_init):
                db_n = normalize_db_name(db_full_name)
                db_parts = db_n.split()
                db_last = db_parts[-1] if db_parts else ""
                db_first = db_parts[0] if len(db_parts) > 1 else ""
                
                te_last_tokens = set(te_last.split())
                db_n_tokens = set(db_parts)

                is_last_match = False
                
                if te_last == db_last:
                    is_last_match = True
                elif db_n_tokens.intersection(te_last_tokens):
                    is_last_match = True
                elif (len(te_last) >= 5 and te_last in db_n) or (len(db_last) >= 5 and db_last in te_last):
                    is_last_match = True

                if is_last_match:
                    if te_init and db_first:
                        if db_first.startswith(te_init): return True
                        else: return False 
                    return True
                return False

            rows = table.find_all('tr')
            i = 0
            pending_p1_raw = None
            pending_p1_row = None
            
            while i < len(rows):
                row = rows[i]
                
                if 'head' in row.get('class', []):
                    pending_p1_raw = None
                    i += 1; continue
                
                cols = row.find_all('td')
                if len(cols) < 2: 
                    i += 1; continue
                
                p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
                if not p_cell: 
                    i += 1; continue
                
                p_raw = clean_player_name(p_cell.get_text(strip=True))
                
                if pending_p1_raw:
                    p2_raw = p_raw
                    if '/' in pending_p1_raw or '/' in p2_raw: 
                        pending_p1_raw = None; i += 1; continue
                    
                    te_last1, te_init1 = parse_te_name(pending_p1_raw)
                    te_last2, te_init2 = parse_te_name(p2_raw)
                    
                    matched_pm = None
                    is_reversed = False
                    
                    for pm in list(safe_to_check):
                        if match_player_db_te(pm['player1_name'], te_last1, te_init1) and match_player_db_te(pm['player2_name'], te_last2, te_init2):
                            matched_pm = pm
                            break
                        elif match_player_db_te(pm['player1_name'], te_last2, te_init2) and match_player_db_te(pm['player2_name'], te_last1, te_init1):
                            matched_pm = pm
                            is_reversed = True
                            break

                    if matched_pm:
                        s1, scores1 = extract_row_data(pending_p1_row)
                        s2, scores2 = extract_row_data(row)
                        
                        winner = None
                        final_score = ""
                        
                        is_ret = "ret." in pending_p1_row.get_text().lower() or "ret." in row.get_text().lower() or "w.o." in pending_p1_row.get_text().lower() or "w.o." in row.get_text().lower()
                        
                        if s1 != -1 and s2 != -1:
                            if s1 > s2: 
                                winner = matched_pm['player2_name'] if is_reversed else matched_pm['player1_name']
                            elif s2 > s1: 
                                winner = matched_pm['player1_name'] if is_reversed else matched_pm['player2_name']
                            
                            score_parts = []
                            min_len = min(len(scores1), len(scores2))
                            for k in range(min_len):
                                if is_reversed:
                                    score_parts.append(f"{scores2[k]}-{scores1[k]}")
                                else:
                                    score_parts.append(f"{scores1[k]}-{scores2[k]}")
                            
                            if score_parts:
                                final_score = " ".join(score_parts)
                                if is_ret: final_score += " ret."

                        if winner:
                            log(f"      🔍 AUDITOR FOUND: {final_score} -> Winner: {winner}")
                            supabase.table("market_odds").update({
                                "actual_winner_name": winner,
                                "score": final_score
                            }).eq("id", matched_pm['id']).execute()

                            # ==========================================
                            # LIVE SKILL & FORM UPDATE ENGINE
                            # ==========================================
                            log(f"🧠 Triggere Live Skill & Form Engine für das Match...")
                            
                            p1_name = matched_pm['player1_name']
                            p2_name = matched_pm['player2_name']
                            
                            p1_id = next((p['id'] for p in players if p.get('last_name') == p1_name), None)
                            p2_id = next((p['id'] for p in players if p.get('last_name') == p2_name), None)
                            
                            if p1_id and p2_id:
                                try:
                                    skills_res = supabase.table('player_skills').select('*').in_('player_id', [p1_id, p2_id]).execute()
                                    db_skills = skills_res.data or []
                                    
                                    p1_skills_db = next((s for s in db_skills if s.get('player_id') == p1_id), None)
                                    p2_skills_db = next((s for s in db_skills if s.get('player_id') == p2_id), None)
                                    
                                    odds1 = to_float(matched_pm.get('odds1', 1.85))
                                    odds2 = to_float(matched_pm.get('odds2', 1.85))
                                    
                                    if p1_skills_db:
                                        new_s1 = LiveSkillEngine.calculate_new_skills(p1_skills_db, odds1, (winner == p1_name), final_score, is_player1=True)
                                        if new_s1:
                                            new_s1['updated_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                            supabase.table('player_skills').update(new_s1).eq('player_id', p1_id).execute()
                                            
                                    if p2_skills_db:
                                        new_s2 = LiveSkillEngine.calculate_new_skills(p2_skills_db, odds2, (winner == p2_name), final_score, is_player1=False)
                                        if new_s2:
                                            new_s2['updated_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                            supabase.table('player_skills').update(new_s2).eq('player_id', p2_id).execute()
                                except Exception as se:
                                    log(f"⚠️ Live Skill Update Error: {se}")

                            # RE-PROFILING HOOK (Form & Surface Update)
                            for p_name_hook in [matched_pm['player1_name'], matched_pm['player2_name']]:
                                p_hist = await fetch_player_history_extended(p_name_hook, limit=80)
                                p_profile = SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name_hook)
                                p_form = MomentumV2Engine.calculate_rating(p_hist[:20], p_name_hook)
                                
                                supabase.table('players').update({
                                    'surface_ratings': p_profile,
                                    'form_rating': p_form 
                                }).ilike('last_name', f"%{p_name_hook}%").execute()

                        safe_to_check = [x for x in safe_to_check if x['id'] != matched_pm['id']]

                    pending_p1_raw = None
                    pending_p1_row = None
                else:
                    first_cell = row.find('td', class_='first')
                    if first_cell and first_cell.get('rowspan') == '2': 
                        pending_p1_raw = p_raw
                        pending_p1_row = row
                    else: 
                        pending_p1_raw = p_raw
                        pending_p1_row = row
                i += 1
        except Exception as e:
            pass
        finally: 
            await page.close()

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
            score_str = str(last_match['score']).lower()
            if 'ret' in score_str or 'wo' in score_str: 
                fatigue_score *= 0.5
            else:
                sets = len(re.findall(r'(\d+)-(\d+)', score_str))
                tiebreaks = len(re.findall(r'7-6|6-7', score_str))
                total_games = 0
                for s in re.findall(r'(\d+)-(\d+)', score_str):
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
                        sets_in_week += len(re.findall(r'\d+-\d+', str(m['score'])))
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

async def fetch_elo_ratings(browser: Browser):
    log("📊 Lade Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    for tour, url in urls.items():
        page = await browser.new_page()
        try:
            await page.goto(f"{url}?t={int(time.time())}", wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table', {'id': 'reportable'})
            
            if table:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) > 4:
                        name = normalize_text(cols[0].get_text(strip=True)).lower()
                        last_name = name.split()[-1] if " " in name else name
                        ELO_CACHE[tour][last_name] = {
                            'Hard': to_float(cols[3].get_text(strip=True), 1500),
                            'Clay': to_float(cols[4].get_text(strip=True), 1500),
                            'Grass': to_float(cols[5].get_text(strip=True), 1500)
                        }
        except: 
            pass
        finally: 
            await page.close()

async def get_db_data():
    try:
        # Fetch dynamic weights from DB for the Neural Optimizer
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

        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        
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
# 8. MATH CORE
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def normal_cdf_prob(elo_diff: float, sigma: float = 280.0) -> float:
    z = elo_diff / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def calculate_value_metrics(fair_prob: float, market_odds: float) -> Dict[str, Any]:
    if market_odds <= 1.01 or fair_prob <= 0: 
        return {"type": "NONE", "edge_percent": 0.0, "is_value": False}
        
    market_odds = min(market_odds, 100.0)
    edge = (fair_prob * market_odds) - 1
    edge_percent = round(edge * 100, 1)
    
    if edge_percent <= 0.5: 
        return {"type": "NONE", "edge_percent": edge_percent, "is_value": False}

    label = "VALUE"
    if edge_percent >= 15.0: 
        label = "🔥 HIGH VALUE" 
    elif edge_percent >= 8.0: 
        label = "✨ GOOD VALUE" 
    elif edge_percent >= 2.0: 
        label = "📈 THIN VALUE" 
    else: 
        label = "👀 WATCH"

    return {"type": label, "edge_percent": edge_percent, "is_value": True}

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, mc_prob_a, market_odds1, market_odds2):
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
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
        
    model_trust_factor = 0.35 
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
async def build_country_city_map(browser: Browser):
    if COUNTRY_TO_CITY_MAP: 
        return
    url = "https://www.unitedcup.com/en/scores/group-standings"
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=20000, wait_until="networkidle")
        text_content = await page.inner_text("body")
        prompt = f"TASK: Map Country to City (United Cup). Text: {text_content[:20000]}. JSON ONLY."
        res = await call_groq(prompt)
        if res:
            try:
                data = json.loads(res.replace("json", "").replace("```", "").strip())
                COUNTRY_TO_CITY_MAP.update(ensure_dict(data))
            except: 
                pass
    except: 
        pass
    finally: 
        await page.close()

async def resolve_united_cup_via_country(p1):
    if not COUNTRY_TO_CITY_MAP: 
        return None
        
    cache_key = f"COUNTRY_{p1}"
    
    if cache_key in TOURNAMENT_LOC_CACHE: 
        country = TOURNAMENT_LOC_CACHE[cache_key]
    else:
        res = await call_groq(f"Country of player {p1}? JSON: {{'country': 'Name'}}")
        try:
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            country = data.get("country", "Unknown")
        except: 
            country = "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
        
    if country in COUNTRY_TO_CITY_MAP: 
        return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
        
    return None

async def resolve_ambiguous_tournament(p1, p2, scraped_name, p1_country, p2_country):
    if scraped_name in TOURNAMENT_LOC_CACHE: 
        return TOURNAMENT_LOC_CACHE[scraped_name]
        
    prompt = f"TASK: Identify tournament location. MATCH: {p1} ({p1_country}) vs {p2} ({p2_country}). SOURCE: '{scraped_name}'. OUTPUT JSON: {{ 'city': 'Name', 'surface': 'Hard/Clay/Grass', 'indoor': true/false }}"
    res = await call_groq(prompt)
    
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
    return f"Serve: {s.get('serve', 50)}, FH: {s.get('forehand', 50)}, BH: {s.get('backhand', 50)}, Volley: {s.get('volley', 50)}, Speed: {s.get('speed', 50)}, Stamina: {s.get('stamina', 50)}, Power: {s.get('power', 50)}, Mental: {s.get('mental', 50)}"

async def analyze_match_with_ai(tour_name, p1, p2, s1, s2, report1, report2, surface, bsi, notes, form1_data, form2_data, weather_data, p1_surface_profile, p2_surface_profile, mc_results):
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
    finalProb = f"{mc_results['probA']}%" if aWins else f"{mc_results['probB']}%"

    # SOTA: Injecting Live Accuracy from Self-Learning Engine
    tour = "WTA" if "WTA" in tour_name.upper() else "ATP"
    sys_acc = SYSTEM_ACCURACY.get(tour, 65.0)

    prompt = f"""
    You are an elite Senior Tennis Analyst (Style: Gil Gross). 
    Your analysis must be grounded EXCLUSIVELY in the provided technical data and scouting reports.
    
    *** SYSTEM SELF-REFLECTION (CRITICAL) ***
    Our internal neural network has an active prediction accuracy of {sys_acc}%. 
    If this accuracy is below 65%, you MUST be more conservative in your tone and acknowledge potential variance. If it is above 70%, be highly assertive about the data trends.
    
    *** DATA GROUNDING (SOURCE OF TRUTH) ***
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
    
    *** CRITICAL DIRECTIVES (MUST OBEY) ***
    1. NO NUMBERS IN TEXT: Strictly forbidden to use percentages (%), numerical ratings (e.g., 8/10), or skill points in 'prediction_text' and 'key_factor'.
    2. TACTICAL PROSA: Use Gil Gross style "Matchup Physics". Explain how the specific "Court Notes" (bounce height, court speed) amplify a player's strengths or expose their weaknesses.
    3. FACTUAL INTEGRITY: If the Scouting Report says a player has poor movement, NEVER describe them as "athletic". Ground your analysis in the provided 'Weaknesses'.
    4. PATTERN ANALYSIS: Explain HOW {predictedMCWinner}'s specific skills interact with {predictedMCLoser}'s specific weaknesses under these exact court conditions.
    5. DO NOT EXPLAIN CALCULATIONS: Output strictly the JSON. No introductory chatter.
    
    OUTPUT JSON:
    {{
        "winner_prediction": "{predictedMCWinner}",
        "key_factor": "One sharp tactical sentence focusing on the primary technical mismatch (NO NUMBERS).",
        "prediction_text": "Deep Gil Gross style analysis (~200 words). Focus on tactical matchup physics, court conditions, and how the scouting report details manifest on court. STRICTLY NO NUMBERS OR PERCENTAGES.",
        "tactical_bullets": ["Tactic 1 based on report", "Tactic 2 based on report", "Tactic 3 based on report"]
    }}
    """
    
    res = await call_groq(prompt)
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
    def run_simulation(p1_skills: Dict, p2_skills: Dict, bsi: float, surface: str, iterations: int = 1000) -> Dict[str, Any]:
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
        
        return {
            "predicted_line": round(sum(total_games_log) / len(total_games_log), 1),
            "median_games": total_games_log[len(total_games_log)//2],
            "probabilities": {
                "over_20_5": sum(1 for x in total_games_log if x > 20.5) / iterations,
                "over_21_5": sum(1 for x in total_games_log if x > 21.5) / iterations,
                "over_22_5": sum(1 for x in total_games_log if x > 22.5) / iterations,
                "over_23_5": sum(1 for x in total_games_log if x > 23.5) / iterations
            },
            "sim_details": {
                "p1_est_hold_pct": round(p1_hold_prob * 100, 1), 
                "p2_est_hold_pct": round(p2_hold_prob * 100, 1)
            }
        }

# =================================================================
# 11. LIVE SKILL ENGINE
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

        score_lower = str(score).lower()
        sets = re.findall(r'(\d+)-(\d+)', score_lower)

        if is_winner and len(sets) == 2 and not "ret." in score_lower and not "w.o." in score_lower:
            for skill in ['power', 'serve', 'forehand', 'backhand', 'volley', 'speed']:
                if skill in new_skills: new_skills[skill] += 0.2

        if is_winner and len(sets) >= 3:
            if 'mental' in new_skills: new_skills['mental'] += 0.3
            if 'stamina' in new_skills: new_skills['stamina'] += 0.3

        lost_tiebreak = False
        for s in sets:
            l, r = int(s[0]), int(s[1])
            if is_player1 and l < r and r == 7: lost_tiebreak = True
            if not is_player1 and r < l and l == 7: lost_tiebreak = True

        if not is_winner and lost_tiebreak:
             if 'mental' in new_skills: new_skills['mental'] -= 0.2

        if not is_winner and "ret." in score_lower:
             if 'stamina' in new_skills: new_skills['stamina'] -= 0.5
             if 'speed' in new_skills: new_skills['speed'] -= 0.5

        new_overall = sum(new_skills[k] for k in skill_fields if k in new_skills) / len([k for k in skill_fields if k in new_skills])
        new_skills['overall_rating'] = new_overall

        for k, v in new_skills.items():
            new_skills[k] = max(1.0, min(99.0, round(v, 2)))

        return new_skills

# =================================================================
# 12. FANTASY SETTLEMENT ENGINE
# =================================================================
class FantasySettlementEngine:
    @staticmethod
    def run_settlement():
        log("🏆 [FANTASY ENGINE] Starte Settlement & Gameweek Management...")
        now = datetime.now(timezone.utc)
        
        res_gw = supabase.table("fantasy_gameweeks").select("*").eq("status", "active").execute()
        active_gws = res_gw.data or []
        
        for gw in active_gws:
            deadline = datetime.fromisoformat(gw['deadline_time'].replace('Z', '+00:00'))
            end_of_week = deadline + timedelta(days=7) 
            
            if now > end_of_week:
                log(f"📉 [FANTASY ENGINE] Deadline + 7 Tage erreicht. Settling Gameweek {gw['week_number']}...")
                FantasySettlementEngine.settle_gameweek(gw, deadline, end_of_week)
                
        res_latest = supabase.table("fantasy_gameweeks").select("*").order("start_time", desc=True).limit(1).execute()
        latest_gw = res_latest.data[0] if res_latest.data else None
        
        if not latest_gw or datetime.fromisoformat(latest_gw['deadline_time'].replace('Z', '+00:00')) < now:
            next_week_num = latest_gw['week_number'] + 1 if latest_gw else int(now.strftime("%V"))
            next_year = latest_gw['year'] if latest_gw else now.year
            if next_week_num > 52:
                next_week_num = 1
                next_year += 1
                
            days_ahead = 0 - now.weekday()
            if days_ahead <= 0: days_ahead += 7
            next_monday = now + timedelta(days=days_ahead)
            next_deadline = next_monday.replace(hour=8, minute=0, second=0, microsecond=0)
            
            new_gw = {
                "week_number": next_week_num,
                "year": next_year,
                "start_time": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "deadline_time": next_deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "active"
            }
            supabase.table("fantasy_gameweeks").insert(new_gw).execute()
            log(f"🌱 [FANTASY ENGINE] Neue Woche generiert: Week {next_week_num}, {next_year}")

    @staticmethod
    def settle_gameweek(gw: Dict, deadline: datetime, end_of_week: datetime):
        gw_id = gw['id']
        
        lineups_res = supabase.table("fantasy_lineups").select("*").eq("gameweek_id", gw_id).execute()
        lineups = lineups_res.data or []
        
        if not lineups:
            supabase.table("fantasy_gameweeks").update({"status": "completed"}).eq("id", gw_id).execute()
            log(f"⏭️ [FANTASY ENGINE] Gameweek {gw['week_number']} geschlossen (Keine Aufstellungen).")
            return

        matches_res = supabase.table("market_odds").select("*").not_.is_("actual_winner_name", "null").gte("created_at", deadline.isoformat()).lte("created_at", end_of_week.isoformat()).execute()
        matches = matches_res.data or []
        
        players_res = supabase.table("players").select("id, last_name, first_name").execute()
        players_map = {p['id']: p for p in (players_res.data or [])}

        for lineup in lineups:
            total_pts = 0
            for pid in [lineup.get('player1_id'), lineup.get('player2_id'), lineup.get('player3_id')]:
                if not pid or pid not in players_map: continue
                p_name = players_map[pid]['last_name'].lower()
                
                p_matches = [m for m in matches if p_name in str(m.get('player1_name')).lower() or p_name in str(m.get('player2_name')).lower()]
                
                for m in p_matches:
                    won = p_name in str(m.get('actual_winner_name')).lower()
                    is_p1 = p_name in str(m.get('player1_name')).lower()
                    odds = float(m.get('odds1', 1.5)) if is_p1 else float(m.get('odds2', 1.5))
                    
                    if won:
                        base = 50
                        bonus = max(0, (odds - 1.5) * 20) 
                        total_pts += (base + bonus)
                    else:
                        total_pts -= 10
                        
            total_pts = round(max(0, total_pts), 1)
            uid = lineup['user_id']
            
            supabase.table("fantasy_lineups").update({"total_points": total_pts}).eq("id", lineup['id']).execute()
            
            xp_gain = int(total_pts * 10)
            credits_gain = int(total_pts / 5) 
            
            prof_res = supabase.table("fantasy_profiles").select("*").eq("user_id", uid).execute()
            if prof_res.data:
                old_xp = prof_res.data[0].get('total_xp', 0)
                supabase.table("fantasy_profiles").update({
                    "total_xp": old_xp + xp_gain
                }).eq("user_id", uid).execute()
            else:
                supabase.table("fantasy_profiles").insert({"user_id": uid, "total_xp": xp_gain}).execute()
                
            try:
                main_prof_res = supabase.table("profiles").select("credits").eq("id", uid).execute()
                if main_prof_res.data:
                    old_credits = main_prof_res.data[0].get('credits') or 0
                    supabase.table("profiles").update({
                        "credits": old_credits + credits_gain
                    }).eq("id", uid).execute()
                    log(f"💰 {credits_gain} Echt-Credits an User {uid} (profiles) ausgeschüttet!")
            except Exception as cred_err:
                log(f"❌ Fehler beim Credit-Update für User {uid}: {cred_err}")
        
        supabase.table("fantasy_gameweeks").update({"status": "completed"}).eq("id", gw_id).execute()
        log(f"✅ [FANTASY ENGINE] Gameweek {gw['week_number']} settled für {len(lineups)} Scouts. Points & Credits verteilt.")

# =================================================================
# PIPELINE EXECUTION
# =================================================================
async def run_pipeline():
    log(f"🚀 Neural Scout V150.3 (AUTONOMOUS SELF-LEARNING LOOP) Starting...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: 
                return
            
            await update_past_results(browser, players)
            
            # THE SELF LEARNING HOOK (Runs dynamically using recent past results)
            try:
                NeuralOptimizer.trigger_learning_cycle(players, all_skills)
            except Exception as opt_err:
                log(f"⚠️ Self-Learning Cycle Exception: {opt_err}")
            
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            
            matches = await fetch_1win_markets_spatial_stream(browser, players)
            
            if not matches:
                log("❌ Keine relevanten DB-Matches mit Quoten gefunden. Beende Zyklus.")
                return
                
            log(f"🔍 Starte Oracle Enrichment für die relevanten DB-Matches...")
            
            db_matched_count = 0
            
            for m in matches:
                try:
                    await asyncio.sleep(0.5) 
                    p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                    p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                    
                    # 🚨 SOTA FIX: Hard Abort bei Zwillingen/Brüdern!
                    if p1_obj == "TIE_BREAKER" or p2_obj == "TIE_BREAKER":
                        log(f"🚨 Tie-Breaker Alarm! Überspringe Match ({m['p1_raw']} vs {m['p2_raw']})")
                        continue

                    if not p1_obj or not p2_obj: 
                        continue
                        
                    n1 = p1_obj['last_name']
                    n2 = p2_obj['last_name']
                    
                    if n1 == n2: 
                        continue
                        
                    db_matched_count += 1
                        
                    if not validate_market_integrity(m['odds1'], m['odds2']):
                        log(f"   ⚠️ REJECTED BAD DATA: {n1} vs {n2} -> {m['odds1']} | {m['odds2']} (Margin Error)")
                        continue 

                    res1 = supabase.table("market_odds").select("*").eq("player1_name", n1).eq("player2_name", n2).order("created_at", desc=True).limit(1).execute()
                    existing_match = res1.data[0] if res1.data else None
                    
                    if not existing_match:
                        res2 = supabase.table("market_odds").select("*").eq("player1_name", n2).eq("player2_name", n1).order("created_at", desc=True).limit(1).execute()
                        existing_match = res2.data[0] if res2.data else None
                        
                        if existing_match:
                            n1, n2 = n2, n1
                            p1_obj, p2_obj = p2_obj, p1_obj
                            m['odds1'], m['odds2'] = m['odds2'], m['odds1']
                    
                    if existing_match:
                        if is_suspicious_movement(to_float(existing_match.get('odds1'), 0), m['odds1'], to_float(existing_match.get('odds2'), 0), m['odds2']):
                            log(f"   ⚠️ REJECTED SPIKE: {n1} ({existing_match.get('odds1')}->{m['odds1']}) vs {n2}")
                            continue

                    db_match_id = existing_match['id'] if existing_match else None
                    if existing_match and existing_match.get('actual_winner_name'): 
                        continue 

                    hist_fair1, hist_fair2 = 0, 0
                    hist_is_value, hist_pick_player = False, None
                    is_signal_locked = has_active_signal(existing_match.get('ai_analysis_text', '')) if existing_match else False
                    
                    if is_signal_locked:
                        log(f"      🔒 DIAMOND LOCK ACTIVE: {n1} vs {n2}")
                        
                        # 🚀 SOTA FIX 1: Zwinge "is_visible_in_scanner: True" bei gelockten Matches!
                        update_data = {"odds1": m['odds1'], "odds2": m['odds2'], "is_visible_in_scanner": True}
                        if m['time'] and m['time'] != "00:00": 
                            update_data["match_time"] = parse_time_to_iso(m['time'])
                        
                        op1 = to_float(existing_match.get('opening_odds1'), 0)
                        
                        if op1 <= 1.01 and m['odds1'] > 1.01: 
                            update_data["opening_odds1"] = m['odds1']
                            update_data["opening_odds2"] = m['odds2']
                            
                        try:
                            supabase.table("market_odds").update(update_data).eq("id", db_match_id).execute()
                        except Exception as up_e:
                            log(f"❌ SUPABASE UPDATE ERROR (Locked) bei {n1} vs {n2}: {up_e}")
                            
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
                        
                        surf, bsi, notes, city_for_weather, matched_tour_name = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2, p1_obj.get('country', 'Unknown'), p2_obj.get('country', 'Unknown'), match_date=datetime.now())
                        weather_data = await fetch_weather_data(city_for_weather)

                        s1 = all_skills.get(p1_obj['id'], {})
                        s2 = all_skills.get(p2_obj['id'], {})
                        
                        report1 = next((r for r in all_reports if r.get('player_id') == p1_obj['id']), None)
                        report2 = next((r for r in all_reports if r.get('player_id') == p2_obj['id']), None)
                        
                        p1_history = await fetch_player_history_extended(n1, limit=80)
                        p2_history = await fetch_player_history_extended(n2, limit=80)
                        
                        p1_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p1_history, n1)
                        p2_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p2_history, n2)
                        p1_form_v2 = MomentumV2Engine.calculate_rating(p1_history[:20], n1)
                        p2_form_v2 = MomentumV2Engine.calculate_rating(p2_history[:20], n2)

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
                                value_tag = f"\n\n[{val_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n1
                            elif val_p2["is_value"]: 
                                value_tag = f"\n\n[{val_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n2
                                
                            ai_text_final = re.sub(r'\[VALUE.*?\]', '', cached_ai['ai_text']).strip() + value_tag
                            hist_fair1 = fair1
                            hist_fair2 = fair2
                            
                            # 🚀 SOTA FIX 2: Zwinge "is_visible_in_scanner: True" auch bei kleinen Odds-Updates
                            if db_match_id:
                                try:
                                    supabase.table("market_odds").update({
                                        "odds1": m['odds1'], 
                                        "odds2": m['odds2'], 
                                        "ai_fair_odds1": fair1, 
                                        "ai_fair_odds2": fair2,
                                        "ai_analysis_text": ai_text_final,
                                        "is_visible_in_scanner": True
                                    }).eq("id", db_match_id).execute()
                                except Exception as up_e:
                                    pass

                        else:
                            log(f"   🧠 Fresh AI Gil Gross Analysis & Monte Carlo Sim: {n1} vs {n2} | T: {matched_tour_name}")
                            
                            # 1. Run Quantum Simulator for O/U Games
                            sim_result = QuantumGamesSimulator.run_simulation(s1, s2, bsi, surf)
                            
                            # 2. Run new SOTA Monte Carlo Win Probability Simulator (NOW DYNAMIC WITH TOUR INFO)
                            current_surf_key = SurfaceIntelligence.normalize_surface_key(surf)
                            surf_rating_a = p1_surface_profile.get(current_surf_key, {}).get('rating', 5.0)
                            surf_rating_b = p2_surface_profile.get(current_surf_key, {}).get('rating', 5.0)
                            
                            tour_identifier = "WTA" if "WTA" in matched_tour_name.upper() else "ATP"
                            
                            mc_results = MonteCarloEngine.run_simulation(
                                tour=tour_identifier,
                                skillA=s1.get('overall_rating', 50), formA=p1_form_v2['score'], surfaceA=surf_rating_a,
                                skillB=s2.get('overall_rating', 50), formB=p2_form_v2['score'], surfaceB=surf_rating_b
                            )
                            
                            # 3. Request Gil Gross AI Text with specific Matchup Math
                            ai = await analyze_match_with_ai(
                                matched_tour_name, p1_obj, p2_obj, s1, s2, report1, report2, surf, bsi, notes, 
                                p1_form_v2, p2_form_v2, weather_data, p1_surface_profile, p2_surface_profile, mc_results
                            )
                            
                            # 4. Integrate MC Prob into the final market physics blend
                            prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai['mc_prob_a'], m['odds1'], m['odds2'])
                            
                            fair1 = round(1/prob, 2) if prob > 0.01 else 99
                            fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                            
                            val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                            val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                            
                            value_tag = ""
                            if val_p1["is_value"]: 
                                value_tag = f"\n\n[{val_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n1
                            elif val_p2["is_value"]: 
                                value_tag = f"\n\n[{val_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n2
                            
                            ai_text_final = f"{ai['ai_text']} {value_tag}\n[🎲 SIM: {sim_result['predicted_line']} Games]"
                            final_time_str = parse_time_to_iso(m['time'])

                            # 🚀 SOTA FIX 3: Haupt-Payload IMMER mit is_visible_in_scanner: True
                            data = {
                                "player1_name": n1, 
                                "player2_name": n2, 
                                "tournament": matched_tour_name,
                                "ai_fair_odds1": fair1, 
                                "ai_fair_odds2": fair2,
                                "ai_analysis_text": ai_text_final, 
                                "games_prediction": sim_result, 
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": final_time_str, 
                                "odds1": m['odds1'], 
                                "odds2": m['odds2'],
                                "is_visible_in_scanner": True 
                            }
                            
                            hist_fair1 = fair1
                            hist_fair2 = fair2
                            
                            if db_match_id: 
                                try:
                                    supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                                except Exception as up_e:
                                    log(f"❌ SUPABASE UPDATE ERROR bei {n1} vs {n2}: {up_e}")
                            else:
                                op1 = to_float(existing_match.get('opening_odds1') if existing_match else 0, 0)
                                if op1 <= 1.01 and m['odds1'] > 1.01: 
                                    data["opening_odds1"] = m['odds1']
                                    data["opening_odds2"] = m['odds2']
                                    
                                try:
                                    res_ins = supabase.table("market_odds").insert(data).execute()
                                    if res_ins.data: 
                                        db_match_id = res_ins.data[0]['id']
                                    log(f"💾 Saved: {n1} vs {n2} (via 1WIN) - Odds: {m['odds1']} | {m['odds2']}")
                                except Exception as ins_e:
                                    log(f"❌ SUPABASE INSERT ERROR bei {n1} vs {n2}: {ins_e}")

                    # L8 SOTA FIX: CATCH-ALL ODDS MOVEMENT TRACKING
                    if db_match_id:
                        should_log_history = False
                        
                        if not existing_match:
                            should_log_history = True
                            log(f"      📍 [ENTRY ODDS] Logging initial odds for {n1} vs {n2}")
                        elif is_signal_locked or hist_is_value: 
                            should_log_history = True
                        else:
                            try:
                                old_o1 = to_float(existing_match.get('odds1'), 0)
                                old_o2 = to_float(existing_match.get('odds2'), 0)
                                
                                if round(old_o1, 3) != round(m['odds1'], 3) or round(old_o2, 3) != round(m['odds2'], 3):
                                    should_log_history = True
                                    log(f"      📈 [ODDS MOVEMENT] {n1} vs {n2} | P1: {old_o1} -> {m['odds1']} | P2: {old_o2} -> {m['odds2']}")
                            except: 
                                pass
                        
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
                            except Exception as hist_err: 
                                log(f"❌ HISTORY INSERT ERROR: {hist_err}")

                except Exception as e: 
                    log(f"⚠️ Match Error bei Iteration: {e}")
            
            log(f"📊 SUMMARY: {db_matched_count} relevante DB-Matches erfolgreich prozessiert.")

        finally: 
            await browser.close()
            
    try:
        FantasySettlementEngine.run_settlement()
    except Exception as e:
        log(f"⚠️ FANTASY ENGINE ERROR: {e}")
        
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
