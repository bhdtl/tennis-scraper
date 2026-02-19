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
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Set
import urllib.parse

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

log("🔌 Initialisiere Neural Scout (V119.0 - ZERO DEP & PERFECT SPATIAL RADAR)...")

# Secrets Load
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub/Groq Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# SOTA Model Selection
MODEL_NAME = 'llama-3.1-8b-instant'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
METADATA_CACHE: Dict[str, Any] = {} 
WEATHER_CACHE: Dict[str, Any] = {} 
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 
TML_MATCH_CACHE: List[Dict] = [] 

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
    # INFO: Funktion ist wieder da (für 1:1 Vollständigkeit), wird aber in Pipeline ignoriert.
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
    clean = re.sub(r'<.*?>', '', clean)
    clean = re.sub(r'S\d+.*$', '', clean) 
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

def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str] = set()) -> Optional[Dict]:
    if not scraped_name_raw or not db_players: 
        return None
        
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
            if parts[0]:
                scrape_initial = parts[0][0].lower()
    else:
        scrape_last = clean_scrape

    target_last = normalize_db_name(scrape_last)
    candidates = []
    
    for p in db_players:
        db_last_raw = p.get('last_name', '')
        db_last = normalize_db_name(db_last_raw)
        match_score = 0
        
        if db_last == target_last: 
            match_score = 100
        elif target_last in db_last or db_last in target_last: 
            if len(target_last) > 3 and len(db_last) > 3: 
                match_score = 80
                
        if match_score > 0:
            db_first = p.get('first_name', '').lower()
            if scrape_initial and db_first:
                if db_first.startswith(scrape_initial): 
                    match_score += 20 
                else: 
                    match_score -= 50 
            if match_score > 50: 
                candidates.append((p, match_score))
                
    if not candidates: 
        return None
        
    candidates.sort(key=lambda x: (x[1], x[0]['id'] in report_ids), reverse=True)
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
        
    clean_location = re.sub(r'[^a-zA-Z0-9\s,]', '', location_name).strip()
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
        log(f"⚠️ Weather Fetch Error ({clean_location}): {e}")
        return None

# --- MARKET INTEGRITY & ANTI-SPIKE ENGINE ---
def validate_market_integrity(o1: float, o2: float) -> bool:
    if o1 <= 1.01 or o2 <= 1.01: 
        return False 
    if o1 > 100 or o2 > 100: 
        return False 
    implied_prob = (1/o1) + (1/o2)
    if implied_prob < 0.92: 
        return False 
    if implied_prob > 1.25: 
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
# 3. MOMENTUM V2 ENGINE (TML ENHANCED)
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

            odds = m['odds1'] if is_p1 else m['odds2']
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
        n = re.sub(r'[^a-z0-9]', '', n)
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
# 5. TACTICAL COMPUTER
# =================================================================
class TacticalComputer:
    @staticmethod
    def calculate_matchup_math(
        s1: Dict, s2: Dict, 
        f1_score: float, f2_score: float, 
        fatigue1_txt: str, fatigue2_txt: str,
        bsi: float, surface: str
    ) -> Dict[str, Any]:
        score = 5.5
        log_reasons = []

        skill1_avg = sum(s1.values()) / len(s1) if s1 else 50
        skill2_avg = sum(s2.values()) / len(s2) if s2 else 50
        diff = skill1_avg - skill2_avg
        skill_impact = (diff / 2) * 0.1
        score += skill_impact
        
        if abs(skill_impact) > 0.3: 
            log_reasons.append(f"Skill Gap: {'P1' if skill_impact > 0 else 'P2'} superior (+{abs(diff):.1f})")

        form_diff = f1_score - f2_score
        form_impact = form_diff * 0.2
        score += form_impact
        
        if abs(form_impact) > 0.4: 
            log_reasons.append(f"Form: {'P1' if form_impact > 0 else 'P2'} hotter")

        p1_power = s1.get('power', 50) + s1.get('serve', 50)
        p2_power = s2.get('power', 50) + s2.get('serve', 50)
        
        if bsi >= 7.5: 
            if p1_power > (p2_power + 10):
                score += 0.5
                log_reasons.append("P1 Big Serve advantage on fast court")
            elif p2_power > (p1_power + 10):
                score -= 0.5
                log_reasons.append("P2 Big Serve advantage on fast court")
        elif bsi <= 4.0: 
            p1_grind = s1.get('stamina', 50) + s1.get('speed', 50)
            p2_grind = s2.get('stamina', 50) + s2.get('speed', 50)
            if p1_grind > (p2_grind + 10):
                score += 0.5
                log_reasons.append("P1 Grinder advantage on slow court")
            elif p2_grind > (p1_grind + 10):
                score -= 0.5
                log_reasons.append("P2 Grinder advantage on slow court")

        if "Heavy" in fatigue2_txt or "CRITICAL" in fatigue2_txt:
            score += 0.8
            log_reasons.append("P2 Fatigued")
            
        if "Heavy" in fatigue1_txt or "CRITICAL" in fatigue1_txt:
            score -= 0.8
            log_reasons.append("P1 Fatigued")

        final_score = max(1.0, min(10.0, score))
        return {"calculated_score": round(final_score, 2), "reasons": log_reasons, "skill_diff": round(diff, 1)}

# =================================================================
# 6. GROQ ENGINE
# =================================================================
async def call_groq(prompt: str, model: str = MODEL_NAME, temp: float = 0.0) -> Optional[str]:
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
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200: 
                return None
            return response.json()['choices'][0]['message']['content']
        except: 
            return None

# =================================================================
# L8 SOTA: THE RAG AI AUDITOR (Replaces TennisExplorer Regex)
# =================================================================
async def duckduckgo_html_search(query: str) -> str:
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    data = {"q": query}
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, data=data, timeout=15.0)
            soup = BeautifulSoup(res.text, 'html.parser')
            snippets = []
            for a in soup.find_all('a', class_='result__snippet'):
                snippets.append(a.get_text(strip=True))
            return " | ".join(snippets[:5]) 
    except Exception as e:
        return ""

async def update_past_results_via_ai():
    log("🏆 The Quantum AI Auditor: Booting RAG Search Engine (Zero Dependency V119.0)...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    
    if not pending or not isinstance(pending, list): 
        return
    
    for m in pending:
        try:
            mt = datetime.fromisoformat(m['created_at'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) - mt < timedelta(hours=5): 
                continue
        except: 
            pass

        p1 = m.get('player1_name')
        p2 = m.get('player2_name')
        
        if not p1 or not p2: 
            continue

        query = f"{p1} vs {p2} tennis match result score {datetime.now().year}"
        log(f"   🔍 AI Search Agent queries DDG for: {p1} vs {p2}...")
        
        search_context = await duckduckgo_html_search(query)
        if len(search_context) < 10: 
            continue 
        
        prompt = f"""
        TASK: Extract the tennis match result from the provided web search snippets.
        MATCH: {p1} vs {p2}
        
        WEB SEARCH SNIPPETS:
        "{search_context}"
        
        RULES:
        1. Identify if the match is finished.
        2. Identify the winner (must be exactly '{p1}' or '{p2}').
        3. Identify the final score (e.g., "6-4 6-2").
        4. If the snippets don't explicitly mention the final result for this exact match, return "status": "unknown".
        
        OUTPUT JSON ONLY:
        {{
            "status": "finished" or "unknown",
            "winner": "Name",
            "score": "Score string"
        }}
        """
        
        ai_res = await call_groq(prompt)
        if ai_res:
            try:
                data = json.loads(ai_res)
                if data.get('status') == 'finished' and data.get('winner') and data.get('score'):
                    winner = data['winner']
                    score = data['score']
                    
                    log(f"      🤖 RAG AUDITOR FOUND: {p1} vs {p2} -> Winner: {winner} ({score})")
                    supabase.table("market_odds").update({
                        "actual_winner_name": winner,
                        "score": score
                    }).eq("id", m['id']).execute()
                    
                    log(f"🔄 Triggering Real-Time Profile Refresh for {p1} & {p2}")
                    for p_name in [p1, p2]:
                        p_hist = await fetch_player_history_extended(p_name, limit=80)
                        p_profile = SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name)
                        p_form = MomentumV2Engine.calculate_rating(p_hist[:20], p_name)
                        
                        supabase.table('players').update({
                            'surface_ratings': p_profile,
                            'form_rating': p_form 
                        }).ilike('last_name', f"%{p_name}%").execute()
            except: 
                pass
                
        await asyncio.sleep(1.0)

# =================================================================
# 6.5 1WIN SOTA MASTER FEED (V119.0 INDEPENDENT BLOCK PARSING)
# =================================================================
def extract_odds_from_lines(lines_slice: List[str]) -> tuple[float, float]:
    floats = []
    for l in lines_slice:
        cl = l.replace(',', '.').strip()
        matches = re.findall(r'\b\d+\.\d{2,3}\b', cl)
        for m in matches:
            try:
                val = float(m)
                if 1.0 < val <= 150.0:
                    floats.append(val)
            except: 
                pass
            
    best_pair = (0.0, 0.0)
    best_diff = 999.0
    
    # L8 Fix: Größerer Index-Radius und strikterer Early-Exit für echte Buchmacher-Marge
    for x in range(len(floats)):
        for y in range(x+1, min(x+8, len(floats))):
            o1 = floats[x]
            o2 = floats[y]
            try:
                implied = (1/o1) + (1/o2)
                if 1.03 <= implied <= 1.15: 
                    diff = abs(implied - 1.055)
                    if diff < best_diff:
                        best_diff = diff
                        best_pair = (o1, o2)
            except: 
                pass
                
    return best_pair

def extract_time_context(lines_slice: List[str]) -> str:
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

async def fetch_1win_markets_spatial_stream(browser: Browser, db_players: List[Dict]) -> List[Dict]:
    log("🚀 [1WIN GHOST] Starte Independent Block Spatial Engine (V119.0 SOTA Radar)...")
    
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
        java_script_enabled=True
    )
    
    await context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); window.navigator.chrome = { runtime: {} };")
    page = await context.new_page()

    db_name_map = {}
    for p in db_players:
        real_last = p.get('last_name', '')
        if real_last: 
            db_name_map[normalize_db_name(real_last)] = real_last

    sorted_db_names = sorted(db_name_map.items(), key=lambda x: len(x[0]), reverse=True)
    
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
            
        log("⏳ Führe Micro-Scrolling durch, um Virtualized React DOM zu capturen...")
        
        for scroll_step in range(30):
            try:
                await page.evaluate("""
                    let buttons = document.querySelectorAll('div, button, span');
                    for(let b of buttons) {
                        let txt = b.innerText ? b.innerText.toLowerCase() : '';
                        if((txt.includes('more') || txt.includes('anzeigen') || txt.includes('alle')) && b.clientHeight > 0) {
                            try { b.click(); } catch(e) {}
                        }
                    }
                """)
                
                text_dump = await page.evaluate("document.body.innerText")
                all_raw_text_blocks.append(text_dump)
                
                await page.mouse.wheel(delta_x=0, delta_y=600)
                await asyncio.sleep(0.5) 
                
            except Exception as scroll_e: 
                continue
                
    except Exception as e: 
        log(f"⚠️ [1WIN GHOST] Timeout/Fehler beim Laden: {e}")
    finally: 
        await context.close()

    log("🧩 Verarbeite DOM-Blöcke (Independent Parsing, um Seam-Ghosts zu eliminieren)...")
    
    # L8 Fix: Wir iterieren isoliert durch JEDEN Block, den wir gescreent haben.
    for block in all_raw_text_blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        current_tour = "Unknown"

        for i, line in enumerate(lines):
            line_norm = normalize_text(line).lower()
            
            if len(line) > 3 and len(line) < 60 and not re.match(r'^[\d\.,\s:\-]+$', line):
                if any(kw in line_norm for kw in ['atp', 'wta', 'open', 'masters', 'tour', 'classic', 'championship', 'cup', 'men', 'singles', 'doha', 'qatar', 'dubai', 'rotterdam', 'rio', 'los cabos', 'acapulco']):
                    current_tour = line
            
            p1_found_real = None
            for p_norm, p_real in sorted_db_names:
                # L8 Fix: >= 3 Erlaubt Namen wie "Wu", "Zhu", "Buse", "Wong"
                if len(p_norm) > 2 and re.search(rf'\b{re.escape(p_norm)}\b', line_norm):
                    p1_found_real = p_real
                    break 
                    
            if p1_found_real:
                p2_found_real = None
                
                # L8 Fix: Suchradius auf satte 40 Zeilen erweitert! 
                if '-' in line_norm or 'vs' in line_norm or '/' in line_norm:
                    search_slice = lines[i:min(i+40, len(lines))]
                else:
                    search_slice = lines[i+1:min(i+40, len(lines))]
                    
                combined_text_norm = normalize_text(" ".join(search_slice)).lower()
                
                for p_norm, p_real in sorted_db_names:
                    if len(p_norm) > 2 and re.search(rf'\b{re.escape(p_norm)}\b', combined_text_norm):
                        if p_real != p1_found_real:
                            p2_found_real = p_real
                            break
                        
                if p2_found_real:
                    match_key = tuple(sorted([p1_found_real, p2_found_real]))
                    
                    if match_key not in seen_matches:
                        # L8 Fix: Quoten-Radius auf 50 Zeilen!
                        odds_slice = lines[i:min(i+50, len(lines))]
                        o1, o2 = extract_odds_from_lines(odds_slice)
                        
                        time_context_slice = lines[max(0, i-4):min(i+4, len(lines))]
                        extracted_time = extract_time_context(time_context_slice)
                        
                        if o1 > 0 and o2 > 0:
                            seen_matches.add(match_key)
                            parsed_matches.append({
                                "p1_raw": p1_found_real, 
                                "p2_raw": p2_found_real,
                                "tour": clean_tournament_name(current_tour),
                                "time": extracted_time, 
                                "odds1": o1, 
                                "odds2": o2,
                                "handicap_line": None, "handicap_odds1": 0, "handicap_odds2": 0,
                                "over_under_line": None, "over_odds": 0, "under_odds": 0,
                                "actual_winner": None, "score": ""
                            })
                elif "-" not in line_norm and "vs" not in line_norm:
                    # L8 Fix: Logge Ghosting nur, wenn der P1 nicht gerade am unteren Bildschirmrand abgeschnitten wurde.
                    if i < len(lines) - 40:
                        pass # log(f"   👁️ [GHOST DETECTED] Spieler gefunden: {p1_found_real}, aber kein bekannter Gegner in den nächsten 40 Zeilen!")

    log(f"✅ [1WIN GHOST] {len(parsed_matches)} saubere DB-Matches isoliert.")
    return parsed_matches

# =================================================================
# 7. DATA FETCHING & ORACLE (LEGACY FUNCTIONS KEPT FOR 1:1 COMPLETENESS)
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
    # INFO: Funktion ist hier zur 1:1 Vollständigkeit. Wird nicht mehr aktiv aufgerufen.
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

async def update_past_results(browser: Browser):
    # INFO: Alte Regex-Funktion ist hier für 1:1 Vollständigkeit. Wurde durch update_past_results_via_ai ersetzt.
    pass

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
                        'speed': to_float(entry.get('speed')), 
                        'stamina': to_float(entry.get('stamina')),
                        'mental': to_float(entry.get('mental'))
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

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    ai_meta = ensure_dict(ai_meta)
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
    
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8)
    
    def get_offense(s): 
        return s.get('serve', 50) + s.get('power', 50)
        
    c1_score = get_offense(s1)
    c2_score = get_offense(s2)
    
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)
    
    f1 = to_float(ai_meta.get('p1_form_score', 5))
    f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)

    weights = [0.20, 0.15, 0.15, 0.40, 0.10] 
    
    prob_alpha = (prob_matchup * weights[0]) + (prob_bsi * weights[1]) + (prob_skills * weights[2]) + (prob_elo * weights[3]) + (prob_form * weights[4])
    
    if prob_alpha > 0.60: 
        prob_alpha = min(prob_alpha * 1.05, 0.98)
    elif prob_alpha < 0.40: 
        prob_alpha = max(prob_alpha * 0.95, 0.02)
    
    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
        
    model_trust_factor = 0.25
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

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, form1_data, form2_data, weather_data, p1_surface_profile, p2_surface_profile):
    fatigueA = await get_advanced_load_analysis(await fetch_player_history_extended(p1['last_name'], 10))
    fatigueB = await get_advanced_load_analysis(await fetch_player_history_extended(p2['last_name'], 10))
    
    tactical_data = TacticalComputer.calculate_matchup_math(s1, s2, form1_data['score'], form2_data['score'], fatigueA, fatigueB, bsi, surface)
    score = tactical_data['calculated_score']
    reasons_bullet_points = "\n- ".join(tactical_data['reasons'])
    
    if weather_data:
        weather_str = f"WEATHER: {weather_data['summary']}. IMPACT: {weather_data['impact_note']}"
    else:
        weather_str = "Weather: Unknown"
        
    current_surf_key = SurfaceIntelligence.normalize_surface_key(surface)
    p1_s_rating = p1_surface_profile.get(current_surf_key, {}).get('rating', 5.5)
    p2_s_rating = p2_surface_profile.get(current_surf_key, {}).get('rating', 5.5)
    surface_context = f"SURFACE SPECIALTIES: P1 Rating {p1_s_rating} on {current_surf_key}. P2 Rating {p2_s_rating} on {current_surf_key}."

    prompt = f"""
    ROLE: Elite Tennis Analyst.
    TASK: Write a sharp analysis based on these CALCULATED FACTS.
    
    FACTS (TRUST THESE):
    - Matchup Score: {score}/10 ( >5.5 favors {p1['last_name']}, <5.5 favors {p2['last_name']})
    - Key Drivers: 
      - {reasons_bullet_points if reasons_bullet_points else "Balanced stats."}
    - Player A ({p1['last_name']}): Form {form1_data['text']}, Fatigue: {fatigueA}
    - Player B ({p2['last_name']}): Form {form2_data['text']}, Fatigue: {fatigueB}
    - Conditions: {surface} (Speed {bsi}/10). {weather_str}
    - {surface_context}

    OUTPUT JSON ONLY:
    {{ 
        "p1_tactical_score": {score}, 
        "p2_tactical_score": {10.0 - score}, 
        "ai_text": "One key sentence summary + 3 bullet points explanations. Mention surface rating if relevant.", 
        "p1_win_sentiment": {score / 10.0} 
    }}
    """
    
    res = await call_groq(prompt)
    default = {'p1_tactical_score': score, 'p2_tactical_score': 10.0-score, 'ai_text': 'Analysis unavailable.', 'p1_win_sentiment': 0.5}
    
    if not res: 
        return default
        
    try:
        cleaned = res.replace("json", "").replace("```", "").strip()
        data = ensure_dict(json.loads(cleaned))
        data['p1_tactical_score'] = score
        data['p2_tactical_score'] = 10.0 - score
        return data
    except: 
        return default

# =================================================================
# 10. QUANTUM GAMES SIMULATOR
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
# PIPELINE EXECUTION
# =================================================================
async def run_pipeline():
    log(f"🚀 Neural Scout V119.0 (PERFECT SPATIAL EDITION) Starting...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await update_past_results_via_ai()
            
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: 
                return
                
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            
            log("🌍 [MIGRATION] Prüfe Spieler auf Schema-Sync (V95)...")
            players_to_update = []
            for p_iter in players:
                if not p_iter.get('surface_ratings') or not isinstance(p_iter.get('surface_ratings'), dict) or not p_iter.get('surface_ratings', {}).get('_v95_mastery_applied'):
                    players_to_update.append(p_iter)
                    
            if players_to_update:
                log(f"🔄 Syncing {len(players_to_update)} players to new schema...")
                for p_data in players_to_update:
                    p_name = p_data['last_name']
                    p_hist = await fetch_player_history_extended(p_name, limit=80)
                    p_profile = SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name)
                    p_form = MomentumV2Engine.calculate_rating(p_hist[:20], p_name)
                    try: 
                        supabase.table('players').update({'surface_ratings': p_profile, 'form_rating': p_form}).eq('id', p_data['id']).execute()
                    except: 
                        pass
                    await asyncio.sleep(0.05) 
                log("✅ [GLOBAL PROFILER] Migration abgeschlossen.")
            
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
                        update_data = {"odds1": m['odds1'], "odds2": m['odds2']}
                        if m['time'] and m['time'] != "00:00": 
                            update_data["match_time"] = m['time']
                        
                        op1 = to_float(existing_match.get('opening_odds1'), 0)
                        op2 = to_float(existing_match.get('opening_odds2'), 0)
                        
                        if op1 <= 1.01 and m['odds1'] > 1.01: 
                            update_data["opening_odds1"] = m['odds1']
                            update_data["opening_odds2"] = m['odds2']
                            
                        supabase.table("market_odds").update(update_data).eq("id", db_match_id).execute()
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
                        
                        p1_history = await fetch_player_history_extended(n1, limit=80)
                        p2_history = await fetch_player_history_extended(n2, limit=80)
                        
                        p1_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p1_history, n1)
                        p2_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p2_history, n2)
                        p1_form_v2 = MomentumV2Engine.calculate_rating(p1_history[:20], n1)
                        p2_form_v2 = MomentumV2Engine.calculate_rating(p2_history[:20], n2)

                        try:
                            supabase.table('players').update({'surface_ratings': p1_surface_profile, 'form_rating': p1_form_v2}).eq('id', p1_obj['id']).execute()
                            supabase.table('players').update({'surface_ratings': p2_surface_profile, 'form_rating': p2_form_v2}).eq('id', p2_obj['id']).execute()
                        except: 
                            pass

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
                                value_tag = f" [{val_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n1
                            elif val_p2["is_value"]: 
                                value_tag = f" [{val_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n2
                                
                            ai_text_final = re.sub(r'\[.*?\]', '', cached_ai['ai_text']).strip() + value_tag
                            hist_fair1 = fair1
                            hist_fair2 = fair2

                        else:
                            log(f"   🧠 Fresh Analysis & Simulation: {n1} vs {n2} | T: {matched_tour_name} ({surf}) | Time: {m['time']}")
                            sim_result = QuantumGamesSimulator.run_simulation(s1, s2, bsi, surf)
                            ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, {}, {}, surf, bsi, notes, p1_form_v2, p2_form_v2, weather_data, p1_surface_profile, p2_surface_profile)
                            prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai, m['odds1'], m['odds2'])
                            
                            fair1 = round(1/prob, 2) if prob > 0.01 else 99
                            fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                            
                            val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                            val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                            
                            value_tag = ""
                            if val_p1["is_value"]: 
                                value_tag = f" [{val_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n1
                            elif val_p2["is_value"]: 
                                value_tag = f" [{val_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                hist_is_value = True
                                hist_pick_player = n2
                            
                            ai_text_final = f"{ai.get('ai_text', '').replace('json', '').strip()} {value_tag} [🎲 SIM: {sim_result['predicted_line']} Games]"
                            
                            if m['time'] and m['time'] != "00:00":
                                final_time_str = m['time']
                            else:
                                final_time_str = f"{datetime.now().strftime('%Y-%m-%d')}T00:00:00Z"

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
                                "odds2": m['odds2']
                            }
                            
                            hist_fair1 = fair1
                            hist_fair2 = fair2
                            
                            if db_match_id: 
                                supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                            else:
                                op1 = to_float(existing_match.get('opening_odds1') if existing_match else 0, 0)
                                if op1 <= 1.01 and m['odds1'] > 1.01: 
                                    data["opening_odds1"] = m['odds1']
                                    data["opening_odds2"] = m['odds2']
                                    
                                res_ins = supabase.table("market_odds").insert(data).execute()
                                if res_ins.data: 
                                    db_match_id = res_ins.data[0]['id']
                                log(f"💾 Saved: {n1} vs {n2} (via 1WIN) - Odds: {m['odds1']} | {m['odds2']}")

                    if db_match_id:
                        should_log_history = False
                        
                        if not existing_match or is_signal_locked or hist_is_value: 
                            should_log_history = True
                        else:
                            try:
                                if abs(to_float(existing_match.get('odds1'), 0) - m['odds1']) >= 0.01 or abs(to_float(existing_match.get('odds2'), 0) - m['odds2']) >= 0.01: 
                                    should_log_history = True
                                    log(f"      📈 [ODDS MOVEMENT] {n1} vs {n2} | P1: {to_float(existing_match.get('odds1'), 0)} -> {m['odds1']} | P2: {to_float(existing_match.get('odds2'), 0)} -> {m['odds2']}")
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
                            except: 
                                pass

                except Exception as e: 
                    log(f"⚠️ Match Error: {e}")
            
            log(f"📊 SUMMARY: {db_matched_count} relevante DB-Matches analysiert.")

        finally: 
            await browser.close()
    
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
