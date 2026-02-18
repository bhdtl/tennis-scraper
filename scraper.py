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
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Set

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

log("🔌 Initialisiere Neural Scout (V97.0 - 1WIN MASTER FEED & TE AUDITOR)...")

# Secrets Load
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub/Groq Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_NAME = 'llama-3.1-8b-instant'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
METADATA_CACHE: Dict[str, Any] = {} 
WEATHER_CACHE: Dict[str, Any] = {} 
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena", "Sydney": "Ken Rosewall Arena",
    "Brisbane": "Pat Rafter Arena", "Adelaide": "Memorial Drive Tennis Centre",
    "Melbourne": "Rod Laver Arena"
}
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def to_float(val: Any, default: float = 50.0) -> float:
    if val is None: return default
    try: return float(val)
    except: return default

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

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip()
    parts = clean.split()
    return parts[-1].lower() if parts else ""

def ensure_dict(data: Any) -> Dict:
    try:
        if isinstance(data, dict): return data
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict): return data[0]
        return {}
    except: return {}

def normalize_db_name(name: str) -> str:
    if not name: return ""
    n = name.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

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
                if db_first.startswith(scrape_initial): match_score += 20 
                else: match_score -= 50 
            if match_score > 50: candidates.append((p, match_score))
    if not candidates: return None
    candidates.sort(key=lambda x: (x[1], x[0]['id'] in report_ids), reverse=True)
    return candidates[0][0]

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
    if "canberra" in s_tokens and "canberra" in d_tokens: score += 30
    return score

def has_active_signal(text: Optional[str]) -> bool:
    if not text: return False
    if "[" in text and "]" in text:
        if any(icon in text for icon in ["💎", "🛡️", "⚖️", "💰", "🔥", "✨", "📈", "👀"]):
            return True
    return False

# --- SOTA WEATHER SERVICE ---
async def fetch_weather_data(location_name: str) -> Optional[Dict]:
    if not location_name or location_name == "Unknown": return None
    
    clean_location = re.sub(r'[^a-zA-Z0-9\s,]', '', location_name).strip()
    if len(clean_location) > 50: clean_location = clean_location[:50] 
    
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
            if not curr: return None

            impact = "Neutral conditions."
            temp = curr.get('temperature_2m', 20)
            hum = curr.get('relative_humidity_2m', 50)
            wind = curr.get('wind_speed_10m', 10)

            if temp > 30: impact = "EXTREME HEAT: Ball flies fast, physically draining."
            elif temp < 12: impact = "COLD: Low bounce, heavy conditions."
            
            if hum > 70: impact += " HIGH HUMIDITY: Air is heavy, ball travels slower."
            if wind > 20: impact += " WINDY: Serve toss difficult, high variance."

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
    if o1 <= 1.01 or o2 <= 1.01: return False 
    if o1 > 100 or o2 > 100: return False 
    implied_prob = (1/o1) + (1/o2)
    if implied_prob < 0.92: return False 
    if implied_prob > 1.25: return False
    return True

def is_suspicious_movement(old_o1: float, new_o1: float, old_o2: float, new_o2: float) -> bool:
    if old_o1 == 0 or old_o2 == 0: return False 
    change_p1 = abs(new_o1 - old_o1) / old_o1
    change_p2 = abs(new_o2 - old_o2) / old_o2
    if change_p1 > 0.35 or change_p2 > 0.35:
        if old_o1 < 1.10 or old_o2 < 1.10: return False
        return True
    return False

# =================================================================
# 3. MOMENTUM V2 ENGINE
# =================================================================
class MomentumV2Engine:
    @staticmethod
    def calculate_rating(matches: List[Dict], player_name: str, max_matches: int = 15) -> Dict[str, Any]:
        if not matches: return {"score": 5.0, "text": "Neutral (No Data)", "history_summary": "", "color_hex": "#808080"}

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
            if not odds or odds <= 1.0: odds = 1.50

            is_recent = idx >= (len(chrono_matches) - 5)
            weight = 1.5 if is_recent else 0.8
            impact = 0.0

            if won:
                if odds < 1.30: impact = 0.4      
                elif odds <= 2.00: impact = 0.8   
                else: impact = 1.8                
                score = str(m.get('score', ''))
                if score and "2-1" not in score and "1-2" not in score:
                    impact += 0.3
                momentum += 0.2 
                history_log.append("W")
            else:
                if odds < 1.30: impact = -0.6     
                elif odds < 1.50: impact = -0.5
                elif odds <= 2.20: impact = -0.6  
                else: impact = -0.2                
                
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
        if final_rating > 8.5: desc = "🔥 ELITE"
        elif final_rating > 7.0: desc = "📈 Strong"
        elif final_rating < 4.0: desc = "❄️ Cold"
        elif final_rating < 5.5: desc = "⚠️ Weak"
        
        color_hex = "#F0C808" 
        if final_rating >= 9.0: color_hex = "#FF00FF" 
        elif final_rating >= 8.0: color_hex = "#3366FF" 
        elif final_rating >= 7.0: color_hex = "#00B25B" 
        elif final_rating >= 6.0: color_hex = "#99CC33" 
        elif final_rating <= 4.0: color_hex = "#CC0000" 
        elif final_rating <= 5.5: color_hex = "#FF9933" 

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
        if not raw_surface: return "unknown"
        s = raw_surface.lower()
        if "grass" in s: return "grass"
        if "clay" in s or "sand" in s: return "clay"
        if "hard" in s or "carpet" in s or "acrylic" in s or "indoor" in s: return "hard"
        return "unknown"

    @staticmethod
    def clean_name_for_matching(name: str) -> str:
        if not name: return ""
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
            if match_hist: found_surface = match_hist.group(1)
            elif "hard court" in ai_text or "hard surface" in ai_text: found_surface = "hard"
            elif "red clay" in ai_text or "clay court" in ai_text: found_surface = "clay"
            elif "grass court" in ai_text: found_surface = "grass"
            elif "clay" in tour_name or "roland garros" in tour_name: found_surface = "clay"
            elif "grass" in tour_name or "wimbledon" in tour_name: found_surface = "grass"
            elif "hard" in tour_name or "us open" in tour_name or "australian open" in tour_name: found_surface = "hard"
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
        log(f"📊 [TELEMETRY] Starte Surface-Profilierung für '{player_name}' mit {len(matches)} historischen Matches.")
        
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
                log(f"   -> {surf.upper()}: 0 Matches (Base 3.5 applied)")
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
            if final_rating >= 8.5: desc = "🔥 SPECIALIST"
            elif final_rating >= 7.0: desc = "📈 Strong"
            elif final_rating >= 5.5: desc = "Solid"
            elif final_rating >= 4.5: desc = "⚠️ Vulnerable"
            else: desc = "❄️ Weakness"
            
            color_hex = "#F0C808" 
            if final_rating >= 8.5: color_hex = "#FF00FF" 
            elif final_rating >= 7.5: color_hex = "#3366FF" 
            elif final_rating >= 6.5: color_hex = "#00B25B" 
            elif final_rating >= 5.5: color_hex = "#99CC33" 
            elif final_rating <= 4.5: color_hex = "#CC0000" 
            elif final_rating < 5.5: color_hex = "#FF9933" 

            profile[surf] = {
                "rating": round(final_rating, 2),
                "color": color_hex,
                "matches_tracked": n_surf,
                "text": desc
            }
            
            log(f"   -> {surf.upper()}: {n_surf} Matches. WinRate: {win_rate:.2f} | Final: {round(final_rating, 2)}")
            
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
        if abs(skill_impact) > 0.3: log_reasons.append(f"Skill Gap: {'P1' if skill_impact > 0 else 'P2'} superior (+{abs(diff):.1f})")

        form_diff = f1_score - f2_score
        form_impact = form_diff * 0.2
        score += form_impact
        if abs(form_impact) > 0.4: log_reasons.append(f"Form: {'P1' if form_impact > 0 else 'P2'} hotter")

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
async def call_groq(prompt: str, model: str = MODEL_NAME) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a tennis analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200: return None
            return response.json()['choices'][0]['message']['content']
        except: return None

call_gemini = call_groq 

# =================================================================
# 6.5 1WIN SOTA MASTER FEED (NEW SOTA ARCHITECTURE)
# =================================================================
async def fetch_1win_api_markets() -> List[Dict]:
    """
    SOTA API Interception: Liest den geheimen JSON-Feed von 1wins Provider 'top-parser' aus.
    Dies ist ab V97.0 die alleinige Source of Truth für anstehende Matches und Quoten.
    """
    api_url = "https://api-gateway.top-parser.com/sports/get?sportId=33&l=de-DE&p=44ba10e5-7df2-47ab-a44d-dc93803c7a6e" 
    
    log(f"📡 [1WIN MASTER API] Fetching direct JSON markets from {api_url}...")
    parsed_1win_matches = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://1win.io/",
        "Origin": "https://1win.io"
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(api_url, headers=headers)
            if res.status_code != 200:
                log(f"⚠️ [1WIN MASTER API] Error {res.status_code}. Response: {res.text[:100]}")
                return []
                
            data = res.json()
            
            # Die Matches liegen im Array, entweder direkt oder unter "matches" / "data"
            matches_list = []
            if isinstance(data, list):
                matches_list = data
            elif isinstance(data, dict):
                matches_list = data.get('matches', data.get('data', []))
            
            if not matches_list:
                log(f"⚠️ [1WIN MASTER API] JSON geladen, aber keine Matches gefunden. Struktur checken: {str(data)[:200]}")
                return []
            
            for m in matches_list:
                match_name = m.get('name', '')
                if ' - ' not in match_name:
                    continue
                
                parts = match_name.split(' - ')
                if len(parts) < 2: continue
                
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                
                tournament_obj = m.get('tournament', {})
                tour_name = tournament_obj.get('slug', tournament_obj.get('name', 'Unknown'))
                
                start_time_ts = m.get('startAt', 0)
                start_time_str = "00:00"
                if start_time_ts > 0:
                    start_time_str = datetime.fromtimestamp(start_time_ts).strftime('%H:%M')

                # Wir extrahieren die Quoten tief aus dem JSON. 
                # (Abhängig von 1wins genauer Struktur können diese Pfade leicht variieren)
                odds_markets = m.get('markets', {})
                moneyline = odds_markets.get('moneyline', {})
                o1 = to_float(moneyline.get('w1', moneyline.get('team1', 0)), 0)
                o2 = to_float(moneyline.get('w2', moneyline.get('team2', 0)), 0)
                
                # Wenn wir die Quoten direkt in der root-Ebene oder unter "odds" finden:
                if o1 == 0 and o2 == 0:
                    o1 = to_float(m.get('w1', 0), 0)
                    o2 = to_float(m.get('w2', 0), 0)
                
                # Falls 1win sie in einer Liste von Events/Selections speichert, nehmen wir an, dass sie zumindest 0 sind 
                # und vom Value-Scanner ignoriert werden, bis die genaue Pfadstruktur der Quoten analysiert wird.
                
                handicap = odds_markets.get('handicap', {})
                hc_line = to_float(handicap.get('line'), None)
                hc_o1 = to_float(handicap.get('w1'), 0)
                hc_o2 = to_float(handicap.get('w2'), 0)
                
                totals = odds_markets.get('totals', {})
                ou_line = to_float(totals.get('line'), None)
                o_odds = to_float(totals.get('over'), 0)
                u_odds = to_float(totals.get('under'), 0)
                
                parsed_1win_matches.append({
                    "p1_raw": p1,
                    "p2_raw": p2,
                    "tour": clean_tournament_name(tour_name),
                    "time": start_time_str,
                    "odds1": o1,
                    "odds2": o2,
                    "handicap_line": hc_line,
                    "handicap_odds1": hc_o1,
                    "handicap_odds2": hc_o2,
                    "over_under_line": ou_line,
                    "over_odds": o_odds,
                    "under_odds": u_odds,
                    "actual_winner": None,
                    "score": ""
                })
                
        log(f"✅ [1WIN MASTER API] Erfolgreich {len(parsed_1win_matches)} Matches geparst.")
        return parsed_1win_matches
    except Exception as e:
        log(f"⚠️ [1WIN MASTER API] Exception: {e}")
        return []

# =================================================================
# 7. DATA FETCHING & ORACLE
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
    except: pass
    finally: await page.close()
    return metadata

async def fetch_player_history_extended(player_last_name: str, limit: int = 80) -> List[Dict]:
    try:
        res = supabase.table("market_odds")\
            .select("player1_name, player2_name, odds1, odds2, actual_winner_name, score, created_at, tournament, ai_analysis_text")\
            .or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order("created_at", desc=True).limit(limit).execute()
        data = res.data or []
        return data
    except: return []

async def fetch_player_form_quantum(matches: List[Dict], player_last_name: str) -> Dict[str, Any]:
    return MomentumV2Engine.calculate_rating(matches[:20], player_last_name)

def get_style_matchup_stats_py(matches: List[Dict], player_name: str, opponent_style_raw: str, supabase_client: Client) -> Optional[Dict]:
    if not player_name or not opponent_style_raw or not matches: return None
    target_style = opponent_style_raw.split(',')[0].split('(')[0].strip()
    if not target_style or target_style == 'Unknown': return None
    try:
        opponent_names_to_fetch = []
        for m in matches:
            if player_name.lower() in m['player1_name'].lower(): opp = get_last_name(m['player2_name']).lower()
            else: opp = get_last_name(m['player1_name']).lower()
            if opp: opponent_names_to_fetch.append(opp)
        if not opponent_names_to_fetch: return None
        unique_opps = list(set(opponent_names_to_fetch))
        opponents_map = {}
        for i in range(0, len(unique_opps), 20):
            chunk = unique_opps[i:i+20]
            p_res = supabase_client.table('players').select('last_name, play_style').in_('last_name', chunk).execute()
            if p_res.data:
                for p in p_res.data:
                    if p.get('play_style'):
                        s = [x.split('(')[0].strip() for x in p['play_style'].split(',')]
                        opponents_map[p['last_name'].lower()] = s
        relevant_matches = 0; wins = 0
        for m in matches:
            if player_name.lower() in m['player1_name'].lower(): opp_name = get_last_name(m['player2_name']).lower()
            else: opp_name = get_last_name(m['player1_name']).lower()
            opp_styles = opponents_map.get(opp_name, [])
            if target_style in opp_styles:
                relevant_matches += 1
                winner = m.get('actual_winner_name', '').lower()
                if player_name.lower() in winner: wins += 1
        if relevant_matches < 3: return None
        win_rate = (wins / relevant_matches) * 100
        verdict = "Neutral"
        if win_rate > 65: verdict = "Dominant vs this style"
        elif win_rate < 40: verdict = "Struggles significantly vs this style"
        return {"win_rate": win_rate, "matches": relevant_matches, "verdict": verdict, "style": target_style}
    except: return None

async def get_advanced_load_analysis(matches: List[Dict]) -> str:
    try:
        recent_matches = matches[:5]
        if not recent_matches: return "Fresh (No recent data)"
        now_ts = datetime.now().timestamp()
        fatigue_score = 0.0
        details = []
        last_match = recent_matches[0]
        try:
            lm_time = datetime.fromisoformat(last_match['created_at'].replace('Z', '+00:00')).timestamp()
            hours_since_last = (now_ts - lm_time) / 3600
        except: return "Unknown"
        if hours_since_last < 24: fatigue_score += 50; details.append("Back-to-back match")
        elif hours_since_last < 48: fatigue_score += 25; details.append("Short rest")
        elif hours_since_last > 336: return "Rusty (2+ weeks break)"
        if hours_since_last < 72 and last_match.get('score'):
            score_str = str(last_match['score']).lower()
            if 'ret' in score_str or 'wo' in score_str: fatigue_score *= 0.5
            else:
                sets = len(re.findall(r'(\d+)-(\d+)', score_str))
                tiebreaks = len(re.findall(r'7-6|6-7', score_str))
                total_games = 0
                for s in re.findall(r'(\d+)-(\d+)', score_str):
                    try: total_games += int(s[0]) + int(s[1])
                    except: pass
                if sets >= 3: fatigue_score += 20; details.append("Last match 3+ sets")
                if total_games > 30: fatigue_score += 15; details.append("Marathon match (>30 games)")
                if tiebreaks > 0: fatigue_score += 5 * tiebreaks; details.append(f"{tiebreaks} Tiebreaks played")
        matches_in_week = 0; sets_in_week = 0
        for m in recent_matches:
            try:
                mt = datetime.fromisoformat(m['created_at'].replace('Z', '+00:00')).timestamp()
                if (now_ts - mt) < (7 * 24 * 3600):
                    matches_in_week += 1
                    if m.get('score'): sets_in_week += len(re.findall(r'\d+-\d+', str(m['score'])))
            except: pass
        if matches_in_week >= 4: fatigue_score += 20; details.append(f"Busy week ({matches_in_week} matches)")
        if sets_in_week > 10: fatigue_score += 15; details.append(f"Heavy leg load ({sets_in_week} sets in 7 days)")
        status = "Fresh"
        if fatigue_score > 75: status = "CRITICAL FATIGUE"
        elif fatigue_score > 50: status = "Heavy Legs"
        elif fatigue_score > 30: status = "In Rhythm (Active)"
        if details: return f"{status} [{', '.join(details)}]"
        return status
    except: return "Unknown"

async def fetch_tennisexplorer_stats(browser: Browser, relative_url: str, surface: str) -> float:
    if not relative_url: return 0.5
    cache_key = f"{relative_url}_{surface}"
    if cache_key in SURFACE_STATS_CACHE: return SURFACE_STATS_CACHE[cache_key]
    if not relative_url.startswith("/"): relative_url = f"/{relative_url}"
    url = f"https://www.tennisexplorer.com{relative_url}?annual=all&t={int(time.time())}"
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        target_header = "Hard"
        if "clay" in surface.lower(): target_header = "Clay"
        elif "grass" in surface.lower(): target_header = "Grass"
        elif "indoor" in surface.lower(): target_header = "Indoors"
        tables = soup.find_all('table', class_='result')
        total_matches = 0; total_wins = 0
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
                except: pass
                break
        if total_matches > 0:
            rate = total_wins / total_matches
            SURFACE_STATS_CACHE[cache_key] = rate
            return rate
    except: pass
    finally: await page.close()
    return 0.5

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
        except: pass
        finally: await page.close()

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
                if t_name and t_surf: GLOBAL_SURFACE_MAP[t_name.lower()] = t_surf
                if t_loc and t_surf:
                    for part in t_loc.split(','):
                        part = part.strip().lower()
                        if part and len(part) > 2: GLOBAL_SURFACE_MAP[part] = t_surf
                            
        clean_skills = {}
        if skills:
            for entry in skills:
                if not isinstance(entry, dict): continue
                pid = entry.get('player_id')
                if pid:
                    clean_skills[pid] = {
                        'serve': to_float(entry.get('serve')), 'power': to_float(entry.get('power')),
                        'forehand': to_float(entry.get('forehand')), 'backhand': to_float(entry.get('backhand')),
                        'speed': to_float(entry.get('speed')), 'stamina': to_float(entry.get('stamina')),
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
    if market_odds <= 1.01 or fair_prob <= 0: return {"type": "NONE", "edge_percent": 0.0, "is_value": False}
    market_odds = min(market_odds, 100.0)
    edge = (fair_prob * market_odds) - 1
    edge_percent = round(edge * 100, 1)
    if edge_percent <= 0.5: return {"type": "NONE", "edge_percent": edge_percent, "is_value": False}

    label = "VALUE"
    if edge_percent >= 15.0: label = "🔥 HIGH VALUE" 
    elif edge_percent >= 8.0: label = "✨ GOOD VALUE" 
    elif edge_percent >= 2.0: label = "📈 THIN VALUE" 
    else: label = "👀 WATCH"

    return {"type": label, "edge_percent": edge_percent, "is_value": True}

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2, has_scouting_reports: bool, style_stats_p1: Optional[Dict], style_stats_p2: Optional[Dict]):
    ai_meta = ensure_dict(ai_meta)
    n1 = get_last_name(p1_name); n2 = get_last_name(p2_name)
    tour = "ATP"; bsi_val = to_float(bsi, 6.0)
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {}); p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo1 = p1_stats.get(elo_surf, 1500); elo2 = p2_stats.get(elo_surf, 1500)
    elo_diff_model = elo1 - elo2
    if market_odds1 > 0 and market_odds2 > 0:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
        implied_p1 = inv1 / (inv1 + inv2)
        if 0.01 < implied_p1 < 0.99:
            try: elo_diff_market = -400 * math.log10(1/implied_p1 - 1)
            except: elo_diff_market = elo_diff_model
        else: elo_diff_market = elo_diff_model 
        elo_diff_final = (elo_diff_model * 0.70) + (elo_diff_market * 0.30)
    else: elo_diff_final = elo_diff_model
    prob_elo = normal_cdf_prob(elo_diff_final, sigma=280.0)
    m1 = to_float(ai_meta.get('p1_tactical_score', 5)); m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8)
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    c1_score = get_offense(s1); c2_score = get_offense(s2)
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)
    f1 = to_float(ai_meta.get('p1_form_score', 5)); f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    style_boost = 0
    if style_stats_p1 and style_stats_p1['verdict'] == "DOMINANT": style_boost += 0.08 
    if style_stats_p1 and style_stats_p1['verdict'] == "STRUGGLES": style_boost -= 0.06
    if style_stats_p2 and style_stats_p2['verdict'] == "DOMINANT": style_boost -= 0.08 
    if style_stats_p2 and style_stats_p2['verdict'] == "STRUGGLES": style_boost += 0.06
    weights = [0.20, 0.15, 0.05, 0.50, 0.10];
    
    model_trust_factor = 0.25 
    
    total_w = sum(weights); weights = [w/total_w for w in weights]
    prob_alpha = (prob_matchup * weights[0]) + (prob_bsi * weights[1]) + (prob_skills * weights[2]) + (prob_elo * weights[3]) + (prob_form * weights[4])
    prob_alpha += style_boost
    if prob_alpha > 0.60: prob_alpha = min(prob_alpha * 1.05, 0.98)
    elif prob_alpha < 0.40: prob_alpha = max(prob_alpha * 0.95, 0.02)
    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
    final_prob = (prob_alpha * model_trust_factor) + (prob_market * (1 - model_trust_factor))
    return final_prob

def recalculate_fair_odds_with_new_market(old_fair_odds1: float, old_market_odds1: float, old_market_odds2: float, new_market_odds1: float, new_market_odds2: float) -> float:
    try:
        old_prob_market = 0.5
        if old_market_odds1 > 1 and old_market_odds2 > 1:
            inv1 = 1/old_market_odds1; inv2 = 1/old_market_odds2
            old_prob_market = inv1 / (inv1 + inv2)
        if old_fair_odds1 <= 1.01: return 0.5
        old_final_prob = 1 / old_fair_odds1
        alpha_part = old_final_prob - (old_prob_market * 0.40)
        prob_alpha = alpha_part / 0.60
        new_prob_market = 0.5
        if new_market_odds1 > 1 and new_market_odds2 > 1:
            inv1 = 1/new_market_odds1; inv2 = 1/new_market_odds2
            new_prob_market = inv1 / (inv1 + inv2)
        new_final_prob = (prob_alpha * 0.60) + (new_prob_market * 0.40)
        if new_market_odds1 < 1.10:
             mkt_prob1 = 1/new_market_odds1
             new_final_prob = (new_final_prob * 0.15) + (mkt_prob1 * 0.85)
        return new_final_prob
    except: return 0.5

# =================================================================
# 9. PIPELINE UTILS
# =================================================================
async def build_country_city_map(browser: Browser):
    if COUNTRY_TO_CITY_MAP: return
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
            except: pass
    except: pass
    finally: await page.close()

async def resolve_united_cup_via_country(p1):
    if not COUNTRY_TO_CITY_MAP: return None
    cache_key = f"COUNTRY_{p1}"
    if cache_key in TOURNAMENT_LOC_CACHE: country = TOURNAMENT_LOC_CACHE[cache_key]
    else:
        res = await call_groq(f"Country of player {p1}? JSON: {{'country': 'Name'}}")
        try:
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            country = data.get("country", "Unknown")
        except: country = "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
    if country in COUNTRY_TO_CITY_MAP: return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
    return None

async def resolve_ambiguous_tournament(p1, p2, scraped_name, p1_country, p2_country):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    p1_meta = METADATA_CACHE.get(normalize_db_name(p1))
    p2_meta = METADATA_CACHE.get(normalize_db_name(p2))
    oracle_data = p1_meta or p2_meta
    if oracle_data:
         real_name = oracle_data.get('tournament', scraped_name)
         prompt = f"TASK: Details for tennis tournament '{real_name}'. CONTEXT: {p1} vs {p2}. Date: {datetime.now().strftime('%B %Y')}. OUTPUT JSON: {{ 'city': 'Name', 'surface': 'Hard/Clay/Grass', 'indoor': true/false }}"
    else:
         prompt = f"TASK: Identify tournament location. MATCH: {p1} ({p1_country}) vs {p2} ({p2_country}). SOURCE: '{scraped_name}'. OUTPUT JSON: {{ 'city': 'Name', 'surface': 'Hard/Clay/Grass', 'indoor': true/false }}"
    res = await call_groq(prompt)
    if res:
        try: 
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            surface_type = data.get('surface', 'Hard')
            if data.get('indoor'): surface_type += " Indoor"
            else: surface_type += " Outdoor"
            est_bsi = 6.5
            if 'clay' in surface_type.lower(): est_bsi = 3.5
            elif 'grass' in surface_type.lower(): est_bsi = 8.0
            elif 'indoor' in surface_type.lower(): est_bsi = 7.5
            city = data.get('city', 'Unknown')
            if "plantation" in city.lower() and p1_country == "USA": city = "Winston-Salem"; surface_type = "Hard Indoor"; est_bsi = 7.5
            simulated_db_entry = {"city": city, "surface_guessed": surface_type, "bsi_estimate": est_bsi, "note": f"AI/Oracle: {city}"}
            TOURNAMENT_LOC_CACHE[scraped_name] = simulated_db_entry
            return simulated_db_entry
        except: pass
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2, p1_country="Unknown", p2_country="Unknown", match_date: datetime = None): 
    s_low = clean_tournament_name(tour).lower().strip()
    if "united cup" in s_low:
        arena_target = await resolve_united_cup_via_country(p1)
        if arena_target:
            for t in db_tours:
                if "united cup" in t['name'].lower() and arena_target.lower() in t.get('location', '').lower():
                    return t['surface'], t['bsi_rating'], f"United Cup ({arena_target})", arena_target
        return "Hard Court Outdoor", 8.3, "United Cup (Sydney Default)", "Sydney"

    if match_date:
        month = match_date.month
        s_clean = s_low.lower()
        if "oeiras" in s_clean:
             if month in [1, 2, 3, 10, 11, 12]: s_low = "oeiras indoor"
             elif month in [4, 5, 6, 7, 8, 9]: s_low = "oeiras red clay"
        elif "nottingham" in s_clean:
             if month in [6, 7]: s_low = "nottingham grass"
             else: s_low = "nottingham" 

    best_match = None; best_score = 0
    for t in db_tours:
        score = calculate_fuzzy_score(s_low, t['name'])
        if score > best_score: best_score = score; best_match = t
    if best_match and best_score >= 20: 
        loc = best_match.get('location', '')
        city_for_weather = loc.split(',')[0] if loc else best_match['name']
        return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', ''), city_for_weather
    
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour, p1_country, p2_country)
    ai_loc = ensure_dict(ai_loc)
    if ai_loc and ai_loc.get('city'):
        surf = ai_loc.get('surface_guessed', 'Hard Court Outdoor')
        bsi = ai_loc.get('bsi_estimate', 6.5)
        city = ai_loc.get('city', 'Unknown')
        return surf, bsi, "AI Guess", city
    return 'Hard Court Outdoor', 6.5, 'Fallback', tour.split()[0]

def get_city_from_note(note):
    if not note: return "Unknown"
    if "AI/Oracle:" in note: return note.split(":")[-1].strip()
    if "(" in note: return note.split("(")[-1].replace(")", "").strip()
    if len(note) > 50: return "Unknown" 
    return note

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1_data, form2_data, weather_data, p1_surface_profile, p2_surface_profile):
    fatigueA = await get_advanced_load_analysis(await fetch_player_history_extended(p1['last_name'], 10))
    fatigueB = await get_advanced_load_analysis(await fetch_player_history_extended(p2['last_name'], 10))
    
    tactical_data = TacticalComputer.calculate_matchup_math(
        s1, s2, 
        form1_data['score'], form2_data['score'], 
        fatigueA, fatigueB, 
        bsi, surface
    )
    
    score = tactical_data['calculated_score']
    reasons_bullet_points = "\n- ".join(tactical_data['reasons'])
    
    weather_str = "Weather: Unknown"
    if weather_data:
        weather_str = f"WEATHER: {weather_data['summary']}. IMPACT: {weather_data['impact_note']}"

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
    
    data = ensure_dict(safe_get_ai_data(res))
    if not data: return default
    
    data['p1_tactical_score'] = score
    data['p2_tactical_score'] = 10.0 - score
    return data

def safe_get_ai_data(res_text: Optional[str]) -> Dict[str, Any]:
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': '', 'p1_win_sentiment': 0.5}
    if not res_text: return default
    try:
        cleaned = res_text.replace("json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return ensure_dict(data)
    except: return default

async def update_past_results(browser: Browser):
    log("🏆 The Auditor: Checking Real-Time Results & Scores via TE...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending or not isinstance(pending, list): return
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
            
            rows = table.find_all('tr')
            for row in rows:
                if 'flags' in str(row) or 'head' in str(row): continue
                
                row_text = row.get_text(separator=" ", strip=True).lower()
                row_norm = normalize_text(row_text).lower()

                for pm in list(safe_to_check):
                    p1_norm = normalize_db_name(pm['player1_name'])
                    p2_norm = normalize_db_name(pm['player2_name'])
                    if p1_norm in row_norm and p2_norm in row_norm:
                        pattern = r'(\d+-\d+(?:\(\d+\))?|ret\.|w\.o\.)'
                        all_matches = re.findall(pattern, row_text, flags=re.IGNORECASE)
                        valid_sets = []
                        for m in all_matches:
                            if "ret" in m or "w.o" in m: valid_sets.append(m)
                            elif "-" in m:
                                try:
                                    l, r = map(int, m.split('(')[0].split('-'))
                                    if (l >= 6 or r >= 6) or (l+r >= 6): 
                                        valid_sets.append(m)
                                except: pass
                        score_cleaned = " ".join(valid_sets).strip()

                        score_matches = re.findall(r'(\d+)-(\d+)', score_cleaned)
                        p1_sets = 0; p2_sets = 0
                        idx_p1 = row_norm.find(p1_norm); idx_p2 = row_norm.find(p2_norm)
                        p1_is_left = idx_p1 < idx_p2
                        
                        for s in score_matches:
                            try:
                                sl = int(s[0]); sr = int(s[1])
                                if sl > sr: 
                                    if p1_is_left: p1_sets += 1
                                    else: p2_sets += 1
                                elif sr > sl: 
                                    if p1_is_left: p2_sets += 1
                                    else: p1_sets += 1
                            except: pass
                        
                        is_ret = "ret." in row_text or "w.o." in row_text
                        sets_needed = 2
                        if "open" in pm['tournament'].lower() and ("atp" in pm['tournament'].lower() or "men" in pm['tournament'].lower()): sets_needed = 3
                        
                        winner = None
                        if p1_sets >= sets_needed or p2_sets >= sets_needed or is_ret:
                            if is_ret:
                                if p1_is_left: winner = pm['player1_name']
                                else: winner = pm['player2_name']
                            else:
                                if p1_sets > p2_sets: winner = pm['player1_name']
                                elif p2_sets > p1_sets: winner = pm['player2_name']
                            
                            if winner:
                                log(f"      🔍 AUDITOR FOUND: {score_cleaned} -> Winner: {winner}")
                                supabase.table("market_odds").update({
                                    "actual_winner_name": winner,
                                    "score": score_cleaned
                                }).eq("id", pm['id']).execute()
                                
                                # RE-PROFILING HOOK (FOR REAL-TIME SYNC)
                                log(f"🔄 Triggering Real-Time Profile Refresh for {pm['player1_name']} & {pm['player2_name']}")
                                for p_name in [pm['player1_name'], pm['player2_name']]:
                                    p_hist = await fetch_player_history_extended(p_name, limit=80)
                                    p_profile = SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name)
                                    p_form = MomentumV2Engine.calculate_rating(p_hist[:20], p_name)
                                    
                                    supabase.table('players').update({
                                        'surface_ratings': p_profile,
                                        'form_rating': p_form 
                                    }).ilike('last_name', f"%{p_name}%").execute()

                                safe_to_check = [x for x in safe_to_check if x['id'] != pm['id']]
                                break
        except: pass
        finally: await page.close()

# =================================================================
# 10. QUANTUM GAMES SIMULATOR (CALIBRATED V83.1)
# =================================================================
class QuantumGamesSimulator:
    
    @staticmethod
    def derive_hold_probability(server_skills: Dict, returner_skills: Dict, bsi: float, surface: str) -> float:
        p_hold = 67.0 
        
        srv = server_skills.get('serve', 50)
        pwr = server_skills.get('power', 50)
        
        p_hold += (srv - 50) * 0.35
        p_hold += (pwr - 50) * 0.10

        ret_speed = returner_skills.get('speed', 50)
        ret_mental = returner_skills.get('mental', 50)
        
        p_hold -= (ret_speed - 50) * 0.15
        p_hold -= (ret_mental - 50) * 0.08

        bsi_delta = (bsi - 6.0) * 1.4 
        p_hold += bsi_delta

        return max(52.0, min(94.0, p_hold)) / 100.0

    @staticmethod
    def simulate_set(p1_prob: float, p2_prob: float) -> tuple[int, int]:
        g1, g2 = 0, 0
        while True:
            if random.random() < p1_prob: g1 += 1
            else: g2 += 1 
            
            if g1 >= 6 and g1 - g2 >= 2: return (1, g1 + g2)
            if g2 >= 6 and g2 - g1 >= 2: return (2, g1 + g2)
            
            if g1 == 6 and g2 == 6:
                tb_prob_p1 = 0.5 + (p1_prob - p2_prob)
                if random.random() < tb_prob_p1: return (1, 13)
                else: return (2, 13)
            
            if random.random() < p2_prob: g2 += 1
            else: g1 += 1 
            
            if g1 >= 6 and g1 - g2 >= 2: return (1, g1 + g2)
            if g2 >= 6 and g2 - g1 >= 2: return (2, g1 + g2)
            
            if g1 == 6 and g2 == 6:
                tb_prob_p1 = 0.5 + (p1_prob - p2_prob)
                if random.random() < tb_prob_p1: return (1, 13)
                else: return (2, 13)

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
        median_games = total_games_log[len(total_games_log)//2]
        avg_games = sum(total_games_log) / len(total_games_log)
        
        probs = {
            "over_20_5": sum(1 for x in total_games_log if x > 20.5) / iterations,
            "over_21_5": sum(1 for x in total_games_log if x > 21.5) / iterations,
            "over_22_5": sum(1 for x in total_games_log if x > 22.5) / iterations,
            "over_23_5": sum(1 for x in total_games_log if x > 23.5) / iterations
        }
        
        return {
            "predicted_line": round(avg_games, 1),
            "median_games": median_games,
            "probabilities": probs,
            "sim_details": {
                "p1_est_hold_pct": round(p1_hold_prob * 100, 1),
                "p2_est_hold_pct": round(p2_hold_prob * 100, 1)
            }
        }

# --- SMART FREEZE HELPER ---
def is_valid_opening_odd(o1: float, o2: float) -> bool:
    if o1 < 1.06 and o2 < 1.06: return False 
    if o1 <= 1.01 or o2 <= 1.01: return False 
    return True

async def run_pipeline():
    log(f"🚀 Neural Scout V97.0 (1WIN MASTER FEED EDITION) Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            # 1. AUDITOR (TE Historie prüfen)
            await update_past_results(browser)
            
            # 2. CACHES & DB INIT
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: return
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            
            # =================================================================
            # 🌍 GLOBAL PROFILER (V95 SCHEMA SYNC)
            # =================================================================
            def needs_surface_update(p_data):
                sr = p_data.get('surface_ratings')
                if not sr: return True
                if isinstance(sr, str):
                    try: sr = json.loads(sr)
                    except: return True
                if not isinstance(sr, dict) or len(sr) == 0: return True
                if not sr.get('_v95_mastery_applied'): return True
                return False

            players_to_update = [p for p in players if needs_surface_update(p)]
            if players_to_update: log(f"🔄 Syncing {len(players_to_update)} players to new schema...")
            
            for p_data in players_to_update:
                p_name = p_data['last_name']
                p_hist = await fetch_player_history_extended(p_name, limit=80)
                p_profile = SurfaceIntelligence.compute_player_surface_profile(p_hist, p_name)
                p_form = MomentumV2Engine.calculate_rating(p_hist[:20], p_name)
                try:
                    supabase.table('players').update({
                        'surface_ratings': p_profile,
                        'form_rating': p_form
                    }).eq('id', p_data['id']).execute()
                except Exception as e: pass
                await asyncio.sleep(0.05) 
            # =================================================================
            
            # --- THE 1WIN MASTER FEED CALL ---
            # Das ist jetzt der einzige Ort, an dem wir neue Spiele beziehen!
            target_date = datetime.now() # Für Date-Referenz des Turniers
            METADATA_CACHE.update(await scrape_oracle_metadata(browser, target_date))
            
            matches = await fetch_1win_api_markets()
            
            if not matches:
                log("❌ Keine Matches vom 1win Feed erhalten. Beende Zyklus.")
                return
                
            log(f"🔍 Starte Oracle Enrichment für {len(matches)} 1win Matches...")
            
            for m in matches:
                try:
                    await asyncio.sleep(0.5) 
                    
                    p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                    p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                    if p1_obj and p2_obj:
                        n1 = p1_obj['last_name']; n2 = p2_obj['last_name']
                        if n1 == n2: continue
                        if p1_obj.get('tour') != p2_obj.get('tour'):
                            if "united cup" not in m['tour'].lower(): continue 
                        
                        # --- V83.4: MARKET SANITY GATEKEEPER ---
                        # Ignoriere Matches, in denen wir (noch) keine sauberen Quoten geparst haben
                        if m['odds1'] > 0 and m['odds2'] > 0:
                            if not validate_market_integrity(m['odds1'], m['odds2']):
                                continue 

                        existing_match = None
                        res1 = supabase.table("market_odds").select("*").eq("player1_name", n1).eq("player2_name", n2).order("created_at", desc=True).limit(1).execute()
                        if res1.data: existing_match = res1.data[0]
                        else:
                            res2 = supabase.table("market_odds").select("*").eq("player1_name", n2).eq("player2_name", n1).order("created_at", desc=True).limit(1).execute()
                            if res2.data: existing_match = res2.data[0]
                        
                        # --- VELOCITY CHECK ---
                        if existing_match and m['odds1'] > 0 and m['odds2'] > 0:
                            prev_o1 = to_float(existing_match.get('odds1'), 0)
                            prev_o2 = to_float(existing_match.get('odds2'), 0)
                            if is_suspicious_movement(prev_o1, m['odds1'], prev_o2, m['odds2']):
                                continue

                        db_match_id = None
                        hist_fair1 = 0; hist_fair2 = 0; hist_is_value = False; hist_pick_player = None; hist_is_locked = False
                        
                        is_signal_locked = False
                        if existing_match:
                            db_match_id = existing_match['id']
                            if existing_match.get('actual_winner_name'): continue 

                            if has_active_signal(existing_match.get('ai_analysis_text', '')):
                                is_signal_locked = True
                                hist_is_locked = True
                                log(f"      🔒 DIAMOND LOCK ACTIVE: {n1} vs {n2}")

                        if is_signal_locked:
                            update_data = {}
                            if m['odds1'] > 0:
                                update_data["odds1"] = m['odds1']
                                update_data["odds2"] = m['odds2']
                            
                            # Update Deep Markets
                            if m.get('handicap_line') is not None:
                                update_data['handicap_line'] = m['handicap_line']
                                update_data['handicap_odds1'] = m['handicap_odds1']
                                update_data['handicap_odds2'] = m['handicap_odds2']
                            if m.get('over_under_line') is not None:
                                update_data['over_under_line'] = m['over_under_line']
                                update_data['over_odds'] = m['over_odds']
                                update_data['under_odds'] = m['under_odds']
                            
                            if m['odds1'] > 0:
                                stored_op1 = to_float(existing_match.get('opening_odds1'), 0)
                                stored_op2 = to_float(existing_match.get('opening_odds2'), 0)
                                if not is_valid_opening_odd(stored_op1, stored_op2) and is_valid_opening_odd(m['odds1'], m['odds2']):
                                     update_data["opening_odds1"] = m['odds1']
                                     update_data["opening_odds2"] = m['odds2']

                            if update_data:
                                supabase.table("market_odds").update(update_data).eq("id", db_match_id).execute()
                            
                            hist_fair1 = to_float(existing_match.get('ai_fair_odds1'), 0)
                            hist_fair2 = to_float(existing_match.get('ai_fair_odds2'), 0)
                            
                        else:
                            cached_ai = {}
                            if existing_match and existing_match.get('ai_analysis_text'):
                                cached_ai = {'ai_text': existing_match.get('ai_analysis_text'), 'ai_fair_odds1': existing_match.get('ai_fair_odds1'), 'old_odds1': existing_match.get('odds1', 0), 'old_odds2': existing_match.get('odds2', 0), 'last_update': existing_match.get('created_at')}
                            
                            c1 = p1_obj.get('country', 'Unknown'); c2 = p2_obj.get('country', 'Unknown')
                            
                            surf, bsi, notes, city_for_weather = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2, c1, c2, match_date=target_date)
                            weather_data = await fetch_weather_data(city_for_weather)

                            s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p2_obj['id']), {})
                            
                            p1_history = await fetch_player_history_extended(n1, limit=80)
                            p2_history = await fetch_player_history_extended(n2, limit=80)
                            
                            style_stats_p1 = get_style_matchup_stats_py(p1_history, n1, p2_obj.get('play_style', ''), supabase)
                            style_stats_p2 = get_style_matchup_stats_py(p2_history, n2, p1_obj.get('play_style', ''), supabase)

                            p1_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p1_history, n1)
                            p2_surface_profile = SurfaceIntelligence.compute_player_surface_profile(p2_history, n2)
                            
                            p1_form_v2 = MomentumV2Engine.calculate_rating(p1_history[:20], n1)
                            p2_form_v2 = MomentumV2Engine.calculate_rating(p2_history[:20], n2)

                            # Dummy HREFs for TE Stats, da wir sie nicht mehr haben
                            surf_rate1 = 0.5
                            surf_rate2 = 0.5
                            
                            is_value_active = False; value_pick_player = None
                            
                            should_run_ai = True
                            if db_match_id and cached_ai:
                                if m['odds1'] > 0:
                                    odds_diff = max(abs(cached_ai['old_odds1'] - m['odds1']), abs(cached_ai['old_odds2'] - m['odds2']))
                                    is_significant_move = odds_diff > (m['odds1'] * 0.05)
                                else:
                                    is_significant_move = False
                                    
                                try:
                                    last_up = datetime.fromisoformat(cached_ai.get('last_update', '').replace('Z', '+00:00'))
                                    is_stale = (datetime.now(timezone.utc) - last_up) > timedelta(hours=6)
                                except: is_stale = True
                                if not is_significant_move and not is_stale: should_run_ai = False
                                
                            if not should_run_ai:
                                ai_text_final = cached_ai['ai_text']
                                new_prob = recalculate_fair_odds_with_new_market(cached_ai['ai_fair_odds1'], cached_ai['old_odds1'], cached_ai['old_odds2'], m['odds1'], m['odds2'])
                                
                                fair1 = round(1/new_prob, 2) if new_prob > 0.01 else 99
                                fair2 = round(1/(1-new_prob), 2) if new_prob < 0.99 else 99
                                
                                val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                                val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                                
                                value_tag = ""
                                if val_p1["is_value"]: 
                                    value_tag = f" [{val_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                    is_value_active = True; value_pick_player = n1
                                elif val_p2["is_value"]: 
                                    value_tag = f" [{val_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                    is_value_active = True; value_pick_player = n2
                                
                                ai_text_base = re.sub(r'\[.*?\]', '', ai_text_final).strip()
                                ai_text_final = ai_text_base + value_tag

                                hist_fair1 = fair1; hist_fair2 = fair2
                                hist_is_value = is_value_active; hist_pick_player = value_pick_player

                            else:
                                log(f"   🧠 Fresh Analysis & Simulation: {n1} vs {n2}")
                                
                                f1_data = p1_form_v2; f2_data = p2_form_v2
                                elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
                                e1 = ELO_CACHE.get("ATP", {}).get(n1.lower(), {}).get(elo_key, 1500)
                                e2 = ELO_CACHE.get("ATP", {}).get(n2.lower(), {}).get(elo_key, 1500)
                                
                                sim_result = QuantumGamesSimulator.run_simulation(s1, s2, bsi, surf)
                                ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, e1, e2, f1_data, f2_data, weather_data, p1_surface_profile, p2_surface_profile)
                                prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai, m['odds1'], m['odds2'], surf_rate1, surf_rate2, bool(r1.get('strengths')), style_stats_p1, style_stats_p2)
                                
                                fair1 = round(1/prob, 2) if prob > 0.01 else 99
                                fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                                
                                val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                                val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                                
                                value_tag = ""
                                if val_p1["is_value"]: 
                                    value_tag = f" [{val_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {val_p1['edge_percent']}%]"
                                    is_value_active = True; value_pick_player = n1
                                elif val_p2["is_value"]: 
                                    value_tag = f" [{val_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {val_p2['edge_percent']}%]"
                                    is_value_active = True; value_pick_player = n2
                                
                                games_tag = f" [🎲 SIM: {sim_result['predicted_line']} Games]"
                                ai_text_base = ai.get('ai_text', '').replace("json", "").strip()
                                ai_text_final = f"{ai_text_base} {value_tag} {games_tag}"
                                if style_stats_p1 and style_stats_p1['verdict'] != "Neutral": ai_text_final += f" (Note: {n1} {style_stats_p1['verdict']})"
                            
                                data = {
                                    "player1_name": n1, "player2_name": n2, "tournament": m['tour'],
                                    "ai_fair_odds1": fair1, "ai_fair_odds2": fair2,
                                    "ai_analysis_text": ai_text_final,
                                    "games_prediction": sim_result, 
                                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "match_time": f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                                }
                                
                                if m['odds1'] > 0:
                                    data["odds1"] = m['odds1']
                                    data["odds2"] = m['odds2']
                                
                                # Deep Markets
                                if m.get('handicap_line') is not None:
                                    data['handicap_line'] = m['handicap_line']
                                    data['handicap_odds1'] = m['handicap_odds1']
                                    data['handicap_odds2'] = m['handicap_odds2']
                                if m.get('over_under_line') is not None:
                                    data['over_under_line'] = m['over_under_line']
                                    data['over_odds'] = m['over_odds']
                                    data['under_odds'] = m['under_odds']
                                
                                hist_fair1 = fair1; hist_fair2 = fair2
                                hist_is_value = is_value_active; hist_pick_player = value_pick_player
                                
                                final_match_id = None
                                if db_match_id:
                                    supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                                    final_match_id = db_match_id
                                    log(f"🔄 Updated: {n1} vs {n2} (via 1WIN)")
                                else:
                                    if is_valid_opening_odd(m['odds1'], m['odds2']):
                                        data["opening_odds1"] = m['odds1']
                                        data["opening_odds2"] = m['odds2']
                                    res_insert = supabase.table("market_odds").insert(data).execute()
                                    if res_insert.data: final_match_id = res_insert.data[0]['id']
                                    log(f"💾 Saved: {n1} vs {n2} (via 1WIN)")
                                
                                db_match_id = final_match_id

                        if db_match_id and m['odds1'] > 0:
                            should_log_history = False
                            if not existing_match: should_log_history = True
                            elif hist_is_locked: should_log_history = True
                            elif hist_is_value: should_log_history = True
                            else:
                                try:
                                    old_o1 = to_float(existing_match.get('odds1'), 0)
                                    if abs(old_o1 - m['odds1']) > 0.001: should_log_history = True
                                except: pass
                            
                            if should_log_history:
                                pick_name = "LOCKED" if hist_is_locked else hist_pick_player
                                h_data = {
                                    "match_id": db_match_id, 
                                    "odds1": m['odds1'], 
                                    "odds2": m['odds2'], 
                                    "fair_odds1": hist_fair1, 
                                    "fair_odds2": hist_fair2, 
                                    "is_hunter_pick": (hist_is_value or hist_is_locked),
                                    "pick_player_name": pick_name,
                                    "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                }
                                try: supabase.table("odds_history").insert(h_data).execute()
                                except: pass

                except Exception as e: log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
