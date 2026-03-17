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
from pywebpush import webpush, WebPushException # 🚀 SOTA: Mobile Push Notifications

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

log("🔌 Initialisiere Neural Scout (V204.9 - ULTIMATE H2H & PUSH EDITION)...")

# Secrets Load
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Zwinge Python, die Master-Rechte zu nutzen (für Push-Sub-Abfrage)
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
API_TENNIS_KEY = os.environ.get("API_TENNIS_KEY") # 🚀 SOTA API KEY

# 🚀 SOTA: WEB PUSH SECRETS (Clean Architecture)
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
if not VAPID_PRIVATE_KEY:
    VAPID_PRIVATE_KEY = "lilvmR9dnAWN-u4G5Skyu-2IYb6n4E_OIRy7IGrGTWo"
VAPID_CLAIMS = {"sub": "mailto:bh.dtl@web.de"} 

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

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

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

# 🚀 SOTA: NATIVE MOBILE PUSH ALERT (iOS Koma-Sicher) - MASS BROADCAST
async def fire_sniper_push(match_data: Dict, edge: float, pick_name: str, odds: float):
    try:
        log(f"🚨 [PUSH ENGINE] Trigger ausgelöst für {pick_name}! Prüfe Datenbank auf Handys...")
        if not VAPID_PRIVATE_KEY:
            log("⚠️ [PUSH ENGINE] VAPID_PRIVATE_KEY fehlt in Secrets!")
            return

        subs_res = supabase.table("push_subscriptions").select("id, subscription").execute()
        subscriptions = subs_res.data or []
        
        log(f"🔍 [PUSH ENGINE] Supabase Antwort: {len(subscriptions)} Token(s) gefunden.")

        if not subscriptions:
            log("❌ [PUSH ENGINE] ABBRUCH! Tabelle 'push_subscriptions' ist leer ODER RLS blockiert den Lesezugriff (Service Role Key fehlt).")
            return

        payload = json.dumps({
            "title": f"🚨 +{round(edge, 1)}% Edge Detected!",
            "body": f"Pick: {pick_name} @ {odds:.2f} \nTournament: {match_data.get('tournament', 'Unknown')}",
            "url": "/sniper-feed",
            "icon": "/icon-192x192.png", 
            "badge": "/badge-72x72.png"  
        })

        success_count = 0
        for row in subscriptions:
            sub_data = row['subscription']
            sub_id = row['id']
            try:
                webpush(
                    subscription_info=sub_data,
                    data=payload,
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims=VAPID_CLAIMS,
                    ttl=43200 
                )
                success_count += 1
            except WebPushException as ex:
                if hasattr(ex, 'response') and ex.response is not None:
                    if ex.response.status_code in [404, 410]:
                        try:
                            supabase.table("push_subscriptions").delete().eq("id", sub_id).execute()
                            log(f"🗑️ [PUSH ENGINE] Totes Push-Abo gelöscht.")
                        except: pass
        
        if success_count > 0:
            log(f"📲 GLOBAL PUSH GESENDET an {success_count} Geräte: {pick_name}")
    except Exception as e:
        log(f"⚠️ Global Push Error: {e}")

# 🚀 SOTA: PERSONALIZED TARGETED PUSH ALERT (Für OddsLab Limit-Orders)
async def fire_targeted_user_push(user_id: str, match_data: Dict, pick_name: str, target_odds: float, actual_odds: float):
    try:
        log(f"🎯 [PUSH ENGINE] Bereite gezielten Push für User {user_id} vor...")
        if not VAPID_PRIVATE_KEY:
            return

        # Wir holen exakt NUR den Token für diesen einen User
        subs_res = supabase.table("push_subscriptions").select("id, subscription").eq("user_id", user_id).execute()
        subscriptions = subs_res.data or []
        
        if not subscriptions:
            log(f"⚠️ [PUSH ENGINE] Kein Push-Token für User {user_id} gefunden. Alarm wird übersprungen.")
            return

        payload = json.dumps({
            "title": f"🎯 SNIPER ALERT TRIGGERED!",
            "body": f"Your target of {target_odds:.2f} for {pick_name} has been hit!\nCurrent odds: {actual_odds:.2f}",
            "url": "/odds-lab",
            "icon": "/icon-192x192.png",
            "badge": "/badge-72x72.png"
        })

        for row in subscriptions:
            sub_data = row['subscription']
            sub_id = row['id']
            try:
                webpush(
                    subscription_info=sub_data,
                    data=payload,
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims=VAPID_CLAIMS,
                    ttl=43200 
                )
                log(f"🎯 TARGETED PUSH GESENDET an User {user_id} für Pick: {pick_name}")
            except WebPushException as ex:
                if hasattr(ex, 'response') and ex.response is not None:
                    if ex.response.status_code in [404, 410]:
                        try:
                            supabase.table("push_subscriptions").delete().eq("id", sub_id).execute()
                        except: pass
    except Exception as e:
        log(f"⚠️ Targeted Push Error: {e}")

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
            
            # 🚀 ARCHITECT FIX: Hybrid Schema Logic (Legacy vs API-Tennis)
            if "ret" in score_str or "w.o" in score_str:
                actual_perf = 0.6 if won else 0.4
            else:
                # PATH A: The API-Tennis Set-Only Format (e.g. "2-0", "1-2")
                # Matches exactly one pair of numbers summing to <= 5
                set_only_match = re.match(r'^(\d+)\s*-\s*(\d+)$', score_str)
                
                if set_only_match and sum(int(x) for x in set_only_match.groups()) <= 5:
                    l, r = int(set_only_match.group(1)), int(set_only_match.group(2))
                    p_sets = l if is_p1 else r
                    o_sets = r if is_p1 else l
                    
                    if p_sets >= o_sets + 2 or (p_sets == 2 and o_sets == 0):
                        actual_perf = 0.85  # Absolute Dominance
                    elif p_sets > o_sets:
                        actual_perf = 0.65  # Grind/Resilience Win
                    elif o_sets >= p_sets + 2 or (o_sets == 2 and p_sets == 0):
                        actual_perf = 0.15  # Outclassed
                    elif o_sets > p_sets:
                        actual_perf = 0.35  # Competitive Loss
                    else:
                        actual_perf = 0.75 if won else 0.25 # Fallback
                
                # PATH B: The Legacy DB Format (e.g. "6-4 6-2", "3-6 7-6 6-2")
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
            
            # Win/Loss Bias (Psychological Momentum)
            if won:
                match_edge += 0.40  
            else:
                match_edge -= 0.20
            
            # Time Decay: Aktuellste Matches zählen bis zu 100%, ältuälteste ~30%
            time_weight = 0.3 + (0.7 * (idx / max(1, len(chrono_matches) - 1)))
            
            cumulative_edge += (match_edge * time_weight)
            total_weight += time_weight
            
            history_log.append("W" if won else "L")

        # Streak Multiplier
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
        
        # Color & Text Generation
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
    def compute_player_surface_profile(matches: List[Dict], player_name: str, api_stats: Dict = None) -> Dict[str, Any]:
        profile = {}
        
        surfaces_data = {
            "hard": SurfaceIntelligence.get_matches_by_surface(matches, "hard"),
            "clay": SurfaceIntelligence.get_matches_by_surface(matches, "clay"),
            "grass": SurfaceIntelligence.get_matches_by_surface(matches, "grass")
        }
        
        for surf, surf_matches in surfaces_data.items():
            n_surf = len(surf_matches)
            wins = 0
            for m in surf_matches:
                winner = str(m.get('actual_winner_name', ""))
                if is_same_player(player_name, winner):
                    wins += 1
            
            if api_stats and isinstance(api_stats, list) and len(api_stats) > 0:
                recent_season = api_stats[0]
                api_won = int(recent_season.get(f"{surf}_won") or 0)
                api_lost = int(recent_season.get(f"{surf}_lost") or 0)
                api_total = api_won + api_lost
                
                if api_total > 5:
                    n_surf = api_total
                    wins = api_won

            if n_surf == 0:
                profile[surf] = {
                    "rating": 3.5, 
                    "color": "#808080",
                    "matches_tracked": 0,
                    "text": "No Experience"
                }
                continue
                
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
# 5. SOTA MARKOV CHAIN ENGINE
# =================================================================
class MarkovChainEngine:
    @staticmethod
    def run_simulation(s1: Dict, s2: Dict, formA: float, formB: float, 
                       bsi: float, styleA: str, styleB: str, 
                       iterations: int = 2500) -> Dict[str, Any]:
        
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
                    if (fair1 < fair2 and p1_name.lower() in winner.lower()) or (fair2 < fair1 and p2_name.lower() in winner.lower()):
                        correct_predictions += 1

                p1_obj = next((p for p in players if p.get('last_name') == p1_name and p['id'] in tour_players), None)
                p2_obj = next((p for p in players if p.get('last_name') == p2_name and p['id'] in tour_players), None)
                
                if p1_obj and p2_obj:
                    s1 = all_skills.get(p1_obj['id'], {}).get('overall_rating', 50)
                    s2 = all_skills.get(p2_obj['id'], {}).get('overall_rating', 50)
                    history_data.append({
                        "skillA": s1, "formA": 5.5, "surfA": 5.5,
                        "skillB": s2, "formB": 5.5, "surfB": 5.5,
                        "winner_is_A": p1_name.lower() in winner.lower()
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
async def fetch_player_history_extended(player_last_name: str, limit: int = 80) -> List[Dict]:
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
            is_reversed = False
            
            for pm in list(safe_to_check):
                # SOTA ARCHITECT UPGRADE: O(1) Key Lookup
                if pm.get('api_match_key') and str(pm['api_match_key']) == str(fix.get('event_key')):
                    matched_pm = pm
                    break

                # Fallback für alte DB-Einträge ohne Key:
                db_p1_last = normalize_db_name(get_last_name(pm['player1_name']))
                db_p2_last = normalize_db_name(get_last_name(pm['player2_name']))
                api_p1_last = normalize_db_name(get_last_name(p1_api))
                api_p2_last = normalize_db_name(get_last_name(p2_api))
                
                if (db_p1_last == api_p1_last and db_p2_last == api_p2_last) or \
                   (get_similarity(db_p1_last, api_p1_last) > 0.80 and get_similarity(db_p2_last, api_p2_last) > 0.80):
                    matched_pm = pm
                    break
                elif (db_p1_last == api_
