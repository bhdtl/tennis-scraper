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

log("🔌 Initialisiere Neural Scout (V85.0 - DIRECT LINKING [GROQ])...")

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
    # Check for V60+ Bracket format or V44 legacy format OR V81 Value Signals
    if "[" in text and "]" in text:
        # Legacy Icons + New Value Icons (Fire, Sparkles, Chart, Eyes)
        if any(icon in text for icon in ["💎", "🛡️", "⚖️", "💰", "🔥", "✨", "📈", "👀"]):
            return True
    return False

# --- MARKET INTEGRITY & ANTI-SPIKE ENGINE (V85.0) ---
def validate_market_integrity(o1: float, o2: float) -> bool:
    """
    Prüft, ob der Markt mathematisch valide ist.
    Filtrert TennisExplorer Glitches wie 1.03 vs 1.03 oder extreme Arbitrage.
    """
    if o1 <= 1.01 or o2 <= 1.01: return False 
    if o1 > 100 or o2 > 100: return False 

    # Berechne Overround (Buchmacher Marge)
    implied_prob = (1/o1) + (1/o2)
    
    # 1. Reject Massive Arbitrage (Fehlerhafte Daten)
    if implied_prob < 0.90: return False # Adjusted for loose markets

    # 2. Reject Massive Juice (Glitch)
    if implied_prob > 1.25: return False

    return True

def is_suspicious_movement(old_o1: float, new_o1: float, old_o2: float, new_o2: float) -> bool:
    """
    Velocity Check: Blockiert unrealistische Sprünge (Spikes).
    """
    if old_o1 == 0 or old_o2 == 0: return False 

    change_p1 = abs(new_o1 - old_o1) / old_o1
    change_p2 = abs(new_o2 - old_o2) / old_o2

    # REGEL: Wenn sich eine Quote um mehr als 50% ändert in EINEM Tick
    if change_p1 > 0.50 or change_p2 > 0.50:
        # Ausnahme: Extreme Favoriten
        if old_o1 < 1.10 or old_o2 < 1.10: return False
        return True
    
    return False

# =================================================================
# 3. QUANTUM FORM ENGINE (VEGAS-BEATER)
# =================================================================
class QuantumFormEngine:
    @staticmethod
    def get_rating_visuals(score: float) -> Dict[str, str]:
        s = round(score, 1)
        if s >= 10.0: return {"color": "RAINBOW_SHINY", "hex": "#FF00FF", "desc": "🦄 MYTHICAL"}
        if s >= 9.5: return {"color": "LIGHT_PINK_PURPLE", "hex": "#E066FF", "desc": "🔮 TRANSCENDENT"}
        if s >= 9.0: return {"color": "PURPLE", "hex": "#800080", "desc": "👿 GODLIKE"}
        if s >= 8.5: return {"color": "DARK_BLUE", "hex": "#00008B", "desc": "🌊 ELITE"}
        if s >= 8.0: return {"color": "BLUE", "hex": "#0000FF", "desc": "🧊 COLD BLOODED"}
        if s >= 7.5: return {"color": "DARK_GREEN", "hex": "#006400", "desc": "🌲 PEAK"}
        if s >= 7.0: return {"color": "GREEN", "hex": "#008000", "desc": "🌿 SOLID"}
        if s >= 6.5: return {"color": "YELLOW", "hex": "#FFFF00", "desc": "⚠️ AVERAGE"}
        if s >= 6.0: return {"color": "LIGHT_RED", "hex": "#FF6666", "desc": "🔥 WARMING UP"}
        if s >= 5.5: return {"color": "RED", "hex": "#FF0000", "desc": "🚩 STRUGGLING"}
        return {"color": "DARK_RED", "hex": "#8B0000", "desc": "🛑 DISASTER"}

    @staticmethod
    def parse_score_details(score_str: str, player_won: bool) -> Dict[str, float]:
        if not score_str or "ret" in score_str.lower() or "wo" in score_str.lower():
            return {"dominance": 0.5}
        matches = re.findall(r'(\d+)-(\d+)', score_str)
        if not matches: return {"dominance": 0.5}
        games_won = 0; games_lost = 0; sets_won = 0; sets_lost = 0
        for s in matches:
            l = int(s[0]); r = int(s[1])
            p_games = l if player_won else r
            o_games = r if player_won else l
            games_won += p_games; games_lost += o_games
            if p_games > o_games: sets_won += 1
            elif o_games > p_games: sets_lost += 1
        total_games = games_won + games_lost
        if total_games == 0: return {"dominance": 0.5}
        dominance = games_won / total_games
        if player_won and sets_lost == 0: dominance += 0.1
        if not player_won and sets_won > 0: dominance += 0.15
        return {"dominance": min(max(dominance, 0.0), 1.0)}

    @staticmethod
    def calculate_match_performance(odds: float, won: bool, score_str: str) -> float:
        if odds <= 1.0: odds = 1.01
        details = QuantumFormEngine.parse_score_details(score_str, won)
        dominance = details.get("dominance", 0.5)
        delta = 0.0
        if won:
            if odds < 1.20: delta = 0.1 + (dominance * 0.1) 
            elif 1.20 <= odds <= 2.00: delta = 0.3 + (dominance * 0.2)
            elif 2.00 < odds <= 3.00: delta = 0.8 + (dominance * 0.3)
            elif odds > 3.00: log_boost = math.log(odds, 2); delta = 1.0 + (log_boost * 0.3)
        else:
            if odds < 1.20: delta = -1.5 - (1.0 - dominance) 
            elif 1.20 <= odds <= 2.00: delta = -0.6 - (0.5 - dominance)
            elif 2.00 < odds <= 3.00:
                if dominance > 0.45: delta = +0.1 
                else: delta = -0.2
            elif odds > 3.00:
                if dominance > 0.4: delta = +0.2 
                else: delta = 0.0
        return delta

    @classmethod
    def calculate_player_form(cls, matches: List[Dict], player_name: str) -> Dict[str, Any]:
        current_rating = 6.5 
        history_log = []
        sorted_matches = sorted(matches, key=lambda x: x.get('created_at', ''))
        for i, m in enumerate(sorted_matches):
            is_p1 = player_name.lower() in m['player1_name'].lower()
            odds = m['odds1'] if is_p1 else m['odds2']
            winner_name = m.get('actual_winner_name', '')
            won = False
            if winner_name: won = player_name.lower() in winner_name.lower()
            score_str = m.get('score', '')
            match_delta = cls.calculate_match_performance(odds, won, score_str)
            weight = 0.5 + (i * 0.2) 
            weighted_delta = match_delta * weight
            current_rating += weighted_delta
            icon = '✅' if won else '❌'
            history_log.append(f"{icon}(@{odds})")
        final_rating = max(0.0, min(10.0, current_rating))
        visuals = cls.get_rating_visuals(final_rating)
        return {"score": round(final_rating, 2), "color_data": visuals, "text": f"{visuals['desc']} ({visuals['color']})", "history_summary": " ".join(history_log[-5:])}

# =================================================================
# 4. GROQ ENGINE
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
            if response.status_code != 200:
                log(f"   ⚠️ Groq API Error: {response.status_code} - {response.text}")
                return None
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            log(f"   ⚠️ Groq Connection Failed: {e}")
            return None

call_gemini = call_groq 

# =================================================================
# 5. DATA FETCHING & ORACLE
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

async def fetch_player_form_quantum(browser: Browser, player_last_name: str) -> Dict[str, Any]:
    try:
        res = supabase.table("market_odds")\
            .select("player1_name, player2_name, odds1, odds2, actual_winner_name, score, created_at")\
            .or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order("created_at", desc=True).limit(8).execute()
        matches = res.data
        if not matches: return {"text": "No Data", "score": 6.5, "history_summary": ""}
        form_data = QuantumFormEngine.calculate_player_form(matches[:5], player_last_name) 
        return form_data
    except Exception as e:
        log(f"   ⚠️ Form Calc Error {player_last_name}: {e}")
        return {"text": "Calc Error", "score": 6.5, "history_summary": ""}

def get_style_matchup_stats_py(supabase_client: Client, player_name: str, opponent_style_raw: str) -> Optional[Dict]:
    if not player_name or not opponent_style_raw: return None
    target_style = opponent_style_raw.split(',')[0].split('(')[0].strip()
    if not target_style or target_style == 'Unknown': return None
    try:
        res = supabase_client.table('market_odds').select('player1_name, player2_name, actual_winner_name')\
            .or_(f"player1_name.ilike.%{player_name}%,player2_name.ilike.%{player_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order('created_at', desc=True).limit(80).execute()
        matches = res.data
        if not matches or len(matches) < 3: return None
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
    except Exception as e: return None

async def get_advanced_load_analysis(supabase_client: Client, player_name: str) -> str:
    try:
        res = supabase_client.table('market_odds').select('created_at, score, actual_winner_name')\
            .or_(f"player1_name.ilike.%{player_name}%,player2_name.ilike.%{player_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order('created_at', desc=True).limit(5).execute()
        recent_matches = res.data
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

        matches_in_week = 0
        sets_in_week = 0
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
    except Exception: return "Unknown"

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
                log(f"   ✅ {tour} Elo geladen: {len(ELO_CACHE[tour])}")
        except: pass
        finally: await page.close()

async def get_db_data():
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
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
# 6. MATH CORE (PURE VALUE ENGINE - NO STAKES)
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
    if edge_percent >= 15.0: label = "🔥 HIGH VALUE" 
    elif edge_percent >= 8.0: label = "✨ GOOD VALUE" 
    elif edge_percent >= 2.0: label = "📈 THIN VALUE" 
    else: label = "👀 WATCH"

    return {
        "type": label, 
        "edge_percent": edge_percent, 
        "is_value": True
    }

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
# 7. PIPELINE UTILS
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
                    return t['surface'], t['bsi_rating'], f"United Cup ({arena_target})"
        return "Hard Court Outdoor", 8.3, "United Cup (Sydney Default)"

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
    if best_match and best_score >= 20: return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', '')
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour, p1_country, p2_country)
    ai_loc = ensure_dict(ai_loc)
    if ai_loc and ai_loc.get('city'):
        surf = ai_loc.get('surface_guessed', 'Hard Court Outdoor')
        bsi = ai_loc.get('bsi_estimate', 6.5)
        note = ai_loc.get('note', 'AI Guess')
        return surf, bsi, note
    return 'Hard Court Outdoor', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1_data, form2_data):
    f1_txt = f"{form1_data['text']} (Rating: {form1_data['score']})"
    f2_txt = f"{form2_data['text']} (Rating: {form2_data['score']})"
    fatigueA = await get_advanced_load_analysis(supabase, p1['last_name'])
    fatigueB = await get_advanced_load_analysis(supabase, p2['last_name'])
    styleA_vs_B = get_style_matchup_stats_py(supabase, p1['last_name'], p2.get('play_style', ''))
    styleB_vs_A = get_style_matchup_stats_py(supabase, p2['last_name'], p1.get('play_style', ''))
    
    prompt = f"""
    Role: Elite Tennis Analyst.
    MATCHUP: {p1['last_name']} vs {p2['last_name']}
    CONTEXT: {surface} (BSI: {bsi}/10) | {notes}
    
    PLAYER A: {p1['last_name']}
    - Style: {p1.get('play_style', 'Unknown')}
    - FORM (Vegas): {f1_txt}
    - BIO-LOAD: {fatigueA}
    - Matchup History: {styleA_vs_B['verdict'] if styleA_vs_B else "No data"}
    
    PLAYER B: {p2['last_name']}
    - Style: {p2.get('play_style', 'Unknown')}
    - FORM (Vegas): {f2_txt}
    - BIO-LOAD: {fatigueB}
    - Matchup History: {styleB_vs_A['verdict'] if styleB_vs_A else "No data"}

    RULES:
    1. FATIGUE: Reduce win% if "Heavy Legs"/"Critical Fatigue" (esp. vs Grinders or high BSI).
    2. STYLE: Server vs Returner: Adv. Returner on Slow/Clay; Adv. Server on Fast/Grass.
    3. SURFACE: High BSI favors Aggressors; Low BSI favors Defenders.

    OUTPUT JSON ONLY:
    {{ 
        "p1_tactical_score": [0-10], 
        "p2_tactical_score": [0-10], 
        "p1_form_score": {form1_data['score']}, 
        "p2_form_score": {form2_data['score']}, 
        "ai_text": "Insight (fatigue/style/surface, max 25 words).", 
        "p1_win_sentiment": [0.0-1.0] 
    }}
    """
    res = await call_groq(prompt)
    data = ensure_dict(safe_get_ai_data(res))
    data['p1_form_score'] = form1_data['score']
    data['p2_form_score'] = form2_data['score']
    return data

def safe_get_ai_data(res_text: Optional[str]) -> Dict[str, Any]:
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': '', 'p1_win_sentiment': 0.5}
    if not res_text: return default
    try:
        cleaned = res_text.replace("json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return ensure_dict(data)
    except: return default

async def scrape_tennis_odds_for_date(browser: Browser, target_date):
    page = await browser.new_page()
    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}&t={int(time.time())}"
        log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        return await page.content()
    except: return None
    finally: await page.close()

def parse_matches_locally_v5(html, p_names): 
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        pending_p1_raw = None; pending_p1_href = None; pending_time = "00:00"
        i = 0
        while i < len(rows):
            row = rows[i]
            
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                pending_p1_raw = None
                i += 1; continue
            
            cols = row.find_all('td')
            if len(cols) < 2: i += 1; continue
            
            first_cell = row.find('td', class_='first')
            if first_cell and ('time' in first_cell.get('class', []) or 't-name' in first_cell.get('class', [])):
                tm = re.search(r'(\d{1,2}:\d{2})', first_cell.get_text(strip=True))
                if tm: pending_time = tm.group(1).zfill(5)
            
            p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
            if not p_cell: i += 1; continue
            p_raw = clean_player_name(p_cell.get_text(strip=True))
            p_href = p_cell.find('a')['href']
            
            raw_odds = []
            for c in row.find_all('td', class_=re.compile(r'course')):
                try:
                    val = float(c.get_text(strip=True))
                    if 1.01 <= val <= 100.0: raw_odds.append(val)
                except: pass

            if pending_p1_raw:
                p2_raw = p_raw; p2_href = p_href
                if '/' in pending_p1_raw or '/' in p2_raw: 
                    pending_p1_raw = None; i += 1; continue
                
                prev_row = rows[i-1]
                prev_odds = []
                for c in prev_row.find_all('td', class_=re.compile(r'course')):
                    try:
                        val = float(c.get_text(strip=True))
                        if 1.01 <= val <= 100.0: prev_odds.append(val)
                    except: pass
                
                all_odds = prev_odds + raw_odds
                if len(all_odds) >= 2:
                    final_o1 = all_odds[0]; final_o2 = all_odds[1]
                    winner_found = None
                    final_score = ""
                    
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

                    s1, scores1 = extract_row_data(prev_row)
                    s2, scores2 = extract_row_data(row)
                    
                    if s1 != -1 and s2 != -1:
                        if s1 > s2: winner_found = pending_p1_raw
                        elif s2 > s1: winner_found = p2_raw
                        
                        score_parts = []
                        min_len = min(len(scores1), len(scores2))
                        for k in range(min_len):
                            score_parts.append(f"{scores1[k]}-{scores2[k]}")
                        
                        if score_parts:
                            final_score = " ".join(score_parts)

                    found.append({
                        "p1_raw": pending_p1_raw, "p2_raw": p2_raw, 
                        "tour": clean_tournament_name(current_tour), 
                        "time": pending_time, "odds1": final_o1, "odds2": final_o2,
                        "p1_href": pending_p1_href, "p2_href": p2_href,
                        "actual_winner": winner_found,
                        "score": final_score
                    })
                pending_p1_raw = None
            else:
                if first_cell and first_cell.get('rowspan') == '2': pending_p1_raw = p_raw; pending_p1_href = p_href
                else: pending_p1_raw = p_raw; pending_p1_href = p_href
            i += 1
    return found

async def update_past_results(browser: Browser):
    log("🏆 The Auditor: Checking Real-Time Results & Scores (V72.0)...")
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
                                safe_to_check = [x for x in safe_to_check if x['id'] != pm['id']]
                                break
        except: pass
        finally: await page.close()

# =================================================================
# 9. QUANTUM GAMES SIMULATOR (CALIBRATED V83.1)
# =================================================================
class QuantumGamesSimulator:
    """
    SOTA Monte Carlo Engine for Tennis Totals.
    """
    
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

# =================================================================
# MAIN PIPELINE (THE FIX)
# =================================================================

async def run_pipeline():
    log(f"🚀 Neural Scout V85.0 DIRECT LINKING (GROQ) Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            # Init Caches
            await update_past_results(browser)
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: return
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            
            # Helper for strict matching
            def find_match_in_db(p1_name, p2_name):
                # Versuche exakten Match
                res = supabase.table("market_odds").select("*")\
                    .or_(f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})")\
                    .is_("actual_winner_name", "null")\
                    .order("created_at", desc=True).limit(1).execute()
                if res.data: return res.data[0]
                return None

            for day_offset in range(-1, 11): 
                target_date = datetime.now() + timedelta(days=day_offset)
                METADATA_CACHE.update(await scrape_oracle_metadata(browser, target_date))
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue
                
                # Lokales Parsing der Seite
                matches = parse_matches_locally_v5(html, [p['last_name'] for p in players])
                log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    try:
                        await asyncio.sleep(0.1) # Fast mode
                        
                        # 1. Identifikation
                        p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                        p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                        
                        if not p1_obj or not p2_obj: continue
                        
                        n1 = p1_obj['last_name']; n2 = p2_obj['last_name']
                        if n1 == n2: continue

                        # 2. Market Integrity Check (Gatekeeper)
                        if not validate_market_integrity(m['odds1'], m['odds2']):
                            continue 

                        # 3. Finde das Match in der DB (oder erstelle es)
                        existing_match = find_match_in_db(n1, n2)
                        
                        # Anti-Spike Check (nur wenn Match existiert)
                        if existing_match:
                            prev_o1 = to_float(existing_match.get('odds1'), 0)
                            prev_o2 = to_float(existing_match.get('odds2'), 0)
                            if is_suspicious_movement(prev_o1, m['odds1'], prev_o2, m['odds2']):
                                log(f"   ⚠️ SPIKE BLOCKED: {n1} vs {n2}")
                                continue

                        db_match_id = None
                        is_signal_locked = False
                        
                        # --- HAUPT LOGIK ---
                        if existing_match:
                            db_match_id = existing_match['id']
                            
                            # Check Lock Status
                            if has_active_signal(existing_match.get('ai_analysis_text', '')):
                                is_signal_locked = True

                            # Update Odds in Market Table
                            update_payload = {"odds1": m['odds1'], "odds2": m['odds2']}
                            
                            # Fix Opening Odds if missing/bad
                            op1 = to_float(existing_match.get('opening_odds1'), 0)
                            if not is_valid_opening_odd(op1, 0):
                                update_payload["opening_odds1"] = m['odds1']
                                update_payload["opening_odds2"] = m['odds2']
                            
                            supabase.table("market_odds").update(update_payload).eq("id", db_match_id).execute()
                            
                            # Update Winner if Live Result detected
                            if m.get('actual_winner') and not existing_match.get('actual_winner_name'):
                                 supabase.table("market_odds").update({
                                     "actual_winner_name": m['actual_winner'],
                                     "score": m.get('score', '')
                                 }).eq("id", db_match_id).execute()
                                 continue # Stop here if match finished

                        else:
                            # CREATE NEW MATCH
                            log(f"   🧠 Fresh Analysis & Simulation: {n1} vs {n2}")
                            f1_data = await fetch_player_form_quantum(browser, n1)
                            f2_data = await fetch_player_form_quantum(browser, n2)
                            
                            c1 = p1_obj.get('country', 'Unknown'); c2 = p2_obj.get('country', 'Unknown')
                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2, c1, c2, match_date=target_date)
                            s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p2_obj['id']), {})
                            style_stats_p1 = get_style_matchup_stats_py(supabase, n1, p2_obj.get('play_style', ''))
                            style_stats_p2 = get_style_matchup_stats_py(supabase, n2, p1_obj.get('play_style', ''))
                            surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
                            surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)
                            
                            elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
                            e1 = ELO_CACHE.get("ATP", {}).get(n1.lower(), {}).get(elo_key, 1500)
                            e2 = ELO_CACHE.get("ATP", {}).get(n2.lower(), {}).get(elo_key, 1500)
                            
                            sim_result = QuantumGamesSimulator.run_simulation(s1, s2, bsi, surf)
                            ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, e1, e2, f1_data, f2_data)
                            prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai, m['odds1'], m['odds2'], surf_rate1, surf_rate2, bool(r1.get('strengths')), style_stats_p1, style_stats_p2)
                            
                            fair1 = round(1/prob, 2) if prob > 0.01 else 99
                            fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                            val_p1 = calculate_value_metrics(1/fair1, m['odds1'])
                            val_p2 = calculate_value_metrics(1/fair2, m['odds2'])
                            
                            value_tag = ""
                            is_value_active = False
                            value_pick_player = None
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
                            
                            new_data = {
                                "player1_name": n1, "player2_name": n2, "tournament": m['tour'],
                                "odds1": m['odds1'], "odds2": m['odds2'],
                                "opening_odds1": m['odds1'], "opening_odds2": m['odds2'],
                                "ai_fair_odds1": fair1, "ai_fair_odds2": fair2,
                                "ai_analysis_text": ai_text_final,
                                "games_prediction": sim_result,
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                            }
                            res_insert = supabase.table("market_odds").insert(new_data).execute()
                            if res_insert.data: db_match_id = res_insert.data[0]['id']
                            log(f"💾 Created: {n1} vs {n2}")

                        # --- HISTORY LOGGING (DER FIX) ---
                        # Wir nutzen EXAKT die db_match_id von oben. Kein erneutes Suchen.
                        # Wir schreiben nur, wenn sich was geändert hat oder es neu ist.
                        
                        if db_match_id:
                            should_log = False
                            if not existing_match: should_log = True # New match
                            elif is_signal_locked: should_log = True # Always track locked matches
                            else:
                                # Check Delta
                                old_o1 = to_float(existing_match.get('odds1'), 0)
                                if abs(old_o1 - m['odds1']) > 0.001: should_log = True
                            
                            if should_log:
                                # Hole Fair Odds aus dem gerade geupdateten/existierenden Match Object
                                # (Oder aus den lokalen Variablen falls neu erstellt)
                                f1 = existing_match.get('ai_fair_odds1') if existing_match else fair1
                                f2 = existing_match.get('ai_fair_odds2') if existing_match else fair2
                                
                                # Pick Name logic
                                pick_name = None
                                if is_signal_locked: pick_name = "LOCKED" 
                                elif 'value_pick_player' in locals() and value_pick_player: pick_name = value_pick_player

                                h_data = {
                                    "match_id": db_match_id, # DIRECT LINK!
                                    "odds1": m['odds1'],
                                    "odds2": m['odds2'],
                                    "fair_odds1": f1,
                                    "fair_odds2": f2,
                                    "is_hunter_pick": (is_signal_locked or (is_value_active if 'is_value_active' in locals() else False)),
                                    "pick_player_name": pick_name,
                                    "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                }
                                
                                try:
                                    supabase.table("odds_history").insert(h_data).execute()
                                except Exception as db_err:
                                    log(f"⚠️ History Insert Error: {db_err}")

                    except Exception as e: log(f"⚠️ Loop Error: {e}")

        finally: await browser.close()
    
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())