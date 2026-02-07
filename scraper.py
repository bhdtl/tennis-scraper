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

log("🔌 Initialisiere Neural Scout (V90.0 - API-INTEGRATION [THE ODDS API])...")

# Secrets Load
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
THE_ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY") # NEU: Hier kommt dein Key rein

if not all([GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY, THE_ODDS_API_KEY]):
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub/Environment Secrets.")
    log("Benötigt: GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY, THE_ODDS_API_KEY")
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
# 2. HELPER FUNCTIONS (ERHALTEN)
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
# 3. QUANTUM FORM ENGINE (ERHALTEN)
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
# 4. GROQ ENGINE (ERHALTEN)
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

# =================================================================
# 5. NEW API-DATA FETCHING (ERSETZT SCRAPING)
# =================================================================

async def fetch_odds_via_the_odds_api(sport_key: str = "tennis_atp") -> List[Dict]:
    """
    Holt Pre-Match Odds von The Odds API.
    Wir nehmen Pinnacle als Gold-Standard für Pre-Match Movement.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    async with httpx.AsyncClient() as client:
        try:
            log(f"📡 API-Request für {sport_key}...")
            res = await client.get(url, params=params, timeout=20.0)
            if res.status_code != 200:
                log(f"   ⚠️ API Error {res.status_code}: {res.text}")
                return []
            
            api_data = res.json()
            found_matches = []
            
            for event in api_data:
                # Wir suchen nach Pinnacle (bestes Movement) oder Bet365
                bookmaker = next((b for b in event['bookmakers'] if b['key'] == 'pinnacle'), None)
                if not bookmaker:
                    bookmaker = next((b for b in event['bookmakers'] if b['key'] == 'bet365'), None)
                if not bookmaker:
                    bookmaker = event['bookmakers'][0] if event['bookmakers'] else None
                
                if bookmaker:
                    market = bookmaker['markets'][0]
                    found_matches.append({
                        "id": event['id'],
                        "p1_raw": event['home_team'],
                        "p2_raw": event['away_team'],
                        "tour": event['sport_title'],
                        "time": event['commence_time'],
                        "odds1": market['outcomes'][0]['price'],
                        "odds2": market['outcomes'][1]['price'],
                        "bookie": bookmaker['title']
                    })
            return found_matches
        except Exception as e:
            log(f"   ⚠️ API Connection Error: {e}")
            return []

async def fetch_scores_via_the_odds_api(sport_key: str = "tennis_atp") -> List[Dict]:
    """
    Nutzt den Scores Endpoint zum automatischen Abrechnen.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    params = {"apiKey": THE_ODDS_API_KEY, "daysFrom": 3}
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(url, params=params)
            return res.json() if res.status_code == 200 else []
        except: return []

# =================================================================
# 6. ORACLE & ELO (WEITERHIN PLAYWRIGHT FÜR DATA ENRICHMENT)
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

async def fetch_tennisexplorer_stats(browser: Browser, player_name: str, surface: str) -> float:
    # Da wir nun die API nutzen, simulieren wir den Stat-Check über eine Suche oder lassen ihn auf 0.5 Fallback
    # In V91 könnten wir hier die TennisExplorer URL über die API ID mappen
    return 0.5

async def fetch_elo_ratings(browser: Browser):
    log("📊 Lade Elo Ratings via Browser (No API available)...")
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
# 7. MATH & SIMULATION CORE (ERHALTEN)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def normal_cdf_prob(elo_diff: float, sigma: float = 280.0) -> float:
    z = elo_diff / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def calculate_value_metrics(fair_prob: float, market_odds: float) -> Dict[str, Any]:
    if market_odds <= 1.01 or fair_prob <= 0: return {"type": "NONE", "edge_percent": 0.0, "is_value": False}
    edge = (fair_prob * market_odds) - 1
    edge_percent = round(edge * 100, 1)
    if edge_percent <= 0.5: return {"type": "NONE", "edge_percent": edge_percent, "is_value": False}
    label = "VALUE"; 
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

def recalculate_fair_odds_with_new_market(old_fair_odds1, old_market_odds1, old_market_odds2, new_market_odds1, new_market_odds2):
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
# 8. SIMULATOR & TOOLS (ERHALTEN)
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
            data = ensure_dict(data); country = data.get("country", "Unknown")
        except: country = "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
    if country in COUNTRY_TO_CITY_MAP: return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
    return None

async def resolve_ambiguous_tournament(p1, p2, tour_name, p1_country, p2_country):
    prompt = f"TASK: Identify tournament location. MATCH: {p1} ({p1_country}) vs {p2} ({p2_country}). SOURCE: '{tour_name}'. OUTPUT JSON: {{ 'city': 'Name', 'surface': 'Hard/Clay/Grass', 'indoor': true/false }}"
    res = await call_groq(prompt)
    if res:
        try: 
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data); surface_type = data.get('surface', 'Hard')
            if data.get('indoor'): surface_type += " Indoor"
            else: surface_type += " Outdoor"
            return {"city": data.get('city', 'Unknown'), "surface_guessed": surface_type, "bsi_estimate": 6.5, "note": f"AI: {data.get('city')}"}
        except: pass
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2, p1_country="Unknown", p2_country="Unknown", match_date: datetime = None): 
    s_low = clean_tournament_name(tour).lower().strip()
    best_match = None; best_score = 0
    for t in db_tours:
        score = calculate_fuzzy_score(s_low, t['name'])
        if score > best_score: best_score = score; best_match = t
    if best_match and best_score >= 20: return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', '')
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour, p1_country, p2_country)
    if ai_loc: return ai_loc.get('surface_guessed'), 6.5, ai_loc.get('note')
    return 'Hard Court Outdoor', 6.5, 'Fallback'

class QuantumGamesSimulator:
    @staticmethod
    def derive_hold_probability(server_skills, returner_skills, bsi, surface):
        p_hold = 67.0 
        srv = server_skills.get('serve', 50); pwr = server_skills.get('power', 50)
        p_hold += (srv - 50) * 0.35 + (pwr - 50) * 0.10
        ret_speed = returner_skills.get('speed', 50); ret_mental = returner_skills.get('mental', 50)
        p_hold -= (ret_speed - 50) * 0.15 + (ret_mental - 50) * 0.08
        p_hold += (bsi - 6.0) * 1.4
        return max(52.0, min(94.0, p_hold)) / 100.0

    @staticmethod
    def simulate_set(p1_prob, p2_prob):
        g1, g2 = 0, 0
        while True:
            if random.random() < p1_prob: g1 += 1
            else: g2 += 1 
            if g1 >= 6 and g1 - g2 >= 2: return (1, g1 + g2)
            if g2 >= 6 and g2 - g1 >= 2: return (2, g1 + g2)
            if g1 == 6 and g2 == 6:
                tb = 0.5 + (p1_prob - p2_prob)
                return (1, 13) if random.random() < tb else (2, 13)

    @staticmethod
    def run_simulation(p1_skills, p2_skills, bsi, surface, iterations = 1000):
        p1_hold = QuantumGamesSimulator.derive_hold_probability(p1_skills, p2_skills, bsi, surface)
        p2_hold = QuantumGamesSimulator.derive_hold_probability(p2_skills, p1_skills, bsi, surface)
        total_games = []
        for _ in range(iterations):
            w1, g1 = QuantumGamesSimulator.simulate_set(p1_hold, p2_hold)
            w2, g2 = QuantumGamesSimulator.simulate_set(p1_hold + 0.02 if w1==1 else -0.01, p2_hold + 0.02 if w1==2 else -0.01)
            total = g1 + g2
            if w1 != w2:
                w3, g3 = QuantumGamesSimulator.simulate_set(p1_hold, p2_hold)
                total += g3
            total_games.append(total)
        return {"predicted_line": round(sum(total_games)/iterations, 1), "median_games": sorted(total_games)[iterations//2]}

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1_data, form2_data):
    fatigueA = await get_advanced_load_analysis(supabase, p1['last_name'])
    fatigueB = await get_advanced_load_analysis(supabase, p2['last_name'])
    prompt = f"Role: Elite Tennis Analyst. MATCHUP: {p1['last_name']} vs {p2['last_name']}. CONTEXT: {surface} (BSI: {bsi}). BIO-LOAD A: {fatigueA}. BIO-LOAD B: {fatigueB}. RULES: Focus on fatigue/style. OUTPUT JSON ONLY."
    res = await call_groq(prompt)
    data = ensure_dict(json.loads(res.replace("json", "").replace("```", "").strip()) if res else {})
    data['p1_form_score'] = form1_data['score']; data['p2_form_score'] = form2_data['score']
    return data

# =================================================================
# 9. MAIN PIPELINE (V90 - API INTEGRATED)
# =================================================================

async def run_pipeline():
    log(f"🚀 Neural Scout V90.0 (API-FIRST) Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: return
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            
            # API FETCH
            for sport_key in ["tennis_atp", "tennis_wta"]:
                matches = await fetch_odds_via_the_odds_api(sport_key)
                log(f"🔍 API lieferte {len(matches)} Matches für {sport_key}")
                
                # AUTOMATIC SETTLEMENT (Score Update)
                scores_data = await fetch_scores_via_the_odds_api(sport_key)
                for score_entry in scores_data:
                    if score_entry.get('completed'):
                        winner = score_entry['home_team'] if score_entry['scores'][0]['score'] > score_entry['scores'][1]['score'] else score_entry['away_team']
                        supabase.table("market_odds").update({
                            "actual_winner_name": winner,
                            "score": f"{score_entry['scores'][0]['score']}-{score_entry['scores'][1]['score']}"
                        }).eq("external_id", score_entry['id']).execute()

                for m in matches:
                    try:
                        p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                        p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                        
                        if p1_obj and p2_obj:
                            n1 = p1_obj['last_name']; n2 = p2_obj['last_name']
                            if not validate_market_integrity(m['odds1'], m['odds2']): continue

                            # CHECK DB VIA EXTERNAL_ID
                            res_db = supabase.table("market_odds").select("*").eq("external_id", m['id']).execute()
                            existing_match = res_db.data[0] if res_db.data else None
                            
                            db_match_id = existing_match['id'] if existing_match else None
                            
                            # LOGIC FOR MOVEMENT / AI
                            if not existing_match:
                                log(f"🧠 Fresh Analysis & Simulation: {n1} vs {n2}")
                                f1_data = await fetch_player_form_quantum(browser, n1)
                                f2_data = await fetch_player_form_quantum(browser, n2)
                                surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2)
                                sim_result = QuantumGamesSimulator.run_simulation(all_skills.get(p1_obj['id'], {}), all_skills.get(p2_obj['id'], {}), bsi, surf)
                                
                                ai = await analyze_match_with_ai(p1_obj, p2_obj, {}, {}, {}, {}, surf, bsi, notes, 1500, 1500, f1_data, f2_data)
                                prob = calculate_physics_fair_odds(n1, n2, all_skills.get(p1_obj['id'], {}), all_skills.get(p2_obj['id'], {}), bsi, surf, ai, m['odds1'], m['odds2'], 0.5, 0.5, False, None, None)
                                
                                fair1 = round(1/prob, 2); fair2 = round(1/(1-prob), 2)
                                
                                data = {
                                    "external_id": m['id'], "player1_name": n1, "player2_name": n2, "tournament": m['tour'],
                                    "odds1": m['odds1'], "odds2": m['odds2'], "opening_odds1": m['odds1'], "opening_odds2": m['odds2'],
                                    "ai_fair_odds1": fair1, "ai_fair_odds2": fair2, "ai_analysis_text": ai.get('ai_text', ''),
                                    "match_time": m['time'], "created_at": datetime.now(timezone.utc).isoformat()
                                }
                                ins_res = supabase.table("market_odds").insert(data).execute()
                                db_match_id = ins_res.data[0]['id'] if ins_res.data else None
                            else:
                                # Update current odds
                                supabase.table("market_odds").update({"odds1": m['odds1'], "odds2": m['odds2']}).eq("id", db_match_id).execute()

                            # HISTORY LOGGING
                            if db_match_id:
                                supabase.table("odds_history").insert({
                                    "match_id": db_match_id, "odds1": m['odds1'], "odds2": m['odds2'],
                                    "recorded_at": datetime.now(timezone.utc).isoformat()
                                }).execute()

                    except Exception as e: log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
