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

log("🔌 Initialisiere Neural Scout (V59.9 - THE INTEGRATED WINNER)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.0-flash'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
METADATA_CACHE: Dict[str, Any] = {} # ORACLE CACHE

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena",
    "Brisbane": "Pat Rafter Arena",
    "Adelaide": "Memorial Drive Tennis Centre",
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

# --- V58.3 UPGRADE: ELITE NAME MATCHING ---
def normalize_db_name(name: str) -> str:
    """Normalizes DB names for robust matching (Mc, De, hyphens)."""
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
                if db_first.startswith(scrape_initial):
                    match_score += 20 
                else:
                    match_score -= 50 
            
            if match_score > 50:
                candidates.append((p, match_score))

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

# =================================================================
# 3. GEMINI ENGINE
# =================================================================
async def call_gemini(prompt: str, model: str = MODEL_NAME) -> Optional[str]:
    await asyncio.sleep(0.5) 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.4}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                log(f"   ⚠️ Gemini API Error: {response.status_code}")
                return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return None

# =================================================================
# 4. DATA FETCHING & ORACLE (TENNISTEMPLE)
# =================================================================

# --- V59.2: THE ORACLE SCRAPER (TennisTemple) ---
async def scrape_oracle_metadata(browser: Browser, target_date: datetime):
    """Fetches real tournament names from TennisTemple to fix 'Futures' ambiguity."""
    date_str = target_date.strftime('%Y-%m-%d')
    url = f"https://de.tennistemple.com/matches/{date_str}"
    
    page = await browser.new_page()
    metadata = {}
    
    try:
        # log(f"   🔮 Consult Oracle (TennisTemple): {date_str}")
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        links = soup.find_all('a', href=True)
        current_tournament = "Unknown"
        
        for element in soup.find_all(['h2', 'a']):
            text = element.get_text(strip=True)
            href = element.get('href', '')
            
            # Detect Tournament Header
            if ('/turnier/' in href or '/tournament/' in href):
                current_tournament = text
                continue
                
            # Detect Player
            if ('/spieler/' in href or '/player/' in href):
                norm_name = normalize_db_name(text)
                if norm_name and current_tournament != "Unknown":
                    metadata[norm_name] = current_tournament
                    
    except Exception as e:
        pass
    finally:
        await page.close()
        
    return metadata

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
        if not matches or len(matches) < 5: return None
        opponents_map = {} 
        opponent_names_to_fetch = []
        for m in matches:
            if player_name.lower() in m['player1_name'].lower():
                opp = get_last_name(m['player2_name']).lower()
            else:
                opp = get_last_name(m['player1_name']).lower()
            if opp: opponent_names_to_fetch.append(opp)
        if not opponent_names_to_fetch: return None
        unique_opps = list(set(opponent_names_to_fetch))
        for i in range(0, len(unique_opps), 20):
            chunk = unique_opps[i:i+20]
            p_res = supabase_client.table('players').select('last_name, play_style').in_('last_name', chunk).execute()
            if p_res.data:
                for p in p_res.data:
                    if p.get('play_style'):
                        s = [x.split('(')[0].strip() for x in p['play_style'].split(',')]
                        opponents_map[p['last_name'].lower()] = s
        relevant_matches = 0
        wins = 0
        for m in matches:
            if player_name.lower() in m['player1_name'].lower():
                opp_name = get_last_name(m['player2_name']).lower()
            else:
                opp_name = get_last_name(m['player1_name']).lower()
            opp_styles = opponents_map.get(opp_name, [])
            if target_style in opp_styles:
                relevant_matches += 1
                winner = m.get('actual_winner_name', '').lower()
                if player_name.lower() in winner:
                    wins += 1
        if relevant_matches < 3: return None
        win_rate = (wins / relevant_matches) * 100
        verdict = "Neutral"
        if win_rate > 65: verdict = "DOMINANT"
        elif win_rate < 40: verdict = "STRUGGLES"
        return {"win_rate": win_rate, "matches": relevant_matches, "verdict": verdict, "style": target_style}
    except Exception as e:
        return None

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

async def fetch_player_form_hybrid(browser: Browser, player_last_name: str) -> Dict[str, Any]:
    try:
        res = supabase.table("market_odds").select("actual_winner_name, match_time").or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%").not_.is_("actual_winner_name", "null").order("match_time", desc=True).limit(5).execute()
        matches = res.data
        if matches and isinstance(matches, list) and len(matches) >= 3: 
            wins = 0
            for m in matches:
                if isinstance(m, dict) and player_last_name.lower() in m.get('actual_winner_name', '').lower(): wins += 1
            trend = "Neutral"
            if wins >= 4: trend = "🔥 ON FIRE"
            elif wins >= 3: trend = "Good"
            elif len(matches) - wins >= 4: trend = "❄️ ICE COLD"
            return {"text": f"{trend} (DB: {wins}/{len(matches)} wins)"}
    except: pass
    return {"text": "No recent DB data."}

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
# 5. MATH CORE & SOTA V57 QUANT ENGINE (Z-SCORE + GRAVITY)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def normal_cdf_prob(elo_diff: float, sigma: float = 280.0) -> float:
    z = elo_diff / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def calculate_dynamic_stake(fair_prob: float, market_odds: float, ai_sentiment_score: float = 0.5) -> Dict[str, Any]:
    if market_odds <= 1.01 or fair_prob <= 0: 
        return {"stake_str": "0u", "type": "NONE", "is_bet": False}

    b = market_odds - 1
    q = 1 - fair_prob
    if b == 0: return {"stake_str": "0u", "type": "NONE", "is_bet": False}
    full_kelly = (b * fair_prob - q) / b

    if full_kelly <= 0: 
        return {"stake_str": "0u", "type": "NONE", "is_bet": False}

    # --- TIERED STRATEGY (V57 Refined) ---
    if 1.30 <= market_odds < 1.70:
        required_edge = 0.06 
        kelly_fraction = 0.25 
        max_stake = 3.0       
        label = "🛡️ BANKER"
    elif 1.70 <= market_odds < 2.40:
        required_edge = 0.125
        kelly_fraction = 0.20 
        max_stake = 2.0
        label = "⚖️ VALUE"
    elif 2.40 <= market_odds <= 6.00:
        required_edge = 0.20 
        if ai_sentiment_score < 0.45:
             return {"stake_str": "0u", "type": "AI_BLOCK", "is_bet": False}
        kelly_fraction = 0.125 
        max_stake = 1.0       
        label = "💎 HUNTER"
    else:
        return {"stake_str": "0u", "type": "SKIP", "is_bet": False}

    edge = (fair_prob * market_odds) - 1
    if edge < required_edge:
         return {"stake_str": "0u", "type": "LOW_EDGE", "is_bet": False}

    safe_stake = full_kelly * kelly_fraction
    raw_units = safe_stake * 100 * 0.5 
    
    units = round(raw_units * 2) / 2
    if units < 0.5: return {"stake_str": "0u", "type": "TOO_SMALL", "is_bet": False}
    if units > max_stake: units = max_stake
    
    return {
        "stake_str": f"{units}u",
        "type": label,
        "is_bet": True,
        "edge_percent": round(edge * 100, 1),
        "units": units
    }

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2, has_scouting_reports: bool, style_stats_p1: Optional[Dict], style_stats_p2: Optional[Dict]):
    ai_meta = ensure_dict(ai_meta)
    n1 = get_last_name(p1_name); n2 = get_last_name(p2_name)
    tour = "ATP"; bsi_val = to_float(bsi, 6.0)
    
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo1 = p1_stats.get(elo_surf, 1500)
    elo2 = p2_stats.get(elo_surf, 1500)
    
    # 1. MARKET GRAVITY (Quant Fix)
    elo_diff_model = elo1 - elo2
    
    if market_odds1 > 0 and market_odds2 > 0:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
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

    # --- PROBABILITY CALCULATION (V57) ---
    prob_elo = normal_cdf_prob(elo_diff_final, sigma=280.0)
    
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8)
    
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    c1_score = get_offense(s1); c2_score = get_offense(s2)
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)
    
    f1 = to_float(ai_meta.get('p1_form_score', 5)); f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    style_boost = 0
    if style_stats_p1 and style_stats_p1['verdict'] == "DOMINANT": 
        style_boost += 0.08 
    if style_stats_p1 and style_stats_p1['verdict'] == "STRUGGLES": style_boost -= 0.06
    if style_stats_p2 and style_stats_p2['verdict'] == "DOMINANT": 
        style_boost -= 0.08 
    if style_stats_p2 and style_stats_p2['verdict'] == "STRUGGLES": style_boost += 0.06
    
    # [V57 WEIGHTING]
    weights = [0.20, 0.15, 0.05, 0.50, 0.10] 
    model_trust_factor = 0.45 
        
    total_w = sum(weights)
    weights = [w/total_w for w in weights]
    
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
        
        # Reverse V57 Ratio (60/40)
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
    except:
        return 0.5

# =================================================================
# 6. PIPELINE UTILS
# =================================================================
async def build_country_city_map(browser: Browser):
    if COUNTRY_TO_CITY_MAP: return
    url = "https://www.unitedcup.com/en/scores/group-standings"
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=20000, wait_until="networkidle")
        text_content = await page.inner_text("body")
        prompt = f"TASK: Map Country to City (United Cup). Text: {text_content[:20000]}. JSON ONLY."
        res = await call_gemini(prompt)
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
        res = await call_gemini(f"Country of player {p1}? JSON: {{'country': 'Name'}}")
        try:
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            country = data.get("country", "Unknown")
        except: country = "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
            
    if country in COUNTRY_TO_CITY_MAP: return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
    return None

# --- V59.3: CONTEXT-AWARE RESOLVER WITH DB ALIGNMENT ---
async def resolve_ambiguous_tournament(p1, p2, scraped_name, p1_country, p2_country):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    
    # 1. ORACLE CHECK (TennisTemple Metadata)
    # Check if we have specific tournament data for these players
    p1_meta = METADATA_CACHE.get(normalize_db_name(p1))
    p2_meta = METADATA_CACHE.get(normalize_db_name(p2))
    
    oracle_data = p1_meta or p2_meta
    
    if oracle_data:
         # Found real data!
         real_name = oracle_data.get('tournament', scraped_name)
         log(f"   🔮 Oracle Hit: {p1} -> {real_name}")
         
         # Now use this REAL name to get details from Gemini
         prompt = f"""
         TASK: Details for tennis tournament '{real_name}'.
         CONTEXT: {p1} vs {p2}. Date: {datetime.now().strftime('%B %Y')}.
         OUTPUT JSON: {{ "city": "Name", "surface": "Hard/Clay/Grass", "indoor": true/false }}
         """
         # Proceed to call Gemini with the BETTER name...
         scraped_name = real_name # Update name for cache key
    else:
         # No Oracle data, use standard fallback prompt
         prompt = f"""
         TASK: Identify tournament location.
         MATCH: {p1} ({p1_country}) vs {p2} ({p2_country}).
         SOURCE: '{scraped_name}'.
         OUTPUT JSON: {{ "city": "Name", "surface": "Hard/Clay/Grass", "indoor": true/false }}
         """

    res = await call_gemini(prompt)
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
            
            # Corrections
            city = data.get('city', 'Unknown')
            if "plantation" in city.lower() and p1_country == "USA":
                 city = "Winston-Salem" # Fix hallucination based on user feedback
                 surface_type = "Hard Indoor"
            
            simulated_db_entry = {
                "city": city,
                "surface_guessed": surface_type,
                "bsi_estimate": est_bsi,
                "note": f"AI/Oracle: {city}"
            }
            TOURNAMENT_LOC_CACHE[scraped_name] = simulated_db_entry
            return simulated_db_entry
        except: pass
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2, p1_country="Unknown", p2_country="Unknown"):
    s_low = clean_tournament_name(tour).lower().strip()
    if "united cup" in s_low:
        arena_target = await resolve_united_cup_via_country(p1)
        if arena_target:
            for t in db_tours:
                if "united cup" in t['name'].lower() and arena_target.lower() in t.get('location', '').lower():
                    return t['surface'], t['bsi_rating'], f"United Cup ({arena_target})"
        return "Hard Court Outdoor", 8.3, "United Cup (Sydney Default)"

    best_match = None; best_score = 0
    for t in db_tours:
        score = calculate_fuzzy_score(s_low, t['name'])
        if score > best_score: best_score = score; best_match = t
    
    if best_match and best_score >= 20:
        log(f"   🏟️ DB HIT: '{s_low}' -> '{best_match['name']}' | BSI: {best_match['bsi_rating']} | Court: {best_match.get('notes', 'N/A')[:50]}...")
        return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', '')

    log(f"   ⚠️ Tournament '{s_low}' vague. Consulting Oracle & Gemini...")
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour, p1_country, p2_country)
    ai_loc = ensure_dict(ai_loc)
    
    if ai_loc and ai_loc.get('city'):
        surf = ai_loc.get('surface_guessed', 'Hard Court Outdoor')
        bsi = ai_loc.get('bsi_estimate', 6.5)
        note = ai_loc.get('note', 'AI Guess')
        log(f"   🤖 RESOLVED: '{s_low}' -> {note} | BSI: {bsi}")
        return surf, bsi, note
    
    return 'Hard Court Outdoor', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1, form2):
    log(f"   🤖 Asking AI for analysis on: {p1['last_name']} vs {p2['last_name']}")
    has_reports = r1.get('strengths') and r2.get('strengths')
    
    prompt = f"""
    ROLE: Elite Tennis Analyst.
    TASK: Analyze {p1['last_name']} vs {p2['last_name']} on {surface} (BSI {bsi}).
    DATA: ELO {elo1} vs {elo2}. FORM {form1['text']} vs {form2['text']}.
    SCOUTING P1: {r1.get('strengths', 'N/A')}
    SCOUTING P2: {r2.get('strengths', 'N/A')}
    COURT: {notes}
    OUTPUT JSON ONLY.
    JSON: {{ 
        "p1_tactical_score": [0-10], 
        "p2_tactical_score": [0-10], 
        "p1_form_score": [0-10], 
        "p2_form_score": [0-10], 
        "ai_text": "Analysis string (max 2 sentences).",
        "p1_win_sentiment": [0.0-1.0] 
    }}
    """
    res = await call_gemini(prompt)
    data = ensure_dict(safe_get_ai_data(res))
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

# --- V59.9: INTEGRATED WINNER PARSER (Hybrid Mode) ---
def parse_matches_locally_v5(html, p_names): 
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        
        # Buffer variables to handle split rows
        pending_p1_raw = None
        pending_p1_href = None
        pending_time = "00:00"
        
        i = 0
        while i < len(rows):
            row = rows[i]
            
            # Update Tournament context
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                # Reset pending if tournament changes
                pending_p1_raw = None
                i += 1; continue
            
            # Basic parsing of the row
            cols = row.find_all('td')
            if len(cols) < 2: 
                i += 1; continue
                
            # Check for Time cell
            first_cell = row.find('td', class_='first')
            has_time = False
            if first_cell and ('time' in first_cell.get('class', []) or 't-name' in first_cell.get('class', [])):
                tm = re.search(r'(\d{1,2}:\d{2})', first_cell.get_text(strip=True))
                if tm: 
                    pending_time = tm.group(1).zfill(5)
                    has_time = True
            
            # Try to extract player from this row
            p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
            
            # If no player link found, skip
            if not p_cell: 
                i += 1; continue
                
            p_raw = clean_player_name(p_cell.get_text(strip=True))
            p_href = p_cell.find('a')['href']
            
            # Odds extraction
            raw_odds = []
            for c in row.find_all('td', class_=re.compile(r'course')):
                try:
                    val = float(c.get_text(strip=True))
                    if 1.01 <= val <= 100.0: raw_odds.append(val)
                except: pass

            # --- PAIRING LOGIC & INTEGRATED WINNER DETECTION ---
            if pending_p1_raw:
                # We have P1 waiting, this row must be P2
                p2_raw = p_raw
                p2_href = p_href
                
                # Check validity
                if '/' in pending_p1_raw or '/' in p2_raw: 
                    # Doubles detected, discard
                    pending_p1_raw = None
                    i += 1; continue
                
                prev_row = rows[i-1]
                prev_odds = []
                for c in prev_row.find_all('td', class_=re.compile(r'course')):
                    try:
                        val = float(c.get_text(strip=True))
                        if 1.01 <= val <= 100.0: prev_odds.append(val)
                    except: pass
                
                all_odds = prev_odds + raw_odds
                if len(all_odds) >= 2:
                    final_o1 = all_odds[0]
                    final_o2 = all_odds[1]
                    
                    # --- V59.9 LIVE WINNER CHECK (The Hybrid Trick) ---
                    winner_found = None
                    
                    # Check scores in BOTH rows (Standard TE layout: Score is often in Row 1)
                    score_cell_p1 = prev_row.find('td', class_='result')
                    score_cell_p2 = row.find('td', class_='result')
                    
                    if score_cell_p1 and score_cell_p2:
                        t1 = score_cell_p1.get_text(strip=True)
                        t2 = score_cell_p2.get_text(strip=True)
                        if t1.isdigit() and t2.isdigit():
                            s1 = int(t1); s2 = int(t2)
                            
                            # Valid Completion Check (2 sets min usually)
                            if s1 >= 2 or s2 >= 2:
                                if s1 > s2: winner_found = pending_p1_raw
                                elif s2 > s1: winner_found = p2_raw
                    
                    found.append({
                        "p1_raw": pending_p1_raw, "p2_raw": p2_raw, 
                        "tour": clean_tournament_name(current_tour), 
                        "time": pending_time, "odds1": final_o1, "odds2": final_o2,
                        "p1_href": pending_p1_href, "p2_href": p2_href,
                        "actual_winner": winner_found 
                    })
                
                # Reset pending
                pending_p1_raw = None
                
            else:
                # No pending P1. This row is P1.
                if first_cell and first_cell.get('rowspan') == '2':
                    pending_p1_raw = p_raw
                    pending_p1_href = p_href
                else:
                    # Single row match? (Rare in TE odds view)
                    pending_p1_raw = p_raw
                    pending_p1_href = p_href
            
            i += 1

    return found

async def update_past_results(browser: Browser):
    log("🏆 The Auditor: Checking Real-Time Results (Today + Past)...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending or not isinstance(pending, list): return
    safe_to_check = []
    for pm in pending:
        try: safe_to_check.append(pm)
        except: continue
    if not safe_to_check: return

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
                        # 1. SCORE VALIDATION
                        score_matches = re.findall(r'(\d+)-(\d+)', row_text)
                        
                        p1_sets = 0
                        p2_sets = 0
                        
                        for s in score_matches:
                            try:
                                sl = int(s[0])
                                sr = int(s[1])
                                if sl > sr: p1_sets += 1
                                elif sr > sl: p2_sets += 1
                            except: pass
                            
                        is_ret = "ret." in row_text or "w.o." in row_text
                        
                        # 2. MATCH TYPE DETECTION (Grand Slam Men = Best of 5)
                        is_gs_men = "open" in pm['tournament'].lower() and ("atp" in pm['tournament'].lower() or "men" in pm['tournament'].lower())
                        sets_needed = 3 if is_gs_men else 2
                        
                        # 3. COMPLETION GATE
                        if p1_sets >= sets_needed or p2_sets >= sets_needed or is_ret:
                            winner = None
                            idx_p1 = row_norm.find(p1_norm)
                            idx_p2 = row_norm.find(p2_norm)
                            
                            if idx_p1 < idx_p2: winner = pm['player1_name']
                            else: winner = pm['player2_name']
                            
                            if not is_ret:
                                if p1_sets > p2_sets: winner = pm['player1_name']
                                elif p2_sets > p1_sets: winner = pm['player2_name']
                            
                            if winner:
                                supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", pm['id']).execute()
                                safe_to_check = [x for x in safe_to_check if x['id'] != pm['id']]
                                log(f"      ✅ SETTLED: {winner} won (vs {pm['player2_name'] if winner==pm['player1_name'] else pm['player1_name']})")
                                break

        except: pass
        finally: await page.close()

async def run_pipeline():
    log(f"🚀 Neural Scout V59.9 THE INTEGRATED WINNER Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await update_past_results(browser)
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: return
            
            report_ids = {r['player_id'] for r in all_reports if isinstance(r, dict) and r.get('player_id')}
            player_names = [p['last_name'] for p in players]
            
            for day_offset in range(-1, 11): 
                target_date = datetime.now() + timedelta(days=day_offset)
                
                METADATA_CACHE.update(await scrape_oracle_metadata(browser, target_date))
                
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue
                matches = parse_matches_locally_v5(html, player_names)
                log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    try:
                        p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                        p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                        
                        if p1_obj and p2_obj:
                            n1 = p1_obj['last_name']
                            n2 = p2_obj['last_name']
                            if n1 == n2: continue
                            
                            if p1_obj.get('tour') != p2_obj.get('tour'):
                                if "united cup" not in m['tour'].lower() and "hopman" not in m['tour'].lower():
                                    continue 

                            # --- V59.9: LIVE SETTLEMENT IF DETECTED ---
                            actual_winner_val = m.get('actual_winner') # Prefer live detection
                            if not actual_winner_val and m.get('actual_winner'): 
                                # Fallback to standard check if live detection missed it but it exists? No, m['actual_winner'] comes from parse_matches
                                actual_winner_val = n1 if m['actual_winner'] == m['p1_raw'] else n2

                            if m['odds1'] < 1.01 and m['odds2'] < 1.01 and not actual_winner_val: 
                                continue
                            
                            existing_match = None
                            res1 = supabase.table("market_odds").select("id, actual_winner_name, odds1, odds2, player2_name, ai_analysis_text, ai_fair_odds1, ai_fair_odds2")\
                                .eq("player1_name", n1).eq("player2_name", n2).order("created_at", desc=True).limit(1).execute()
                            
                            if res1.data and len(res1.data) > 0: existing_match = res1.data[0]
                            else:
                                res2 = supabase.table("market_odds").select("id, actual_winner_name, odds1, odds2, player2_name, ai_analysis_text, ai_fair_odds1, ai_fair_odds2")\
                                    .eq("player1_name", n2).eq("player2_name", n1).order("created_at", desc=True).limit(1).execute()
                                if res2.data and len(res2.data) > 0: existing_match = res2.data[0]
                            
                            db_match_id = None
                            cached_ai = {}
                            
                            if existing_match:
                                db_match_id = existing_match['id']
                                # If we found a winner live, update DB immediately
                                if actual_winner_val and not existing_match.get('actual_winner_name'):
                                     supabase.table("market_odds").update({"actual_winner_name": actual_winner_val}).eq("id", db_match_id).execute()
                                     log(f"      🏆 LIVE SETTLEMENT: {actual_winner_val} won (vs {n2 if actual_winner_val==n1 else n1})")
                                     continue # Done, no need to recalc odds for finished match
                                     
                                if existing_match.get('actual_winner_name'): continue 
                                old_o1 = existing_match.get('odds1', 0)
                                if existing_match.get('ai_analysis_text'):
                                    cached_ai = {
                                        'ai_text': existing_match.get('ai_analysis_text'),
                                        'ai_fair_odds1': existing_match.get('ai_fair_odds1'),
                                        'ai_fair_odds2': existing_match.get('ai_fair_odds2'),
                                        'old_odds1': existing_match.get('odds1', 0),
                                        'old_odds2': existing_match.get('odds2', 0)
                                    }

                            c1 = p1_obj.get('country', 'Unknown')
                            c2 = p2_obj.get('country', 'Unknown')
                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2, c1, c2)
                            
                            log(f"      ⚖️ Physics Context: Surface={surf}, BSI={bsi}, Location={notes}")
                            
                            s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p2_obj['id']), {})
                            
                            style_stats_p1 = get_style_matchup_stats_py(supabase, n1, p2_obj.get('play_style', ''))
                            style_stats_p2 = get_style_matchup_stats_py(supabase, n2, p1_obj.get('play_style', ''))
                            
                            has_real_report = bool(r1.get('strengths') and r2.get('strengths'))

                            surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
                            surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)
                            
                            is_hunter_pick_active = False
                            hunter_pick_player = None

                            if db_match_id and cached_ai and not actual_winner_val:
                                log(f"   💰 Token Saver: Reusing AI Text")
                                ai_text_final = cached_ai['ai_text']
                                new_prob = recalculate_fair_odds_with_new_market(
                                    old_fair_odds1=cached_ai['ai_fair_odds1'],
                                    old_market_odds1=cached_ai['old_odds1'],
                                    old_market_odds2=cached_ai['old_odds2'],
                                    new_market_odds1=m['odds1'],
                                    new_market_odds2=m['odds2']
                                )
                                fair1 = round(1/new_prob, 2) if new_prob > 0.01 else 99
                                fair2 = round(1/(1-new_prob), 2) if new_prob < 0.99 else 99
                                
                                bet_p1 = calculate_dynamic_stake(1/fair1, m['odds1'], 0.5)
                                bet_p2 = calculate_dynamic_stake(1/fair2, m['odds2'], 0.5)
                                
                                kelly_advice = ""
                                if bet_p1["is_bet"]:
                                    kelly_advice = f" | {bet_p1['type']}: {n1} ({fair1}) -> {bet_p1['stake_str']}"
                                    is_hunter_pick_active = True
                                    hunter_pick_player = n1
                                elif bet_p2["is_bet"]:
                                    kelly_advice = f" | {bet_p2['type']}: {n2} ({fair2}) -> {bet_p2['stake_str']}"
                                    is_hunter_pick_active = True
                                    hunter_pick_player = n2
                                
                                if "VALUE" not in ai_text_final and "HUNTER" not in ai_text_final and "BANKER" not in ai_text_final:
                                    ai_text_final += kelly_advice
                            else:
                                f1_d = await fetch_player_form_hybrid(browser, n1)
                                f2_d = await fetch_player_form_hybrid(browser, n2)
                                elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
                                e1 = ELO_CACHE.get("ATP", {}).get(n1.lower(), {}).get(elo_key, 1500)
                                e2 = ELO_CACHE.get("ATP", {}).get(n2.lower(), {}).get(elo_key, 1500)
                                
                                ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, e1, e2, f1_d, f2_d)
                                
                                prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai, m['odds1'], m['odds2'], surf_rate1, surf_rate2, has_real_report, style_stats_p1, style_stats_p2)
                                
                                ai_text_base = ai.get('ai_text', 'No detailed analysis available.')
                                fair1 = round(1/prob, 2) if prob > 0.01 else 99
                                fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                                
                                p1_sentiment = to_float(ai.get('p1_win_sentiment', 0.5), 0.5)
                                p2_sentiment = 1.0 - p1_sentiment
                                
                                betting_advice = ""
                                bet_p1 = calculate_dynamic_stake(1/fair1, m['odds1'], p1_sentiment)
                                bet_p2 = calculate_dynamic_stake(1/fair2, m['odds2'], p2_sentiment)
                                
                                if bet_p1["is_bet"]:
                                    betting_advice = f" [💎 {bet_p1['type']}: {n1} @ {m['odds1']} | Fair: {fair1} | Edge: {bet_p1['edge_percent']}% | Stake: {bet_p1['stake_str']}]"
                                    is_hunter_pick_active = True
                                    hunter_pick_player = n1
                                elif bet_p2["is_bet"]:
                                    betting_advice = f" [💎 {bet_p2['type']}: {n2} @ {m['odds2']} | Fair: {fair2} | Edge: {bet_p2['edge_percent']}% | Stake: {bet_p2['stake_str']}]"
                                    is_hunter_pick_active = True
                                    hunter_pick_player = n2
                                
                                ai_text_final = ai_text_base + betting_advice
                                
                                if style_stats_p1 and style_stats_p1['verdict'] != "Neutral":
                                    ai_text_final += f" (Note: {n1} {style_stats_p1['verdict']} vs {style_stats_p1['style']})"
                                if style_stats_p2 and style_stats_p2['verdict'] != "Neutral":
                                    ai_text_final += f" (Note: {n2} {style_stats_p2['verdict']} vs {style_stats_p2['style']})"
                            
                            data = {
                                "player1_name": n1, "player2_name": n2, "tournament": m['tour'],
                                "odds1": m['odds1'], "odds2": m['odds2'],
                                "ai_fair_odds1": fair1, "ai_fair_odds2": fair2,
                                "ai_analysis_text": ai_text_final,
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z",
                                "actual_winner_name": actual_winner_val
                            }
                            
                            final_match_id = None
                            
                            if db_match_id:
                                supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                                final_match_id = db_match_id
                                log(f"🔄 Updated Odds: {n1} vs {n2} (Winner: {actual_winner_val})")
                            else:
                                res_insert = supabase.table("market_odds").insert(data).execute()
                                if res_insert.data and len(res_insert.data) > 0:
                                    final_match_id = res_insert.data[0]['id']
                                    log(f"💾 Saved: {n1} vs {n2} (Winner: {actual_winner_val})")

                            if final_match_id and is_hunter_pick_active:
                                history_payload = {
                                    "match_id": final_match_id,
                                    "odds1": m['odds1'],
                                    "odds2": m['odds2'],
                                    "fair_odds1": fair1,
                                    "fair_odds2": fair2,
                                    "is_hunter_pick": is_hunter_pick_active,
                                    "pick_player_name": hunter_pick_player,
                                    "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                                }
                                try:
                                    supabase.table("odds_history").insert(history_payload).execute()
                                except Exception as hist_err:
                                    log(f"      ⚠️ History Log Error: {hist_err}")
                        else:
                            pass
                                    
                    except Exception as e: log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
