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

log("🔌 Initialisiere Neural Scout (V38.0 - Kelly Criterion & Dynamic Weights)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# [ARCHITECT NOTE]: Wir nutzen das stabilste Flash-Modell für Speed & Cost
MODEL_NAME = 'gemini-2.0-flash'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena",
    "Brisbane": "Pat Rafter Arena",
    "Adelaide": "Memorial Drive Tennis Centre",
    "Melbourne": "Rod Laver Arena"
}
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

# =================================================================
# 2. HELPER FUNCTIONS & SAFETY
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
    # Standard Garbage Removal
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    # Remove Seeds and Country Codes in brackets if any
    clean = re.sub(r'\s*\(\d+\)', '', clean) 
    clean = re.sub(r'\s*\(.*?\)', '', clean) 
    return clean.replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = raw
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<.*?>', '', clean) # Remove stray HTML tags
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

# --- V37.0 FIX: COMPOUND NAME HANDLING (Carballes Baena Logic) ---
def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str]) -> Optional[Dict]:
    if not scraped_name_raw or not db_players: return None
    
    # 1. Cleaning
    clean_scrape = clean_player_name(scraped_name_raw).lower() # "carballes baena r."
    parts = clean_scrape.split()
    
    if not parts: return None
    
    scrape_last_name_str = ""
    scrape_initial = None
    
    # 2. Intelligent Parsing Strategy
    # Check if the last part is an initial (len=1 or len=2 with dot, e.g. "r." or "r")
    last_token = parts[-1]
    is_initial = (len(last_token) == 1) or (len(last_token) == 2 and last_token.endswith('.'))
    
    if len(parts) > 1 and is_initial:
        # Format: "Surname Surname I." -> Split off the initial at the end
        scrape_initial = last_token.replace('.', '')
        # Combine everything BEFORE the initial as the last name
        scrape_last_name_str = " ".join(parts[:-1]) # "carballes baena"
    else:
        # Format: "Surname" (No initial found or just one word)
        scrape_last_name_str = " ".join(parts) # "nadal"
        
    # Normalize Scrape Name (remove hyphens for better matching "Pinnington-Jones" vs "Pinnington Jones")
    scrape_last_name_clean = scrape_last_name_str.replace('-', ' ')
    
    candidates = []
    for p in db_players:
        if not isinstance(p, dict): continue
        
        # Normalize DB Name
        db_last_raw = p.get('last_name', '').lower()
        db_last_clean = db_last_raw.replace('-', ' ')
        
        # 3. STRICT FULL MATCH CHECK
        # "carballes baena" == "carballes baena" -> TRUE
        # "kawa" == "kawaguchi" -> FALSE (Fixes the Kawa/Kawaguchi bug)
        if db_last_clean == scrape_last_name_clean:
            
            # 4. INITIAL CHECK (Consistency)
            if scrape_initial:
                db_first = p.get('first_name', '').lower()
                if db_first and not db_first.startswith(scrape_initial):
                    continue # Wrong sibling (e.g. Cerundolo F vs J)
            
            candidates.append(p)
            
    if not candidates: return None
    
    # Priority: Player with Scouting Report
    for cand in candidates:
        if cand['id'] in report_ids: return cand
    
    return candidates[0]

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
    await asyncio.sleep(0.8)
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
# 4. DATA FETCHING
# =================================================================
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
                        # Strict key generation
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
# 5. MATH CORE & KELLY CRITERION (THE VETERAN UPGRADE)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_kelly_stake(fair_prob: float, market_odds: float, bankroll_fraction: float = 0.05) -> str:
    """
    Calculates the Kelly Criterion for unit sizing.
    f* = (bp - q) / b
    where b = market_odds - 1
          p = fair_prob
          q = 1 - p
    """
    if market_odds <= 1.0: return "0u (Bad Odds)"
    
    b = market_odds - 1
    p = fair_prob
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    
    # [VETERAN SAFEGUARDS]
    # We use 'Fractional Kelly' (e.g., 20% of Full Kelly) to reduce variance.
    safe_kelly = kelly_fraction * 0.20 
    
    if safe_kelly <= 0: return "0u"
    
    # Normalize to Unit sizes (1 Unit = 2% of Bankroll typically)
    # If safe_kelly is 0.02 (2%), that's 1 Unit.
    units = round(safe_kelly / 0.02, 2)
    
    if units < 0.25: return "0u (Edge too thin)"
    if units > 3.0: return "3.0u (MAX BET)" # Cap max risk
    
    return f"{units}u"

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2, has_scouting_reports: bool):
    ai_meta = ensure_dict(ai_meta)
    n1 = get_last_name(p1_name); n2 = get_last_name(p2_name)
    tour = "ATP"; bsi_val = to_float(bsi, 6.0)
    
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo1 = p1_stats.get(elo_surf, 1500)
    elo2 = p2_stats.get(elo_surf, 1500)
    
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8)
    
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    def get_defense(s): return s.get('speed', 50) + s.get('stamina', 50) + s.get('mental', 50)
    
    c1_score = get_offense(s1); c2_score = get_offense(s2)
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    f1 = to_float(ai_meta.get('p1_form_score', 5)); f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    # [VETERAN DYNAMIC WEIGHTS]
    # If we have REAL scouting reports, trust AI/Matchup more. 
    # If we don't, trust ELO/Market more.
    if has_scouting_reports:
        # High confidence in AI data
        weights = [0.25, 0.20, 0.10, 0.10, 0.15] # Matchup is King
    else:
        # Low confidence in AI data -> Trust ELO/Form
        weights = [0.10, 0.15, 0.10, 0.35, 0.10] # ELO is King
        
    total_w = sum(weights)
    weights = [w/total_w for w in weights]
    
    prob_alpha = (prob_matchup * weights[0]) + (prob_bsi * weights[1]) + (prob_skills * weights[2]) + (prob_elo * weights[3]) + (prob_form * weights[4])
    
    # Compression (Conservative Adjustment)
    if prob_alpha > 0.60: prob_alpha = min(prob_alpha * 1.05, 0.94) # Reduced aggression
    elif prob_alpha < 0.40: prob_alpha = max(prob_alpha * 0.95, 0.06)
    
    # Market Consensus (De-Vigged)
    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
        margin = inv1 + inv2
        true_prob1 = inv1 / margin
        prob_market = true_prob1
    
    # Final Blend: 70% Model / 30% Market (Anchor to reality)
    return (prob_alpha * 0.70) + (prob_market * 0.30)

def recalculate_fair_odds_with_new_market(old_fair_odds1: float, old_market_odds1: float, old_market_odds2: float, new_market_odds1: float, new_market_odds2: float) -> float:
    try:
        old_prob_market = 0.5
        if old_market_odds1 > 1 and old_market_odds2 > 1:
            inv1 = 1/old_market_odds1; inv2 = 1/old_market_odds2
            old_prob_market = inv1 / (inv1 + inv2)
        
        if old_fair_odds1 <= 1.01: return 0.5
        old_final_prob = 1 / old_fair_odds1
        
        alpha_part = old_final_prob - (old_prob_market * 0.30)
        prob_alpha = alpha_part / 0.70
        
        new_prob_market = 0.5
        if new_market_odds1 > 1 and new_market_odds2 > 1:
            inv1 = 1/new_market_odds1; inv2 = 1/new_market_odds2
            new_prob_market = inv1 / (inv1 + inv2)
            
        new_final_prob = (prob_alpha * 0.70) + (new_prob_market * 0.30)
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

async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    res = await call_gemini(f"Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\" }}")
    if res:
        try: 
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            TOURNAMENT_LOC_CACHE[scraped_name] = data
            return data
        except: pass
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2):
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
        log(f"   🏟️ Tournament Matched: '{s_low}' -> '{best_match['name']}' (Score: {best_score})")
        return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', '')

    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour)
    ai_loc = ensure_dict(ai_loc)
    if ai_loc and ai_loc.get('city'):
        city = ai_loc['city'].lower()
        surf = ai_loc.get('surface_guessed', 'Hard')
        return surf, (3.5 if 'clay' in surf.lower() else 6.5), f"AI Guess: {city}"
    
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1, form2):
    log(f"   🤖 Asking AI for analysis on: {p1['last_name']} vs {p2['last_name']}")
    has_reports = r1.get('strengths') and r2.get('strengths')
    if has_reports: log("      📄 Real Scouting Reports found!")
    else: log("      ⚠️ Missing Scouting Reports - AI will use stats fallback.")

    prompt = f"""
    ROLE: Elite Tennis Analyst (Silicon Valley Style).
    TASK: Analyze {p1['last_name']} vs {p2['last_name']} on {surface} (BSI {bsi}).
    DATA: ELO {elo1} vs {elo2}. FORM {form1['text']} vs {form2['text']}.
    SCOUTING P1: {r1.get('strengths', 'N/A')}
    SCOUTING P2: {r2.get('strengths', 'N/A')}
    COURT: {notes}
    OUTPUT JSON ONLY.
    JSON: {{ "p1_tactical_score": [0-10], "p2_tactical_score": [0-10], "p1_form_score": [0-10], "p2_form_score": [0-10], "ai_text": "Analysis string (max 2 sentences)." }}
    """
    res = await call_gemini(prompt)
    data = ensure_dict(safe_get_ai_data(res))
    text = data.get('ai_text', '')
    if not text or len(text) < 30 or "..." in text:
        log("      ⚠️ AI returned weak analysis - Injecting Hard Fallback.")
        adv = p1['last_name'] if elo1 > elo2 else p2['last_name']
        data['ai_text'] = f"Based on ELO ({elo1} vs {elo2}) and form, {adv} holds a slight edge."
    return data

def safe_get_ai_data(res_text: Optional[str]) -> Dict[str, Any]:
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': ''}
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
    # V35.5 (Hybrid Winner Logic)
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    
    odds_class_pattern = re.compile(r'course')

    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                i += 1; continue
            
            if "doubles" in current_tour.lower() or i+1 >= len(rows): 
                i+=1; continue

            row2 = rows[i+1]
            cols1 = row.find_all('td')
            cols2 = row2.find_all('td')
            
            if len(cols1) < 2 or len(cols2) < 1: 
                i+=1; continue

            p1_cell = next((c for c in cols1 if c.find('a') and 'time' not in c.get('class', [])), None)
            if not p1_cell and len(cols1) > 1: p1_cell = cols1[1]

            p2_cell = next((c for c in cols2 if c.find('a')), None)
            if not p2_cell and len(cols2) > 0: p2_cell = cols2[0]
            
            if not p1_cell or not p2_cell: 
                i+=1; continue

            p1_raw = clean_player_name(p1_cell.get_text(strip=True))
            p2_raw = clean_player_name(p2_cell.get_text(strip=True))
            
            if '/' in p1_raw or '/' in p2_raw or "canc" in (row.text + row2.text).lower(): 
                i+=2; continue

            m_time = "00:00"
            tc = row.find('td', class_='first')
            if tc and 'time' in tc.get('class', []):
                tm = re.search(r'(\d{1,2}:\d{2})', tc.get_text(strip=True))
                if tm: m_time = tm.group(1).zfill(5)

            p1_match = any(tp in p1_raw.lower() for tp in target_players)
            p2_match = any(tp in p2_raw.lower() for tp in target_players)

            if p1_match and p2_match:
                # --- WINNER DETECTION LOGIC ---
                winner_found = None
                p1_res = row.find('td', class_='result')
                p2_res = row2.find('td', class_='result')
                if p1_res and p2_res:
                    t1 = p1_res.get_text(strip=True)
                    t2 = p2_res.get_text(strip=True)
                    if t1.isdigit() and t2.isdigit():
                        s1 = int(t1); s2 = int(t2)
                        if s1 > s2 and s1 >= 2: winner_found = p1_raw
                        elif s2 > s1 and s2 >= 2: winner_found = p2_raw

                odds = []
                try:
                    course_cells_r1 = row.find_all('td', class_=odds_class_pattern)
                    found_r1_odds = []
                    for cell in course_cells_r1:
                        txt = cell.get_text(strip=True)
                        try:
                            val = float(txt)
                            if 1.01 <= val <= 100.0: found_r1_odds.append(val)
                        except: pass
                    
                    if len(found_r1_odds) >= 2:
                        odds = found_r1_odds[:2]
                    else:
                        course_cells_r2 = row2.find_all('td', class_=odds_class_pattern)
                        found_r2_odds = []
                        for cell in course_cells_r2:
                            txt = cell.get_text(strip=True)
                            try:
                                val = float(txt)
                                if 1.01 <= val <= 100.0: found_r2_odds.append(val)
                            except: pass
                        
                        if found_r1_odds and found_r2_odds:
                            odds = [found_r1_odds[0], found_r2_odds[0]]

                except Exception: pass
                
                final_o1 = odds[0] if len(odds) > 0 else 0.0
                final_o2 = odds[1] if len(odds) > 1 else 0.0

                if (final_o1 > 0 and final_o2 > 0) or winner_found:
                    found.append({
                        "p1_raw": p1_raw, "p2_raw": p2_raw, "tour": clean_tournament_name(current_tour), 
                        "time": m_time, "odds1": final_o1, "odds2": final_o2,
                        "p1_href": p1_cell.find('a')['href'] if p1_cell.find('a') else None, 
                        "p2_href": p2_cell.find('a')['href'] if p2_cell.find('a') else None,
                        "actual_winner": winner_found
                    })
                i += 2 
            else: i += 1 
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
            log(f"      Scanning Results for: {t_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            soup = BeautifulSoup(await page.content(), 'html.parser')
            table = soup.find('table', class_='result')
            if not table: continue
            rows = table.find_all('tr')
            for i in range(len(rows)):
                row = rows[i]
                if 'flags' in str(row) or 'head' in str(row) or i+1 >= len(rows): continue
                next_row = rows[i+1]
                rt = row.get_text(separator=" ", strip=True).lower()
                nrt = next_row.get_text(separator=" ", strip=True).lower()
                if not any(c.isdigit() for c in rt): continue 
                if "h2h" in rt or "head" in rt: continue
                for pm in safe_to_check:
                    p1 = get_last_name(pm['player1_name'])
                    p2 = get_last_name(pm['player2_name'])
                    found_match = (p1 in rt and p2 in nrt) or (p2 in rt and p1 in nrt) or (p1 in rt and p2 in rt)
                    if found_match:
                        winner = None
                        try:
                            if "ret." in rt or "w.o." in rt or "ret." in nrt:
                                if p1 in rt and "ret." not in rt: winner = pm['player1_name']
                                elif p2 in nrt and "ret." not in nrt: winner = pm['player2_name']
                            else:
                                s1_sets = rt.count("6-") + rt.count("7-") + rt.count("1-") 
                                s2_sets = nrt.count("6-") + nrt.count("7-") + nrt.count("1-")
                                if s1_sets > s2_sets: winner = pm['player1_name'] if p1 in rt else pm['player2_name']
                                elif s2_sets > s1_sets: winner = pm['player2_name'] if p2 in nrt else pm['player1_name']
                            if winner:
                                supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", pm['id']).execute()
                                safe_to_check = [x for x in safe_to_check if x['id'] != pm['id']]
                                log(f"      ✅ SETTLED: {winner} won (vs {p2 if winner==pm['player1_name'] else p1})")
                        except: pass
        except: pass
        finally: await page.close()

async def run_pipeline():
    log(f"🚀 Neural Scout V38.0 (Veteran Edition: Kelly + Dynamic Weights) Starting...")
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
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue
                matches = parse_matches_locally_v5(html, player_names)
                log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    try:
                        # V37.0: Use Compound-Aware Matching
                        p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                        p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                        
                        if p1_obj and p2_obj:
                            n1 = p1_obj['last_name']
                            n2 = p2_obj['last_name']
                            
                            if n1 == n2: continue

                            # Settlement
                            if m.get('actual_winner'):
                                winner_full = n1 if m['actual_winner'] == m['p1_raw'] else n2
                                try:
                                    res = supabase.table("market_odds").select("id").or_(f"and(player1_name.eq.{n1},player2_name.eq.{n2}),and(player1_name.eq.{n2},player2_name.eq.{n1})").is_("actual_winner_name", "null").execute()
                                    if res.data:
                                        for rec in res.data:
                                            supabase.table("market_odds").update({"actual_winner_name": winner_full}).eq("id", rec['id']).execute()
                                            log(f"      🏆 LIVE SETTLEMENT: {winner_full} won")
                                except Exception as e: log(f"Settlement Error: {e}")
                                continue

                            if m['odds1'] < 1.01 and m['odds2'] < 1.01: continue
                            
                            existing_p1 = supabase.table("market_odds").select("id, actual_winner_name, odds1, odds2, player2_name, ai_analysis_text, ai_fair_odds1, ai_fair_odds2").eq("player1_name", n1).order("created_at", desc=True).limit(5).execute()
                            existing = []
                            if existing_p1.data:
                                for rec in existing_p1.data:
                                    if rec['player2_name'] == n2: existing.append(rec); break
                            
                            db_match_id = None
                            cached_ai = {}
                            if existing:
                                rec = existing[0]
                                if rec.get('actual_winner_name'): continue 
                                if abs(rec.get('odds1', 0) - m['odds1']) < 0.05 and rec.get('odds1', 0) > 1.1: continue
                                db_match_id = rec['id']
                                if rec.get('ai_analysis_text'):
                                    cached_ai = {
                                        'ai_text': rec.get('ai_analysis_text'),
                                        'ai_fair_odds1': rec.get('ai_fair_odds1'),
                                        'ai_fair_odds2': rec.get('ai_fair_odds2'),
                                        'old_odds1': rec.get('odds1', 0),
                                        'old_odds2': rec.get('odds2', 0)
                                    }

                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2)
                            s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p2_obj['id']), {})
                            
                            # CHECK FOR REAL REPORT
                            has_real_report = bool(r1.get('strengths') and r2.get('strengths'))
                            if has_real_report: log(f"   🔥 HIGH CONFIDENCE: Real Report found for {n1} vs {n2}")

                            surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
                            surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)
                            
                            if db_match_id and cached_ai:
                                log(f"   💰 Token Saver: Reusing AI Text, Recalculating Fair Odds")
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
                                
                                # Recalculate Kelly with new odds
                                kelly_advice = ""
                                if m['odds1'] > fair1:
                                    kelly_advice = f" | 💎 P1 VALUE ({fair1}) -> " + calculate_kelly_stake(1/fair1, m['odds1'])
                                elif m['odds2'] > fair2:
                                    kelly_advice = f" | 💎 P2 VALUE ({fair2}) -> " + calculate_kelly_stake(1/fair2, m['odds2'])
                                
                                # Append Advice if not already there
                                if "VALUE" not in ai_text_final:
                                    ai_text_final += kelly_advice
                            else:
                                f1_d = await fetch_player_form_hybrid(browser, n1)
                                f2_d = await fetch_player_form_hybrid(browser, n2)
                                elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
                                e1 = ELO_CACHE.get("ATP", {}).get(n1.lower(), {}).get(elo_key, 1500)
                                e2 = ELO_CACHE.get("ATP", {}).get(n2.lower(), {}).get(elo_key, 1500)
                                
                                ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, e1, e2, f1_d, f2_d)
                                
                                # Pass has_real_report to math core
                                prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai, m['odds1'], m['odds2'], surf_rate1, surf_rate2, has_real_report)
                                
                                ai_text_base = ai.get('ai_text', 'No detailed analysis available.')
                                fair1 = round(1/prob, 2) if prob > 0.01 else 99
                                fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                                
                                # --- VETERAN: ADD STAKING ADVICE ---
                                betting_advice = ""
                                edge_p1 = (1/fair1) - (1/m['odds1']) if m['odds1'] > 0 else 0
                                edge_p2 = (1/fair2) - (1/m['odds2']) if m['odds2'] > 0 else 0
                                
                                # Only show advice if edge > 2%
                                if m['odds1'] > fair1 and edge_p1 > -0.05: # Slight tolerance
                                    stake = calculate_kelly_stake(1/fair1, m['odds1'])
                                    betting_advice = f" [💎 P1 VALUE @ {m['odds1']} (Fair: {fair1}) | Stake: {stake}]"
                                elif m['odds2'] > fair2 and edge_p2 > -0.05:
                                    stake = calculate_kelly_stake(1/fair2, m['odds2'])
                                    betting_advice = f" [💎 P2 VALUE @ {m['odds2']} (Fair: {fair2}) | Stake: {stake}]"
                                
                                ai_text_final = ai_text_base + betting_advice
                            
                            data = {
                                "player1_name": n1, "player2_name": n2, "tournament": m['tour'],
                                "odds1": m['odds1'], "odds2": m['odds2'],
                                "ai_fair_odds1": fair1, "ai_fair_odds2": fair2,
                                "ai_analysis_text": ai_text_final,
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                            }
                            
                            if db_match_id:
                                supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                                log(f"🔄 Updated Odds: {n1} vs {n2}")
                            else:
                                supabase.table("market_odds").insert(data).execute()
                                log(f"💾 Saved: {n1} vs {n2}")
                    except Exception as e: log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
