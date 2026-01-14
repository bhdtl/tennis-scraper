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

log("🔌 Initialisiere Neural Scout (V40.0 - SOTA Weighted Model)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.0-flash'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 

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

def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str]) -> Optional[Dict]:
    if not scraped_name_raw or not db_players: return None
    clean_scrape = clean_player_name(scraped_name_raw).lower()
    parts = clean_scrape.split()
    if not parts: return None
    
    last_token = parts[-1]
    is_initial = (len(last_token) == 1) or (len(last_token) == 2 and last_token.endswith('.'))
    
    if len(parts) > 1 and is_initial:
        scrape_initial = last_token.replace('.', '')
        scrape_last_name_str = " ".join(parts[:-1])
    else:
        scrape_last_name_str = " ".join(parts)
        
    scrape_last_name_clean = scrape_last_name_str.replace('-', ' ')
    
    candidates = []
    for p in db_players:
        if not isinstance(p, dict): continue
        db_last_raw = p.get('last_name', '').lower()
        db_last_clean = db_last_raw.replace('-', ' ')
        
        if db_last_clean == scrape_last_name_clean:
            if is_initial and scrape_initial:
                db_first = p.get('first_name', '').lower()
                if db_first and not db_first.startswith(scrape_initial):
                    continue
            candidates.append(p)
            
    if not candidates: return None
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
# 4. MATH CORE & SOTA WEIGHTS
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_kelly_stake(fair_prob: float, market_odds: float) -> str:
    if market_odds <= 1.0 or fair_prob <= 0: return "0u"
    b = market_odds - 1
    p = fair_prob
    q = 1 - p
    kelly_fraction = (b * p - q) / b
    safe_kelly = kelly_fraction * 0.25 
    if safe_kelly <= 0: return "0u"
    raw_units = safe_kelly / 0.02
    units = round(raw_units * 4) / 4
    if units < 0.25: return "0u"
    if units > 3.0: units = 3.0
    return f"{units}u"

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2, has_scouting_reports: bool, style_stats_p1: Optional[Dict], style_stats_p2: Optional[Dict]):
    ai_meta = ensure_dict(ai_meta)
    n1 = get_last_name(p1_name); n2 = get_last_name(p2_name)
    tour = "ATP"; bsi_val = to_float(bsi, 6.0)
    
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo1 = p1_stats.get(elo_surf, 1500)
    elo2 = p2_stats.get(elo_surf, 1500)
    
    # 1. MATCHUP (AI)
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8)
    
    # 2. BSI / ATTRIBUTES
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    c1_score = get_offense(s1); c2_score = get_offense(s2)
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    
    # 3. SKILLS
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)
    
    # 4. ELO (Physics)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    # 5. FORM (AI + DB)
    f1 = to_float(ai_meta.get('p1_form_score', 5)); f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    # [STYLE MODIFIER]
    style_boost = 0
    if style_stats_p1 and style_stats_p1['verdict'] == "DOMINANT": style_boost += 0.05
    if style_stats_p1 and style_stats_p1['verdict'] == "STRUGGLES": style_boost -= 0.05
    if style_stats_p2 and style_stats_p2['verdict'] == "DOMINANT": style_boost -= 0.05
    if style_stats_p2 and style_stats_p2['verdict'] == "STRUGGLES": style_boost += 0.05
    
    # [SILICON VALLEY OPTIMIZED WEIGHTS]
    # ELO is king (30%), AI Matchup (20%), Speed Fit (20%), Form (15%), Raw Skills (15%)
    # This reduces variance and respects the baseline probability more.
    if has_scouting_reports:
        weights = [0.20, 0.20, 0.15, 0.30, 0.15] # Matchup, BSI, Skills, ELO, Form
    else:
        # Fallback if no report: Rely heavily on ELO and Form
        weights = [0.10, 0.15, 0.10, 0.45, 0.20]
        
    total_w = sum(weights)
    weights = [w/total_w for w in weights]
    
    prob_alpha = (prob_matchup * weights[0]) + (prob_bsi * weights[1]) + (prob_skills * weights[2]) + (prob_elo * weights[3]) + (prob_form * weights[4])
    prob_alpha += style_boost
    
    # Compression
    if prob_alpha > 0.60: prob_alpha = min(prob_alpha * 1.05, 0.94)
    elif prob_alpha < 0.40: prob_alpha = max(prob_alpha * 0.95, 0.06)
    
    # Market Wisdom (30%)
    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
        margin = inv1 + inv2
        prob_market = inv1 / margin
    
    return (prob_alpha * 0.70) + (prob_market * 0.30)

def recalculate_fair_odds_with_new_market(old_fair_odds1: float, old_market_odds1: float, old_market_odds2: float, new_market_odds1: float, new_market_odds2: float) -> float:
    try:
        old_prob_market = 0.5
        if old_market_odds1 > 1 and old_market_odds2 > 1:
            inv1 = 1/old_market_odds1; inv2 = 1/old_market_odds2
            old_prob_market = inv1 / (inv1 + inv2)
        
        if old_fair_odds1 <= 1.01: return 0.5
        old_final_prob = 1 / old_fair_odds1
        
        # Reverse Engineering the Alpha (Model) Probability
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
# 5. DATA FETCHING (STYLE & ELO)
# =================================================================
def get_style_matchup_stats_py(supabase_client: Client, player_name: str, opponent_style_raw: str) -> Optional[Dict]:
    if not player_name or not opponent_style_raw: return None
    target_style = opponent_style_raw.split(',')[0].split('(')[0].strip()
    if not target_style or target_style == 'Unknown': return None

    try:
        res = supabase_client.table('market_odds').select('player1_name, player2_name, actual_winner_name')\
            .or_(f"player1_name.ilike.%{player_name}%,player2_name.ilike.%{player_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order('created_at', desc=True)\
            .limit(80)\
            .execute()
        matches = res.data
        if not matches or len(matches) < 5: return None

        opponents_map = {}
        opponent_names_to_fetch = []
        for m in matches:
            if player_name.lower() in m['player1_name'].lower(): opp = get_last_name(m['player2_name']).lower()
            else: opp = get_last_name(m['player1_name']).lower()
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

        relevant_matches = 0; wins = 0
        for m in matches:
            if player_name.lower() in m['player1_name'].lower(): opp_name = get_last_name(m['player2_name']).lower()
            else: opp_name = get_last_name(m['player1_name']).lower()
            opp_styles = opponents_map.get(opp_name, [])
            if target_style in opp_styles:
                relevant_matches += 1
                if player_name.lower() in m.get('actual_winner_name', '').lower(): wins += 1
        
        if relevant_matches < 3: return None
        win_rate = (wins / relevant_matches) * 100
        verdict = "Neutral"
        if win_rate > 65: verdict = "DOMINANT"
        elif win_rate < 40: verdict = "STRUGGLES"
        return {"win_rate": win_rate, "matches": relevant_matches, "verdict": verdict, "style": target_style}
    except Exception as e: return None

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
        if matches and len(matches) >= 3: 
            wins = 0
            for m in matches:
                if player_last_name.lower() in m.get('actual_winner_name', '').lower(): wins += 1
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
    except Exception: return [], {}, [], []

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
    if not text or len(text) < 30:
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
    # V40.0: Structural DOM Anchor (Rowspan Check)
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
            
            if i + 1 >= len(rows): i += 1; continue

            first_cell = row.find('td', class_='first')
            is_match_start = False
            
            if first_cell:
                if first_cell.get('rowspan') == '2': is_match_start = True
                elif 'time' in first_cell.get('class', []) and len(first_cell.get_text(strip=True)) > 2:
                     row2_first = rows[i+1].find('td', class_='first')
                     if not row2_first: is_match_start = True

            if not is_match_start: i += 1; continue

            row2 = rows[i+1]
            cols1 = row.find_all('td')
            cols2 = row2.find_all('td')
            if len(cols1) < 2 or len(cols2) < 1: i += 2; continue

            p1_cell = next((c for c in cols1 if c.find('a') and 'time' not in c.get('class', [])), None)
            if not p1_cell and len(cols1) > 1: p1_cell = cols1[1]
            p2_cell = next((c for c in cols2 if c.find('a')), None)
            if not p2_cell and len(cols2) > 0: p2_cell = cols2[0]
            
            if not p1_cell or not p2_cell: i += 2; continue

            p1_raw = clean_player_name(p1_cell.get_text(strip=True))
            p2_raw = clean_player_name(p2_cell.get_text(strip=True))
            
            if '/' in p1_raw or '/' in p2_raw or "canc" in (row.text + row2.text).lower(): i += 2; continue

            m_time = "00:00"
            if first_cell:
                tm = re.search(r'(\d{1,2}:\d{2})', first_cell.get_text(strip=True))
                if tm: m_time = tm.group(1).zfill(5)

            p1_match = any(tp in p1_raw.lower() for tp in target_players)
            p2_match = any(tp in p2_raw.lower() for tp in target_players)

            if p1_match and p2_match:
                winner_found = None
                # Check results for settlement
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
                    # Extract Odds
                    found_odds = []
                    for r_temp in [row, row2]:
                        cells = r_temp.find_all('td', class_=odds_class_pattern)
                        for cell in cells:
                            try:
                                val = float(cell.get_text(strip=True))
                                if 1.01 <= val <= 100.0: found_odds.append(val)
                            except: pass
                    if len(found_odds) >= 2: odds = found_odds[:2]
                except: pass
                
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
            
    return found

async def update_past_results(browser: Browser):
    log("🏆 The Auditor: Checking Real-Time Results...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending: return
    safe_to_check = [pm for pm in pending if pm]
    
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
            for i in range(len(rows)):
                row = rows[i]
                if 'flags' in str(row) or 'head' in str(row) or i+1 >= len(rows): continue
                next_row = rows[i+1]
                rt = row.get_text(separator=" ", strip=True).lower()
                nrt = next_row.get_text(separator=" ", strip=True).lower()
                if not any(c.isdigit() for c in rt): continue 
                
                for pm in safe_to_check:
                    p1 = get_last_name(pm['player1_name'])
                    p2 = get_last_name(pm['player2_name'])
                    if (p1 in rt and p2 in nrt) or (p2 in rt and p1 in nrt):
                        winner = None
                        if "ret." in rt or "w.o." in rt:
                            if p1 in rt and "ret." not in rt: winner = pm['player1_name']
                            elif p2 in nrt and "ret." not in nrt: winner = pm['player2_name']
                        else:
                            s1_sets = rt.count("6-") + rt.count("7-") 
                            s2_sets = nrt.count("6-") + nrt.count("7-")
                            if s1_sets > s2_sets: winner = pm['player1_name'] if p1 in rt else pm['player2_name']
                            elif s2_sets > s1_sets: winner = pm['player2_name'] if p2 in nrt else pm['player1_name']
                        if winner:
                            supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", pm['id']).execute()
                            safe_to_check = [x for x in safe_to_check if x['id'] != pm['id']]
                            log(f"      ✅ SETTLED: {winner} won")
        except: pass
        finally: await page.close()

async def run_pipeline():
    log(f"🚀 Neural Scout V40.0 Starting...")
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
                        p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
                        p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
                        
                        if p1_obj and p2_obj:
                            n1 = p1_obj['last_name']
                            n2 = p2_obj['last_name']
                            if n1 == n2: continue

                            # Settlement check directly from match page if odds scraped also have results
                            if m.get('actual_winner'):
                                winner_full = n1 if m['actual_winner'] == m['p1_raw'] else n2
                                supabase.table("market_odds").update({"actual_winner_name": winner_full}).eq("player1_name", n1).eq("player2_name", n2).is_("actual_winner_name", "null").execute()

                            if m['odds1'] < 1.01: continue
                            
                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, n1, n2)
                            s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if r.get('player_id') == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if r.get('player_id') == p2_obj['id']), {})
                            
                            style_stats_p1 = get_style_matchup_stats_py(supabase, n1, p2_obj.get('play_style', ''))
                            style_stats_p2 = get_style_matchup_stats_py(supabase, n2, p1_obj.get('play_style', ''))
                            
                            has_real_report = bool(r1.get('strengths') and r2.get('strengths'))
                            surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
                            surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)
                            
                            f1_d = await fetch_player_form_hybrid(browser, n1)
                            f2_d = await fetch_player_form_hybrid(browser, n2)
                            
                            # ELO Lookup
                            elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
                            e1 = ELO_CACHE.get("ATP", {}).get(n1.lower(), {}).get(elo_key, 1500)
                            e2 = ELO_CACHE.get("ATP", {}).get(n2.lower(), {}).get(elo_key, 1500)
                            
                            ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, e1, e2, f1_d, f2_d)
                            
                            prob = calculate_physics_fair_odds(n1, n2, s1, s2, bsi, surf, ai, m['odds1'], m['odds2'], surf_rate1, surf_rate2, has_real_report, style_stats_p1, style_stats_p2)
                            
                            fair1 = round(1/prob, 2) if prob > 0.01 else 99
                            fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99
                            
                            ai_text = ai.get('ai_text', '')
                            edge_p1 = (1/fair1) - (1/m['odds1']) if m['odds1'] > 0 else 0
                            
                            if m['odds1'] > fair1 and edge_p1 > -0.05:
                                stake = calculate_kelly_stake(1/fair1, m['odds1'])
                                ai_text += f" [💎 P1 VALUE @ {m['odds1']} (Fair: {fair1}) | Stake: {stake}]"
                            elif m['odds2'] > fair2:
                                stake = calculate_kelly_stake(1/fair2, m['odds2'])
                                ai_text += f" [💎 P2 VALUE @ {m['odds2']} (Fair: {fair2}) | Stake: {stake}]"
                            
                            if style_stats_p1 and style_stats_p1['verdict'] != "Neutral":
                                ai_text += f" (Note: {n1} {style_stats_p1['verdict']} vs Style)"

                            data = {
                                "player1_name": n1, "player2_name": n2, "tournament": m['tour'],
                                "odds1": m['odds1'], "odds2": m['odds2'],
                                "ai_fair_odds1": fair1, "ai_fair_odds2": fair2,
                                "ai_analysis_text": ai_text,
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                            }
                            
                            # SOTA FIX: Robust Upsert for Live Scraper too
                            try:
                                supabase.table("market_odds").upsert(data, on_conflict="player1_name,player2_name,match_time").execute()
                                log(f"💾 Upserted: {n1} vs {n2}")
                            except Exception as e:
                                log(f"⚠️ Write Error: {e}")

                    except Exception as e: log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
