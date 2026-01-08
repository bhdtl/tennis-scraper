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
from typing import List, Dict, Optional, Any, Tuple

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
logger = logging.getLogger("NeuralScout")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V8.1 - Stable Baseline + AI Fix)...")

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
FORM_CACHE: Dict[str, Dict[str, Any]] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 

# --- TOPOLOGY MAP ---
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena"
}

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
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = re.sub(r'S\d+[A-Z0-9]*$', '', raw).strip()
    return clean

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip()
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# --- SILICON VALLEY IDENTITY FIX ---
def find_player_safe(scraped_name_raw: str, db_players: List[Dict]) -> Optional[Dict]:
    clean_scrape = clean_player_name(scraped_name_raw).lower()
    candidates = []
    for p in db_players:
        if p['last_name'].lower() in clean_scrape:
            candidates.append(p)
            
    if not candidates: return None
    if len(candidates) == 1: return candidates[0]
    
    for cand in candidates:
        first_name = cand.get('first_name', '').lower()
        if first_name:
            initial = first_name[0]
            if f"{initial}." in clean_scrape or f" {initial} " in clean_scrape or clean_scrape.startswith(f"{initial} "):
                return cand
    return candidates[0]

# =================================================================
# 3. GEMINI ENGINE
# =================================================================
async def call_gemini(prompt: str, model: str = MODEL_NAME) -> Optional[str]:
    await asyncio.sleep(1.0)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                log(f"⚠️ Gemini API Error: {response.status_code} - {response.text}")
                return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            log(f"⚠️ Gemini Connection Failed: {e}")
            return None

# =================================================================
# 4. CORE LOGIC & DATA FETCHING
# =================================================================

async def fetch_tennisexplorer_stats(browser: Browser, relative_url: str, surface: str) -> float:
    if not relative_url: return 0.5
    
    cache_key = f"{relative_url}_{surface}"
    if cache_key in SURFACE_STATS_CACHE: return SURFACE_STATS_CACHE[cache_key]

    # CACHE BUSTING HERE
    timestamp = int(time.time())
    url = f"https://www.tennisexplorer.com{relative_url}?annual=all&t={timestamp}"
    
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
        
        total_wins = 0
        total_matches = 0
        
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.get_text(strip=True) for h in headers]
            
            if "Summary" in header_texts and target_header in header_texts:
                try:
                    col_idx = header_texts.index(target_header)
                    rows = table.find_all('tr')
                    for row in rows:
                        if "Summary" in row.get_text():
                            cols = row.find_all(['td', 'th'])
                            if len(cols) > col_idx:
                                stats_text = cols[col_idx].get_text(strip=True)
                                if "/" in stats_text:
                                    w, l = map(int, stats_text.split('/'))
                                    total_wins = w
                                    total_matches = w + l
                                    break
                except: pass
                break
        
        if total_matches > 0:
            rate = total_wins / total_matches
            log(f"   📊 Direct Stats ({target_header}): {rate:.2f} ({total_matches} matches)")
            SURFACE_STATS_CACHE[cache_key] = rate
            return rate
            
    except Exception as e:
        # log(f"   ⚠️ TE Stats Error: {e}")
        pass
    finally:
        await page.close()
        
    return 0.5

async def fetch_elo_ratings(browser: Browser):
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    for tour, url in urls.items():
        page = await browser.new_page()
        try:
            # CACHE BUSTING
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
                        try:
                            ELO_CACHE[tour][name] = {
                                'Hard': to_float(cols[3].get_text(strip=True), 1500),
                                'Clay': to_float(cols[4].get_text(strip=True), 1500),
                                'Grass': to_float(cols[5].get_text(strip=True), 1500)
                            }
                        except: continue
                log(f"   ✅ {tour} Elo Ratings geladen: {len(ELO_CACHE[tour])} Spieler.")
        except Exception as e:
            log(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
        finally:
            await page.close()

async def fetch_player_form_hybrid(browser: Browser, player_last_name: str) -> Dict[str, Any]:
    try:
        res = supabase.table("market_odds")\
            .select("actual_winner_name, match_time")\
            .or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order("match_time", desc=True)\
            .limit(5)\
            .execute()
            
        matches = res.data
        if matches and len(matches) >= 3: 
            wins = 0
            for m in matches:
                if player_last_name.lower() in m['actual_winner_name'].lower(): wins += 1
            
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
        for entry in skills:
            pid = entry.get('player_id')
            if pid:
                clean_skills[pid] = {
                    'serve': to_float(entry.get('serve')), 'power': to_float(entry.get('power')),
                    'forehand': to_float(entry.get('forehand')), 'backhand': to_float(entry.get('backhand')),
                    'speed': to_float(entry.get('speed')), 'stamina': to_float(entry.get('stamina')),
                    'mental': to_float(entry.get('mental'))
                }
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# 5. MATH CORE (SILICON VALLEY VETERAN EDITION - V5.3)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2):
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    bsi_val = to_float(bsi, 6.0)

    # --- SPECIALIST DELTA CHECK ---
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    p1_hard_elo = p1_stats.get('Hard', 1500)
    p1_clay_elo = p1_stats.get('Clay', 1500)
    p2_hard_elo = p2_stats.get('Hard', 1500)
    p2_clay_elo = p2_stats.get('Clay', 1500)

    p1_is_clay_specialist = (p1_clay_elo - p1_hard_elo) > 80
    p2_is_clay_specialist = (p2_clay_elo - p2_hard_elo) > 80

    # FIX: Safety check for ai_meta (List vs Dict)
    if isinstance(ai_meta, list):
        ai_meta = ai_meta[0] if len(ai_meta) > 0 else {}

    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8) 

    # 2. PHYSICS LAYER (ADAPTIVE)
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    
    def get_defense(s, is_clay_spec, bsi_now): 
        base_def = s.get('speed', 50) + s.get('stamina', 50) + s.get('mental', 50)
        if bsi_now > 7.0 and is_clay_spec:
            return base_def * 0.75 
        return base_def 

    def get_tech(s): return s.get('forehand', 50) + s.get('backhand', 50)

    off1 = get_offense(s1); def1 = get_defense(s1, p1_is_clay_specialist, bsi_val); tech1 = get_tech(s1)
    off2 = get_offense(s2); def2 = get_defense(s2, p2_is_clay_specialist, bsi_val); tech2 = get_tech(s2)

    c1_score = 0; c2_score = 0

    if bsi_val < 4.0: 
        c1_score = (def1 * 0.7) + (tech1 * 0.3)
        c2_score = (def2 * 0.7) + (tech2 * 0.3)
    elif 4.0 <= bsi_val < 5.5: 
        c1_score = (def1 * 0.5) + (tech1 * 0.4) + (off1 * 0.1)
        c2_score = (def2 * 0.5) + (tech2 * 0.4) + (off2 * 0.1)
    elif 5.5 <= bsi_val < 7.0: 
        c1_score = def1 + tech1 + off1
        c2_score = def2 + tech2 + off2
    elif 7.0 <= bsi_val < 8.0: 
        c1_score = (off1 * 0.5) + (tech1 * 0.3) + (def1 * 0.2)
        c2_score = (off2 * 0.5) + (tech2 * 0.3) + (def2 * 0.2)
    elif 8.0 <= bsi_val < 9.0: 
        c1_score = (off1 * 0.8) + (tech1 * 0.2)
        c2_score = (off2 * 0.8) + (tech2 * 0.2)
    else: 
        c1_score = off1
        c2_score = off2

    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)

    score_p1 = sum(s1.values())
    score_p2 = sum(s2.values())
    prob_skills = sigmoid_prob(score_p1 - score_p2, sensitivity=0.08)

    # 4. ELO HISTORICAL LAYER
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    elo1 = p1_stats.get(elo_surf, 1500.0)
    elo2 = p2_stats.get(elo_surf, 1500.0)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    f1 = to_float(ai_meta.get('p1_form_score', 5))
    f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    surf_diff = surf_rate1 - surf_rate2
    prob_surface_stats = 0.5 + (surf_diff * 0.9) 
    prob_surface_stats = max(0.1, min(0.9, prob_surface_stats))

    physics_weight = 0.20
    elo_weight = 0.10
    matchup_weight = 0.20
    form_weight = 0.10
    surface_stats_weight = 0.30 
    
    if abs(surf_diff) > 0.2: 
        surface_stats_weight = 0.45
        matchup_weight = 0.10
    
    if bsi_val > 7.5:
        physics_weight = 0.25 
    
    skill_weight = 1.0 - (physics_weight + elo_weight + matchup_weight + form_weight + surface_stats_weight)

    prob_alpha = (prob_matchup * matchup_weight) + \
                 (prob_bsi * physics_weight) + \
                 (prob_skills * skill_weight) + \
                 (prob_elo * elo_weight) + \
                 (prob_form * form_weight) + \
                 (prob_surface_stats * surface_stats_weight)

    if prob_alpha > 0.60: prob_alpha = min(prob_alpha * 1.10, 0.94)
    elif prob_alpha < 0.40: prob_alpha = max(prob_alpha * 0.90, 0.06)

    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
        
    final_prob = (prob_alpha * 0.75) + (prob_market * 0.25)
    return final_prob

# =================================================================
# 6. RESULT VERIFICATION ENGINE
# =================================================================
async def update_past_results(browser: Browser):
    log("🏆 Checking for Match Results (Restored v95.0 Aggressive Logic)...")
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    
    if not pending_matches: return

    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending_matches:
        try:
            created_at_str = pm['created_at'].replace('Z', '+00:00')
            created_at = datetime.fromisoformat(created_at_str)
            if (now_utc - created_at).total_seconds() / 60 > 65:
                safe_matches.append(pm)
        except: continue

    if not safe_matches: return

    for day_offset in range(3):
        target_date = datetime.now() - timedelta(days=day_offset)
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table', class_='result')
            if not table: continue

            rows = table.find_all('tr')
            for i in range(len(rows)):
                row = rows[i]
                if 'flags' in str(row) or 'head' in str(row): continue
                
                for pm in safe_matches:
                    p1_last = get_last_name(pm['player1_name'])
                    p2_last = get_last_name(pm['player2_name'])
                    row_text = row.get_text(separator=" ", strip=True).lower()
                    next_row_text = rows[i+1].get_text(separator=" ", strip=True).lower() if i+1 < len(rows) else ""
                    
                    match_found = (p1_last in row_text and p2_last in next_row_text) or \
                                  (p2_last in row_text and p1_last in next_row_text) or \
                                  (p1_last in row_text and p2_last in row_text)
                    
                    if match_found:
                        try:
                            is_retirement = "ret." in row_text or "w.o." in row_text
                            cols1 = row.find_all('td')
                            cols2 = rows[i+1].find_all('td') if i+1 < len(rows) else []
                            
                            def extract_scores_aggressive(columns):
                                scores = []
                                for col in columns:
                                    txt = col.get_text(strip=True)
                                    if len(txt) > 4: continue
                                    if '(' in txt: txt = txt.split('(')[0]
                                    if txt.isdigit() and len(txt) == 1 and int(txt) <= 7: scores.append(int(txt))
                                return scores

                            p1_scores = extract_scores_aggressive(cols1)
                            p2_scores = extract_scores_aggressive(cols2)
                            
                            p1_sets = 0; p2_sets = 0
                            for k in range(min(len(p1_scores), len(p2_scores))):
                                if p1_scores[k] > p2_scores[k]: p1_sets += 1
                                elif p2_scores[k] > p1_scores[k]: p2_sets += 1
                            
                            winner_name = None
                            if (p1_sets >= 2 and p1_sets > p2_sets) or (is_retirement and p1_sets > p2_sets):
                                if p1_last in row_text: winner_name = pm['player1_name']
                                elif p2_last in row_text: winner_name = pm['player2_name']
                            elif (p2_sets >= 2 and p2_sets > p1_sets) or (is_retirement and p2_sets > p1_sets):
                                if p1_last in next_row_text: winner_name = pm['player1_name']
                                elif p2_last in next_row_text: winner_name = pm['player2_name']
                            
                            if winner_name:
                                supabase.table("market_odds").update({"actual_winner_name": winner_name}).eq("id", pm['id']).execute()
                                safe_matches = [x for x in safe_matches if x['id'] != pm['id']]
                                log(f"      ✅ Verified Winner: {winner_name}")
                        except: pass
        except: pass
        finally: await page.close()

# =================================================================
# 7. MAIN PIPELINE
# =================================================================
async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    prompt = f"TASK: Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\", \"is_indoor\": bool }}"
    res = await call_gemini(prompt)
    if not res: return None
    try:
        data = json.loads(res.replace("json", "").replace("", "").strip())
        TOURNAMENT_LOC_CACHE[scraped_name] = data
        return data
    except: return None

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
            COUNTRY_TO_CITY_MAP.update(json.loads(res.replace("json", "").replace("", "").strip()))
    except: pass
    finally: await page.close()

async def resolve_united_cup_via_country(p1):
    if not COUNTRY_TO_CITY_MAP: return None
    cache_key = f"COUNTRY_{p1}"
    if cache_key in TOURNAMENT_LOC_CACHE: country = TOURNAMENT_LOC_CACHE[cache_key]
    else:
        res = await call_gemini(f"Country of player {p1}? JSON: {{'country': 'Name'}}")
        country = json.loads(res.replace("json", "").replace("", "").strip()).get("country", "Unknown") if res else "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
            
    if country in COUNTRY_TO_CITY_MAP: return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
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

    for t in db_tours:
        if t['name'].lower() == s_low: return t['surface'], t['bsi_rating'], t.get('notes', '')
    
    if "clay" in s_low: return "Red Clay", 3.5, "Local"
    if "hard" in s_low: return "Hard", 6.5, "Local"
    if "indoor" in s_low: return "Indoor", 8.0, "Local"
    
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour)
    if ai_loc and ai_loc.get('city'):
        city = ai_loc['city'].lower()
        surf = ai_loc.get('surface_guessed', 'Hard')
        for t in db_tours:
            if city in t['name'].lower(): return t['surface'], t['bsi_rating'], f"AI: {city}"
        return surf, (3.5 if 'clay' in surf.lower() else 6.5), f"AI Guess: {city}"
    return 'Hard', 6.5, 'Fallback'

# --- UPDATE START: V7.0 RAW INTEL + LIST FIX ---
async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1, form2):
    # OPTIMIZED V7.0 PROMPT: RAW INTEL
    prompt = f"""
    ROLE: Elite Tennis Analyst.
    MATCH: {p1['last_name']} vs {p2['last_name']} ({surface}).
    
    DATA:
    - ELO ({surface}): {p1['last_name']}={elo1} vs {p2['last_name']}={elo2}
    - FORM: {p1['last_name']}={form1['text']}, {p2['last_name']}={form2['text']}
    
    TASK: Generate "RAW INTEL" for database storage. 
    NO PROSE. ONLY FACTS.
    
    OUTPUT JSON ONLY: 
    {{ 
      "p1_tactical_score": 7, 
      "p2_tactical_score": 5, 
      "p1_form_score": 8, 
      "p2_form_score": 4, 
      "ai_text": "SURFACE_ADVANTAGE: {p1['last_name']} (Higher ELO). KEY: {r1.get('strengths','Serve')} vs {r2.get('weaknesses','Backhand')}. FORM: {p1['last_name']} is {form1['text']}." 
    }}
    """
    res = await call_gemini(prompt)
    default_res = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': 'No intel.'}
    if not res: return default_res
    try: 
        cleaned = res.replace("json", "").replace("```", "").strip() # Added backticks clean
        data = json.loads(cleaned)
        
        # --- THE FIX: HANDLE LISTS ---
        if isinstance(data, list):
            data = data[0] if len(data) > 0 else default_res
            
        return data
    except: return default_res
# --- UPDATE END ---

async def scrape_tennis_odds_for_date(browser: Browser, target_date):
    page = await browser.new_page()
    try:
        # Cache Busting
        timestamp = int(time.time())
        url = f"[https://www.tennisexplorer.com/matches/?type=all&year=](https://www.tennisexplorer.com/matches/?type=all&year=){target_date.year}&month={target_date.month}&day={target_date.day}&t={timestamp}"
        
        log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        return await page.content()
    except Exception as e:
        log(f"❌ Scrape Error: {e}")
        return None
    finally: await page.close()

def parse_matches_locally_v5(html, p_names): 
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                i += 1; continue
            if "doubles" in current_tour.lower(): i += 1; continue
            if i + 1 >= len(rows): break

            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                raw_time = first_col.get_text(strip=True)
                time_match = re.search(r'(\d{1,2}:\d{2})', raw_time)
                if time_match: match_time_str = time_match.group(1).zfill(5) 

            # EXTRACT LINKS
            p1_cell = row.find_all('td')[1] 
            p2_cell = rows[i+1].find_all('td')[0] 

            p1_link_tag = p1_cell.find('a')
            p2_link_tag = p2_cell.find('a')
            
            p1_raw = clean_player_name(p1_cell.get_text(strip=True))
            p2_raw = clean_player_name(p2_cell.get_text(strip=True))
            
            p1_href = p1_link_tag['href'] if p1_link_tag else None
            p2_href = p2_link_tag['href'] if p2_link_tag else None

            if '/' in p1_raw or '/' in p2_raw: i += 1; continue

            if any(tp in p1_raw.lower() for tp in target_players) and any(tp in p2_raw.lower() for tp in target_players):
                odds = []
                try:
                    nums = re.findall(r'\d+\.\d+', row_text)
                    valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                    if len(valid) >= 2: odds = valid[:2]
                    else:
                        nums2 = re.findall(r'\d+\.\d+', rows[i+1].get_text())
                        valid2 = [float(x) for x in nums2 if 1.0 < float(x) < 50.0]
                        if valid and valid2: odds = [valid[0], valid2[0]]
                except: pass
                
                found.append({
                    "p1_raw": p1_raw, "p2_raw": p2_raw, "tour": clean_tournament_name(current_tour), 
                    "time": match_time_str, "odds1": odds[0] if odds else 0.0, "odds2": odds[1] if len(odds)>1 else 0.0,
                    "p1_href": p1_href, "p2_href": p2_href 
                })
                i += 2 
            else: i += 1 
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v8.1 (Stable Baseline + AI Fix) Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await update_past_results(browser)
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            
            if not players: return

            current_date = datetime.now()
            player_names = [p['last_name'] for p in players]
            
            for day_offset in range(11): 
                target_date = current_date + timedelta(days=day_offset)
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue

                matches = parse_matches_locally_v5(html, player_names)
                log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    try:
                        p1_obj = find_player_safe(m['p1_raw'], players)
                        p2_obj = find_player_safe(m['p2_raw'], players)
                        
                        if p1_obj and p2_obj:
                            m_odds1 = m['odds1']; m_odds2 = m['odds2']
                            iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                            # --- UPDATED DB LOGIC: UPSERT OR UPDATE ---
                            # Wir prüfen nicht nur ob es existiert, sondern aktualisieren die Odds
                            
                            s1 = all_skills.get(p1_obj['id'], {})
                            s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                            
                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                            
                            surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
                            surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)

                            f1_data = await fetch_player_form_hybrid(browser, p1_obj['last_name'])
                            f2_data = await fetch_player_form_hybrid(browser, p2_obj['last_name'])
                            
                            elo_surf = 'Hard'
                            if 'clay' in surf.lower(): elo_surf = 'Clay'
                            elif 'grass' in surf.lower(): elo_surf = 'Grass'
                            elo1_val = ELO_CACHE.get("ATP", {}).get(p1_obj['last_name'].lower(), {}).get(elo_surf, 1500)
                            elo2_val = ELO_CACHE.get("ATP", {}).get(p2_obj['last_name'].lower(), {}).get(elo_surf, 1500)

                            # CALL MODIFIED AI
                            ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, elo1_val, elo2_val, f1_data, f2_data)
                            
                            prob_p1 = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta, m_odds1, m_odds2, surf_rate1, surf_rate2)
                            
                            entry = {
                                "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                                "odds1": m_odds1, "odds2": m_odds2,
                                "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0.01 else 99,
                                "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 0.99 else 99,
                                "ai_analysis_text": ai_meta.get('ai_text', 'No analysis'),
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": iso_timestamp 
                            }
                            
                            # UPDATED UPSERT LOGIC
                            existing = supabase.table("market_odds").select("id").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                            
                            if existing.data and len(existing.data) > 0:
                                match_id = existing.data[0]['id']
                                supabase.table("market_odds").update(entry).eq("id", match_id).execute()
                                log(f"🔄 Updated: {entry['player1_name']} vs {entry['player2_name']} (New Odds: {m_odds1}/{m_odds2})")
                            else:
                                supabase.table("market_odds").insert(entry).execute()
                                log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']} (BSI: {bsi})")

                    except Exception as e:
                        log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
