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
from typing import List, Dict, Optional, Any

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

log("🔌 Initialisiere Neural Scout (V11.1 - Hotfix Restore)...")

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

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena"
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

def find_player_safe(scraped_name_raw: str, db_players: List[Dict]) -> Optional[Dict]:
    if not scraped_name_raw or not db_players: return None
    clean_scrape = clean_player_name(scraped_name_raw).lower()
    candidates = []
    
    for p in db_players:
        if not isinstance(p, dict): continue
        if p.get('last_name', '').lower() in clean_scrape:
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
    await asyncio.sleep(1.0) # Rate limiting respect
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
                log(f"⚠️ Gemini API Error: {response.status_code}")
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
        
        total_matches = 0; total_wins = 0
        tables = soup.find_all('table', class_='result')
        
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
                                    total_wins = w
                                    total_matches = w + l
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
                        ELO_CACHE[tour][name] = {
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
# 5. MATH CORE
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2):
    if not isinstance(ai_meta, dict): ai_meta = {}
    n1 = p1_name.lower().split()[-1]; n2 = p2_name.lower().split()[-1]
    tour = "ATP"; bsi_val = to_float(bsi, 6.0)

    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    p1_hard = p1_stats.get('Hard', 1500); p1_clay = p1_stats.get('Clay', 1500)
    p2_hard = p2_stats.get('Hard', 1500); p2_clay = p2_stats.get('Clay', 1500)
    p1_clay_spec = (p1_clay - p1_hard) > 80
    p2_clay_spec = (p2_clay - p2_hard) > 80

    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8) 

    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    def get_defense(s, is_clay, bsi): 
        base = s.get('speed', 50) + s.get('stamina', 50) + s.get('mental', 50)
        return base * 0.75 if bsi > 7.0 and is_clay else base
    def get_tech(s): return s.get('forehand', 50) + s.get('backhand', 50)

    off1 = get_offense(s1); def1 = get_defense(s1, p1_clay_spec, bsi_val); tech1 = get_tech(s1)
    off2 = get_offense(s2); def2 = get_defense(s2, p2_clay_spec, bsi_val); tech2 = get_tech(s2)

    c1_score = 0; c2_score = 0
    if bsi_val < 4.0: c1_score = (def1 * 0.7) + (tech1 * 0.3); c2_score = (def2 * 0.7) + (tech2 * 0.3)
    elif 4.0 <= bsi_val < 5.5: c1_score = (def1 * 0.5) + (tech1 * 0.4) + (off1 * 0.1); c2_score = (def2 * 0.5) + (tech2 * 0.4) + (off2 * 0.1)
    elif 5.5 <= bsi_val < 7.0: c1_score = def1 + tech1 + off1; c2_score = def2 + tech2 + off2
    elif 7.0 <= bsi_val < 8.0: c1_score = (off1 * 0.5) + (tech1 * 0.3) + (def1 * 0.2); c2_score = (off2 * 0.5) + (tech2 * 0.3) + (def2 * 0.2)
    elif 8.0 <= bsi_val < 9.0: c1_score = (off1 * 0.8) + (tech1 * 0.2); c2_score = (off2 * 0.8) + (tech2 * 0.2)
    else: c1_score = off1; c2_score = off2

    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)

    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    prob_elo = 1 / (1 + 10 ** ((p2_stats.get(elo_surf, 1500) - p1_stats.get(elo_surf, 1500)) / 400))

    f1 = to_float(ai_meta.get('p1_form_score', 5)); f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    surf_diff = surf_rate1 - surf_rate2
    prob_surface_stats = max(0.1, min(0.9, 0.5 + (surf_diff * 0.9)))

    weights = [0.20, 0.20, 0.10, 0.10, 0.10, 0.30] 
    if abs(surf_diff) > 0.2: weights = [0.10, 0.20, 0.10, 0.10, 0.05, 0.45]
    if bsi_val > 7.5: weights[1] = 0.25

    total_w = sum(weights)
    weights = [w/total_w for w in weights]

    prob_alpha = (prob_matchup * weights[0]) + (prob_bsi * weights[1]) + (prob_skills * weights[2]) + \
                 (prob_elo * weights[3]) + (prob_form * weights[4]) + (prob_surface_stats * weights[5])

    if prob_alpha > 0.60: prob_alpha = min(prob_alpha * 1.10, 0.94)
    elif prob_alpha < 0.40: prob_alpha = max(prob_alpha * 0.90, 0.06)

    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
        
    return (prob_alpha * 0.75) + (prob_market * 0.25)

# =================================================================
# 6. PIPELINE UTILS & RESTORED FUNCTIONS
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
            COUNTRY_TO_CITY_MAP.update(json.loads(res.replace("json", "").replace("```", "").strip()))
    except: pass
    finally: await page.close()

async def resolve_united_cup_via_country(p1):
    if not COUNTRY_TO_CITY_MAP: return None
    cache_key = f"COUNTRY_{p1}"
    if cache_key in TOURNAMENT_LOC_CACHE: country = TOURNAMENT_LOC_CACHE[cache_key]
    else:
        res = await call_gemini(f"Country of player {p1}? JSON: {{'country': 'Name'}}")
        country = json.loads(res.replace("json", "").replace("```", "").strip()).get("country", "Unknown") if res else "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
            
    if country in COUNTRY_TO_CITY_MAP: return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
    return None

async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    res = await call_gemini(f"Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\" }}")
    if res:
        try: TOURNAMENT_LOC_CACHE[scraped_name] = json.loads(res.replace("json","").replace("```","").strip()); return TOURNAMENT_LOC_CACHE[scraped_name]
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

    for t in db_tours:
        if t['name'].lower() == s_low: return t['surface'], t['bsi_rating'], t.get('notes', '')
    
    if "clay" in s_low: return "Red Clay", 3.5, "Local"
    if "hard" in s_low: return "Hard", 6.5, "Local"
    if "indoor" in s_low: return "Indoor", 8.0, "Local"
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1, form2):
    prompt = f"""
    ROLE: Elite Tennis Analyst. MATCH: {p1['last_name']} vs {p2['last_name']} ({surface}).
    DATA: ELO: {elo1} vs {elo2}. FORM: {form1['text']} vs {form2['text']}.
    OUTPUT JSON ONLY: {{ "p1_tactical_score": 5, "p2_tactical_score": 5, "p1_form_score": 5, "p2_form_score": 5, "ai_text": "Analysis." }}
    """
    res = await call_gemini(prompt)
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': 'No intel.'}
    
    if not res: return default
    
    # CRASH FIX: LIST HANDLING
    try: 
        cleaned = res.replace("json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return data[0]
            else:
                return default
        
        if not isinstance(data, dict):
            return default
            
        return data
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
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []): current_tour = row.get_text(strip=True); i+=1; continue
            if "doubles" in current_tour.lower() or i+1 >= len(rows): i+=1; continue

            row2 = rows[i+1]
            cols1 = row.find_all('td'); cols2 = row2.find_all('td')
            if len(cols1) < 2 or len(cols2) < 1: i+=1; continue

            # Robust name extraction
            p1_cell = next((c for c in cols1 if c.find('a') and 'time' not in c.get('class', [])), cols1[1] if len(cols1)>1 else None)
            p2_cell = next((c for c in cols2 if c.find('a')), cols2[0] if len(cols2)>0 else None)
            
            if not p1_cell or not p2_cell: i+=1; continue

            p1_raw = clean_player_name(p1_cell.get_text(strip=True))
            p2_raw = clean_player_name(p2_cell.get_text(strip=True))
            
            if '/' in p1_raw or '/' in p2_raw or "canc" in (row.text + row2.text).lower(): i+=2; continue

            # Time Extraction
            m_time = "00:00"
            tc = row.find('td', class_='first')
            if tc and 'time' in tc.get('class', []):
                tm = re.search(r'(\d{1,2}:\d{2})', tc.get_text(strip=True))
                if tm: m_time = tm.group(1).zfill(5)

            p1_match = any(tp in p1_raw.lower() for tp in target_players)
            p2_match = any(tp in p2_raw.lower() for tp in target_players)

            if p1_match and p2_match:
                odds = []
                try:
                    # FIX: LAX REGEX FOR JODAR (Accepts 1.5, 12, etc)
                    txt = row.get_text(" ") + " " + row2.get_text(" ")
                    nums = [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', txt) if 1.01 <= float(x) < 50.0 and ":" not in x]
                    if len(nums) >= 2:
                        odds = [nums[-2], nums[-1]]
                except: pass
                
                found.append({
                    "p1_raw": p1_raw, "p2_raw": p2_raw, "tour": clean_tournament_name(current_tour), 
                    "time": m_time, "odds1": odds[0] if odds else 0.0, "odds2": odds[1] if len(odds)>1 else 0.0,
                    "p1_href": p1_cell.find('a')['href'] if p1_cell.find('a') else None, 
                    "p2_href": p2_cell.find('a')['href'] if p2_cell.find('a') else None 
                })
                i += 2 
            else: i += 1 
    return found

# =================================================================
# 7. MAIN LOOP & RESULT CHECKER (TIME AWARE)
# =================================================================
async def update_past_results(browser: Browser):
    log("🏆 Checking for Match Results (V11.1)...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending or not isinstance(pending, list): return

    safe = []
    now = datetime.now(timezone.utc)
    for pm in pending:
        if not isinstance(pm, dict): continue
        try:
            # FUTURE MATCH GUARD
            m_time = pm.get('match_time')
            if m_time:
                mdt = datetime.fromisoformat(m_time.replace('Z', '+00:00'))
                if mdt > (now - timedelta(hours=2)): continue # Skip Future
            else:
                cat = datetime.fromisoformat(pm['created_at'].replace('Z', '+00:00'))
                if (now - cat).total_seconds() < 3600*3: continue
            safe.append(pm)
        except: continue

    if not safe: 
        log("   💤 Keine Matches zur Auswertung.")
        return

    log(f"   🔍 Prüfe {len(safe)} Matches...")
    for day_off in range(2): 
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
                if "h2h" in rt or "head" in rt: continue

                for pm in safe:
                    p1 = get_last_name(pm['player1_name']); p2 = get_last_name(pm['player2_name'])
                    found = (p1 in rt and p2 in nrt) or (p2 in rt and p1 in nrt) or (p1 in rt and p2 in rt)
                    
                    if found:
                        def get_scores(cols):
                            s = []
                            for c in cols:
                                t = c.get_text(strip=True)
                                if ":" in t or "(" in t: continue
                                if t.isdigit() and len(t)==1 and int(t)<=7: s.append(int(t))
                            return s

                        s1 = get_scores(row.find_all('td'))
                        s2 = get_scores(next_row.find_all('td'))
                        if len(s1) < 1: continue

                        w1 = 0; w2 = 0
                        for k in range(min(len(s1), len(s2))):
                            if s1[k] > s2[k]: w1+=1
                            elif s2[k] > s1[k]: w2+=1
                        
                        winner = None
                        ret = "ret." in rt or "ret." in nrt
                        if (w1 >= 2 and w1 > w2) or (ret and w1 > w2):
                            winner = pm['player1_name'] if p1 in rt else pm['player2_name']
                        elif (w2 >= 2 and w2 > w1) or (ret and w2 > w1):
                            winner = pm['player1_name'] if p1 in nrt else pm['player2_name']
                        
                        if winner:
                            supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", pm['id']).execute()
                            safe = [x for x in safe if x['id'] != pm['id']]
                            log(f"      ✅ Winner: {winner}")
        except: pass
        finally: await page.close()

async def run_pipeline():
    log(f"🚀 Neural Scout v11.1 (Final) Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await update_past_results(browser)
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            if not players: return

            player_names = [p['last_name'] for p in players]
            
            for day_offset in range(11): 
                target_date = datetime.now() + timedelta(days=day_offset)
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue

                matches = parse_matches_locally_v5(html, player_names)
                log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    try:
                        p1_obj = find_player_safe(m['p1_raw'], players)
                        p2_obj = find_player_safe(m['p2_raw'], players)
                        
                        if p1_obj and p2_obj:
                            if m['odds1'] < 1.01 and m['odds2'] < 1.01: continue

                            res = supabase.table("market_odds").select("id, actual_winner_name, odds1").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                            existing = res.data if res else []
                            
                            db_match_id = None
                            if existing and isinstance(existing, list) and len(existing) > 0:
                                rec = existing[0]
                                if rec.get('actual_winner_name'): continue 
                                if abs(rec.get('odds1', 0) - m['odds1']) < 0.05 and rec.get('odds1', 0) > 1.1: continue
                                db_match_id = rec['id']

                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                            s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                            
                            r1 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if isinstance(r, dict) and r.get('player_id') == p2_obj['id']), {})

                            surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
                            surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)
                            f1_d = await fetch_player_form_hybrid(browser, p1_obj['last_name'])
                            f2_d = await fetch_player_form_hybrid(browser, p2_obj['last_name'])
                            
                            elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
                            e1 = ELO_CACHE.get("ATP", {}).get(p1_obj['last_name'].lower(), {}).get(elo_key, 1500)
                            e2 = ELO_CACHE.get("ATP", {}).get(p2_obj['last_name'].lower(), {}).get(elo_key, 1500)

                            ai = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, e1, e2, f1_d, f2_d)
                            prob = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai, m['odds1'], m['odds2'], surf_rate1, surf_rate2)
                            
                            data = {
                                "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                                "odds1": m['odds1'], "odds2": m['odds2'],
                                "ai_fair_odds1": round(1/prob, 2) if prob > 0.01 else 99,
                                "ai_fair_odds2": round(1/(1-prob), 2) if prob < 0.99 else 99,
                                "ai_analysis_text": ai.get('ai_text', 'No analysis'),
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                            }
                            
                            if db_match_id:
                                supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
                                log(f"🔄 Updated: {data['player1_name']} vs {data['player2_name']}")
                            else:
                                supabase.table("market_odds").insert(data).execute()
                                log(f"💾 Saved: {data['player1_name']} vs {data['player2_name']}")
                        else:
                             missing = []
                             if not p1_obj: missing.append(m['p1_raw'])
                             if not p2_obj: missing.append(m['p2_raw'])
                             log(f"   ❌ Unbekannt: {', '.join(missing)}")

                    except Exception as e: log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
