# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import numpy as np 
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("NeuralScout")

def log(msg):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V92.1 - Clean Input Protocol)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.0-flash-exp' 

ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# HELPER FUNCTIONS (Sanitization & Matching)
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text): 
    # Normalisiert Text: entfernt Akzente, lowercase
    if not text: return ""
    norm = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return norm.lower()

def clean_player_name(raw): 
    if not raw: return ""
    raw = re.sub(r'\(.*?\)', '', raw) 
    raw = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    return raw.replace('|', '').strip()

def clean_tournament_name(raw):
    """
    Entfernt den Tabellen-Müll (S 1 2 3 H2H...) aus dem Turniernamen.
    Input: 'United CupS12345H2HHA'
    Output: 'United Cup'
    """
    if not raw: return ""
    # Entferne alles ab dem Muster 'S' gefolgt von Zahlen (Sets Headers)
    # TennisExplorer Header sind oft: Tournament Name + "S 1 2 3..."
    clean = re.sub(r'S\s*\d.*', '', raw) 
    # Falls das 'S' Pattern nicht greift, versuche H2H oder Bwin noise
    clean = re.sub(r'H2H.*', '', clean)
    return clean.strip()

def get_last_name(full_name):
    if not full_name: return ""
    clean = re.sub(r'^\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

def get_name_tokens(full_name):
    if not full_name: return set()
    clean = normalize_text(full_name)
    clean = re.sub(r'[^\w\s]', '', clean)
    parts = clean.split()
    tokens = set()
    for p in parts:
        if len(p) > 2 or (len(parts) == 1): 
            tokens.add(p)
    return tokens

def names_match(db_name_tokens, scraped_text):
    scraped_clean = normalize_text(scraped_text)
    for token in db_name_tokens:
        if re.search(r'\b' + re.escape(token) + r'\b', scraped_clean):
            return True
    return False

def sanitize_timestamp(dirty_ts):
    if not dirty_ts: return None
    try:
        return datetime.fromisoformat(dirty_ts.replace('Z', '+00:00'))
    except:
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})', dirty_ts)
        if match:
            clean_str = match.group(1)
            return datetime.fromisoformat(clean_str + ":00+00:00")
        return None

# =================================================================
# GEMINI ENGINE
# =================================================================
async def call_gemini(prompt):
    await asyncio.sleep(0.5) 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200: return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except: return None

# =================================================================
# DATA FETCHING
# =================================================================
async def fetch_elo_ratings():
    log("📊 Lade Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font"] else route.continue_())
        
        for tour, url in urls.items():
            try:
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
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
                await page.close()
            except Exception as e:
                log(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
        await browser.close()

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
                    'mental': to_float(entry.get('mental')), 'volley': to_float(entry.get('volley'))
                }
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# INTELLIGENT COURT RESOLUTION
# =================================================================
async def resolve_venue_with_ai(p1, p2, tour_name, candidates):
    """Fragt AI, welcher Court für dieses Matchup genutzt wird."""
    cache_key = f"{tour_name}_{p1}_{p2}"
    if cache_key in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[cache_key]

    candidates_str = "\n".join([f"- ID: {c['id']}, Name: {c['name']}, City: {c.get('city', 'Unknown')}" for c in candidates])
    
    prompt = f"""
    ROLE: Tennis Data Engineer. TASK: Identify specific venue.
    MATCH: {p1} vs {p2} | TOURNAMENT: {tour_name}
    CANDIDATES:
    {candidates_str}
    INSTRUCTION: Pick the ID where this match is most likely played based on players/groups.
    RETURN JSON: {{ "selected_id": "uuid" }}
    """
    try:
        res = await call_gemini(prompt)
        if not res: return candidates[0]
        data = json.loads(res.replace("```json", "").replace("```", "").strip())
        best_match = next((c for c in candidates if str(c['id']) == str(data.get("selected_id"))), candidates[0])
        TOURNAMENT_LOC_CACHE[cache_key] = best_match
        return best_match
    except: return candidates[0]

async def find_best_court_match_smart(scraped_tour_name, db_tours, p1_name, p2_name):
    """Findet den exakten Court (beachtet Multi-Venues wie United Cup)."""
    # 1. CLEANING: Entferne "S123..." Müll
    s_clean = clean_tournament_name(scraped_tour_name)
    s_low = s_clean.lower()
    
    # 2. Broad Search
    candidates = []
    for t in db_tours:
        db_name = t['name'].lower()
        # Prüfen ob cleaned Name im DB-Namen ist ODER umgekehrt
        if s_low in db_name or db_name in s_low:
            candidates.append(t)
    
    selected_court = None
    if not candidates:
        log(f"   ⚠️ Unknown Tournament: '{s_clean}' (Raw: '{scraped_tour_name}'). Using fallback.")
        surface = "Hard"
        if "clay" in s_low: surface = "Red Clay"
        elif "grass" in s_low: surface = "Grass"
        elif "indoor" in s_low: surface = "Indoor Hard"
        return surface, 5.0, "Generic Fallback"

    elif len(candidates) == 1:
        selected_court = candidates[0]
    else:
        # 3. AI Disambiguation (z.B. United Cup Perth vs Sydney)
        log(f"   🤔 Ambiguous Venue for '{s_clean}' ({len(candidates)} candidates). Asking AI...")
        selected_court = await resolve_venue_with_ai(p1_name, p2_name, s_clean, candidates)
        log(f"   🤖 AI Selected: {selected_court['name']}")

    bsi = to_float(selected_court.get('bsi_rating'), 5.0)
    surf = selected_court.get('surface', 'Hard')
    return surf, bsi, f"DB: {selected_court['name']}"

# =================================================================
# SCIENTIFIC FAIR ODDS ENGINE
# =================================================================
def solve_shin_probabilities(o1, o2):
    if o1 <= 1 or o2 <= 1: return 0.5, 0.5
    ip1, ip2 = 1.0/o1, 1.0/o2
    if ip1 + ip2 <= 1.0: return ip1/(ip1+ip2), ip2/(ip1+ip2)
    
    low, high, k = 1.0, 10.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        s = pow(ip1, mid) + pow(ip2, mid)
        if abs(s - 1.0) < 1e-6: k = mid; break
        if s > 1.0: low = mid
        else: high = mid
            
    p1, p2 = pow(ip1, k), pow(ip2, k)
    s_final = p1 + p2
    return p1 / s_final, p2 / s_final

def sigmoid(x, k=1.0): return 1 / (1 + math.exp(-k * x))

def get_dynamic_court_weights(bsi, surface):
    bsi = float(bsi)
    w = {'serve': 1.0, 'power': 1.0, 'rally': 1.0, 'movement': 1.0, 'mental': 0.8, 'volley': 0.5}
    if bsi >= 7.0: 
        speed_factor = (bsi - 5.0) * 0.35 
        w['serve'] += speed_factor * 1.5; w['power'] += speed_factor * 1.2; w['volley'] += speed_factor * 1.0 
        w['rally'] -= speed_factor * 0.5; w['movement'] -= speed_factor * 0.3 
    elif bsi <= 4.0: 
        slow_factor = (5.0 - bsi) * 0.4
        w['serve'] -= slow_factor * 0.8; w['power'] -= slow_factor * 0.5 
        w['rally'] += slow_factor * 1.2; w['movement'] += slow_factor * 1.5; w['volley'] -= slow_factor * 0.5
    return w

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    n1 = p1_name.lower().split()[-1]; n2 = p2_name.lower().split()[-1]
    tour = "ATP"
    bsi_val = to_float(bsi, 5.0)
    weights = get_dynamic_court_weights(bsi_val, surface)
    
    def get_player_score(skills):
        if not skills: return 50.0 
        score_serve = (skills.get('serve', 50) * 0.7 + skills.get('power', 50) * 0.3) * weights['serve']
        score_rally = (skills.get('forehand', 50) + skills.get('backhand', 50)) / 2 * weights['rally']
        score_move  = (skills.get('speed', 50) * 0.6 + skills.get('stamina', 50) * 0.4) * weights['movement']
        score_net   = skills.get('volley', 50) * weights['volley']
        score_ment  = skills.get('mental', 50) * weights['mental']
        total_weight = sum(weights.values())
        return (score_serve + score_rally + score_move + score_net + score_ment) / (total_weight / 3.5)

    p1_score = get_player_score(s1)
    p2_score = get_player_score(s2)
    prob_physics = sigmoid((p1_score - p2_score) / 12.0)

    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_tactical = 0.5 + ((m1 - m2) * 0.15)

    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    if market_odds1 > 1 and market_odds2 > 1:
        prob_market_p1, _ = solve_shin_probabilities(market_odds1, market_odds2)
        overround = (1/market_odds1 + 1/market_odds2) - 1.0
    else:
        prob_market_p1 = 0.5; overround = 0.05

    w_market = 0.35; w_elo = 0.20; w_phys = 0.30; w_ai = 0.15
    if overround < 0.04: w_market = 0.45; w_elo = 0.15; w_phys = 0.25
    elif overround > 0.08: w_market = 0.20; w_elo = 0.25; w_phys = 0.40

    total_w = w_market + w_elo + w_phys + w_ai
    raw_prob = ((prob_market_p1 * w_market) + (prob_elo * w_elo) + (prob_physics * w_phys) + (prob_tactical * w_ai)) / total_w

    if raw_prob > 0.5: final_prob = raw_prob - (raw_prob - 0.5) * 0.05 
    else: final_prob = raw_prob + (0.5 - raw_prob) * 0.05
    return final_prob

# =================================================================
# SCAPING & VERIFICATION (V92.1 Logic)
# =================================================================
async def scrape_single_date(browser, target_date):
    try:
        context = await browser.new_context()
        await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_())
        page = await context.new_page()
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        content = await page.content()
        await page.close(); await context.close()
        return (target_date, content)
    except Exception as e:
        log(f"⚠️ Scrape Fail {target_date.strftime('%d.%m')}: {e}")
        return (target_date, None)

def parse_matches_locally(html, p_names):
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []): 
                # CLEAN THE TOURNAMENT NAME HERE
                raw_tour = row.get_text(strip=True)
                current_tour = clean_tournament_name(raw_tour)
                continue
            
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match = re.search(r'(\d{2}:\d{2})', first_col.get_text(strip=True))
                if match: match_time_str = match.group(1)

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                if any(tp in p1_raw.lower() for tp in target_players) and any(tp in p2_raw.lower() for tp in target_players):
                    odds = []
                    try:
                        def extract_odds(r):
                            found_o = []
                            for c in r.find_all('td'):
                                txt = c.get_text(strip=True)
                                if (re.match(r'^\d+\.\d{2}$', txt)) and 1.0 < float(txt) < 50.0:
                                    found_o.append(float(txt))
                            return found_o
                        
                        row1_odds = extract_odds(row)
                        row2_odds = extract_odds(rows[i+1])
                        if row1_odds and row2_odds: odds = [row1_odds[0], row2_odds[0]]
                        else:
                            nums = re.findall(r'\d+\.\d{2}', row_text + " " + rows[i+1].get_text())
                            valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                            if len(valid) >= 2: odds = valid[:2]
                    except: pass
                    
                    found.append({"p1": p1_raw, "p2": p2_raw, "tour": current_tour, "time": match_time_str, "odds1": odds[0] if odds else 0.0, "odds2": odds[1] if len(odds)>1 else 0.0})
    return found

async def update_past_results():
    log("🏆 Checking for Results (V92 Token-Match)...")
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending: return

    matches_by_date = defaultdict(list)
    for pm in pending:
        ts = sanitize_timestamp(pm.get('match_time'))
        if ts: matches_by_date[(ts.year, ts.month, ts.day)].append(pm)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for (y, m, d), match_list in matches_by_date.items():
            page = await browser.new_page()
            try:
                await page.goto(f"https://www.tennisexplorer.com/results/?type=all&year={y}&month={m}&day={d}", wait_until="domcontentloaded", timeout=60000)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                rows = []
                for t in soup.find_all('table', class_='result'): rows.extend(t.find_all('tr'))
                
                processed = set()
                i = 0
                while i < len(rows) - 1:
                    r1, r2 = rows[i], rows[i+1]
                    if 'head' in r1.get('class', []) or 'bott' in r1.get('class', []): i+=1; continue
                    
                    t1 = normalize_text(r1.get_text(separator=" ", strip=True))
                    t2 = normalize_text(r2.get_text(separator=" ", strip=True))
                    
                    matched = None
                    for pm in match_list:
                        if pm['id'] in processed: continue
                        tok1 = get_name_tokens(pm['player1_name'])
                        tok2 = get_name_tokens(pm['player2_name'])
                        
                        match_a = names_match(tok1, t1) and names_match(tok2, t2)
                        match_b = names_match(tok2, t1) and names_match(tok1, t2)
                        
                        if match_a or match_b:
                            matched = pm; break
                    
                    if matched:
                        c1 = r1.find_all('td'); c2 = r2.find_all('td')
                        winner = None
                        
                        def get_sets(cols):
                            for c in cols:
                                txt = c.get_text(strip=True)
                                if txt.isdigit():
                                    v = int(txt)
                                    if v <= 3: return v
                            return -1
                        
                        s1 = get_sets(c1[2:]) if len(c1)>2 else get_sets(c1)
                        s2 = get_sets(c2[2:]) if len(c2)>2 else get_sets(c2)
                        
                        if s1 != -1 and s2 != -1 and s1 != s2:
                            r1_wins = s1 > s2
                            tok_p1 = get_name_tokens(matched['player1_name'])
                            row1_is_p1 = names_match(tok_p1, t1)
                            
                            if r1_wins: winner = matched['player1_name'] if row1_is_p1 else matched['player2_name']
                            else: winner = matched['player2_name'] if row1_is_p1 else matched['player1_name']
                        
                        if not winner and ("ret." in t1 or "ret." in t2):
                            tok_p1 = get_name_tokens(matched['player1_name'])
                            row1_is_p1 = names_match(tok_p1, t1)
                            if "ret." in t2: winner = matched['player1_name'] if row1_is_p1 else matched['player2_name']
                            elif "ret." in t1: winner = matched['player2_name'] if row1_is_p1 else matched['player1_name']

                        if winner:
                            log(f"   ✅ Result: {winner} ({s1}-{s2})")
                            supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", matched['id']).execute()
                            processed.add(matched['id'])
                        i+=2
                    else: i+=1
            except: pass
            await page.close()
        await browser.close()

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}) - {notes}. 
    METRICS (0-10): TACTICAL, FORM.
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "ai_text": "Brief nuance check" }}
    """
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5}
    if not res: return d
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return d

async def run_pipeline():
    log(f"🚀 Neural Scout v92.1 (Cleaned) Starting...")
    await update_past_results() 
    await fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    player_names = [p['last_name'] for p in players]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--disable-gpu', '--no-sandbox'])
        tasks = [scrape_single_date(browser, current_date + timedelta(days=i)) for i in range(14)]
        scraped_results = await asyncio.gather(*tasks)
        await browser.close()

    for target_date, html in scraped_results:
        if not html: continue
        matches = parse_matches_locally(html, player_names)
        log(f"🔍 {target_date.strftime('%d.%m')}: {len(matches)} Matches")
        
        for m in matches:
            try:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    m_odds1 = m['odds1']; m_odds2 = m['odds2']
                    iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                    existing = supabase.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                    
                    if existing.data:
                        match_data = existing.data[0]
                        if match_data.get('actual_winner_name'): continue 
                        supabase.table("market_odds").update({ "odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp }).eq("id", match_data['id']).execute()
                        continue

                    if m_odds1 <= 1.01: continue
                    
                    s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {}); r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    prob_p1 = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta, m_odds1, m_odds2)
                    
                    entry = {
                        "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                        "odds1": m_odds1, "odds2": m_odds2,
                        "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0.01 else 99,
                        "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 0.99 else 99,
                        "ai_analysis_text": ai_meta.get('ai_text', 'No analysis'),
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "match_time": iso_timestamp 
                    }
                    supabase.table("market_odds").insert(entry).execute()
                    log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']} (BSI: {bsi})")

            except Exception as e: log(f"⚠️ Match Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
