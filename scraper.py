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
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION & LOGGING
# =================================================================
# Konfiguriere Logging für Docker/Cloud Logs
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("NeuralScout")

def log(msg):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V82.0 - Result Fixing Engine)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.0-flash-exp' # Upgrade to faster model if available, else 'gemini-1.5-pro'

ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# HELPER FUNCTIONS
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text): 
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn') if text else ""

def clean_player_name(raw): 
    """Entfernt Bookie-Namen und Müll aus Spielernamen."""
    if not raw: return ""
    # Entferne typische Störwörter
    raw = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    # Entferne alles in Klammern (oft Seed oder Land)
    raw = re.sub(r'\(.*?\)', '', raw)
    return raw.replace('|', '').strip()

def get_last_name(full_name):
    """Extrahiert den Nachnamen (lowercase) für robusten Vergleich."""
    if not full_name: return ""
    # Entferne Initiale wie "A." am Anfang
    clean = re.sub(r'^\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# GEMINI ENGINE
# =================================================================
async def call_gemini(prompt):
    await asyncio.sleep(0.5) # Rate Limit Protection
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200: 
                log(f"⚠️ Gemini Error {response.status_code}: {response.text}")
                return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e: 
            log(f"⚠️ Gemini Exception: {e}")
            return None

# =================================================================
# CORE LOGIC
# =================================================================
async def fetch_elo_ratings():
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    async with async_playwright() as p:
        # Launch browser ONCE specifically for ELO
        browser = await p.chromium.launch(headless=True)
        for tour, url in urls.items():
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
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
        # Pagination handling omitted for brevity, assuming <1000 active items or adjusting range
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
# QUANTITATIVE FAIR ODDS ENGINE V81.0 (Silicon Valley Grade)
# =================================================================

def sigmoid(x, k=1.0):
    return 1 / (1 + math.exp(-k * x))

def get_dynamic_court_weights(bsi, surface):
    bsi = float(bsi)
    w = {
        'serve': 1.0, 'power': 1.0, 
        'rally': 1.0, 'movement': 1.0, 
        'mental': 0.8, 'volley': 0.5
    }

    if bsi >= 7.0: # Fast
        speed_factor = (bsi - 5.0) * 0.35 
        w['serve'] += speed_factor * 1.5
        w['power'] += speed_factor * 1.2
        w['volley'] += speed_factor * 1.0 
        w['rally'] -= speed_factor * 0.5 
        w['movement'] -= speed_factor * 0.3 

    elif bsi <= 4.0: # Slow
        slow_factor = (5.0 - bsi) * 0.4
        w['serve'] -= slow_factor * 0.8
        w['power'] -= slow_factor * 0.5 
        w['rally'] += slow_factor * 1.2 
        w['movement'] += slow_factor * 1.5 
        w['volley'] -= slow_factor * 0.5

    return w

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
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

    p1_phys_score = get_player_score(s1)
    p2_phys_score = get_player_score(s2)
    
    phys_diff = (p1_phys_score - p2_phys_score) / 12.0
    prob_physics = sigmoid(phys_diff)

    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    tactical_diff = (m1 - m2) * 0.15 
    prob_tactical = 0.5 + tactical_diff

    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
        
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        margin = inv1 + inv2
        prob_market = inv1 / margin
    else:
        prob_market = 0.5

    w_market = 0.35
    w_elo    = 0.20
    w_phys   = 0.30
    w_ai     = 0.15

    raw_prob = (prob_market * w_market) + (prob_elo * w_elo) + (prob_physics * w_phys) + (prob_tactical * w_ai)

    if raw_prob > 0.5:
        final_prob = raw_prob - (raw_prob - 0.5) * 0.05
    else:
        final_prob = raw_prob + (0.5 - raw_prob) * 0.05

    return final_prob

# =================================================================
# RESULT VERIFICATION ENGINE (Fixed for TennisExplorer Structure)
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Structured Table Parser)...")
    
    # 1. Fetch Matches from DB needing results
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    
    # Filter for matches older than 2 hours to ensure result might exist
    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending_matches:
        try:
            created_at = datetime.fromisoformat(pm['match_time'].replace('Z', '+00:00'))
            # Wenn Match vor mehr als 3 Stunden war, sollte ein Ergebnis da sein
            if (now_utc - created_at).total_seconds() > (3 * 3600): 
                safe_matches.append(pm)
        except Exception as e:
            # Fallback falls parsing fails
            continue

    if not safe_matches:
        log("   ✅ No pending matches ready for result check.")
        return

    log(f"   🔎 Scanning results for {len(safe_matches)} matches...")

    # Performance Optimization: Launch Browser ONCE
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Wir checken die letzten 3 Tage
        for day_offset in range(4): 
            target_date = datetime.now() - timedelta(days=day_offset)
            page = await browser.new_page()
            
            try:
                url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                log(f"   📅 Fetching Results: {target_date.strftime('%Y-%m-%d')}")
                
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Finde alle Result Tables
                tables = soup.find_all('table', class_='result')
                
                for table in tables:
                    rows = table.find_all('tr')
                    
                    # Iterate rows. In TennisExplorer, matches are often 2 rows (one per player)
                    # Or single row with data. Structure: 
                    # Row 1: Player A | Score (S) | Set1 | Set2 ...
                    # Row 2: Player B | Score (S) | Set1 | Set2 ...
                    
                    i = 0
                    while i < len(rows) - 1:
                        row1 = rows[i]
                        row2 = rows[i+1]
                        i += 1 # Advance basic, inside logic we might jump another

                        # Skip Headers / Flags
                        if 'flags' in str(row1) or 'head' in str(row1) or 'bott' in str(row1):
                            continue
                        
                        cols1 = row1.find_all('td')
                        cols2 = row2.find_all('td')

                        # Struktur Check: Wir brauchen Namen und Scores
                        # Normalerweise: col[0]=?, col[1]=Name, col[2]=Sets(S), col[3]=Set1...
                        if len(cols1) < 3 or len(cols2) < 3:
                            continue

                        p1_text = normalize_text(row1.get_text(separator=" ")).lower()
                        p2_text = normalize_text(row2.get_text(separator=" ")).lower()
                        
                        # Match Finder Logic
                        matched_db_entry = None
                        for pm in safe_matches:
                            last1 = get_last_name(pm['player1_name'])
                            last2 = get_last_name(pm['player2_name'])
                            
                            # Prüfen ob beide Namen in den beiden Zeilen vorkommen
                            if (last1 in p1_text and last2 in p2_text) or (last2 in p1_text and last1 in p2_text):
                                matched_db_entry = pm
                                break
                        
                        if matched_db_entry:
                            # 🎯 MATCH FOUND - EXTRACT WINNER
                            try:
                                # TennisExplorer hat oft die Klasse 'score' oder man nimmt den fixen Index
                                # Column Index 2 ist oft "S" (Sets) in der Result View
                                # Besser: Suche nach Zelleninhalt, der rein numerisch ist und klein (0, 1, 2, 3)
                                
                                s1_val = -1
                                s2_val = -1

                                # Versuch, die Sätze aus der "S" Spalte zu lesen (oft Index 2 oder 3 nach Name)
                                # Wir iterieren rückwärts von den Set-Scores? Nein, Index basiert.
                                # Name ist meistens in td mit class "t-name"
                                
                                def get_sets_from_row(cols):
                                    for col in cols:
                                        txt = col.get_text(strip=True)
                                        # Suche nach der Spalte, die die Sätze anzeigt (S). Ist meistens fett oder einfach eine einzelne Ziffer.
                                        if txt.isdigit() and len(txt) == 1:
                                            # Wir nehmen an, die erste Einzelziffer nach dem Namen ist das Satz-Ergebnis
                                            return int(txt)
                                    return 0

                                # Retirement Check
                                is_ret = "ret." in p1_text or "ret." in p2_text or "wo." in p1_text

                                s1_val = get_sets_from_row(cols1[2:]) # Skip first 2 cols (Flag/Name often)
                                s2_val = get_sets_from_row(cols2[2:])

                                winner = None
                                
                                # Logic: Wer hat mehr Sätze?
                                if s1_val > s2_val:
                                    # Row 1 Winner
                                    # Map Row 1 name back to DB name
                                    if get_last_name(matched_db_entry['player1_name']) in p1_text:
                                        winner = matched_db_entry['player1_name']
                                    else:
                                        winner = matched_db_entry['player2_name']
                                        
                                elif s2_val > s1_val:
                                    # Row 2 Winner
                                    if get_last_name(matched_db_entry['player1_name']) in p2_text:
                                        winner = matched_db_entry['player1_name']
                                    else:
                                        winner = matched_db_entry['player2_name']
                                
                                # Fallback für Retirement, falls Score 0-0 oder unklar
                                if winner is None and is_ret:
                                    # Wer ist nicht retired?
                                    if "ret." in p2_text:
                                        if get_last_name(matched_db_entry['player1_name']) in p1_text: winner = matched_db_entry['player1_name']
                                        else: winner = matched_db_entry['player2_name']
                                    elif "ret." in p1_text:
                                        if get_last_name(matched_db_entry['player1_name']) in p2_text: winner = matched_db_entry['player1_name']
                                        else: winner = matched_db_entry['player2_name']

                                if winner:
                                    log(f"   ✅ RESULT: {winner} won against {matched_db_entry['player1_name'] if winner != matched_db_entry['player1_name'] else matched_db_entry['player2_name']}")
                                    supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", matched_db_entry['id']).execute()
                                    # Remove from safe_matches to avoid double processing
                                    safe_matches = [x for x in safe_matches if x['id'] != matched_db_entry['id']]
                                    i += 1 # Skip the second row explicitly since we processed the pair

                            except Exception as ex:
                                log(f"   ⚠️ Parsing Error for match {matched_db_entry['player1_name']}: {ex}")

            except Exception as e:
                log(f"   ❌ Page Error: {e}")
                await page.close()
            
            await page.close()
        await browser.close()

# =================================================================
# MAIN PIPELINE
# =================================================================
async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    prompt = f"TASK: Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\", \"is_indoor\": bool }}"
    res = await call_gemini(prompt)
    if not res: return None
    try: 
        data = json.loads(res.replace("```json", "").replace("```", "").strip())
        TOURNAMENT_LOC_CACHE[scraped_name] = data
        return data
    except: return None

async def find_best_court_match_smart(tour, db_tours, p1, p2):
    s_low = tour.lower().strip()
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

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}). P1 Style: {p1.get('play_style')}. P2 Style: {p2.get('play_style')}.
    METRICS (0-10): TACTICAL (25%), FORM (10%), UTR (5%).
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "..." }}
    """
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return d
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return d

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            log(f"📡 Scanning Odds: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            log(f"❌ Scrape Error: {e}")
            await browser.close()
            return None

def parse_matches_locally(html, p_names):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            # Detect Tournament Header
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                continue
            
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # Extract Time
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            if i + 1 < len(rows):
                # Ensure we are looking at match rows, not results or other info
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                
                if any(tp in p1_raw.lower() for tp in target_players) and any(tp in p2_raw.lower() for tp in target_players):
                    odds = []
                    # Robust Odds Parsing: Look for 'course' class or specific odds columns
                    try:
                        # Strategy: Find cells that contain typical odds format (float 1.01 - 50.0)
                        # Avoid '6.4' which might be a score. Odds usually have 2 decimal places in TE
                        def extract_odds_from_row(r):
                            cols = r.find_all('td')
                            found_odds = []
                            for c in cols:
                                txt = c.get_text(strip=True)
                                if 'course' in c.get('class', []) or (re.match(r'^\d+\.\d{2}$', txt)):
                                    try:
                                        val = float(txt)
                                        if 1.0 < val < 50.0: found_odds.append(val)
                                    except: pass
                            return found_odds

                        row1_odds = extract_odds_from_row(row)
                        row2_odds = extract_odds_from_row(rows[i+1])
                        
                        if row1_odds and row2_odds:
                            odds = [row1_odds[0], row2_odds[0]]
                        else:
                            # Fallback regex if classes fail
                            nums = re.findall(r'\d+\.\d{2}', row_text) # Strict 2 decimals
                            valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                            if len(valid) >= 2: odds = valid[:2]

                    except: pass
                    
                    if len(odds) >= 2:
                        found.append({
                            "p1": p1_raw, 
                            "p2": p2_raw, 
                            "tour": current_tour, 
                            "time": match_time_str, 
                            "odds1": odds[0], 
                            "odds2": odds[1]
                        })
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v82.0 Starting...")
    
    # 1. Update Results first (Close the loop)
    await update_past_results()
    
    # 2. Update Data
    await fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    player_names = [p['last_name'] for p in players]
    
    # 3. Look for Future Matches
    for day_offset in range(35): 
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        matches = parse_matches_locally(html, player_names)
        log(f"🔍 Found: {len(matches)} potential matches on {target_date.strftime('%d.%m.')}")
        
        for m in matches:
            try:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    m_odds1 = m['odds1']
                    m_odds2 = m['odds2']
                    iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                    # IMMUTABLE HISTORY CHECK
                    # We utilize Supabase OR logic to find match regardless of order
                    existing = supabase.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                    
                    # Logic: If match exists but is OLD (diff date), create new. If same date, update.
                    match_exists = False
                    if existing.data:
                        # Check if it's the same match (approx time check or just assume same event if active)
                        # For simplicity, if it exists and has no winner, we update odds.
                        match_data = existing.data[0]
                        if match_data.get('actual_winner_name'):
                            log(f"🔒 Locked: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                            continue 
                        
                        # Update odds
                        supabase.table("market_odds").update({ 
                            "odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp 
                        }).eq("id", match_data['id']).execute()
                        match_exists = True

                    if match_exists: continue
                    if m_odds1 <= 1.01: continue
                    
                    # NEW MATCH CALCULATION
                    log(f"✨ New Match Analysis: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                    s1 = all_skills.get(p1_obj['id'], {})
                    s2 = all_skills.get(p2_obj['id'], {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    prob_p1 = calculate_physics_fair_odds(
                        p1_obj['last_name'], p2_obj['last_name'], 
                        s1, s2, bsi, surf, ai_meta, 
                        m_odds1, m_odds2
                    )
                    
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
                    log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']}")

            except Exception as e:
                log(f"⚠️ Loop Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
