# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION & LOGGING
# =================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("🔌 Initialisiere Neural Scout (V104.0 - Identity Fingerprinting)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.5-pro' 

ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# --- V100.0: THE GLOBAL TRUTH MAP ---
COUNTRY_TO_CITY_MAP = {}

# DB Mapping
CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena"
}

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
    # Behalte Initialen wie "B." oder "Billy", entferne nur Wett-Müll
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def clean_tournament_name(raw):
    if not raw: return "Unknown"
    clean = re.sub(r'S\d+[A-Z0-9]*$', '', raw).strip()
    return clean

def get_last_name(full_name):
    """Extrahiert den Nachnamen (ignoriert Initialen am Anfang/Ende)."""
    if not full_name: return ""
    # Entferne "B." am Anfang oder Ende, um den reinen Nachnamen zu finden
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name) # Remove "B. "
    clean = re.sub(r'\s+[A-Z]\.$', '', clean)      # Remove " B."
    parts = clean.strip().split()
    return parts[-1].lower() if parts else ""

# =================================================================
# GEMINI ENGINE
# =================================================================
async def call_gemini(prompt, model=MODEL_NAME):
    await asyncio.sleep(0.5) 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
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
# CORE LOGIC
# =================================================================
async def fetch_elo_ratings():
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for tour, url in urls.items():
            try:
                page = await browser.new_page()
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
                    'mental': to_float(entry.get('mental'))
                }
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# MATH CORE V7
# =================================================================
def sigmoid_prob(diff, sensitivity=0.1):
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    bsi_val = to_float(bsi, 6.0)

    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8) 

    c1_score = 0; c2_score = 0
    if bsi_val <= 4.0:
        c1_score = s1.get('stamina',50) + s1.get('speed',50) + s1.get('mental',50)
        c2_score = s2.get('stamina',50) + s2.get('speed',50) + s2.get('mental',50)
    elif bsi_val >= 7.5:
        c1_score = s1.get('serve',50) + s1.get('power',50)
        c2_score = s2.get('serve',50) + s2.get('power',50)
    else:
        c1_score = sum(s1.values())
        c2_score = sum(s2.values())
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)

    score_p1 = sum(s1.values())
    score_p2 = sum(s2.values())
    prob_skills = sigmoid_prob(score_p1 - score_p2, sensitivity=0.08)

    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    prob_alpha = (prob_matchup * 0.50) + (prob_bsi * 0.20) + (prob_skills * 0.15) + (prob_elo * 0.15)

    if prob_alpha > 0.60:
        prob_alpha = min(prob_alpha * 1.10, 0.92)
    elif prob_alpha < 0.40:
        prob_alpha = max(prob_alpha * 0.90, 0.08)

    prob_market = 0.5 
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
      
    final_prob = (prob_alpha * 0.75) + (prob_market * 0.25)
    return final_prob

# =================================================================
# RESULT VERIFICATION ENGINE
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Deep Scan V6)...")
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending_matches:
        log("   ✅ No pending matches to verify.")
        return

    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending_matches:
        try:
            created_at_str = pm['created_at'].replace('Z', '+00:00')
            created_at = datetime.fromisoformat(created_at_str)
            if (now_utc - created_at).total_seconds() / 60 > 65: 
                safe_matches.append(pm)
        except: continue

    if not safe_matches:
        log("   ⏳ Waiting for matches to finish (Time-Lock active)...")
        return

    for day_offset in range(3): 
        target_date = datetime.now() - timedelta(days=day_offset)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                content = await page.content()
                await browser.close()
                
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

                            except Exception: pass
            except Exception: await browser.close()

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

# --- NEW: STRICT IDENTITY RESOLUTION V104.0 ---
def resolve_player_identity(scraped_name, db_players):
    """
    Identifiziert den korrekten Spieler in der DB.
    Verhindert "Billy Harris" vs "Lloyd Harris" Fehler durch strikte Checks.
    """
    # 1. Clean & Normalize Scraped Name
    # scraped_name could be "Harris B.", "B. Harris", "Billy Harris", "Harris"
    
    scraped_clean = clean_player_name(scraped_name).lower()
    
    # Extract Last Name only for fast filtering
    # "harris b." -> "harris"
    last_name_only = get_last_name(scraped_clean)
    
    # 2. Find Candidates (Filter by last name)
    candidates = [p for p in db_players if get_last_name(p['last_name']).lower() == last_name_only]
    
    if not candidates:
        # Fallback: Fuzzy search if exact last name match fails
        candidates = [p for p in db_players if p['last_name'].lower() in scraped_clean]
        
    if not candidates:
        return None
        
    if len(candidates) == 1:
        return candidates[0]
        
    # 3. DISAMBIGUATION (The Silicon Valley Logic)
    # We have multiple players with same last name (e.g. Harris).
    # We must check if the First Name or Initial matches.
    
    for cand in candidates:
        first_name = cand['first_name'].lower()
        if not first_name: continue
        
        first_initial = first_name[0]
        
        # Case A: Full First Name Match ("Billy Harris" -> "Billy")
        if first_name in scraped_clean:
            return cand
            
        # Case B: Initial Match ("Harris B." -> "B")
        # Check for "b " or " b" or "b." patterns
        # Note: We need to be careful not to match the last letter of surname as initial
        
        # Check specific initial patterns in scraped name
        if f" {first_initial}." in scraped_clean or \
           f"{first_initial}. " in scraped_clean or \
           scraped_clean.endswith(f" {first_initial}") or \
           scraped_clean.startswith(f"{first_initial} "):
            return cand
            
    # 4. Fallback: If no specific initial found in scraped string, use Rank/Popularity logic?
    # For now, we LOG WARNING and skip to avoid bad data.
    # OR we pick the one with most DB entries (if available).
    # Safe approach: Return None if ambiguous.
    # Aggressive approach: Return candidates[0] (usually the more famous one if sorted).
    
    # Let's try to match Lloyd vs Billy specifically if Harris
    if "harris" in last_name_only:
        # If scraped is just "Harris", it's usually Lloyd (ATP level).
        # If it's "Harris B.", we would have caught it above.
        for cand in candidates:
            if "lloyd" in cand['first_name'].lower():
                return cand
                
    log(f"      ⚠️ Ambiguous Player '{scraped_name}'. Candidates: {[c['first_name'] for c in candidates]}. Skipping to be safe.")
    return None

async def build_country_city_map():
    if COUNTRY_TO_CITY_MAP: return 
    
    url = "https://www.unitedcup.com/en/scores/group-standings"
    log("   📥 Building United Cup Knowledge Graph...")
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=20000, wait_until="networkidle")
            await page.wait_for_timeout(3000) 
            text = await page.inner_text("body")
            await browser.close()
            
            prompt = f"""
            TASK: Map Countries to Host Cities.
            TEXT: {text[:20000]}
            RULES: 
            - Group A, Group C, Group E -> Played in PERTH.
            - Group B, Group D, Group F -> Played in SYDNEY.
            
            OUTPUT JSON: {{ "Germany": "Perth", ... }}
            """
            res = await call_gemini(prompt)
            if res:
                data = json.loads(res.replace("```json", "").replace("```", "").strip())
                COUNTRY_TO_CITY_MAP.update(data)
                log(f"      ✅ Mapped {len(COUNTRY_TO_CITY_MAP)} Countries.")
        except Exception: pass

async def resolve_venue_by_country(p1):
    await build_country_city_map()
    
    cache_key = f"COUNTRY_{p1}"
    if cache_key in TOURNAMENT_LOC_CACHE:
        country = TOURNAMENT_LOC_CACHE[cache_key]
    else:
        prompt = f"What country does tennis player '{p1}' represent? Output JSON: {{ \"country\": \"Name\" }}"
        res = await call_gemini(prompt)
        country = "Unknown"
        if res:
            try:
                country = json.loads(res.replace("```json", "").replace("```", "").strip()).get("country")
                TOURNAMENT_LOC_CACHE[cache_key] = country
            except: pass
            
    for map_country, map_city in COUNTRY_TO_CITY_MAP.items():
        if country.lower() in map_country.lower() or map_country.lower() in country.lower():
            return CITY_TO_DB_STRING.get(map_city)

    return "Ken Rosewall Arena" # Fallback

async def find_best_court_match_smart(tour, db_tours, p1, p2):
    s_low = clean_tournament_name(tour).lower().strip()
    
    if "united cup" in s_low:
        arena_target = await resolve_venue_by_country(p1)
        if arena_target:
            for t in db_tours:
                if "united cup" in t['name'].lower() and arena_target.lower() in t.get('location', '').lower():
                    return t['surface'], t['bsi_rating'], f"United Cup ({arena_target})"
        return "Hard Court Outdoor", 8.0, "United Cup (Default)"

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
    CTX: {surface} (BSI {bsi}).
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
            log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            content = await page.content()
            await browser.close()
            return content
        except:
            await browser.close()
            return None

def parse_matches_locally(html, p_names):
    # Dumb parser -> Grabs everything, filtering happens in main loop via resolve_identity
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                continue
            
            if "doubles" in current_tour.lower(): continue

            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                raw_time = first_col.get_text(strip=True)
                time_match = re.search(r'(\d{1,2}:\d{2})', raw_time)
                if time_match: match_time_str = time_match.group(1).zfill(5) 

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))

                if '/' in p1_raw or '/' in p2_raw: continue

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
                    "p1": p1_raw, 
                    "p2": p2_raw, 
                    "tour": clean_tournament_name(current_tour), 
                    "time": match_time_str, 
                    "odds1": odds[0] if odds else 0.0, 
                    "odds2": odds[1] if len(odds)>1 else 0.0
                })
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v104.0 (Identity Fingerprinting) Starting...")
    await update_past_results()
    await fetch_elo_ratings()
    
    await build_country_city_map()
    
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    
    for day_offset in range(11): 
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        matches = parse_matches_locally(html, []) # Empty list -> Get ALL matches
        log(f"🔍 Gefunden: {len(matches)} Raw Matches am {target_date.strftime('%d.%m.')}")
        
        for m in matches:
            try:
                # --- V104.0 IDENTITY CHECK ---
                p1_obj = resolve_player_identity(m['p1'], players)
                p2_obj = resolve_player_identity(m['p2'], players)
                
                if p1_obj and p2_obj:
                    m_odds1 = m['odds1']
                    m_odds2 = m['odds2']
                    iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                    existing = supabase.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                    
                    if existing.data:
                        match_data = existing.data[0]
                        match_id = match_data['id']
                        if match_data.get('actual_winner_name'):
                            continue 
                        supabase.table("market_odds").update({"odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp}).eq("id", match_id).execute()
                        log(f"🔄 Updated: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        continue

                    if m_odds1 <= 1.0: continue
                    
                    log(f"✨ New Match: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                    s1 = all_skills.get(p1_obj['id'], {})
                    s2 = all_skills.get(p2_obj['id'], {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
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

            except Exception as e:
                log(f"⚠️ Match Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
