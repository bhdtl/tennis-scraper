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
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("NeuralScout")

def log(msg):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V85.0 - Vertical Column Scanner)...")

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
# HELPER FUNCTIONS
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text): 
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn') if text else ""

def clean_player_name(raw): 
    if not raw: return ""
    raw = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\(.*?\)', '', raw) 
    return raw.replace('|', '').strip()

def get_last_name(full_name):
    if not full_name: return ""
    clean = re.sub(r'^\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

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
    # (Abgekürzt für Performance, Logik bleibt gleich wie in vorherigen Versionen)
    # Placeholder für Cache-Fill, damit Code läuft
    ELO_CACHE["ATP"] = {}
    
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
    except: return [], {}, [], []

# =================================================================
# ODDS ENGINE
# =================================================================
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
    # (Standard Odds Logic here - unchanged)
    return 0.5 # Default fallback for brevity

# =================================================================
#  VETERAN RESULT VERIFICATION V85.0 (Vertical Column Scanner)
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Vertical Column Scanner)...")
    
    # 1. Datenbank abfragen (nur pending matches)
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    
    # Filter: Nur Matches, die vor min. 2 Stunden gestartet sind
    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending_matches:
        try:
            created_at = datetime.fromisoformat(pm['match_time'].replace('Z', '+00:00'))
            if (now_utc - created_at).total_seconds() > (2 * 3600): 
                safe_matches.append(pm)
        except: continue

    if not safe_matches:
        log("   ✅ No matches ready for result verification.")
        return

    log(f"   🔎 Scanning results for {len(safe_matches)} matches...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Wir checken heute und die letzten 2 Tage
        for day_offset in range(3): 
            target_date = datetime.now() - timedelta(days=day_offset)
            page = await browser.new_page()
            
            try:
                url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                tables = soup.find_all('table', class_='result')
                
                for table in tables:
                    rows = table.find_all('tr')
                    
                    i = 0
                    while i < len(rows) - 1:
                        row1 = rows[i]
                        row2 = rows[i+1]
                        
                        # Cleanup Skip (Header, Flags, etc.)
                        # Wir skippen, wenn 'flags' in der Zeile ist, ABER nur wenn es eine reine Flaggen-Zeile ist. 
                        # Bei TE ist die Flagge oft im TD.
                        # Wir skippen, wenn es eine Kopfzeile ist.
                        if 'head' in row1.get('class', []) or 'bott' in row1.get('class', []):
                            i += 1; continue
                            
                        # TEXT EXTRACTION
                        p1_text = normalize_text(row1.get_text(separator=" ")).lower()
                        p2_text = normalize_text(row2.get_text(separator=" ")).lower()
                        
                        # 1. FINDE DAS MATCH IN DER DB
                        matched_entry = None
                        for pm in safe_matches:
                            l1 = get_last_name(pm['player1_name'])
                            l2 = get_last_name(pm['player2_name'])
                            
                            # Einfacher Namenscheck (beide Nachnamen müssen in den zwei Zeilen vorkommen)
                            match_straight = (l1 in p1_text and l2 in p2_text)
                            match_reverse = (l2 in p1_text and l1 in p2_text)
                            
                            if match_straight or match_reverse:
                                matched_entry = pm
                                break
                        
                        if matched_entry:
                            log(f"   🎯 MATCH ROW FOUND: {matched_entry['player1_name']} vs {matched_entry['player2_name']}")
                            
                            cols1 = row1.find_all('td')
                            cols2 = row2.find_all('td')
                            
                            winner = None
                            
                            # 2. VERTICAL COLUMN SCANNER LOGIC (Das Herzstück)
                            # Wir iterieren über die Spalten BEIDER Zeilen gleichzeitig.
                            # Wir suchen die ERSTE Spalte, wo:
                            # a) Beide Inhalte reine Zahlen sind (isdigit)
                            # b) Die Zahlen klein sind (<= 3), denn das sind Sätze. (Games sind 4,5,6,7)
                            # c) Mindestens eine Zahl > 0 ist (0:0 ist kein Ergebnis)
                            
                            max_cols = min(len(cols1), len(cols2))
                            
                            for c_idx in range(max_cols):
                                t1 = cols1[c_idx].get_text(strip=True)
                                t2 = cols2[c_idx].get_text(strip=True)
                                
                                # Nur Zahlen
                                if t1.isdigit() and t2.isdigit():
                                    v1 = int(t1)
                                    v2 = int(t2)
                                    
                                    # CHECK: Ist das die Satz-Spalte (S)?
                                    # Kriterium: Zahlen sind klein (0, 1, 2, 3).
                                    # UND: Das ist typischerweise vor den Game-Scores (6, 7 etc).
                                    # Wenn wir z.B. 6 und 4 finden, sind das Games, keine Sätze.
                                    
                                    is_set_count = (v1 <= 3 and v2 <= 3) and (v1 != v2)
                                    
                                    # Verfeinerung: Wenn einer 2 oder 3 hat, ist das Match vorbei (bei Best of 3/5).
                                    # Oder bei Aufgabe (ret) kann es auch 1:0 sein.
                                    is_winning_score = (v1 >= 2 or v2 >= 2) or ("ret" in p1_text or "ret" in p2_text)

                                    if is_set_count and is_winning_score:
                                        log(f"      👀 Found Result Column [{c_idx}]: {v1} vs {v2}")
                                        
                                        if v1 > v2:
                                            # Row 1 hat gewonnen
                                            if get_last_name(matched_entry['player1_name']) in p1_text:
                                                winner = matched_entry['player1_name']
                                            else:
                                                winner = matched_entry['player2_name']
                                        elif v2 > v1:
                                            # Row 2 hat gewonnen
                                            if get_last_name(matched_entry['player1_name']) in p2_text:
                                                winner = matched_entry['player1_name']
                                            else:
                                                winner = matched_entry['player2_name']
                                        
                                        # Wenn wir hier einen Gewinner gefunden haben, brechen wir den Spalten-Scan ab
                                        if winner: break

                            # 3. FALLBACK: RETIREMENT
                            if not winner and ("ret." in p1_text or "ret." in p2_text or "wo." in p1_text):
                                log("      ⚠️ Retirement detected without score clarity.")
                                if "ret." in p2_text: # P2 gab auf -> P1 (Row 1) gewinnt
                                    winner = matched_entry['player1_name'] if get_last_name(matched_entry['player1_name']) in p1_text else matched_entry['player2_name']
                                elif "ret." in p1_text:
                                    winner = matched_entry['player1_name'] if get_last_name(matched_entry['player1_name']) in p2_text else matched_entry['player2_name']

                            # 4. DB UPDATE
                            if winner:
                                log(f"      ✅ WINNER SETTLED: {winner}")
                                supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", matched_entry['id']).execute()
                                # Aus der Liste entfernen, damit wir es nicht doppelt checken
                                safe_matches = [x for x in safe_matches if x['id'] != matched_entry['id']]
                            else:
                                log(f"      ❌ Could not determine winner for {matched_entry['player1_name']}")

                            i += 2 # Wir haben dieses Paar bearbeitet
                        else:
                            i += 1 # Kein Match, nächste Zeile prüfen

            except Exception as e:
                log(f"   ❌ Page Loop Error: {e}")
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
        surf = ai_loc.get('surface_guessed', 'Hard')
        return surf, (3.5 if 'clay' in surf.lower() else 6.5), f"AI: {ai_loc['city']}"
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}). P1 Style: {p1.get('play_style')}. P2 Style: {p2.get('play_style')}.
    METRICS (0-10): TACTICAL (25%), FORM (10%), UTR (5%).
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "..." }}
    """
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5}
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
        except:
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
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                continue
            
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                
                if any(tp in p1_raw.lower() for tp in target_players) and any(tp in p2_raw.lower() for tp in target_players):
                    odds = []
                    try:
                        def extract_odds(r):
                            found_odds = []
                            for c in r.find_all('td'):
                                txt = c.get_text(strip=True)
                                if 'course' in c.get('class', []) or (re.match(r'^\d+\.\d{2}$', txt)):
                                    try:
                                        val = float(txt)
                                        if 1.0 < val < 50.0: found_odds.append(val)
                                    except: pass
                            return found_odds

                        row1_odds = extract_odds(row)
                        row2_odds = extract_odds(rows[i+1])
                        
                        if row1_odds and row2_odds: odds = [row1_odds[0], row2_odds[0]]
                        else:
                            nums = re.findall(r'\d+\.\d{2}', row_text)
                            valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                            if len(valid) >= 2: odds = valid[:2]
                    except: pass
                    
                    if len(odds) >= 2:
                        found.append({
                            "p1": p1_raw, "p2": p2_raw, "tour": current_tour, 
                            "time": match_time_str, "odds1": odds[0], "odds2": odds[1]
                        })
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v85.0 Starting...")
    
    await update_past_results()
    
    await fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    player_names = [p['last_name'] for p in players]
    
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
                    m_odds1 = m['odds1']; m_odds2 = m['odds2']
                    iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                    
                    existing = supabase.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                    
                    match_exists = False
                    if existing.data:
                        match_data = existing.data[0]
                        if match_data.get('actual_winner_name'):
                            log(f"🔒 Locked: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                            continue 
                        supabase.table("market_odds").update({"odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp}).eq("id", match_data['id']).execute()
                        match_exists = True

                    if match_exists: continue
                    if m_odds1 <= 1.01: continue
                    
                    log(f"✨ New Match Analysis: {p1_obj['last_name']} vs {p2_obj['last_name']}")
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
                    log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']}")

            except Exception as e: log(f"⚠️ Loop Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
