# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
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

log("🔌 Initialisiere Neural Scout (V91.0 - Token-Based Matching)...")

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
    # Normalisiert Text: entfernt Akzente, lowercase
    if not text: return ""
    norm = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return norm.lower()

def clean_player_name(raw): 
    if not raw: return ""
    raw = re.sub(r'\(.*?\)', '', raw) 
    raw = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    return raw.replace('|', '').strip()

def get_name_tokens(full_name):
    """
    Zerlegt einen Namen in seine Bestandteile (Tokens).
    Filtert kurze Füllwörter (außer bei kurzen asiatischen Namen).
    'Bouzas Maneiro' -> {'bouzas', 'maneiro'}
    'Sierra' -> {'sierra'}
    """
    if not full_name: return set()
    # Entferne Satzzeichen und mache lowercase
    clean = normalize_text(full_name)
    clean = re.sub(r'[^\w\s]', '', clean)
    parts = clean.split()
    
    # Filter Tokens: Wir behalten Tokens, die > 2 Zeichen sind, ODER wenn der Name sehr kurz ist.
    tokens = set()
    for p in parts:
        if len(p) > 2 or (len(parts) == 1): 
            tokens.add(p)
    return tokens

def names_match(db_name_tokens, scraped_text):
    """
    Prüft, ob signifikante Teile des DB-Namens im gescrapeten Text vorkommen.
    """
    scraped_clean = normalize_text(scraped_text)
    # Check if ANY significant token from DB name exists in scraped text as a whole word
    for token in db_name_tokens:
        # Regex suche nach Word Boundary, damit "Li" nicht in "Live" matcht
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
def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    return 0.5 

# =================================================================
#  VETERAN RESULT VERIFICATION V91.0 (Token-Matching)
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Token-Based Matching)...")
    
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending_matches:
        log("   ✅ No pending matches found.")
        return

    matches_by_date = defaultdict(list)
    failed_count = 0
    for pm in pending_matches:
        ts = sanitize_timestamp(pm.get('match_time'))
        if ts:
            date_key = (ts.year, ts.month, ts.day)
            matches_by_date[date_key].append(pm)
        else:
            failed_count += 1

    log(f"   🔎 Targeting {len(matches_by_date)} unique dates ({len(pending_matches)} matches)...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for (year, month, day), match_list in matches_by_date.items():
            page = await browser.new_page()
            try:
                url = f"https://www.tennisexplorer.com/results/?type=all&year={year}&month={month}&day={day}"
                log(f"   📅 Visiting: {day}.{month}.{year} (Seeking {len(match_list)} matches)")
                
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                tables = soup.find_all('table', class_='result')
                if not tables:
                    log("      ⚠️ No result tables found.")
                    await page.close(); continue

                all_rows = []
                for t in tables: all_rows.extend(t.find_all('tr'))

                processed_ids = set()
                
                # Iteration über Zeilen-Paare (besser für Performance)
                i = 0
                while i < len(all_rows) - 1:
                    row1 = all_rows[i]
                    row2 = all_rows[i+1]
                    
                    # Cleanup check
                    if 'head' in row1.get('class', []) or 'bott' in row1.get('class', []):
                        i += 1; continue

                    t1_text = normalize_text(row1.get_text(separator=" ", strip=True))
                    t2_text = normalize_text(row2.get_text(separator=" ", strip=True))
                    
                    matched_entry = None
                    
                    # Suche in unserer Match-Liste für diesen Tag
                    for pm in match_list:
                        if pm['id'] in processed_ids: continue
                        
                        # Hole Tokens
                        tok_p1 = get_name_tokens(pm['player1_name'])
                        tok_p2 = get_name_tokens(pm['player2_name'])
                        
                        # MATCH LOGIC:
                        # Fall A: Zeile 1 = P1, Zeile 2 = P2
                        match_a = names_match(tok_p1, t1_text) and names_match(tok_p2, t2_text)
                        
                        # Fall B: Zeile 1 = P2, Zeile 2 = P1 (Umgekehrte Reihenfolge)
                        match_b = names_match(tok_p2, t1_text) and names_match(tok_p1, t2_text)
                        
                        if match_a or match_b:
                            matched_entry = pm
                            # Extra Info für Debugging
                            log(f"      🎯 MATCH HIT: '{pm['player1_name']}' vs '{pm['player2_name']}'")
                            log(f"         Html Row 1: {t1_text[:30]}...")
                            log(f"         Html Row 2: {t2_text[:30]}...")
                            break
                    
                    if matched_entry:
                        cols1 = row1.find_all('td')
                        cols2 = row2.find_all('td')
                        winner = None
                        
                        try:
                            # 3. SCORE PARSING (Visual Logic)
                            def get_sets_won(columns):
                                for col in columns:
                                    txt = col.get_text(strip=True)
                                    if txt.isdigit():
                                        val = int(txt)
                                        if val <= 3: return val
                                return -1

                            s1 = get_sets_won(cols1[2:]) # Skip Time/Flag
                            s2 = get_sets_won(cols2[2:])
                            
                            if s1 == -1: s1 = get_sets_won(cols1)
                            if s2 == -1: s2 = get_sets_won(cols2)
                            
                            valid_score = (s1 != -1 and s2 != -1 and s1 != s2)
                            
                            if valid_score:
                                # Wer hat in der Tabelle gewonnen?
                                r1_wins = s1 > s2
                                
                                # Wir müssen wissen, wer in Zeile 1 steht
                                tok_p1 = get_name_tokens(matched_entry['player1_name'])
                                row1_is_p1 = names_match(tok_p1, t1_text)
                                
                                if r1_wins:
                                    # Zeile 1 hat gewonnen.
                                    if row1_is_p1: winner = matched_entry['player1_name']
                                    else: winner = matched_entry['player2_name']
                                else:
                                    # Zeile 2 hat gewonnen.
                                    if row1_is_p1: winner = matched_entry['player2_name'] # Wenn R1 P1 ist, dann hat P2 (R2) gewonnen
                                    else: winner = matched_entry['player1_name']
                            
                            # Fallback Retirement
                            if not winner and ("ret." in t1_text or "ret." in t2_text):
                                log("         ⚠️ Retirement detected.")
                                tok_p1 = get_name_tokens(matched_entry['player1_name'])
                                row1_is_p1 = names_match(tok_p1, t1_text)
                                
                                # Wer "ret." hat, hat verloren.
                                if "ret." in t2_text: # R2 retired -> R1 gewinnt
                                    winner = matched_entry['player1_name'] if row1_is_p1 else matched_entry['player2_name']
                                elif "ret." in t1_text: # R1 retired -> R2 gewinnt
                                    winner = matched_entry['player2_name'] if row1_is_p1 else matched_entry['player1_name']

                            if winner:
                                log(f"      ✅ WINNER CONFIRMED: {winner} ({s1}-{s2})")
                                supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", matched_entry['id']).execute()
                                processed_ids.add(matched_entry['id'])
                            else:
                                log(f"      ❌ Match found but ambiguous result ({s1}-{s2})")

                        except Exception as e:
                            log(f"      ❌ Error evaluating winner: {e}")

                        i += 2 # Match bearbeitet, springe 2 Zeilen weiter
                    else:
                        i += 1 # Nächste Zeile prüfen
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
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5}
    return d

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
                raw_time = first_col.get_text(strip=True)
                match = re.search(r'(\d{2}:\d{2})', raw_time)
                if match: match_time_str = match.group(1)

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
    log(f"🚀 Neural Scout v91.0 Starting...")
    
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
