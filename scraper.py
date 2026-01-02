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

log("🔌 Initialisiere Neural Scout (V90.0 - Pivot-Based Search)...")

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
    raw = re.sub(r'\(.*?\)', '', raw) 
    raw = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    return raw.replace('|', '').strip()

def get_normalized_last_name(full_name):
    if not full_name: return ""
    clean = re.sub(r'^\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    if not parts: return ""
    last_part = parts[-1].replace('.', '')
    if len(last_part) == 1 and len(parts) > 1:
        return parts[-2].lower()
    return parts[-1].lower()

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
#  VETERAN RESULT VERIFICATION V90.0 (Pivot-Search)
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Pivot-Search)...")
    
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

                # Collect all rows
                all_rows = []
                for t in tables: all_rows.extend(t.find_all('tr'))

                # DEBUG: Print first few rows to see what scraper sees
                for d_idx, d_row in enumerate(all_rows[:3]):
                    d_text = normalize_text(d_row.get_text(separator=" ", strip=True))
                    log(f"      👀 [DEBUG Row {d_idx}]: {d_text[:50]}...")

                processed_ids = set()
                
                # PIVOT SEARCH: Iterate every row. If it matches a Player 1 or Player 2, check the NEXT row.
                for i in range(len(all_rows) - 1):
                    row_pivot = all_rows[i]
                    # Skip Headers
                    if 'head' in row_pivot.get('class', []) or 'bott' in row_pivot.get('class', []): continue

                    pivot_text = normalize_text(row_pivot.get_text(separator=" ")).lower()
                    
                    # Check if this row contains ANY player from our target list
                    possible_matches = []
                    for pm in match_list:
                        if pm['id'] in processed_ids: continue
                        l1 = get_normalized_last_name(pm['player1_name'])
                        l2 = get_normalized_last_name(pm['player2_name'])
                        
                        if l1 in pivot_text or l2 in pivot_text:
                            possible_matches.append(pm)
                    
                    if not possible_matches: continue

                    # We found a potential player. Now check the NEXT row for the opponent.
                    row_partner = all_rows[i+1]
                    partner_text = normalize_text(row_partner.get_text(separator=" ")).lower()
                    
                    confirmed_match = None
                    for pm in possible_matches:
                        l1 = get_normalized_last_name(pm['player1_name'])
                        l2 = get_normalized_last_name(pm['player2_name'])
                        
                        # Case 1: Pivot is P1, Partner is P2
                        if (l1 in pivot_text and l2 in partner_text):
                            confirmed_match = pm; break
                        # Case 2: Pivot is P2, Partner is P1
                        if (l2 in pivot_text and l1 in partner_text):
                            confirmed_match = pm; break
                            
                    if confirmed_match:
                        log(f"      🎯 MATCH IDENTIFIED: {confirmed_match['player1_name']} vs {confirmed_match['player2_name']}")
                        
                        # SCORING LOGIC
                        cols1 = row_pivot.find_all('td')
                        cols2 = row_partner.find_all('td')
                        winner = None
                        
                        try:
                            def get_sets_won(columns):
                                for col in columns:
                                    txt = col.get_text(strip=True)
                                    if txt.isdigit():
                                        val = int(txt)
                                        if val <= 3: return val
                                return -1

                            # Skip first 2 columns (often flag/name)
                            s1 = get_sets_won(cols1[2:]) 
                            s2 = get_sets_won(cols2[2:])
                            
                            # Fallback scan all
                            if s1 == -1: s1 = get_sets_won(cols1)
                            if s2 == -1: s2 = get_sets_won(cols2)
                            
                            if s1 != -1 and s2 != -1 and s1 != s2:
                                # Who won this specific pairing?
                                pivot_won = s1 > s2
                                
                                # Map back to DB names
                                l1 = get_normalized_last_name(confirmed_match['player1_name'])
                                
                                if pivot_won:
                                    # The Pivot Row won. Is Pivot P1 or P2?
                                    if l1 in pivot_text: winner = confirmed_match['player1_name']
                                    else: winner = confirmed_match['player2_name']
                                else:
                                    # The Partner Row won.
                                    if l1 in partner_text: winner = confirmed_match['player1_name']
                                    else: winner = confirmed_match['player2_name']
                            
                            # Retirement Check
                            if not winner and ("ret." in pivot_text or "ret." in partner_text):
                                log("         ⚠️ Retirement detected.")
                                l1 = get_normalized_last_name(confirmed_match['player1_name'])
                                if "ret." in partner_text: # Partner retired -> Pivot won
                                    if l1 in pivot_text: winner = confirmed_match['player1_name']
                                    else: winner = confirmed_match['player2_name']
                                elif "ret." in pivot_text: # Pivot retired -> Partner won
                                    if l1 in partner_text: winner = confirmed_match['player1_name']
                                    else: winner = confirmed_match['player2_name']

                            if winner:
                                log(f"      ✅ WINNER FOUND: {winner}")
                                supabase.table("market_odds").update({"actual_winner_name": winner}).eq("id", confirmed_match['id']).execute()
                                processed_ids.add(confirmed_match['id'])
                            else:
                                log(f"      ❌ Score Ambiguous ({s1}-{s2})")

                        except Exception as e:
                            log(f"      ❌ Parse Error: {e}")

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
    log(f"🚀 Neural Scout v90.0 Starting...")
    
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
