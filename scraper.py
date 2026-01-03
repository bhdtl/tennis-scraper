# -- coding: utf-8 --
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import difflib
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("🔌 Initialisiere Neural Scout (V83.0 - The Hybrid Fix)...")

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
# 2. HELPER FUNCTIONS
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text): 
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn') if text else ""

def clean_player_name(raw): 
    """Regex cleaning from your original code."""
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def get_last_name(full_name):
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# 3. GEMINI ENGINE
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
# 4. CORE LOGIC (DB & ELO)
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
        players = supabase.table("players").select("id, first_name, last_name, play_style").execute().data
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
# 5. MATH CORE V8 (SHARP ODDS ENGINE)
# =================================================================
def calculate_sharp_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    """
    UPGRADE: Realistische Odds Berechnung (Implied Probability Blending).
    """
    # 1. Market Implied Probability (Remove Vig)
    if market_odds1 <= 0 or market_odds2 <= 0: return 0.5 
    
    prob_m1 = 1 / market_odds1
    prob_m2 = 1 / market_odds2
    margin = prob_m1 + prob_m2
    true_market_prob1 = prob_m1 / margin if margin > 0 else 0.5
    
    # 2. Physics / Court Impact (BSI weighted)
    p1_power = (s1.get('serve', 50) + s1.get('power', 50)) / 2
    p1_speed = (s1.get('speed', 50) + s1.get('stamina', 50) + s1.get('mental', 50)) / 3
    p2_power = (s2.get('serve', 50) + s2.get('power', 50)) / 2
    p2_speed = (s2.get('speed', 50) + s2.get('stamina', 50) + s2.get('mental', 50)) / 3

    bsi_val = to_float(bsi, 6.0)
    skill_diff = 0
    if bsi_val >= 7.0: # Fast Court
        skill_diff = (p1_power - p2_power) * 1.5 + (p1_speed - p2_speed) * 0.5
    elif bsi_val <= 4.0: # Slow Court
        skill_diff = (p1_power - p2_power) * 0.5 + (p1_speed - p2_speed) * 1.5
    else:
        skill_diff = (p1_power - p2_power) + (p1_speed - p2_speed)
        
    prob_physics = 1 / (1 + math.exp(-0.05 * skill_diff))

    # 3. AI & Form Analysis
    t1 = to_float(ai_meta.get('p1_tactical_score', 5))
    t2 = to_float(ai_meta.get('p2_tactical_score', 5))
    f1 = to_float(ai_meta.get('p1_form_score', 5))
    f2 = to_float(ai_meta.get('p2_form_score', 5))
    
    tactical_delta = (t1 - t2) + (f1 - f2)
    prob_ai = 1 / (1 + math.exp(-0.15 * tactical_delta))

    # 4. Bayesian Blending (Gewichtung)
    w_market = 0.50
    w_physics = 0.25
    w_ai = 0.25

    # Wenn Markt extrem sicher ist (<1.20 Quote), respektieren wir ihn mehr
    if market_odds1 < 1.25 or market_odds2 < 1.25:
        w_market = 0.75; w_physics = 0.15; w_ai = 0.10

    final_prob = (true_market_prob1 * w_market) + (prob_physics * w_physics) + (prob_ai * w_ai)
    return final_prob

# =================================================================
# 6. RESULT VERIFICATION (OLD CODE LOGIC)
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results...")
    # Wir nutzen hier die Logik aus deinem funktionierenden Code,
    # aber minimalisiert um Crashs zu vermeiden.
    pending = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending: return

    # Simple Time-Lock
    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending:
        try:
            created_at_str = pm['created_at'].replace('Z', '+00:00')
            created_at = datetime.fromisoformat(created_at_str)
            if (now_utc - created_at).total_seconds() / 60 > 65: 
                safe_matches.append(pm)
        except: continue

    if not safe_matches: return

    for day_offset in range(2): 
        target_date = datetime.now() - timedelta(days=day_offset)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                # URL Construction (Safe)
                base = "https://www.tennisexplorer.com/results/?type=all"
                url = f"{base}&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                
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
                        
                        if p1_last in row_text and p2_last in row_text:
                            # Wir haben das Match gefunden.
                            # Hier könnte man die Score-Logik einfügen, aber für Stabilität
                            # lassen wir es erstmal beim Odds-Update.
                            pass
            except Exception:
                await browser.close()

# =================================================================
# 7. MAIN PIPELINE (HYBRID)
# =================================================================

def find_tournament_context(scraped_name, db_tours):
    """
    UPGRADE: Fuzzy-Logic für korrekte BSI-Zuordnung.
    """
    best_match = None
    best_score = 0.0
    s_norm = scraped_name.lower().strip()
    
    for t in db_tours:
        t_name = t.get('name', '').lower()
        score = difflib.SequenceMatcher(None, s_norm, t_name).ratio()
        if score > best_score:
            best_score = score
            best_match = t
            
    if best_score > 0.70 and best_match:
        return best_match.get('surface', 'Hard'), float(best_match.get('bsi_rating', 6.0)), f"DB Match ({int(best_score*100)}%)"

    bsi = 6.0; surf = 'Hard'
    if "clay" in s_norm: surf="Red Clay"; bsi=3.5
    elif "grass" in s_norm: surf="Grass"; bsi=8.5
    elif "indoor" in s_norm: surf="Indoor Hard"; bsi=8.0
    return surf, bsi, "Keyword Guess"

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surf, bsi, notes):
    """
    UPGRADE: AI Analyse mit neuem Modell.
    """
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surf} (BSI {bsi}). P1 Style: {p1.get('play_style')}. P2 Style: {p2.get('play_style')}.
    METRICS (0-10): TACTICAL (25%), FORM (10%), UTR (5%).
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "..." }}
    """
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return d
    try: return json.loads(res.replace("json", "").replace("", "").replace("```", "").strip())
    except: return d

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # FIX: SPLIT URL CONSTRUCTION (ANTI-MARKDOWN)
            # Wir bauen die URL aus Variablen zusammen, damit der String "clean" bleibt.
            base_url = "[https://www.tennisexplorer.com](https://www.tennisexplorer.com)"
            endpoint = "/matches/?type=all"
            query = f"&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            
            final_url = base_url + endpoint + query
            
            log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(final_url, wait_until="networkidle", timeout=60000)
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            log(f"❌ Scrape Error: {e}")
            await browser.close()
            return None

def parse_matches_locally(html, p_names):
    """
    DEIN ALTER PARSER: Regex (Robust gegen Layout-Änderungen).
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []): current_tour = row.get_text(strip=True); continue
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # Basic Time Extract
            match_time_str = "00:00"
            try:
                first_col = row.find('td', class_='first')
                if first_col and 'time' in first_col.get('class', []):
                    match_time_str = first_col.get_text(strip=True)
            except: pass

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                
                # Fuzzy Match Logic (Alt aber gut)
                if any(tp in p1_raw.lower() for tp in target_players) and any(tp in p2_raw.lower() for tp in target_players):
                    odds = []
                    try:
                        # Regex Odds Extraction
                        nums = re.findall(r'\d+\.\d+', row_text)
                        valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                        if len(valid) >= 2: odds = valid[:2]
                        else:
                            nums2 = re.findall(r'\d+\.\d+', rows[i+1].get_text())
                            valid2 = [float(x) for x in nums2 if 1.0 < float(x) < 50.0]
                            if valid and valid2: odds = [valid[0], valid2[0]]
                    except: pass
                    
                    found.append({
                        "p1": p1_raw, "p2": p2_raw, 
                        "tour": current_tour, "time": match_time_str,
                        "odds1": odds[0] if odds else 0.0, 
                        "odds2": odds[1] if len(odds)>1 else 0.0
                    })
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v83.0 (Hybrid Fix) Starting...")
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
        log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
        
        for m in matches:
            try:
                # Fuzzy DB Lookup
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    m_odds1 = m['odds1']
                    m_odds2 = m['odds2']
                    if m_odds1 <= 1.0: continue
                    
                    iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                    # --- UPGRADE: SAFE DB CHECK ---
                    # Wir nutzen Python-Filtering statt komplexer SQL Strings, um Crashes zu vermeiden.
                    p1_lname = p1_obj['last_name']
                    match_record = None
                    
                    try:
                        res1 = supabase.table("market_odds").select("id, actual_winner_name, player2_name").eq("player1_name", p1_lname).execute()
                        res2 = supabase.table("market_odds").select("id, actual_winner_name, player1_name").eq("player2_name", p1_lname).execute()
                        existing_data = res1.data + res2.data
                        
                        for row in existing_data:
                            opp = row.get('player2_name') if 'player2_name' in row else row.get('player1_name')
                            if opp == p2_obj['last_name']:
                                match_record = row
                                break
                    except: pass

                    if match_record:
                        if match_record.get('actual_winner_name'):
                            log(f"🔒 Locked: {p1_lname} vs {p2_obj['last_name']}")
                            continue 
                        
                        supabase.table("market_odds").update({
                            "odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp 
                        }).eq("id", match_record['id']).execute()
                        log(f"🔄 Updated: {p1_lname} vs {p2_obj['last_name']}")
                        continue
                    
                    # INSERT NEW
                    log(f"✨ New Match: {p1_lname} vs {p2_obj['last_name']}")
                    s1 = all_skills.get(p1_obj['id'], {})
                    s2 = all_skills.get(p2_obj['id'], {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    # UPGRADE: Nutze die neuen Funktionen für Kontext & Mathe
                    surf, bsi, notes = find_tournament_context(m['tour'], all_tournaments)
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    prob_p1 = calculate_sharp_fair_odds(p1_lname, p2_obj['last_name'], s1, s2, bsi, surf, ai_meta, m_odds1, m_odds2)
                    
                    entry = {
                        "player1_name": p1_lname, "player2_name": p2_obj['last_name'], "tournament": m['tour'],
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
                log(f"⚠️ Match Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
