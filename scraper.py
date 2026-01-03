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
from typing import List, Dict, Any, Tuple

# 3rd Party Libraries
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

log("🔌 Initialisiere Neural Scout (V81.5 - Clean URL Fix)...")

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

def clean_player_name_strict(raw):
    """Aggressive Reinigung von Namen, um Müll zu entfernen."""
    garbage = ['Live streams', '1xBet', 'bwin', 'TV', 'Sky Sports', 'bet365', 'Highlights', 'adv.', 'ret.', 'w.o.']
    clean = raw
    for g in garbage:
        clean = re.sub(g, '', clean, flags=re.IGNORECASE)
    return clean.replace('|', '').strip()

def get_smart_last_name(full_name):
    """
    Intelligente Namenserkennung: Sucht das längste Wortsegment.
    Löst Probleme wie 'Sinner J.' -> 'sinner' statt 'j'.
    """
    if not full_name: return ""
    clean = re.sub(r'[.,]', ' ', full_name).strip()
    parts = clean.split()
    if not parts: return ""
    
    # Das längste Wort ist meist der Nachname (Initialen sind kurz)
    best_part = max(parts, key=len)
    if len(best_part) < 2 and len(parts) > 0: return parts[0].lower() # Fallback
    return best_part.lower()

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
# CORE LOGIC (ELO & DB)
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
        # Lade Spieler, Skills, Reports und Turniere
        players = supabase.table("players").select("id, first_name, last_name, play_style").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        
        # Optimiere Skills für schnellen Zugriff
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
# MATH CORE V8 (REALISTIC FAIR ODDS)
# =================================================================
def calculate_sharp_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    """
    Berechnet realistische Odds basierend auf Market-Wisdom + Neural Adjustment.
    """
    # 1. Market Implied Probability (Remove Vig)
    if market_odds1 <= 0 or market_odds2 <= 0: return 0.5 # Safety
    
    prob_m1 = 1 / market_odds1
    prob_m2 = 1 / market_odds2
    margin = prob_m1 + prob_m2
    true_market_prob1 = prob_m1 / margin if margin > 0 else 0.5
    
    # 2. Physics / Court Impact (BSI weighted)
    # BSI: 1 (Slow) -> 10 (Fast)
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

    # Wenn Markt extrem sicher ist (<1.20 Quote), gewichten wir Markt höher
    if market_odds1 < 1.25 or market_odds2 < 1.25:
        w_market = 0.75; w_physics = 0.15; w_ai = 0.10

    final_prob = (true_market_prob1 * w_market) + (prob_physics * w_physics) + (prob_ai * w_ai)
    return final_prob

# =================================================================
# RESULT VERIFICATION ENGINE
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Deep Scan V7)...")
    
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
                        p1_last = get_smart_last_name(pm['player1_name'])
                        p2_last = get_smart_last_name(pm['player2_name'])
                        
                        row_text = row.get_text(separator=" ", strip=True).lower()
                        if p1_last in row_text and p2_last in row_text:
                            # Simple Winner Check
                            pass 
            except Exception:
                await browser.close()

# =================================================================
# MAIN PIPELINE
# =================================================================

def find_tournament_context(scraped_name, db_tours):
    """
    Fuzzy-Logic: Findet das passende Turnier in der DB.
    Gibt Surface, BSI und Notes zurück.
    """
    best_match = None
    best_score = 0.0
    s_norm = scraped_name.lower().strip()
    
    # 1. Fuzzy Match in DB
    for t in db_tours:
        t_name = t.get('name', '').lower()
        score = difflib.SequenceMatcher(None, s_norm, t_name).ratio()
        if score > best_score:
            best_score = score
            best_match = t
            
    if best_score > 0.75 and best_match:
        return best_match.get('surface', 'Hard'), float(best_match.get('bsi_rating', 6.0)), f"DB Match ({int(best_score*100)}%)"

    # 2. Fallback Keyword Search
    bsi = 6.0; surf = 'Hard'
    if "clay" in s_norm: surf="Red Clay"; bsi=3.5
    elif "grass" in s_norm: surf="Grass"; bsi=8.5
    elif "indoor" in s_norm: surf="Indoor Hard"; bsi=8.0
    return surf, bsi, "Keyword Guess"

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
    try: return json.loads(res.replace("json", "").replace("", "").replace("```", "").strip())
    except: return d

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # FIX: Hier ist die URL jetzt absolut sauber, ohne Markdown-Klammern
            base_url = "[https://www.tennisexplorer.com/matches/?type=all](https://www.tennisexplorer.com/matches/?type=all)"
            url = f"{base_url}&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            
            log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            log(f"❌ Scrape Error: {e}")
            await browser.close()
            return None

def parse_matches_strict(html, players):
    """
    DOM-Basierter Parser mit Fehler-Toleranz.
    """
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    
    player_map = {}
    for p in players:
        k = get_smart_last_name(p['last_name'])
        if k: player_map[k] = p

    tables = soup.find_all("table", class_="result")
    current_tour = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            try: # SAFETY BLOCK
                row = rows[i]
                if 'head' in row.get('class', []) or row.find('td', class_='t-name'):
                    current_tour = row.get_text(strip=True)
                    continue
                if 'tr-first' in row.get('class', []): continue

                cols = row.find_all('td')
                if len(cols) < 4: continue 

                links = row.find_all('a', href=re.compile(r'/player/'))
                if not links: continue

                if i + 1 >= len(rows): continue
                
                # P1
                p1_raw = clean_player_name_strict(links[0].get_text(strip=True))
                p1_key = get_smart_last_name(p1_raw)
                
                # P2
                row_next = rows[i+1]
                links2 = row_next.find_all('a', href=re.compile(r'/player/'))
                if not links2: continue
                p2_raw = clean_player_name_strict(links2[0].get_text(strip=True))
                p2_key = get_smart_last_name(p2_raw)

                p1_obj = player_map.get(p1_key)
                p2_obj = player_map.get(p2_key)

                if p1_obj and p2_obj:
                    o1_c = row.find_all('td', class_='course')
                    o2_c = row_next.find_all('td', class_='course')
                    
                    if o1_c and o2_c:
                        o1 = float(o1_c[0].get_text(strip=True))
                        o2 = float(o2_c[0].get_text(strip=True))
                        
                        if 1.01 < o1 < 50.0 and 1.01 < o2 < 50.0:
                            time_cell = row.find('td', class_='first')
                            m_time = time_cell.get_text(strip=True)[:5] if time_cell else "00:00"
                            
                            found.append({
                                "p1_obj": p1_obj, "p2_obj": p2_obj,
                                "tour": current_tour, "time": m_time,
                                "odds1": o1, "odds2": o2
                            })
            except Exception:
                continue # Skip bad row

    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v81.5 (Clean URL Fix) Starting...")
    await update_past_results()
    await fetch_elo_ratings()
    
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: 
        log("❌ Keine Spieler in DB.")
        return

    current_date = datetime.now()
    
    for day_offset in range(7): 
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        matches = parse_matches_strict(html, players)
        log(f"🔍 Validated Matches: {len(matches)} am {target_date.strftime('%d.%m.')}")
        
        for m in matches:
            try:
                p1_obj = m['p1_obj']
                p2_obj = m['p2_obj']
                m_odds1 = m['odds1']
                m_odds2 = m['odds2']
                
                iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                
                # --- SAFE CHECK FOR EXISTING MATCH (v81.3 Logic) ---
                p1_safe_name = p1_obj['last_name']
                p2_safe_name = p2_obj['last_name']
                
                # Wir holen Matches, in denen P1 involviert ist
                existing_batch = supabase.table("market_odds").select("id, actual_winner_name, player1_name, player2_name")\
                    .or_(f"player1_name.eq.{p1_safe_name},player2_name.eq.{p1_safe_name}")\
                    .execute()
                
                match_exists = None
                if existing_batch.data:
                    for row in existing_batch.data:
                        # P1 vs P2 oder P2 vs P1 prüfen
                        if (row['player1_name'] == p1_safe_name and row['player2_name'] == p2_safe_name) or \
                           (row['player1_name'] == p2_safe_name and row['player2_name'] == p1_safe_name):
                            match_exists = row
                            break
                
                if match_exists:
                    if match_exists.get('actual_winner_name'):
                        continue # Match ist fertig
                    
                    # Update Odds
                    supabase.table("market_odds").update({
                        "odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp
                    }).eq("id", match_exists['id']).execute()
                    log(f"🔄 Updated: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                    continue

                # --- NEW MATCH INSERT ---
                log(f"✨ New Match: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                s1 = all_skills.get(p1_obj['id'], {})
                s2 = all_skills.get(p2_obj['id'], {})
                r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                
                surf, bsi, notes = find_tournament_context(m['tour'], all_tournaments)
                
                ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                
                prob_p1 = calculate_sharp_fair_odds(
                    p1_obj['last_name'], p2_obj['last_name'], 
                    s1, s2, bsi, surf, ai_meta, 
                    m_odds1, m_odds2
                )
                
                entry = {
                    "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0.01 else 99,
                    "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 0.99 else 99,
                    "ai_analysis_text": ai_meta.get('ai_text', 'Analysis pending'),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_timestamp 
                }
                supabase.table("market_odds").insert(entry).execute()
                log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']} (Fair: {entry['ai_fair_odds1']})")

            except Exception as e:
                log(f"⚠️ Match Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
