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
from typing import List, Dict, Optional, Tuple, Any

# 3rd Party Libraries
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Tag
from supabase import create_client, Client
import httpx

# =================================================================
# 1. CONFIGURATION & LOGGING (Enterprise Standards)
# =================================================================

# Setup Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (v81.0 - Sharp Architecture)...")

# Environment Validation
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

# Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.0-flash-exp' # Speed & Intelligence Balance

# In-Memory Caches
ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# 2. HELPER FUNCTIONS & UTILS
# =================================================================

def to_float(val: Any, default: float = 50.0) -> float:
    if val is None: return default
    try:
        f = float(val)
        return f if not math.isnan(f) else default
    except:
        return default

def normalize_text(text: str) -> str: 
    if not text: return ""
    # Remove accents/diacritics
    n = unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o'))
    return "".join(c for c in n if unicodedata.category(c) != 'Mn')

def clean_player_name_strict(raw: str) -> str:
    """
    Entfernt aggressiv Müll aus Namen.
    """
    garbage = ['Live streams', '1xBet', 'bwin', 'TV', 'Sky Sports', 'bet365', 'Highlights', 'adv.', 'ret.']
    clean = raw
    for g in garbage:
        clean = re.sub(g, '', clean, flags=re.IGNORECASE)
    
    # Entferne Klammern (z.B. Ländercodes)
    clean = re.sub(r'\(.*?\)', '', clean)
    return clean.replace('|', '').strip()

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# 3. GEMINI AI ENGINE (Robust Retry)
# =================================================================

async def call_gemini(prompt: str, retries=2) -> Optional[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
    }
    
    async with httpx.AsyncClient() as client:
        for attempt in range(retries + 1):
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                if response.status_code == 200:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                elif response.status_code == 429:
                    await asyncio.sleep(2 * (attempt + 1)) # Backoff
                    continue
                else:
                    log(f"⚠️ Gemini Error {response.status_code}")
                    return None
            except Exception as e:
                if attempt == retries:
                    log(f"❌ Gemini Exception: {e}")
                    return None
                await asyncio.sleep(1)
    return None

# =================================================================
# 4. MATH CORE V8 (The "Sharp" Model)
# =================================================================

def implied_probability(odds: float) -> float:
    """Konvertiert Decimal Odds zu Wahrscheinlichkeit (ohne Vig-Bereinigung)."""
    return 1.0 / odds if odds > 0 else 0.0

def calculate_sharp_fair_odds(
    p1_name: str, p2_name: str, 
    s1: Dict, s2: Dict, 
    bsi_rating: float, 
    surface: str, 
    ai_meta: Dict, 
    market_odds1: float, market_odds2: float
) -> float:
    """
    Berechnet die Wahrscheinlichkeit für P1 basierend auf Markt-Weisheit + AI-Adjustment.
    Dies verhindert unrealistische "Ghost Odds".
    """
    
    # --- 1. MARKET IMPLIED PROBABILITY (De-Vigged) ---
    raw_prob1 = implied_probability(market_odds1)
    raw_prob2 = implied_probability(market_odds2)
    margin = raw_prob1 + raw_prob2
    
    # Echte Markt-Wahrscheinlichkeit (entfernt die Marge des Buchmachers)
    true_market_prob1 = raw_prob1 / margin
    
    # --- 2. THE FUNDAMENTAL MODEL (Physics & Skills) ---
    # BSI Logic: 1 (Slow Clay) -> 10 (Fast Indoor)
    
    # Base Power vs Speed Delta
    p1_power_score = (s1.get('serve', 50) + s1.get('power', 50)) / 2
    p2_power_score = (s2.get('serve', 50) + s2.get('power', 50)) / 2
    
    p1_speed_score = (s1.get('speed', 50) + s1.get('stamina', 50) + s1.get('mental', 50)) / 3
    p2_speed_score = (s2.get('speed', 50) + s2.get('stamina', 50) + s2.get('mental', 50)) / 3
    
    # Surface Weighting
    if bsi_rating >= 7.0: # Fast Court -> Favors Power
        skill_diff = (p1_power_score - p2_power_score) * 1.5 + (p1_speed_score - p2_speed_score) * 0.5
    elif bsi_rating <= 4.0: # Slow Court -> Favors Speed/Grind
        skill_diff = (p1_power_score - p2_power_score) * 0.5 + (p1_speed_score - p2_speed_score) * 1.5
    else: # Neutral
        skill_diff = (p1_power_score - p2_power_score) + (p1_speed_score - p2_speed_score)

    # Sigmoid Transformation for Skills (-50 to +50 range mapped to prob shift)
    # Sensitivity 0.05 means a 10 point skill gap creates a ~62% win prob (without other factors)
    prob_skills = 1 / (1 + math.exp(-0.05 * skill_diff))

    # --- 3. AI TACTICAL ANALYSIS ---
    # Holt Werte aus der Gemini Analyse (0-10 Scale)
    t1 = ai_meta.get('p1_tactical_score', 5)
    t2 = ai_meta.get('p2_tactical_score', 5)
    f1 = ai_meta.get('p1_form_score', 5)
    f2 = ai_meta.get('p2_form_score', 5)
    
    tactical_diff = (t1 - t2) + (f1 - f2) # Range approx -10 to +10
    prob_ai = 1 / (1 + math.exp(-0.15 * tactical_diff))

    # --- 4. BLENDING (Bayesian Approach) ---
    # Wir nutzen den Markt als "Prior" und updaten ihn mit unseren Daten ("Posterior")
    
    # Gewichte: Wie sehr vertrauen wir wem?
    w_market = 0.55  # Markt ist sehr effizient, respektiere ihn.
    w_skills = 0.25  # Unsere Datenbank
    w_ai = 0.20      # Tagesform/Analyse
    
    # Sonderfall: Wenn Markt sehr einseitig ist (<1.20), erhöhe Markt-Gewicht
    if market_odds1 < 1.25 or market_odds2 < 1.25:
        w_market = 0.80
        w_skills = 0.10
        w_ai = 0.10

    final_prob_p1 = (true_market_prob1 * w_market) + (prob_skills * w_skills) + (prob_ai * w_ai)
    
    return final_prob_p1

# =================================================================
# 5. DATA FETCHING LAYER
# =================================================================

async def fetch_elo_ratings():
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for tour, url in urls.items():
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', {'id': 'reportable'})
                
                count = 0
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
                                count += 1
                            except: continue
                    log(f"   ✅ {tour} Elo geladen: {count} Einträge.")
                await page.close()
            except Exception as e:
                log(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
        await browser.close()

async def get_db_context():
    """Lädt ALLE relevanten Kontext-Daten für Matching."""
    try:
        # Lade nur notwendige Spalten für Performance
        players = supabase.table("players").select("id, first_name, last_name, play_style").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        
        # Mappe Skills für schnellen Zugriff
        skill_map = {}
        for s in skills:
            pid = s.get('player_id')
            if pid: skill_map[pid] = s
            
        return players, skill_map, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, []

def find_tournament_bsi_fuzzy(scraped_name: str, db_tournaments: List[Dict]) -> Tuple[str, float, str]:
    """
    Findet das passende Turnier via Fuzzy Matching und gibt Surface + BSI zurück.
    Gibt (Surface, BSI, Notes) zurück.
    """
    best_match = None
    best_score = 0.0
    
    s_norm = scraped_name.lower().strip()
    
    # 1. Direct Fuzzy Search in DB
    for t in db_tournaments:
        t_name = t.get('name', '').lower()
        score = difflib.SequenceMatcher(None, s_norm, t_name).ratio()
        if score > best_score:
            best_score = score
            best_match = t
            
    # Threshold: 70% Match
    if best_score > 0.70 and best_match:
        return best_match.get('surface', 'Hard'), float(best_match.get('bsi_rating', 6.0)), f"DB Match ({int(best_score*100)}%)"

    # 2. Fallback: Keyword Inference
    bsi = 6.0
    surf = 'Hard'
    if "clay" in s_norm: 
        surf = "Red Clay"
        bsi = 3.5
    elif "grass" in s_norm:
        surf = "Grass"
        bsi = 8.5
    elif "indoor" in s_norm:
        surf = "Indoor Hard"
        bsi = 8.0
        
    return surf, bsi, "Keyword Guess"

# =================================================================
# 6. PARSING ENGINE (Strict DOM Traversal)
# =================================================================

def parse_matches_strict(html: str, valid_players: List[Dict]) -> List[Dict]:
    """
    Strict Parser, der Ghost-Matches verhindert.
    Akzeptiert nur Matches, wo BEIDE Spieler in der DB existieren.
    """
    soup = BeautifulSoup(html, 'html.parser')
    found_matches = []
    
    # Indiziere DB Spieler für O(1) Lookup (Lastname -> Full Object)
    # Wir speichern eine Map: "nadal": [PlayerObj, PlayerObj] (falls Namen doppelt)
    player_map = {}
    for p in valid_players:
        lname = normalize_text(p['last_name']).lower()
        if lname not in player_map: player_map[lname] = []
        player_map[lname].append(p)

    tables = soup.find_all("table", class_="result")
    
    current_tournament = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        
        for i in range(len(rows)):
            row = rows[i]
            
            # --- Tournament Header Detection ---
            # TennisExplorer headers haben oft class="head" oder "flags"
            if 'head' in row.get('class', []) or row.find('td', class_='t-name'):
                current_tournament = row.get_text(strip=True)
                continue
                
            # Ignore sub-headers or ads
            if 'tr-first' in row.get('class', []): continue # Manchmal Werbung
            
            # --- Check Structure ---
            cols = row.find_all('td')
            if len(cols) < 4: continue # Keine Match Zeile
            
            # Extract Player 1 Name
            # Suche nach Links zu Spielern, das ist sicherer
            p1_links = row.find_all('a', href=re.compile(r'/player/'))
            if not p1_links: continue
            
            p1_raw = clean_player_name_strict(p1_links[0].get_text(strip=True))
            p1_lname = get_last_name(p1_raw)
            
            # Lookahead for Player 2 (Next Row)
            if i + 1 >= len(rows): continue
            row_next = rows[i+1]
            p2_links = row_next.find_all('a', href=re.compile(r'/player/'))
            if not p2_links: continue
            
            p2_raw = clean_player_name_strict(p2_links[0].get_text(strip=True))
            p2_lname = get_last_name(p2_raw)
            
            # --- VALIDATION GATE ---
            # Nur fortfahren, wenn BEIDE Namen in unserer DB bekannt sind (Partial Match)
            # Das verhindert "Joint vs Krejcikova" wenn wir Joint nicht kennen.
            p1_db = player_map.get(p1_lname)
            p2_db = player_map.get(p2_lname)
            
            if not p1_db or not p2_db:
                # Optional: Logging für Debugging, aber kein Match erstellen
                # log(f"Ignoriere: {p1_lname} vs {p2_lname} (Nicht in DB)")
                continue

            # --- Odds Extraction ---
            try:
                # TennisExplorer Odds sind oft in den hinteren Spalten
                # Wir suchen Zellen mit "course" class oder Data-Attributen
                odds_cells_row1 = row.find_all('td', class_='course')
                odds_cells_row2 = row_next.find_all('td', class_='course')
                
                if not odds_cells_row1 or not odds_cells_row2: continue
                
                # Nimm erste verfügbare Quote (meist Bet365 oder Avg)
                o1 = float(odds_cells_row1[0].get_text(strip=True))
                o2 = float(odds_cells_row2[0].get_text(strip=True))
                
                if not (1.01 <= o1 <= 50.0 and 1.01 <= o2 <= 50.0): continue
                
                # Extract Time
                time_cell = row.find('td', class_='first')
                match_time_str = "00:00"
                if time_cell:
                    match_time_str = time_cell.get_text(strip=True)[:5]

                found_matches.append({
                    "p1_raw": p1_raw, "p1_obj": p1_db[0], # Nimm den ersten Treffer
                    "p2_raw": p2_raw, "p2_obj": p2_db[0],
                    "tournament": current_tournament,
                    "time": match_time_str,
                    "odds1": o1, "odds2": o2
                })
                
            except Exception:
                continue

    return found_matches

# =================================================================
# 7. MAIN PIPELINE LOGIC
# =================================================================

async def analyze_match_with_ai(p1, p2, s1, s2, surface, bsi, notes):
    """
    Fragt Gemini nach einer taktischen Einschätzung.
    """
    prompt = f"""
    ROLE: Senior Tennis Analyst. 
    MATCH: {p1['first_name']} {p1['last_name']} vs {p2['first_name']} {p2['last_name']}.
    CONTEXT: Tournament Surface: {surface} (Speed Rating BSI: {bsi}/10). Notes: {notes}.
    
    PLAYER 1 ({p1['last_name']}): Style: {p1.get('play_style', 'Unknown')}. Skills: Serve={s1.get('serve')}, Power={s1.get('power')}.
    PLAYER 2 ({p2['last_name']}): Style: {p2.get('play_style', 'Unknown')}. Skills: Serve={s2.get('serve')}, Power={s2.get('power')}.
    
    TASK: Analyze matchup mechanics. Does the court speed favor one style? Who has the tactical edge?
    OUTPUT JSON ONLY:
    {{
        "p1_tactical_score": [0-10 integer, 10=Dominant tactical advantage],
        "p2_tactical_score": [0-10 integer],
        "p1_form_score": [0-10 integer based on recent hypothetical form],
        "p2_form_score": [0-10 integer],
        "ai_text": "Short concise analysis (max 2 sentences). Focus on BSI impact."
    }}
    """
    res = await call_gemini(prompt)
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': 'AI Unavailable'}
    
    if not res: return default
    try:
        # Cleanup JSON formatting issues often returned by LLMs
        clean_json = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return default

async def run_pipeline():
    log("🚀 Starting Deep Scan...")
    
    # 1. Load Data
    players, all_skills, all_tournaments = await get_db_context()
    if not players:
        log("❌ Keine Spieler in DB gefunden. Abbruch.")
        return

    current_date = datetime.now()
    
    # 2. Scrape Loop (Next 3 Days only to save resources/errors)
    for day_offset in range(3): 
        target_date = current_date + timedelta(days=day_offset)
        date_str = target_date.strftime('%Y-%m-%d')
        log(f"📡 Scanning Date: {date_str}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                html = await page.content()
            except Exception as e:
                log(f"❌ Network Error: {e}")
                await browser.close()
                continue
            await browser.close()

        # 3. Parse Strict
        matches = parse_matches_strict(html, players)
        log(f"   found {len(matches)} VALID matches (filtered against DB).")

        # 4. Process Matches
        for m in matches:
            p1_obj = m['p1_obj']
            p2_obj = m['p2_obj']
            
            # Check DB for existing match (Idempotency)
            existing = supabase.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),"
                f"and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})"
            ).execute()
            
            # ISO Time construction
            match_iso = f"{date_str}T{m['time']}:00Z"
            
            if existing.data:
                # Update Odds Only if not finished
                row = existing.data[0]
                if not row.get('actual_winner_name'):
                    supabase.table("market_odds").update({
                        "odds1": m['odds1'], "odds2": m['odds2'], "match_time": match_iso
                    }).eq("id", row['id']).execute()
                    # log(f"   🔄 Updated Odds: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                continue
            
            # --- NEW MATCH ANALYSIS ---
            log(f"   ✨ Analyzing New Match: {p1_obj['last_name']} vs {p2_obj['last_name']}")
            
            # Resolve Court Context
            surf, bsi, notes = find_tournament_bsi_fuzzy(m['tournament'], all_tournaments)
            
            # Get Skills
            s1 = all_skills.get(p1_obj['id'], {})
            s2 = all_skills.get(p2_obj['id'], {})
            
            # AI Analysis
            ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, surf, bsi, notes)
            
            # Calculate Sharp Odds
            prob_p1 = calculate_sharp_fair_odds(
                p1_obj['last_name'], p2_obj['last_name'],
                s1, s2, bsi, surf, ai_meta,
                m['odds1'], m['odds2']
            )
            
            # Save
            entry = {
                "player1_name": p1_obj['last_name'], 
                "player2_name": p2_obj['last_name'], 
                "tournament": m['tournament'],
                "odds1": m['odds1'], 
                "odds2": m['odds2'],
                "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0.01 else 99.0,
                "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 0.99 else 99.0,
                "ai_analysis_text": ai_meta.get('ai_text', 'Analysis pending'),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "match_time": match_iso
            }
            
            try:
                supabase.table("market_odds").insert(entry).execute()
                log(f"      💾 Saved to DB | Fair: {entry['ai_fair_odds1']} vs {entry['ai_fair_odds2']}")
            except Exception as db_err:
                log(f"      ❌ DB Insert Fail: {db_err}")

    log("🏁 Scraper Run Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
