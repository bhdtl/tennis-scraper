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
from typing import Dict, List, Any, Optional, Union

# Third-party imports
from playwright.async_api import async_playwright, Browser, BrowserContext
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION & LOGGING
# =================================================================
# Configure Logging format for Production
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.5-pro'

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Environment Variables prüfen.")
    sys.exit(1)

# =================================================================
# DATABASE MANAGER (Async Wrapper)
# =================================================================
class DatabaseManager:
    """
    Wraps the synchronous Supabase client in asyncio.to_thread
    to prevent blocking the main event loop during DB operations.
    """
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def fetch_players(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data)

    async def fetch_skills(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data)

    async def fetch_reports(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data)

    async def fetch_tournaments(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data)

    async def fetch_pending_matches(self) -> List[Dict]:
        return await asyncio.to_thread(
            lambda: self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
        )

    async def check_existing_match(self, p1_name: str, p2_name: str) -> List[Dict]:
        # Complex query needs safe wrapping
        def _query():
            return self.client.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})"
            ).execute().data
        return await asyncio.to_thread(_query)

    async def update_match(self, match_id: int, payload: Dict):
        await asyncio.to_thread(
            lambda: self.client.table("market_odds").update(payload).eq("id", match_id).execute()
        )

    async def insert_match(self, payload: Dict):
        await asyncio.to_thread(
            lambda: self.client.table("market_odds").insert(payload).execute()
        )

# Initialize global DB instance
db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# UTILITIES
# =================================================================
def to_float(val: Any, default: float = 50.0) -> float:
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# AI & LOGIC ENGINE
# =================================================================
class AIEngine:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(5) # Rate Limit concurrent AI calls

    async def call_gemini(self, prompt: str) -> Optional[str]:
        async with self.semaphore: # Prevent hitting rate limits
            await asyncio.sleep(1.0) # Conservative spacing
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
            }
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                    if response.status_code != 200: 
                        logger.error(f"Gemini API Error: {response.status_code}")
                        return None
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except Exception as e:
                    logger.error(f"Gemini Request Failed: {e}")
                    return None

ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)
ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# MATH & PHYSICS CORE (V7)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name: str, p2_name: str, s1: Dict, s2: Dict, bsi: Any, surface: str, ai_meta: Dict, market_odds1: float, market_odds2: float) -> float:
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    bsi_val = to_float(bsi, 6.0)

    # 1. AI MATCHUP (50%)
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8) 

    # 2. COURT PHYSICS (20%)
    c1_score = 0.0; c2_score = 0.0
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

    # 3. SKILLS (15%)
    score_p1 = sum(s1.values())
    score_p2 = sum(s2.values())
    prob_skills = sigmoid_prob(score_p1 - score_p2, sensitivity=0.08)

    # 4. ELO (15%)
    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    # Robust ELO lookup
    for tour_key in ["ATP", "WTA"]: # Check both just in case
        for name, stats in ELO_CACHE.get(tour_key, {}).items():
            if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
            if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
            
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    prob_alpha = (prob_matchup * 0.50) + (prob_bsi * 0.20) + (prob_skills * 0.15) + (prob_elo * 0.15)

    # Clamp extreme probabilities
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
# SCRAPING & PIPELINE LOGIC
# =================================================================

class ScraperBot:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def start(self):
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
        # Reuse context to save resources
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

    async def stop(self):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()

    async def fetch_page(self, url: str) -> Optional[str]:
        if not self.context: await self.start()
        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            content = await page.content()
            await page.close()
            return content
        except Exception as e:
            logger.warning(f"⚠️ Page Fetch Error ({url}): {e}")
            return None

async def fetch_elo_ratings_optimized(bot: ScraperBot):
    logger.info("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    for tour, url in urls.items():
        content = await bot.fetch_page(url)
        if not content: continue
        
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
            logger.info(f"   ✅ {tour} Elo Ratings geladen: {len(ELO_CACHE[tour])} Spieler.")

async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    prompt = f"TASK: Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\", \"is_indoor\": bool }}"
    res = await ai_engine.call_gemini(prompt)
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
    
    logger.info(f"   🤖 AI resolving location for {p1} vs {p2}...")
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
    res = await ai_engine.call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return d
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return d

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
                current_tour = row.get_text(strip=True)
                continue
                
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # --- TIME EXTRACTION ---
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                
                # Check intersection with our DB players
                p1_match = any(tp in p1_raw.lower() for tp in target_players)
                p2_match = any(tp in p2_raw.lower() for tp in target_players)

                if p1_match and p2_match:
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
                        "tour": current_tour, 
                        "time": match_time_str,
                        "odds1": odds[0] if odds else 0.0, 
                        "odds2": odds[1] if len(odds)>1 else 0.0
                    })
    return found

# =================================================================
# RESULT VERIFICATION (V80.6 - Refactored)
# =================================================================
async def update_past_results(bot: ScraperBot):
    logger.info("🏆 Checking for Match Results (Deep Scan V6)...")
    
    pending_matches = await db_manager.fetch_pending_matches()
    if not pending_matches:
        logger.info("   ✅ No pending matches to verify.")
        return

    # Filter by time lock (65 mins passed)
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
        logger.info("   ⏳ Waiting for matches to finish (Time-Lock active)...")
        return

    logger.info(f"   🔎 Target List ({len(safe_matches)}): {[m['player1_name'] + ' vs ' + m['player2_name'] for m in safe_matches[:3]]}...")

    for day_offset in range(3): 
        target_date = datetime.now() - timedelta(days=day_offset)
        url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        
        content = await bot.fetch_page(url)
        if not content: continue
        
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
                next_row_text = ""
                if i+1 < len(rows):
                    next_row_text = rows[i+1].get_text(separator=" ", strip=True).lower()
                
                match_found = (p1_last in row_text and p2_last in next_row_text) or \
                              (p2_last in row_text and p1_last in next_row_text) or \
                              (p1_last in row_text and p2_last in row_text)
                
                if match_found:
                    try:
                        is_retirement = "ret." in row_text or "w.o." in row_text
                        
                        # --- SCORE EXTRACTION (Nested helper) ---
                        def extract_scores_aggressive(columns):
                            scores = []
                            for col in columns:
                                txt = col.get_text(strip=True)
                                if len(txt) > 4 or '(' in txt: 
                                    if '(' in txt: txt = txt.split('(')[0] # clean tiebreak
                                if txt.isdigit() and len(txt) == 1 and int(txt) <= 7:
                                    scores.append(int(txt))
                            return scores

                        cols1 = row.find_all('td')
                        cols2 = rows[i+1].find_all('td') if i+1 < len(rows) else []
                        p1_scores = extract_scores_aggressive(cols1)
                        p2_scores = extract_scores_aggressive(cols2)

                        # DERIVE WINNER
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
                            await db_manager.update_match(pm['id'], {"actual_winner_name": winner_name})
                            logger.info(f"   ✅ WINNER SETTLED: {winner_name}")
                            safe_matches = [x for x in safe_matches if x['id'] != pm['id']] # Remove from list

                    except Exception as e:
                        logger.error(f"     ❌ Parsing Error: {e}")

# =================================================================
# MAIN EXECUTION FLOW
# =================================================================
async def process_day_scan(bot: ScraperBot, target_date: datetime, players: List[Dict], all_skills: Dict, all_reports: List, all_tournaments: List):
    """
    Processes a single day. Designed to be run in parallel.
    """
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    html = await bot.fetch_page(url)
    if not html: return

    player_names = [p['last_name'] for p in players]
    matches = parse_matches_locally(html, player_names)
    
    if matches:
        logger.info(f"🔍 {target_date.strftime('%d.%m.')}: {len(matches)} Matches gefunden.")
    
    for m in matches:
        try:
            p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
            p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
            
            if p1_obj and p2_obj:
                m_odds1 = m['odds1']
                m_odds2 = m['odds2']
                iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                # Check Existing
                existing = await db_manager.check_existing_match(p1_obj['last_name'], p2_obj['last_name'])
                
                if existing:
                    match_data = existing[0]
                    if match_data.get('actual_winner_name'):
                        continue # Immutable if finished
                    
                    # Update active match
                    await db_manager.update_match(match_data['id'], {
                        "odds1": m_odds1, 
                        "odds2": m_odds2, 
                        "match_time": iso_timestamp 
                    })
                    continue

                if m_odds1 <= 1.0: continue
                
                # Insert New
                logger.info(f"✨ New Match: {p1_obj['last_name']} vs {p2_obj['last_name']}")
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
                await db_manager.insert_match(entry)
                
        except Exception as e:
            logger.error(f"⚠️ Match Process Error: {e}")

async def run_pipeline():
    logger.info(f"🚀 Neural Scout v80.8 (Architect Edition) Starting...")
    
    bot = ScraperBot()
    await bot.start()

    try:
        # 1. Update Results
        await update_past_results(bot)
        
        # 2. Load Static Data
        await fetch_elo_ratings_optimized(bot)
        
        # 3. Load DB Data (Parallel Fetch)
        logger.info("📡 Fetching Database Context...")
        players_t, skills_t, reports_t, tourneys_t = await asyncio.gather(
            db_manager.fetch_players(),
            db_manager.fetch_skills(),
            db_manager.fetch_reports(),
            db_manager.fetch_tournaments()
        )
        
        # Process Skills into Dict for O(1) Access
        clean_skills = {}
        for entry in skills_t:
            pid = entry.get('player_id')
            if pid:
                clean_skills[pid] = {
                    'serve': to_float(entry.get('serve')), 'power': to_float(entry.get('power')),
                    'forehand': to_float(entry.get('forehand')), 'backhand': to_float(entry.get('backhand')),
                    'speed': to_float(entry.get('speed')), 'stamina': to_float(entry.get('stamina')),
                    'mental': to_float(entry.get('mental'))
                }

        if not players_t:
            logger.error("❌ No players found in DB.")
            return

        # 4. Main Scraping Loop (Parallelized)
        logger.info("🔥 Starting Parallel Scraping Engine...")
        current_date = datetime.now()
        
        # Batch requests to avoid killing the browser or getting IP banned
        batch_size = 5
        days_to_scan = 35
        
        for i in range(0, days_to_scan, batch_size):
            tasks = []
            for j in range(batch_size):
                if i + j >= days_to_scan: break
                target_date = current_date + timedelta(days=i+j)
                tasks.append(process_day_scan(
                    bot, target_date, players_t, clean_skills, reports_t, tourneys_t
                ))
            
            logger.info(f"⚡ Batch Processing Days {i} to {min(i+batch_size, days_to_scan)}...")
            await asyncio.gather(*tasks)
            await asyncio.sleep(2) # Brief cool-down between batches

    except Exception as e:
        logger.critical(f"❌ PIPELINE CRASH: {e}", exc_info=True)
    finally:
        await bot.stop()
        logger.info("🏁 Cycle Finished.")

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        pass
