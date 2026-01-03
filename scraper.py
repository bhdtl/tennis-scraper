# -*- coding: utf-8 -*-
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
from typing import Dict, List, Any, Optional, Tuple

# Third-party imports
from playwright.async_api import async_playwright, Browser, BrowserContext
from bs4 import BeautifulSoup, Tag
from supabase import create_client, Client
import httpx

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout_v96_Anchor")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# --- STRICT USER REQUIREMENT ---
# Warning: If this model name is invalid on Google's side, the fallback logic will catch it.
MODEL_NAME = 'gemini-2.5-flash-lite' 

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Prüfe Environment Variables.")
    sys.exit(1)

# Initialize Global DB Manager
db_manager = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. DATABASE MANAGER (WRAPPER CLASS RESTORED)
# =================================================================
class DatabaseManagerWrapper:
    def __init__(self, client):
        self.client = client

    async def fetch_all_context_data(self):
        logger.info("📡 Fetching Database Context (Parallel)...")
        return await asyncio.gather(
            asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data)
        )

    async def check_existing_match(self, p1_name: str, p2_name: str) -> List[Dict]:
        def _query():
            return self.client.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})"
            ).execute().data
        return await asyncio.to_thread(_query)

    async def insert_match(self, payload: Dict):
        await asyncio.to_thread(lambda: self.client.table("market_odds").insert(payload).execute())

    async def update_match(self, match_id: int, payload: Dict):
        await asyncio.to_thread(lambda: self.client.table("market_odds").update(payload).eq("id", match_id).execute())

# Wrap the client
db = DatabaseManagerWrapper(db_manager)

ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# 3. MATH CORE (WEIGHTED HYBRID MODEL)
# =================================================================
class QuantumMathEngine:
    @staticmethod
    def sigmoid(x: float, sensitivity: float = 0.1) -> float:
        return 1 / (1 + math.exp(-sensitivity * x))

    @staticmethod
    def calculate_fair_probability(ai_tac_p1, ai_phy_p1, skill_p1, skill_p2, elo_p1, elo_p2) -> float:
        prob_ai = ai_tac_p1
        skill_diff = skill_p1 - skill_p2
        prob_skills = QuantumMathEngine.sigmoid(skill_diff, sensitivity=0.12)
        prob_physics = ai_phy_p1
        prob_elo = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))
        
        final_prob = (
            (prob_ai * 0.40) +
            (prob_skills * 0.25) +
            (prob_physics * 0.20) +
            (prob_elo * 0.15)
        )
        return max(0.05, min(0.95, final_prob))

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        if odds1 <= 1 or odds2 <= 1: return 0.5, 0.5
        inv1, inv2 = 1.0/odds1, 1.0/odds2
        margin = inv1 + inv2
        return inv1/margin, inv2/margin

# =================================================================
# 4. UTILITIES
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365|\(\d+\)', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def validate_market(odds1: float, odds2: float) -> bool:
    if odds1 <= 1.01 or odds2 <= 1.01: return False
    if odds1 > 50.0 or odds2 > 50.0: return False 
    margin = (1/odds1) + (1/odds2)
    return 0.85 < margin < 1.30

def clean_time_str(raw_time: str) -> str:
    if not raw_time: return "12:00"
    match = re.search(r'(\d{1,2}:\d{2})', raw_time)
    if match: return match.group(1)
    return "12:00"

# =================================================================
# 5. AI ENGINE (ROBUST FALLBACK)
# =================================================================
class AIEngine:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(1) 

    async def _call_gemini(self, prompt: str, timeout: float = 30.0) -> Optional[Dict]:
        async with self.semaphore:
            await asyncio.sleep(2.0)
            
            # Models to try (Preferred -> Fallback)
            models_to_try = [self.model, 'gemini-1.5-flash', 'gemini-1.0-pro']
            
            for current_model in models_to_try:
                for attempt in range(2):
                    try:
                        async with httpx.AsyncClient() as client:
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self.api_key}"
                            resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
                            
                            if resp.status_code == 200:
                                raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                                return json.loads(raw.replace("```json", "").replace("```", "").strip())
                            
                            if resp.status_code == 404: # Model not found
                                logger.warning(f"⚠️ Model {current_model} not found (404). Switching...")
                                break # Break inner loop to try next model
                                
                            if resp.status_code == 429:
                                await asyncio.sleep(5)
                                continue
                                
                    except Exception:
                        await asyncio.sleep(1)
            
            logger.error("❌ AI completely failed on all models.")
            return None

    async def select_best_court(self, tour_name: str, p1: str, p2: str, candidates: List[Dict]) -> Optional[Dict]:
        if not candidates: return None
        cand_str = "\n".join([f"ID {i}: {c['name']} ({c.get('location', 'Unknown')})" for i, c in enumerate(candidates)])
        prompt = f"""
        TASK: Match Venue Resolution.
        MATCH: {p1} vs {p2} at {tour_name}.
        CANDIDATES:
        {cand_str}
        INSTRUCTIONS: Identify the correct venue (e.g. United Cup city).
        OUTPUT JSON ONLY: {{ "selected_id": 0 }}
        """
        res = await self._call_gemini(prompt, timeout=15.0)
        if res and "selected_id" in res:
            idx = res["selected_id"]
            if 0 <= idx < len(candidates): return candidates[idx]
        return None

    async def analyze_matchup_weighted(self, p1: Dict, p2: Dict, s1: Dict, s2: Dict, r1: Dict, r2: Dict, court: Dict) -> Dict:
        bsi = court.get('bsi_rating', 6.0)
        bounce = court.get('bounce', 'Medium')
        
        prompt = f"""
        ROLE: Expert Tennis Analyst.
        MATCH: {p1['last_name']} ({p1.get('play_style')}) vs {p2['last_name']} ({p2.get('play_style')}).
        
        COURT PHYSICS:
        - Name: {court.get('name')} ({court.get('surface')})
        - BSI (Speed): {bsi}/10
        - Bounce: {bounce}
        - Notes: {court.get('notes', 'N/A')}

        PLAYER DATA:
        - P1: Srv {s1.get('serve')}, Ret {s1.get('speed')}, Men {s1.get('mental')}. Report: {r1.get('strengths')}
        - P2: Srv {s2.get('serve')}, Ret {s2.get('speed')}, Men {s2.get('mental')}. Report: {r2.get('strengths')}

        TASK:
        1. TACTICAL MATCHUP (40% Weight): Analyze styles. Who wins the pattern battle?
           - Score P1 from 0-100 (50 is equal).
        2. COURT FIT (20% Weight): Analyze BSI/Bounce interaction. Who does the court favor?
           - Score P1 from 0-100 (50 is equal).
        3. VERDICT: Detailed analysis of why.

        OUTPUT JSON ONLY:
        {{
            "tactical_score_p1": 55,  
            "physics_score_p1": 45,   
            "analysis_detail": "Detailed breakdown..."
        }}
        """
        return await self._call_gemini(prompt, timeout=30.0) or {
            "tactical_score_p1": 50, "physics_score_p1": 50, "analysis_detail": "AI Timeout"
        }

ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)

# =================================================================
# 6. CONTEXT RESOLVER
# =================================================================
class ContextResolver:
    def __init__(self, db_tournaments):
        self.db_tournaments = db_tournaments
        self.name_map = {t['name'].lower(): t for t in db_tournaments}
        self.lookup_keys = list(self.name_map.keys())

    async def resolve_court_rag(self, scraped_name, p1_name, p2_name):
        s_clean = scraped_name.lower().replace("atp", "").replace("wta", "").strip()
        if s_clean in self.name_map: return self.name_map[s_clean], "Exact"

        candidates = []
        fuzzy = difflib.get_close_matches(s_clean, self.lookup_keys, n=3, cutoff=0.5)
        for fn in fuzzy: 
            if self.name_map[fn] not in candidates: candidates.append(self.name_map[fn])
            
        if "united cup" in s_clean:
            for t in self.db_tournaments:
                if "united cup" in t['name'].lower() and t not in candidates: candidates.append(t)
        
        if candidates:
            selected = await ai_engine.select_best_court(scraped_name, p1_name, p2_name, candidates)
            if selected: return selected, "AI-RAG"
            return candidates[0], "Fuzzy-Fallback"
        
        return {'name': scraped_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium', 'notes': 'Fallback'}, "Default"

# =================================================================
# 7. SCRAPER & ELO
# =================================================================
class ScraperBot:
    def __init__(self):
        self.browser = None
        self.context = None

    async def start(self):
        logger.info("🔌 Starting Playwright Engine...")
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
        self.context = await self.browser.new_context(user_agent="Mozilla/5.0")

    async def stop(self):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()

    async def fetch_page(self, url):
        if not self.context: await self.start()
        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            return await page.content()
        except: return None

async def fetch_elo_ratings_optimized(bot):
    logger.info("📊 Updating Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    for tour, url in urls.items():
        content = await bot.fetch_page(url)
        if not content: continue
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', {'id': 'reportable'})
        if table:
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) > 5:
                    name = normalize_text(cols[0].get_text(strip=True)).lower()
                    try:
                        ELO_CACHE[tour][name] = {
                            'Hard': to_float(cols[3].get_text(strip=True), 1500), 
                            'Clay': to_float(cols[4].get_text(strip=True), 1500), 
                            'Grass': to_float(cols[5].get_text(strip=True), 1500)
                        }
                    except: continue

# =================================================================
# 8. MAIN LOGIC (ANCHOR PARSING - THE FIX)
# =================================================================
async def process_day_url(bot, target_date, players, skills_map, reports, resolver):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    logger.info(f"📅 Scanning: {target_date.strftime('%Y-%m-%d')}")
    
    html = await bot.fetch_page(url)
    if not html: return

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    current_tour_name = "Unknown"
    current_active_date = target_date 

    for table in tables:
        rows = table.find_all("tr")
        
        # --- SILICON VALLEY PARSER: ANCHOR LOGIC ---
        pending_p1 = None
        pending_p1_odds = []
        pending_time = "12:00"

        for i, row in enumerate(rows):
            # 1. Date Header
            if "flags" in row.get("class", []) and "head" not in row.get("class", []):
                txt = row.get_text()
                if "Tomorrow" in txt: 
                    current_active_date = target_date + timedelta(days=1)
                pending_p1 = None # Reset on header
                continue

            # 2. Tournament Header
            if "head" in row.get("class", []):
                link = row.find('a')
                current_tour_name = link.get_text(strip=True) if link else row.get_text(strip=True)
                pending_p1 = None # Reset on tour change
                continue

            # 3. Match Row Parsing
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # CHECK FOR TIME (THE ANCHOR)
            first_col = row.find('td', class_='first')
            has_time = False
            if first_col and 'time' in first_col.get('class', []):
                raw_time = first_col.get_text(strip=True)
                if ':' in raw_time:
                    has_time = True
                    match_time_str = clean_time_str(raw_time)

            # EXTRACT PLAYER NAME CLEANLY
            # Look for strict player links if possible, else text
            p_name_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            
            # Identify Player Object
            p_obj = next((p for p in players if p['last_name'].lower() in p_name_raw.lower()), None)

            if p_obj:
                # IS THIS A START OF A MATCH (P1)?
                if has_time or pending_p1 is None:
                    # New Anchor
                    pending_p1 = p_obj
                    pending_time = match_time_str if has_time else "12:00"
                    
                    # Grab Odds for P1
                    pending_p1_odds = []
                    try:
                        course_tds = row.find_all('td', class_='course')
                        for td in course_tds:
                            val = float(td.get_text(strip=True))
                            if 1.01 <= val < 50.0: pending_p1_odds.append(val)
                    except: pass
                
                # IS THIS THE PARTNER (P2)?
                elif pending_p1 is not None:
                    # We have a P1 waiting, this must be P2
                    p2_obj = p_obj
                    
                    # VALIDATION 1: No self-play
                    if p2_obj['id'] == pending_p1['id']: continue 
                    
                    # VALIDATION 2: Gender/Tour Check (The Phantom Killer)
                    # If Tours are different (ATP vs WTA) and not explicitly Mixed (hard to detect), skip
                    if pending_p1.get('tour') != p2_obj.get('tour'):
                        logger.warning(f"⚠️ Gender Mismatch ignored: {pending_p1['last_name']} ({pending_p1['tour']}) vs {p2_obj['last_name']} ({p2_obj['tour']})")
                        pending_p1 = None # Reset
                        continue

                    # Grab Odds for P2
                    p2_odds_val = 0.0
                    try:
                        course_tds = row.find_all('td', class_='course')
                        for td in course_tds:
                            val = float(td.get_text(strip=True))
                            if 1.01 <= val < 50.0: 
                                p2_odds_val = val
                                break
                    except: pass

                    # Construct Odds Pair
                    m_odds1 = pending_p1_odds[0] if pending_p1_odds else 0.0
                    m_odds2 = p2_odds_val
                    
                    # VALIDATION 3: Market Sanity
                    if validate_market(m_odds1, m_odds2):
                        # --- MATCH FOUND & VALIDATED ---
                        
                        iso_time = f"{current_active_date.strftime('%Y-%m-%d')}T{pending_time}:00Z"
                        
                        # Process Match (DB Check -> Update or Insert)
                        # ... (DB Logic Same as Before) ...
                        existing = await db.check_existing_match(pending_p1['last_name'], p2_obj['last_name'])
                        
                        should_analyze = True
                        if existing:
                            if existing[0].get('actual_winner_name'): should_analyze = False
                            # Result Check
                            if re.search(r'\b[0-7]-[0-7]\b', row_text): should_analyze = False # Finished
                            
                            if not should_analyze:
                                pass # Skip analysis
                            else:
                                await db.update_match(existing[0]['id'], {"match_time": iso_time, "odds1": m_odds1, "odds2": m_odds2})
                                should_analyze = False # Just updated

                        if should_analyze:
                            logger.info(f"✨ Analyzing: {pending_p1['last_name']} vs {p2_obj['last_name']} @ {current_tour_name}")
                            
                            court_db, _ = await resolver.resolve_court_rag(current_tour_name, pending_p1['last_name'], p2_obj['last_name'])
                            s1, s2 = skills_map.get(pending_p1['id'], {}), skills_map.get(p2_obj['id'], {})
                            r1 = next((r for r in reports if r['player_id'] == pending_p1['id']), {})
                            r2 = next((r for r in reports if r['player_id'] == p2_obj['id']), {})
                            
                            ai_data = await ai_engine.analyze_matchup_weighted(pending_p1, p2_obj, s1, s2, r1, r2, court_db)
                            
                            # Math & Save
                            ai_tac_prob = ai_data.get('tactical_score_p1', 50) / 100.0
                            ai_phy_prob = ai_data.get('physics_score_p1', 50) / 100.0
                            skill1 = to_float(s1.get('overall_rating', 50))
                            skill2 = to_float(s2.get('overall_rating', 50))
                            
                            elo_key = 'Hard' # Simplified lookup for brevity
                            e1 = ELO_CACHE.get("ATP", {}).get(pending_p1['last_name'].lower(), {}).get(elo_key) or (skill1 * 15 + 500)
                            e2 = ELO_CACHE.get("ATP", {}).get(p2_obj['last_name'].lower(), {}).get(elo_key) or (skill2 * 15 + 500)
                            
                            prob_final = QuantumMathEngine.calculate_fair_probability(ai_tac_prob, ai_phy_prob, skill1, skill2, e1, e2)
                            mp1, _ = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                            
                            entry = {
                                "player1_name": pending_p1['last_name'], "player2_name": p2_obj['last_name'],
                                "tournament": court_db['name'], "odds1": m_odds1, "odds2": m_odds2,
                                "ai_fair_odds1": round(1/prob_final, 2) if prob_final > 0 else 99,
                                "ai_fair_odds2": round(1/(1-prob_final), 2) if prob_final < 1 else 99,
                                "ai_analysis_text": json.dumps({"edge": f"{(prob_final-mp1)*100:.1f}%", "verdict": ai_data.get("analysis_detail")}),
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": iso_time
                            }
                            await db.insert_match(entry)
                            logger.info(f"   💾 Saved. Edge: {(prob_final-mp1)*100:.1f}%")

                    # RESET AFTER PAIRING
                    pending_p1 = None 
                    pending_p1_odds = []

            # If row was empty or didn't contain a player, we keep pending_p1 for the next row
            # Unless it was a header (handled at top)

# =================================================================
# 9. RUNNER
# =================================================================
async def run_pipeline():
    logger.info("🚀 Neural Scout v96.0 (Anchor Parsing) STARTING...")
    bot = ScraperBot()
    await bot.start()
    try:
        await fetch_elo_ratings_optimized(bot)
        data = await db.fetch_all_context_data()
        players, skills_list, reports, tournaments, _ = data
        if not players: return
        skills_map = {s['player_id']: s for s in skills_list}
        resolver = ContextResolver(tournaments)
        
        today = datetime.now()
        for i in range(14):
            await process_day_url(bot, today + timedelta(days=i), players, skills_map, reports, resolver)
            await asyncio.sleep(1)
    except Exception as e: logger.critical(f"🔥 CRASH: {e}", exc_info=True)
    finally: await bot.stop()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
