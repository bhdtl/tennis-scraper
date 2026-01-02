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
logger = logging.getLogger("NeuralScout_Workflow")

# Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.5-pro'

# Fail-Safe Check
if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Environment Variables prüfen.")
    sys.exit(1)

# =================================================================
# 2. DATABASE MANAGER
# =================================================================
class DatabaseManager:
    """
    Manages all Supabase interactions.
    """
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def fetch_all_context_data(self):
        """Fetches all critical context data in parallel."""
        logger.info("📡 Fetching Database Context (Parallel)...")
        return await asyncio.gather(
            asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data)
        )

    async def check_existing_match(self, p1_name: str, p2_name: str) -> List[Dict]:
        """Checks if match exists to allow UPSERT logic."""
        def _query():
            return self.client.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})"
            ).execute().data
        return await asyncio.to_thread(_query)

    async def insert_match(self, payload: Dict):
        await asyncio.to_thread(lambda: self.client.table("market_odds").insert(payload).execute())

    async def update_match(self, match_id: int, payload: Dict):
        await asyncio.to_thread(lambda: self.client.table("market_odds").update(payload).eq("id", match_id).execute())

db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 3. UTILITIES & CONTEXT
# =================================================================
def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365|\(\d+\)', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def validate_market(odds1: float, odds2: float) -> bool:
    """Checks for valid market margins to filter data noise."""
    if odds1 <= 1.01 or odds2 <= 1.01: return False
    if odds1 > 50.0 or odds2 > 50.0: return False 
    margin = (1/odds1) + (1/odds2)
    return 0.85 < margin < 1.30

class ContextResolver:
    """
    Resolves Tournament Names to Database Entities.
    """
    def __init__(self, db_tournaments: List[Dict]):
        self.db_tournaments = db_tournaments
        self.name_map = {t['name'].lower(): t for t in db_tournaments}
        self.lookup_keys = list(self.name_map.keys())

    def resolve_court(self, scraped_tour_name: str, p1_country: str = None, p2_country: str = None) -> Tuple[Optional[Dict], str]:
        s_clean = scraped_tour_name.lower().replace("atp", "").replace("wta", "").strip()
        
        # United Cup Logic
        if "united cup" in s_clean:
            if "sydney" in s_clean: return self._find_exact("Sydney"), "Explicit (Sydney)"
            if "perth" in s_clean: return self._find_exact("Perth"), "Explicit (Perth)"
            generic = self._find_fuzzy("United Cup")
            return (generic, "Generic United Cup") if generic else (None, "Missing")

        # Standard Matching
        if s_clean in self.name_map: return self.name_map[s_clean], "Exact"
        matches = difflib.get_close_matches(s_clean, self.lookup_keys, n=1, cutoff=0.6)
        if matches: return self.name_map[matches[0]], f"Fuzzy ({matches[0]})"
        for key in self.lookup_keys:
            if key in s_clean or s_clean in key: return self.name_map[key], f"Substring ({key})"
        return None, "Fail"

    def _find_exact(self, name_part):
        for key, val in self.name_map.items():
            if name_part.lower() in key: return val
        return None
    
    def _find_fuzzy(self, name_part):
        matches = difflib.get_close_matches(name_part.lower(), self.lookup_keys, n=1, cutoff=0.5)
        return self.name_map[matches[0]] if matches else None

# =================================================================
# 4. DEEP AI ENGINE
# =================================================================
class AIEngine:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(4) 

    async def analyze_matchup_deep(self, p1: Dict, p2: Dict, s1: Dict, s2: Dict, r1: Dict, r2: Dict, court: Dict) -> Dict:
        """
        Deep Analysis Prompt for Gemini.
        """
        bsi = court.get('bsi_rating', 6.0)
        bounce = court.get('bounce', 'Medium')
        
        prompt = f"""
        ROLE: World-Class Tennis Analyst.
        TASK: Analyze {p1['last_name']} vs {p2['last_name']}.

        COURT: {court.get('name')} ({court.get('surface')}) | BSI: {bsi} | Bounce: {bounce}
        Notes: {court.get('notes', 'N/A')}

        P1: {p1['last_name']} | Hand: {p1.get('plays_hand')} | Style: {p1.get('play_style')}
        Skills: Srv {s1.get('serve')}, Ret {s1.get('speed')}, Ment {s1.get('mental')}
        
        P2: {p2['last_name']} | Hand: {p2.get('plays_hand')} | Style: {p2.get('play_style')}
        Skills: Srv {s2.get('serve')}, Ret {s2.get('speed')}, Ment {s2.get('mental')}

        OUTPUT JSON:
        {{
            "physics_analysis": "Sentence on court fit.",
            "tactical_analysis": "Sentence on matchup.",
            "mental_analysis": "Sentence on mental state.",
            "p1_serve_adjust": 0.04,
            "p2_serve_adjust": -0.02,
            "final_verdict": "Summary."
        }}
        """
        async with self.semaphore:
            await asyncio.sleep(1.0) 
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=90.0)
                    if resp.status_code != 200: return {}
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(raw.replace("```json", "").replace("```", "").strip())
            except: return {}

ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)

# =================================================================
# 5. MATH CORE
# =================================================================
class QuantumMathEngine:
    @staticmethod
    def calculate_match_probabilities(p_serve_1: float, p_serve_2: float) -> float:
        """Hierarchical Markov Model for Match Probabilities."""
        def prob_game(p):
            p = max(0.40, min(0.95, p))
            q = 1.0 - p
            deuce = 20 * (p**3) * (q**3) * ((p**2) / (p**2 + q**2))
            return (p**4) + (4 * (p**4) * q) + (10 * (p**4) * (q**2)) + deuce

        p_hold_1 = prob_game(p_serve_1)
        p_hold_2 = prob_game(p_serve_2)
        diff = p_hold_1 - p_hold_2
        p_set_1 = 1.0 / (1.0 + math.exp(-12.0 * diff))
        return (p_set_1**2) + (2 * (p_set_1**2) * (1.0 - p_set_1))

    @staticmethod
    def get_elo_based_p_serve(elo1: float, elo2: float, surface_factor: float) -> float:
        return surface_factor + (0.0003 * (elo1 - elo2))

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        if odds1 <= 1 or odds2 <= 1: return 0.5, 0.5
        inv1, inv2 = 1.0/odds1, 1.0/odds2
        return inv1/(inv1+inv2), inv2/(inv1+inv2)

# =================================================================
# 6. SCRAPER
# =================================================================
class ScraperBot:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def start(self):
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
        self.context = await self.browser.new_context(user_agent="Mozilla/5.0")

    async def stop(self):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()

    async def fetch_page(self, url: str) -> Optional[str]:
        if not self.context: await self.start()
        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            return await page.content()
        except: return None

# ... ELO CACHE ...
ELO_CACHE = {"ATP": {}, "WTA": {}}
async def fetch_elo_ratings_optimized(bot: ScraperBot):
    logger.info("📊 Updating Elo...")
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
                            'Hard': float(cols[3].get_text(strip=True) or 1500), 
                            'Clay': float(cols[4].get_text(strip=True) or 1500), 
                            'Grass': float(cols[5].get_text(strip=True) or 1500)
                        }
                    except: continue

# =================================================================
# 8. PROCESS DAY (RESULT & TOURNAMENT AWARE)
# =================================================================
async def process_day_url(bot: ScraperBot, target_date: datetime, players: List[Dict], skills_map: Dict, reports: List[Dict], resolver: ContextResolver):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    logger.info(f"📅 Scanning: {target_date.strftime('%Y-%m-%d')}")
    
    html = await bot.fetch_page(url)
    if not html: return

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    
    current_tour_name = "Unknown"
    matches_processed = 0

    for table in tables:
        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            
            # --- 1. TOURNAMENT HEADER ---
            if "head" in row.get("class", []):
                link = row.find('a')
                if link: current_tour_name = link.get_text(strip=True)
                else: current_tour_name = row.get_text(strip=True)
                continue

            # --- 2. MATCH ROW PARSING ---
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            if i+1 >= len(rows): continue

            # Time
            match_time_str = "12:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            # Names
            p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
            
            p1 = next((p for p in players if p['last_name'].lower() in p1_raw.lower()), None)
            p2 = next((p for p in players if p['last_name'].lower() in p2_raw.lower()), None)

            if p1 and p2:
                # --- 3. RESULT DETECTION ---
                is_finished = False
                # If Score string detected (e.g. "6-4" or "2 : 0")
                if re.search(r'\b[0-7]-[0-7]\b', row_text): is_finished = True

                # --- 4. ODDS ---
                odds = []
                try:
                    course_tds = row.find_all('td', class_='course') + rows[i+1].find_all('td', class_='course')
                    for td in course_tds:
                        val = float(td.get_text(strip=True))
                        if 1.01 <= val < 50.0: odds.append(val)
                    if len(odds) < 2:
                         nums = [float(x) for x in re.findall(r'\d+\.\d+', row_text + " " + rows[i+1].get_text()) if 1.01 < float(x) < 50.0]
                         if len(nums) >= 2: odds = nums[:2]
                except: pass
                
                m_odds1 = odds[0] if odds else 0.0
                m_odds2 = odds[1] if len(odds)>1 else 0.0
                
                if not validate_market(m_odds1, m_odds2): continue

                # --- 5. UPDATE OR ANALYZE ---
                iso_time = f"{target_date.strftime('%Y-%m-%d')}T{match_time_str}:00Z"
                matches_processed += 1

                existing = await db_manager.check_existing_match(p1['last_name'], p2['last_name'])
                
                if existing:
                    # UPDATE LOGIC:
                    # 1. If winner already set in DB, skip (immutable).
                    if existing[0].get('actual_winner_name'): continue
                    
                    # 2. If finished in Scrape, maybe log it, but don't overwrite winner blindly.
                    if is_finished:
                        logger.info(f"   🏁 Match finished on scrape: {p1['last_name']} vs {p2['last_name']}")
                        continue 

                    # 3. Otherwise, UPDATE ODDS & TIME
                    await db_manager.update_match(existing[0]['id'], {"match_time": iso_time, "odds1": m_odds1, "odds2": m_odds2})
                    continue

                # NEW ANALYSIS (Only if not finished)
                if is_finished: continue

                logger.info(f"✨ Analyzing: {p1['last_name']} vs {p2['last_name']} @ {current_tour_name}")
                
                court_db, _ = resolver.resolve_court(current_tour_name)
                if not court_db: court_db = {'name': current_tour_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium'}
                
                s1 = skills_map.get(p1['id'], {})
                s2 = skills_map.get(p2['id'], {})
                r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                
                ai_data = await ai_engine.analyze_matchup_deep(p1, p2, s1, s2, r1, r2, court_db)
                
                # Math
                elo_key = 'Hard'
                if 'clay' in court_db.get('surface','').lower(): elo_key = 'Clay'
                
                e1 = ELO_CACHE.get("ATP", {}).get(p1['last_name'].lower(), {}).get(elo_key, 1500)
                e2 = ELO_CACHE.get("ATP", {}).get(p2['last_name'].lower(), {}).get(elo_key, 1500)
                
                p1_srv = QuantumMathEngine.get_elo_based_p_serve(e1, e2, 0.64) + ai_data.get('p1_serve_adjust', 0.0)
                p2_srv = QuantumMathEngine.get_elo_based_p_serve(e2, e1, 0.64) + ai_data.get('p2_serve_adjust', 0.0)
                
                prob = QuantumMathEngine.calculate_match_probabilities(p1_srv, p2_srv)
                m_p1, _ = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                
                entry = {
                    "player1_name": p1['last_name'], "player2_name": p2['last_name'], 
                    "tournament": court_db['name'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    "ai_fair_odds1": round(1/prob, 2) if prob > 0.01 else 99,
                    "ai_fair_odds2": round(1/(1-prob), 2) if prob < 0.99 else 99,
                    "ai_analysis_text": json.dumps({
                        "edge": f"{(prob-m_p1)*100:.1f}%",
                        "verdict": ai_data.get("final_verdict"),
                        "physics": ai_data.get("physics_analysis"),
                        "tactics": ai_data.get("tactical_analysis")
                    }),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_time
                }
                await db_manager.insert_match(entry)
                logger.info(f"   💾 Saved. Edge: {(prob-m_p1)*100:.1f}%")

    logger.info(f"✅ Day {target_date.strftime('%d.%m')} finished: {matches_processed} matches checked.")

# =================================================================
# 9. RUNNER (ONE-SHOT FOR WORKFLOW)
# =================================================================
async def run_pipeline():
    logger.info("🚀 Neural Scout v86.2 (Workflow Edition) STARTING...")
    
    bot = ScraperBot()
    await bot.start()
    
    try:
        # 1. Update Elo Cache
        await fetch_elo_ratings_optimized(bot)
        
        # 2. Load DB Context
        players, skills_list, reports, tournaments, _ = await db_manager.fetch_all_context_data()
        if not players: return
        
        skills_map = {s['player_id']: s for s in skills_list}
        resolver = ContextResolver(tournaments)
        
        # 3. 14-Day Scan (One Pass)
        today = datetime.now()
        for i in range(14):
            await process_day_url(bot, today + timedelta(days=i), players, skills_map, reports, resolver)
            # Polite delay between days
            await asyncio.sleep(2)
            
    except Exception as e: logger.critical(f"🔥 CRASH: {e}", exc_info=True)
    finally: await bot.stop()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
