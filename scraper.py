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
# Install: pip install playwright beautifulsoup4 supabase httpx
from playwright.async_api import async_playwright, Browser, BrowserContext
from bs4 import BeautifulSoup
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
logger = logging.getLogger("NeuralScout_v90")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.5-pro'

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

# Initialize Global DB Instance
db_manager = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. MATH CORE (HIERARCHICAL MARKOV MODELS)
# =================================================================
class QuantumMathEngine:
    """
    Implements the professional hierarchical model: 
    Point Prob (p) -> Game (Hold) -> Set -> Match.
    This replaces the simple sigmoid approach for realistic odds.
    """
    
    @staticmethod
    def calculate_match_probabilities(p_serve_1: float, p_serve_2: float) -> float:
        """
        Calculates P(Player 1 wins match) given both players' serve win %.
        Uses exact O'Malley derivation for games and logistic approximation for sets.
        """
        # 1. Game Probability (O'Malley Formula)
        def prob_game(p):
            # Clamp for realism (ATP avg ~64%)
            p = max(0.40, min(0.95, p))
            q = 1.0 - p
            # P(Deuce) = 20 * p^3 * q^3
            # P(Win|Deuce) = p^2 / (p^2 + q^2)
            prob_deuce_win = (p**2) / (p**2 + q**2)
            deuce_term = 20 * (p**3) * (q**3) * prob_deuce_win
            
            # Sum of winning paths: Love, 15, 30, Deuce
            return (p**4) + (4 * (p**4) * q) + (10 * (p**4) * (q**2)) + deuce_term

        p_hold_1 = prob_game(p_serve_1)
        p_hold_2 = prob_game(p_serve_2)

        # 2. Set Probability (Logistic Approximation)
        # Sensitivity 12.0 is empirically derived for standard sets
        diff = p_hold_1 - p_hold_2
        p_set_1 = 1.0 / (1.0 + math.exp(-12.0 * diff))

        # 3. Match Probability (Best of 3)
        # P(2-0) + P(2-1)
        p_match_1 = (p_set_1**2) + (2 * (p_set_1**2) * (1.0 - p_set_1))
        
        return p_match_1

    @staticmethod
    def get_base_p_serve(elo1: float, elo2: float, surface_factor: float) -> float:
        """ 
        Converts Elo Difference to Base Serve %.
        Standard Slope: 0.0003 (3% swing per 100 points difference).
        """
        return surface_factor + (0.0003 * (elo1 - elo2))

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        """
        Removes Bookmaker Vigorish using Multiplicative Method.
        Returns true implied probabilities.
        """
        if odds1 <= 1 or odds2 <= 1: return 0.5, 0.5
        inv1, inv2 = 1.0/odds1, 1.0/odds2
        margin = inv1 + inv2
        return inv1/margin, inv2/margin

# =================================================================
# 3. UTILITIES & HELPER FUNCTIONS
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

def get_last_name(full_name):
    """Extracts last name for robust comparison."""
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

def validate_market(odds1: float, odds2: float) -> bool:
    """Sanity Check: Do these odds make sense? Filters bad data."""
    if odds1 <= 1.01 or odds2 <= 1.01: return False
    if odds1 > 50.0 or odds2 > 50.0: return False 
    margin = (1/odds1) + (1/odds2)
    # Valid market margin typically 1.01 - 1.25. Allow slight arb (0.85) for errors.
    return 0.85 < margin < 1.30

# =================================================================
# 4. AI ENGINE (RAG-LITE & DEEP ANALYSIS)
# =================================================================
class AIEngine:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(4)

    async def _call_gemini(self, prompt: str, timeout: float = 30.0) -> Optional[Dict]:
        async with self.semaphore:
            await asyncio.sleep(0.5) # Polite spacing
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
                    
                    if resp.status_code != 200: return None
                    
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    clean_json = raw.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_json)
            except Exception as e:
                logger.error(f"AI Error: {e}")
                return None

    async def select_best_court(self, tour_name: str, p1: str, p2: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        RAG Logic: Asks Gemini to pick the best DB court from a list of candidates.
        Crucial for United Cup (Perth vs Sydney).
        """
        if not candidates: return None
        
        # Prepare context for AI
        cand_str = "\n".join([f"ID {i}: {c['name']} ({c.get('location', 'Unknown')}) - Surface: {c['surface']}" for i, c in enumerate(candidates)])
        
        prompt = f"""
        TASK: Match Venue Resolution.
        CONTEXT: Match '{p1} vs {p2}' at tournament '{tour_name}'.
        PROBLEM: We need to assign the correct court physics profile from our database.
        
        AVAILABLE CANDIDATES:
        {cand_str}
        
        INSTRUCTIONS:
        1. Identify the likely location based on the tournament and players (e.g. United Cup groups are split by city).
        2. Select the BEST matching candidate ID.
        
        OUTPUT JSON ONLY: {{ "selected_id": 0 }}
        """
        
        res = await self._call_gemini(prompt, timeout=15.0)
        if res and "selected_id" in res:
            idx = res["selected_id"]
            if 0 <= idx < len(candidates):
                return candidates[idx]
        return None

    async def analyze_matchup_deep(self, p1: Dict, p2: Dict, s1: Dict, s2: Dict, r1: Dict, r2: Dict, court: Dict) -> Dict:
        """
        Deep Analysis: Physics, Tactics, Mental. Returns adjustments for the Math Engine.
        """
        bsi = court.get('bsi_rating', 6.0)
        bounce = court.get('bounce', 'Medium')
        
        prompt = f"""
        ROLE: ATP Quantitative Analyst.
        MATCH: {p1['last_name']} vs {p2['last_name']}.
        
        VENUE: {court.get('name')} ({court.get('location')})
        PHYSICS: Surface {court.get('surface')} | BSI {bsi} (Speed) | Bounce {bounce}.
        NOTES: {court.get('notes', 'N/A')}

        P1 ({p1.get('play_style')}): Srv {s1.get('serve')}, Ret {s1.get('speed')}, Men {s1.get('mental')}
        P2 ({p2.get('play_style')}): Srv {s2.get('serve')}, Ret {s2.get('speed')}, Men {s2.get('mental')}

        TASK:
        1. PHYSICS: How does BSI {bsi} affect the matchup?
        2. TACTICS: Key tactical pattern.
        3. ADJUST: Serve win % shift (-0.08 to +0.08) based on physics fit.

        OUTPUT JSON ONLY:
        {{
            "physics_analysis": "Sentence.",
            "tactical_analysis": "Sentence.",
            "mental_analysis": "Sentence.",
            "p1_serve_adjust": 0.02,
            "p2_serve_adjust": -0.01,
            "final_verdict": "Summary."
        }}
        """
        return await self._call_gemini(prompt, timeout=60.0) or {
            "physics_analysis": "AI Timeout", "p1_serve_adjust": 0.0, "p2_serve_adjust": 0.0
        }

ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)

# =================================================================
# 5. CONTEXT RESOLVER
# =================================================================
class ContextResolver:
    """
    Resolves Tournament Names using Database Candidates + AI Selection.
    """
    def __init__(self, db_tournaments: List[Dict]):
        self.db_tournaments = db_tournaments
        self.name_map = {t['name'].lower(): t for t in db_tournaments}
        self.lookup_keys = list(self.name_map.keys())

    async def resolve_court_rag(self, scraped_name: str, p1_name: str, p2_name: str) -> Tuple[Dict, str]:
        s_clean = scraped_name.lower().replace("atp", "").replace("wta", "").strip()
        
        # 1. Exact Match (Fast Path)
        if s_clean in self.name_map: return self.name_map[s_clean], "Exact"

        # 2. Candidate Generation
        candidates = []
        
        # Fuzzy candidates
        fuzzy_names = difflib.get_close_matches(s_clean, self.lookup_keys, n=3, cutoff=0.5)
        for fn in fuzzy_names:
            if self.name_map[fn] not in candidates: candidates.append(self.name_map[fn])
            
        # Specific Logic: United Cup / Australia
        if "united cup" in s_clean:
            for t in self.db_tournaments:
                t_name = t['name'].lower()
                if "united cup" in t_name:
                    if t not in candidates: candidates.append(t)
        
        # 3. AI Selection (if candidates found)
        if candidates:
            selected = await ai_engine.select_best_court(scraped_name, p1_name, p2_name, candidates)
            if selected: return selected, "AI-RAG"
            # Fallback to first candidate if AI fails
            return candidates[0], "Fuzzy-Fallback"
        
        # 4. Generic Default
        return {'name': scraped_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium', 'notes': 'Fallback'}, "Default"

# =================================================================
# 6. SCRAPER & ELO
# =================================================================
class ScraperBot:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def start(self):
        logger.info("🔌 Starting Playwright Engine...")
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

ELO_CACHE = {"ATP": {}, "WTA": {}}

async def fetch_elo_ratings_optimized(bot: ScraperBot):
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
# 7. MAIN LOGIC (DATE & WORKFLOW AWARE)
# =================================================================
async def process_day_url(bot: ScraperBot, target_date: datetime, players: List[Dict], skills_map: Dict, reports: List[Dict], resolver: ContextResolver):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    logger.info(f"📅 Scanning: {target_date.strftime('%Y-%m-%d')}")
    
    html = await bot.fetch_page(url)
    if not html: return

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    current_tour_name = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            
            # --- 1. HEADER PARSING ---
            if "head" in row.get("class", []):
                link = row.find('a')
                current_tour_name = link.get_text(strip=True) if link else row.get_text(strip=True)
                continue

            # --- 2. MATCH PARSING ---
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
                # Result Check
                is_finished = False
                if re.search(r'\b[0-7]-[0-7]\b', row_text): is_finished = True

                # Odds
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

                # --- 3. WORKFLOW LOGIC ---
                iso_time = f"{target_date.strftime('%Y-%m-%d')}T{match_time_str}:00Z"
                
                # Check Existing ID
                existing = db_manager.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1['last_name']},player2_name.eq.{p2['last_name']}),and(player1_name.eq.{p2['last_name']},player2_name.eq.{p1['last_name']})").execute()

                if existing.data:
                    if existing.data[0].get('actual_winner_name'): continue # Finished
                    
                    # Update odds/time
                    db_manager.table("market_odds").update({"match_time": iso_time, "odds1": m_odds1, "odds2": m_odds2}).eq("id", existing.data[0]['id']).execute()
                    continue

                if is_finished: continue

                # --- 4. NEW MATCH ANALYSIS ---
                logger.info(f"✨ New: {p1['last_name']} vs {p2['last_name']} @ {current_tour_name}")
                
                # Resolve Court (AI RAG)
                court_db, method = await resolver.resolve_court_rag(current_tour_name, p1['last_name'], p2['last_name'])
                
                # Prepare AI Data
                s1 = skills_map.get(p1['id'], {})
                s2 = skills_map.get(p2['id'], {})
                r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                
                # Deep AI Analysis
                ai_data = await ai_engine.analyze_matchup_deep(p1, p2, s1, s2, r1, r2, court_db)
                
                # Math Model (O'Malley)
                elo_key = 'Hard'
                if 'clay' in court_db.get('surface','').lower(): elo_key = 'Clay'
                elif 'grass' in court_db.get('surface','').lower(): elo_key = 'Grass'
                
                e1 = ELO_CACHE.get("ATP", {}).get(p1['last_name'].lower(), {}).get(elo_key)
                e2 = ELO_CACHE.get("ATP", {}).get(p2['last_name'].lower(), {}).get(elo_key)
                
                # Fallback Elo (Fix for 1.16 odds bug)
                if not e1: e1 = float(s1.get('overall_rating', 50)) * 15 + 500
                if not e2: e2 = float(s2.get('overall_rating', 50)) * 15 + 500
                
                base_surf = 0.64 if elo_key == 'Hard' else 0.60
                p1_srv = QuantumMathEngine.get_base_p_serve(e1, e2, base_surf) + ai_data.get('p1_serve_adjust', 0.0)
                p2_srv = QuantumMathEngine.get_base_p_serve(e2, e1, base_surf) + ai_data.get('p2_serve_adjust', 0.0)
                
                prob = QuantumMathEngine.calculate_match_probabilities(p1_srv, p2_srv)
                market_p1, _ = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                
                entry = {
                    "player1_name": p1['last_name'], "player2_name": p2['last_name'], 
                    "tournament": court_db['name'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    "ai_fair_odds1": round(1/prob, 2) if prob > 0.01 else 99,
                    "ai_fair_odds2": round(1/(1-prob), 2) if prob < 0.99 else 99,
                    "ai_analysis_text": json.dumps({
                        "edge": f"{(prob-market_p1)*100:.1f}%",
                        "verdict": ai_data.get("final_verdict"),
                        "physics": ai_data.get("physics_analysis"),
                        "tactics": ai_data.get("tactical_analysis"),
                        "math": {"e1": e1, "e2": e2, "p1_srv": round(p1_srv, 3)}
                    }),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_time
                }
                
                db_manager.table("market_odds").insert(entry).execute()
                logger.info(f"   💾 Saved. Edge: {(prob-market_p1)*100:.1f}%")

# =================================================================
# 8. RUNNER
# =================================================================
async def run_pipeline():
    logger.info("🚀 Neural Scout v90.0 (Elite Architect Monolith) STARTING...")
    
    bot = ScraperBot()
    await bot.start()
    
    try:
        await fetch_elo_ratings_optimized(bot)
        
        # Load Context (Parallel)
        p_data = db_manager.table("players").select("*").execute().data
        s_data = db_manager.table("player_skills").select("*").execute().data
        r_data = db_manager.table("scouting_reports").select("*").execute().data
        t_data = db_manager.table("tournaments").select("*").execute().data
        
        if not p_data: return
        
        skills_map = {s['player_id']: s for s in s_data}
        resolver = ContextResolver(t_data)
        
        # 14 Day Loop
        today = datetime.now()
        for i in range(14):
            await process_day_url(bot, today + timedelta(days=i), p_data, skills_map, r_data, resolver)
            await asyncio.sleep(2)

    except Exception as e: logger.critical(f"🔥 CRASH: {e}", exc_info=True)
    finally: await bot.stop()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
