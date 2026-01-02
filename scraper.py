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
from typing import Dict, List, Any, Optional, Tuple, Union

# Third-party imports
# pip install playwright beautifulsoup4 supabase httpx
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
logger = logging.getLogger("NeuralScout_v82")

# Load Secrets from Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.5-pro'

# Fail-Safe Check
if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Bitte GEMINI_API_KEY, SUPABASE_URL und SUPABASE_KEY setzen.")
    sys.exit(1)

# =================================================================
# 2. DATABASE MANAGER (Async Wrapper)
# =================================================================
class DatabaseManager:
    """
    Manages all interactions with Supabase. 
    Uses asyncio.to_thread to prevent blocking the event loop during HTTP requests.
    """
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def fetch_all_context_data(self):
        """
        Fetches Players, Skills, Reports, Tournaments, and Odds in parallel.
        This reduces startup time significantly.
        """
        logger.info("📡 Fetching heavy context data from Supabase...")
        return await asyncio.gather(
            asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data)
        )

    async def check_existing_match(self, p1_name: str, p2_name: str) -> List[Dict]:
        """Checks if a match between these two already exists to avoid duplicates."""
        def _query():
            return self.client.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})"
            ).execute().data
        return await asyncio.to_thread(_query)

    async def insert_match(self, payload: Dict):
        """Inserts a new match record."""
        await asyncio.to_thread(lambda: self.client.table("market_odds").insert(payload).execute())

    async def update_match(self, match_id: int, payload: Dict):
        """Updates an existing match (e.g. odds change, time update)."""
        await asyncio.to_thread(lambda: self.client.table("market_odds").update(payload).eq("id", match_id).execute())

# Initialize Global DB Instance
db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 3. UTILITIES & ENTITY RESOLUTION
# =================================================================
def normalize_text(text: str) -> str:
    """Removes accents and special characters for better matching."""
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    """Removes betting spam from player names scraped from HTML."""
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

class TournamentResolver:
    """
    Silicon Valley Logic: Fuzzy Matching to link 'Scraped Name' to 'DB Entity'.
    This ensures we get the correct BSI (Speed Index) and Physics Notes.
    """
    def __init__(self, db_tournaments: List[Dict]):
        self.db_tournaments = db_tournaments
        # Create a map for O(1) exact lookup
        self.name_map = {t['name'].lower(): t for t in db_tournaments}
        self.lookup_keys = list(self.name_map.keys())

    def resolve(self, scraped_name: str) -> Tuple[Optional[Dict], str]:
        """
        Tries to find the DB tournament object based on the scraped name.
        Returns: (TournamentDict, Method_Used)
        """
        if not scraped_name: return None, "Empty"
        
        s_clean = scraped_name.lower().replace("atp", "").replace("wta", "").replace("challaenger", "").strip()
        
        # 1. Exact Match
        if s_clean in self.name_map:
            return self.name_map[s_clean], "Exact"

        # 2. Fuzzy Match (Levenshtein Distance)
        # cutoff=0.6 means 60% similarity required
        matches = difflib.get_close_matches(s_clean, self.lookup_keys, n=1, cutoff=0.6)
        if matches:
            return self.name_map[matches[0]], f"Fuzzy ({matches[0]})"
        
        # 3. Substring fallback (e.g. 'Brisbane' in 'Brisbane International')
        for key in self.lookup_keys:
            if key in s_clean or s_clean in key:
                return self.name_map[key], f"Substring ({key})"

        return None, "Fail"

# =================================================================
# 4. AI & LOGIC ENGINE
# =================================================================
class AIEngine:
    """
    Handles communication with Gemini API.
    Includes Rate Limiting (Semaphore) and Structured Prompting.
    """
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        # Limit to 5 concurrent requests to avoid API bans
        self.semaphore = asyncio.Semaphore(5)

    async def analyze_matchup(self, p1: Dict, p2: Dict, s1: Dict, s2: Dict, r1: Dict, r2: Dict, court: Dict) -> Dict:
        """
        Generates a deep tactical analysis and physics-based adjustments.
        It does NOT predict the winner directly, but adjusts the 'Serve Win %' for the Math Engine.
        """
        bsi = court.get('bsi_rating', 'Unknown')
        bounce = court.get('bounce', 'Unknown')
        notes = court.get('notes', 'No specific notes.')
        
        # Construct a detailed context for the LLM
        prompt = f"""
        ACT AS: Senior ATP Quantitative Analyst & Physics Expert.
        
        CONTEXT:
        We are modeling a professional tennis match to find betting value (+EV).
        
        COURT PHYSICS (Crucial):
        - Tournament: {court.get('name', 'Unknown')} ({court.get('surface', 'Hard')})
        - BSI (Bounce Speed Index): {bsi}/10 (Higher is faster)
        - Bounce Height: {bounce}
        - Court Notes: {notes}
        
        PLAYER A: {p1['last_name']} ({p1.get('play_style', 'Unknown')})
        - Skills (0-100): Serve {s1.get('serve')}, Return {s1.get('speed')}, Mental {s1.get('mental')}, Power {s1.get('power')}
        - Strengths: {r1.get('strengths', 'N/A')}
        - Weaknesses: {r1.get('weaknesses', 'N/A')}
        
        PLAYER B: {p2['last_name']} ({p2.get('play_style', 'Unknown')})
        - Skills (0-100): Serve {s2.get('serve')}, Return {s2.get('speed')}, Mental {s2.get('mental')}, Power {s2.get('power')}
        - Strengths: {r2.get('strengths', 'N/A')}
        - Weaknesses: {r2.get('weaknesses', 'N/A')}
        
        TASK:
        1. Analyze how the COURT PHYSICS (Speed/Bounce) interact with Player Styles. (e.g. Does the low bounce hurt P1's extreme grip? Does the speed favor P2's flat serve?)
        2. Identify ONE specific tactical mismatch.
        3. Estimate a "Physics Adjustment" for P1's Serve Win % (Base is usually ~64%).
           - If P1 is favored by conditions/matchup: +0.02 to +0.06
           - If P1 is disadvantaged: -0.02 to -0.06
           - If Neutral: 0.0
           
        OUTPUT JSON ONLY (No markdown, no intro):
        {{
            "analysis_short": "One concise, sharp sentence for the betting dashboard.",
            "p1_serve_adjust": 0.02, 
            "p2_serve_adjust": -0.01,
            "confidence_score": 8.5
        }}
        """
        
        async with self.semaphore:
            # Random sleep to jitter requests and be polite to API
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
                    }
                    resp = await client.post(url, json=payload, timeout=60.0)
                    
                    if resp.status_code != 200: 
                        logger.error(f"AI API Error {resp.status_code}: {resp.text}")
                        return {}
                    
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    # Clean potential markdown wrappers
                    clean_json = raw.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_json)
            except Exception as e:
                logger.error(f"AI Exception: {e}")
                return {}

# Initialize AI
ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)

# =================================================================
# 5. MATH CORE (Hierarchical Markov Models)
# =================================================================
class QuantumMathEngine:
    """
    Implements the O'Malley/Barnett & Clarke hierarchical model.
    Chain: Elo -> p_serve -> P_game -> P_set -> P_match.
    """
    
    @staticmethod
    def probability_game_win(p_serve: float) -> float:
        """
        Calculates probability of winning a service game given p_serve.
        Uses exact derivation including Deuce.
        """
        # Clamp input to realistic tennis values
        p_serve = max(0.40, min(0.95, p_serve))
        q = 1.0 - p_serve
        
        # Probability to reach Deuce (40-40)
        # Combinations(6,3) * p^3 * q^3 = 20 * p^3 * q^3
        # Win from Deuce: p^2 / (p^2 + q^2)
        prob_deuce = 20 * (p_serve**3) * (q**3) * ((p_serve**2) / (p_serve**2 + q**2))
        
        # Sum of winning paths before Deuce + Winning via Deuce
        # Love (4-0): p^4
        # 15 (4-1): 4 * p^4 * q
        # 30 (4-2): 10 * p^4 * q^2
        return (p_serve**4) + (4 * (p_serve**4) * q) + (10 * (p_serve**4) * (q**2)) + prob_deuce

    @staticmethod
    def probability_set_win(p_hold_a: float, p_hold_b: float) -> float:
        """
        Approximation of Set Win Probability based on Hold percentages.
        Using a logistic regression derived from ATP data.
        """
        diff = p_hold_a - p_hold_b
        # Sensitivity factor 12.0 is empirically derived for standard sets
        return 1 / (1 + math.exp(-12.0 * diff))

    @staticmethod
    def probability_match_win(p_set_a: float, p_set_b: float) -> float:
        """
        Calculates Match Win Probability for Best of 3 Sets.
        P_match = P(2-0) + P(2-1)
        """
        p = p_set_a 
        # Note: This assumes p_set is constant, simplified from p_set1 vs p_set2
        return (p*p) + (2 * (p*p) * (1-p))

    @staticmethod
    def get_base_p_serve(elo_diff: float, surface_factor: float) -> float:
        """
        Regresses Elo Difference to Expected Service Points Won %.
        slope = 0.0003 implies 3% swing for every 100 Elo points difference.
        """
        return surface_factor + (0.0003 * elo_diff)

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        """
        Removes bookmaker margin using the Multiplicative method.
        Returns the 'True' implied probabilities.
        """
        if odds1 <= 1 or odds2 <= 1: return 0.5, 0.5
        inv1, inv2 = 1.0/odds1, 1.0/odds2
        margin = inv1 + inv2
        
        # Calculate true probabilities
        true_p1 = inv1 / margin
        true_p2 = inv2 / margin
        
        return true_p1, true_p2

# =================================================================
# 6. SCRAPING LAYER
# =================================================================
class ScraperBot:
    """
    Manages Playwright Browser instances.
    Implements reuse of BrowserContext for performance.
    """
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def start(self):
        logger.info("🔌 Starting Playwright Engine...")
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
        # Use a realistic user agent to avoid bot detection
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

    async def stop(self):
        logger.info("🔌 Stopping Playwright Engine...")
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()

    async def fetch_page(self, url: str) -> Optional[str]:
        if not self.context: await self.start()
        try:
            page = await self.context.new_page()
            # 60s timeout for slow TennisExplorer pages
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            await page.close()
            return content
        except Exception as e:
            logger.warning(f"⚠️ Page Fetch Error ({url}): {e}")
            return None

# =================================================================
# 7. ELO RATING CACHE
# =================================================================
ELO_CACHE = {"ATP": {}, "WTA": {}}

async def fetch_elo_ratings_optimized(bot: ScraperBot):
    """
    Scrapes current Elo ratings from TennisAbstract.
    Fills the global ELO_CACHE.
    """
    logger.info("📊 Updating Elo Ratings from TennisAbstract...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    for tour, url in urls.items():
        content = await bot.fetch_page(url)
        if not content: continue
        
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', {'id': 'reportable'})
        
        if table:
            count = 0
            rows = table.find_all('tr')[1:] 
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 5:
                    name = normalize_text(cols[0].get_text(strip=True)).lower()
                    try:
                        # Extract Surface Specific Elos
                        # Col 3: Hard, Col 4: Clay, Col 5: Grass
                        hard = float(cols[3].get_text(strip=True) or 1500)
                        clay = float(cols[4].get_text(strip=True) or 1500)
                        grass = float(cols[5].get_text(strip=True) or 1500)
                        
                        ELO_CACHE[tour][name] = {
                            'Hard': hard, 'Clay': clay, 'Grass': grass
                        }
                        count += 1
                    except: continue
            logger.info(f"   ✅ Loaded {count} {tour} ratings.")
        else:
            logger.error(f"   ❌ Could not find Elo table for {tour}.")

# =================================================================
# 8. MAIN PROCESSING PIPELINE
# =================================================================
async def process_day(bot: ScraperBot, date_target: datetime, players: List[Dict], skills_map: Dict, reports: List[Dict], resolver: TournamentResolver):
    """
    Processes a single day of matches.
    Scrapes -> Resolves Entities -> AI Analysis -> Math Model -> DB Update.
    """
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={date_target.year}&month={date_target.month}&day={date_target.day}"
    logger.info(f"📅 Scanning Date: {date_target.strftime('%Y-%m-%d')}")
    
    html = await bot.fetch_page(url)
    if not html: return

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    
    current_tour_name = "Unknown"
    match_count = 0
    
    for table in tables:
        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            # 1. Detect Tournament Header
            if "head" in row.get("class", []): 
                current_tour_name = row.get_text(strip=True)
                continue # Skip header row

            # 2. Basic Row Text Extraction
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            if i+1 >= len(rows): continue
            
            # 3. Check for Time
            match_time_str = "12:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            # 4. Extract Player Names
            # Logic: Player 1 is in current row, Player 2 is in next row
            p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
            
            # 5. Match Players to DB (In-Memory Lookup)
            p1 = next((p for p in players if p['last_name'].lower() in p1_raw.lower()), None)
            p2 = next((p for p in players if p['last_name'].lower() in p2_raw.lower()), None)
            
            # Only process if both players are tracked in our system
            if p1 and p2:
                # 6. Extract Market Odds
                odds = []
                try:
                    # Combine texts to find odds
                    txt = row_text + " " + rows[i+1].get_text()
                    # Find floats like 1.50, 2.30
                    nums = [float(x) for x in re.findall(r'\d+\.\d+', txt) if 1.0 < float(x) < 50.0]
                    if len(nums) >= 2: odds = nums[:2]
                except: pass
                
                m_odds1 = odds[0] if odds else 0.0
                m_odds2 = odds[1] if len(odds)>1 else 0.0
                
                # Skip invalid odds
                if m_odds1 <= 1.0 or m_odds2 <= 1.0: continue

                # 7. Check for Existing Match in DB
                existing = await db_manager.check_existing_match(p1['last_name'], p2['last_name'])
                
                # Construct ISO Timestamp
                iso_time = f"{date_target.strftime('%Y-%m-%d')}T{match_time_str}:00Z"
                
                if existing:
                    # Match exists. If it has a winner, it's finished -> LOCK.
                    if existing[0].get('actual_winner_name'):
                        continue
                    
                    # If active, update odds & time
                    await db_manager.update_match(existing[0]['id'], {
                        "odds1": m_odds1, "odds2": m_odds2, "match_time": iso_time
                    })
                    continue # Skip recalculation for speed
                
                # =====================================================
                # NEW MATCH DETECTED - START DEEP ANALYSIS
                # =====================================================
                match_count += 1
                logger.info(f"✨ Analyzing: {p1['last_name']} vs {p2['last_name']} @ {current_tour_name}")

                # A. RESOLVE COURT (Context)
                court_db, method = resolver.resolve(current_tour_name)
                
                # Default / Fallback Court Data
                if not court_db:
                    court_db = {'name': current_tour_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium', 'notes': 'Default fallback'}
                    logger.warning(f"   ⚠️ Court not found: {current_tour_name}. Using Defaults.")
                else:
                    logger.info(f"   🏟️ Court Matched: {court_db['name']} (BSI: {court_db.get('bsi_rating')}) via {method}")
                
                # B. PREPARE AI DATA
                s1 = skills_map.get(p1['id'], {})
                s2 = skills_map.get(p2['id'], {})
                r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                
                # C. CALL AI ENGINE
                ai_data = await ai_engine.analyze_matchup(p1, p2, s1, s2, r1, r2, court_db)
                
                # D. QUANTUM MATH CALCULATION
                # 1. Determine Surface Key for Elo
                elo_key = 'Hard'
                base_surf_factor = 0.64
                
                surf_lower = court_db.get('surface', '').lower()
                if 'clay' in surf_lower: 
                    elo_key = 'Clay'
                    base_surf_factor = 0.60
                elif 'grass' in surf_lower: 
                    elo_key = 'Grass'
                    base_surf_factor = 0.67
                elif 'indoor' in surf_lower:
                    elo_key = 'Hard'
                    base_surf_factor = 0.68 # Indoor is faster, higher hold rate
                
                # 2. Get Elo
                e1 = ELO_CACHE.get("ATP", {}).get(p1['last_name'].lower(), {}).get(elo_key, 1500)
                e2 = ELO_CACHE.get("ATP", {}).get(p2['last_name'].lower(), {}).get(elo_key, 1500)
                
                # 3. Calculate Base P_Serve (Elo Regression)
                p_serve_1_base = QuantumMathEngine.get_base_p_serve(e1 - e2, base_surf_factor)
                p_serve_2_base = QuantumMathEngine.get_base_p_serve(e2 - e1, base_surf_factor)
                
                # 4. Apply AI Physics Adjustments
                adj1 = ai_data.get('p1_serve_adjust', 0.0)
                adj2 = ai_data.get('p2_serve_adjust', 0.0)
                
                p_serve_1_final = p_serve_1_base + adj1
                p_serve_2_final = p_serve_2_base + adj2
                
                # 5. Hierarchical Markov Chain
                # Game Probability
                p_hold_1 = QuantumMathEngine.probability_game_win(p_serve_1_final)
                p_hold_2 = QuantumMathEngine.probability_game_win(p_serve_2_final)
                
                # Set Probability
                p_set_1 = QuantumMathEngine.probability_set_win(p_hold_1, p_hold_2)
                
                # Match Probability (Best of 3)
                p_match = QuantumMathEngine.probability_match_win(p_set_1, 1.0 - p_set_1)
                
                # E. VALUE CALCULATION (Edge)
                market_prob_p1, _ = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                edge = p_match - market_prob_p1
                
                # F. SAVE TO DB
                entry = {
                    "player1_name": p1['last_name'], "player2_name": p2['last_name'], 
                    "tournament": court_db['name'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    # Convert Prob to Odds (1/P)
                    "ai_fair_odds1": round(1/p_match, 2) if p_match > 0.01 else 99,
                    "ai_fair_odds2": round(1/(1-p_match), 2) if p_match < 0.99 else 99,
                    # Store structured analysis
                    "ai_analysis_text": json.dumps({
                        "edge": f"{edge*100:.1f}%",
                        "analysis": ai_data.get("analysis_short", "Analysis Pending"),
                        "math_details": {
                            "elo_diff": e1-e2,
                            "surface": elo_key,
                            "ai_adjust": adj1,
                            "p_serve_projected": round(p_serve_1_final, 3)
                        }
                    }),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_time
                }
                
                await db_manager.insert_match(entry)
                logger.info(f"   💾 Saved: Edge {edge*100:.1f}% | AI: {entry['ai_fair_odds1']}")

    logger.info(f"✅ Finished Day: {match_count} matches analyzed.")

# =================================================================
# 9. RUNNER
# =================================================================
async def run_pipeline():
    logger.info("🚀 Neural Scout v82.0 (Silicon Valley Architect Edition) STARTING...")
    
    # 1. Start Scraper Engine
    bot = ScraperBot()
    await bot.start()
    
    try:
        # 2. Update Elo Cache First
        await fetch_elo_ratings_optimized(bot)
        
        # 3. Load Context Data (Parallel DB Fetch)
        logger.info("📥 Loading Database Context...")
        players, skills_list, reports, tournaments, _ = await db_manager.fetch_all_context_data()
        
        # Create fast lookup map for Skills
        skills_map = {s['player_id']: s for s in skills_list if s.get('player_id')}
        
        if not players:
            logger.critical("❌ No Players in DB. Aborting.")
            return

        # 4. Initialize Entity Resolver
        resolver = TournamentResolver(tournaments)
        
        # 5. Main Processing Loop (Next 7 Days)
        today = datetime.now()
        days_to_scan = 7 
        
        for i in range(days_to_scan):
            target_date = today + timedelta(days=i)
            # Await each day sequentially to manage memory/rate-limits, 
            # but internal HTTP/DB calls are async.
            await process_day(bot, target_date, players, skills_map, reports, resolver)
            
            # Cool-down to prevent scraping bans
            await asyncio.sleep(2)

    except Exception as e:
        logger.critical(f"🔥 PIPELINE CRASH: {e}", exc_info=True)
    finally:
        await bot.stop()
        logger.info("🏁 Neural Scout Pipeline Finished.")

if __name__ == "__main__":
    # Windows Selector Event Loop Fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_pipeline())
