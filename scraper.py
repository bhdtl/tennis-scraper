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
logger = logging.getLogger("NeuralScout_v83")

# Environment Variables
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
    Manages all Supabase interactions.
    Wraps blocking calls in asyncio.to_thread to keep the event loop running.
    """
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def fetch_all_context_data(self):
        """
        Fetches all critical context data in parallel to speed up startup.
        """
        logger.info("📡 Fetching Database Context (Parallel)...")
        return await asyncio.gather(
            asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data)
        )

    async def check_existing_match(self, p1_name: str, p2_name: str) -> List[Dict]:
        """Checks for existing matches to prevent duplicates or identify updates."""
        def _query():
            return self.client.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})"
            ).execute().data
        return await asyncio.to_thread(_query)

    async def insert_match(self, payload: Dict):
        """Inserts a new match record."""
        await asyncio.to_thread(lambda: self.client.table("market_odds").insert(payload).execute())

    async def update_match(self, match_id: int, payload: Dict):
        """Updates odds or time for an existing match."""
        await asyncio.to_thread(lambda: self.client.table("market_odds").update(payload).eq("id", match_id).execute())

# Initialize Global DB Instance
db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 3. UTILITIES & CONTEXT RESOLUTION
# =================================================================
def normalize_text(text: str) -> str:
    """Normalizes unicode characters (e.g. ø -> o)."""
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    """Cleans up scraper noise from player names."""
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365|\(\d+\)', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

class ContextResolver:
    """
    The 'Brain' that connects scraping data to database entities.
    Handles Fuzzy Matching AND Specific Tournament Logic (e.g. United Cup).
    """
    def __init__(self, db_tournaments: List[Dict]):
        self.db_tournaments = db_tournaments
        self.name_map = {t['name'].lower(): t for t in db_tournaments}
        self.lookup_keys = list(self.name_map.keys())

    def resolve_court(self, scraped_tour_name: str, p1_country: str = None, p2_country: str = None) -> Tuple[Optional[Dict], str]:
        """
        Determines the EXACT court/venue.
        Handles multi-venue events like 'United Cup' by checking for city names in the string.
        """
        s_clean = scraped_tour_name.lower().replace("atp", "").replace("wta", "").strip()

        # --- SPECIAL LOGIC: UNITED CUP / MULTI-VENUE ---
        if "united cup" in s_clean:
            # Check if the scraper string explicitly mentions the city
            if "sydney" in s_clean:
                return self._find_exact("Sydney"), "Explicit (Sydney)"
            if "perth" in s_clean:
                return self._find_exact("Perth"), "Explicit (Perth)"
            
            # If the string is just "United Cup", we look for a generic "United Cup" entry in DB
            # which should represent an average Hard Court (BSI 7.5).
            generic_match = self._find_fuzzy("United Cup")
            if generic_match:
                return generic_match, "Generic United Cup Profile"
            
            # Fallback if no generic profile exists
            return None, "United Cup (Missing DB Entry)"

        # --- STANDARD FUZZY MATCHING ---
        # 1. Exact Match
        if s_clean in self.name_map: 
            return self.name_map[s_clean], "Exact"
        
        # 2. Fuzzy Match (Levenshtein)
        matches = difflib.get_close_matches(s_clean, self.lookup_keys, n=1, cutoff=0.6)
        if matches: 
            return self.name_map[matches[0]], f"Fuzzy ({matches[0]})"
        
        # 3. Substring Match
        for key in self.lookup_keys:
            if key in s_clean or s_clean in key: 
                return self.name_map[key], f"Substring ({key})"

        # 4. Fallback
        return None, "Fail"

    def _find_exact(self, name_part):
        for key, val in self.name_map.items():
            if name_part.lower() in key: return val
        return None
    
    def _find_fuzzy(self, name_part):
        matches = difflib.get_close_matches(name_part.lower(), self.lookup_keys, n=1, cutoff=0.5)
        return self.name_map[matches[0]] if matches else None

# =================================================================
# 4. DEEP AI ENGINE (CHAIN OF THOUGHT)
# =================================================================
class AIEngine:
    """
    Handles Gemini API with advanced prompting for deep analysis.
    """
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        # Limit concurrency to ensure quality and respect rate limits
        self.semaphore = asyncio.Semaphore(4) 

    async def analyze_matchup_deep(self, p1: Dict, p2: Dict, s1: Dict, s2: Dict, r1: Dict, r2: Dict, court: Dict) -> Dict:
        """
        Performs a deep-dive analysis with specific weighting instructions.
        """
        bsi = court.get('bsi_rating', 6.0)
        bounce = court.get('bounce', 'Medium')
        
        # Extended Prompt for Detail
        prompt = f"""
        ROLE: World-Class Tennis Handicapper & Data Scientist.
        TASK: Analyze {p1['last_name']} vs {p2['last_name']} for +EV betting.

        === DATA LAYER ===
        [COURT PHYSICS]
        Tournament: {court.get('name')} | Surface: {court.get('surface')}
        BSI (Speed): {bsi}/10 (Higher is faster) | Bounce: {bounce}
        Notes: {court.get('notes', 'N/A')}

        [PLAYER A: {p1['last_name']}]
        Style: {p1.get('play_style')} | Hand: {p1.get('plays_hand')}
        Skills (0-100): Serve {s1.get('serve')}, Return {s1.get('speed')}, Mental {s1.get('mental')}, Power {s1.get('power')}
        Report Strengths: {r1.get('strengths', 'N/A')}
        Report Weaknesses: {r1.get('weaknesses', 'N/A')}

        [PLAYER B: {p2['last_name']}]
        Style: {p2.get('play_style')} | Hand: {p2.get('plays_hand')}
        Skills (0-100): Serve {s2.get('serve')}, Return {s2.get('speed')}, Mental {s2.get('mental')}, Power {s2.get('power')}
        Report Strengths: {r2.get('strengths', 'N/A')}
        Report Weaknesses: {r2.get('weaknesses', 'N/A')}

        === ANALYSIS PROTOCOL (CHAIN OF THOUGHT) ===
        1. PHYSICS FIT (Weight: 30%): Analyze how BSI/Bounce interacts with swing paths. (e.g. Does high bounce target a one-handed backhand?)
        2. TACTICAL MATRIX (Weight: 40%): Specific matchup patterns (e.g. Lefty Serve vs weak Backhand Return).
        3. INTANGIBLES (Weight: 30%): Mental resilience, motivation, recent form signals.
        
        === OUTPUT REQUIREMENT ===
        Return VALID JSON ONLY. No markdown.
        {{
            "physics_analysis": "Detailed 2 sentences on court interaction...",
            "tactical_analysis": "Detailed 2 sentences on strategic patterns...",
            "mental_analysis": "Assessment of mental edge...",
            "p1_serve_adjust": 0.04,  // Float: -0.10 to +0.10 adjustment to base Serve Win %
            "p2_serve_adjust": -0.02, // Negative means player struggles
            "final_verdict": "Comprehensive summary explaining the edge."
        }}
        """
        
        async with self.semaphore:
            await asyncio.sleep(1.0) # Polite spacing
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                    # Increased timeout for longer generation
                    resp = await client.post(url, json=payload, timeout=90.0)
                    
                    if resp.status_code != 200: 
                        logger.error(f"AI API Error: {resp.status_code} - {resp.text}")
                        return {}
                    
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    clean_json = raw.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_json)
            except Exception as e:
                logger.error(f"AI Deep Analysis Failed: {e}")
                return {
                    "physics_analysis": "AI Service Unreachable", 
                    "tactical_analysis": "N/A", 
                    "p1_serve_adjust": 0.0, 
                    "p2_serve_adjust": 0.0,
                    "final_verdict": "Analysis failed due to API error."
                }

# Initialize AI
ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)

# =================================================================
# 5. MATH & DE-VIGGING CORE
# =================================================================
class QuantumMathEngine:
    """
    The Mathematical Core.
    Uses Hierarchical Markov Models to derive Match Probability from Service Points.
    """
    
    @staticmethod
    def calculate_match_probabilities(p_serve_1: float, p_serve_2: float) -> float:
        """
        Calculates P(Player 1 wins match) given both players' serve win %.
        """
        # 1. Game Probability (O'Malley Formula)
        def prob_game(p):
            # Clamp for realism
            p = max(0.40, min(0.95, p))
            q = 1.0 - p
            # P(Deuce) * P(Win|Deuce)
            deuce = 20 * (p**3) * (q**3) * ((p**2) / (p**2 + q**2))
            # Sum of winning paths
            return (p**4) + (4 * (p**4) * q) + (10 * (p**4) * (q**2)) + deuce

        p_hold_1 = prob_game(p_serve_1)
        p_hold_2 = prob_game(p_serve_2)

        # 2. Set Probability (Logistic Approximation for Speed)
        # Using sensitivity 12.0 based on ATP hard court stats
        diff = p_hold_1 - p_hold_2
        p_set_1 = 1.0 / (1.0 + math.exp(-12.0 * diff))

        # 3. Match Probability (Best of 3)
        # P(2-0) + P(2-1)
        # P(2-0) = p_set^2
        # P(2-1) = 2 * p_set^2 * (1-p_set)
        p_match_1 = (p_set_1**2) + (2 * (p_set_1**2) * (1.0 - p_set_1))
        
        return p_match_1

    @staticmethod
    def get_elo_based_p_serve(elo1: float, elo2: float, surface_factor: float) -> float:
        """ 
        Converts Elo Difference to Base Serve %.
        Standard Slope: 0.0003 (3% swing per 100 points) 
        """
        return surface_factor + (0.0003 * (elo1 - elo2))

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        """
        Removes Bookmaker Vigorish using Multiplicative Method.
        """
        if odds1 <= 1 or odds2 <= 1: return 0.5, 0.5
        inv1, inv2 = 1.0/odds1, 1.0/odds2
        margin = inv1 + inv2
        return inv1/margin, inv2/margin

# =================================================================
# 6. SCRAPER (DATE & CONTEXT AWARE)
# =================================================================
class ScraperBot:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def start(self):
        logger.info("🔌 Starting Playwright Engine...")
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
        # User Agent to avoid anti-bot blocks
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
            # 60s timeout for stability
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            await page.close()
            return content
        except Exception as e:
            logger.warning(f"⚠️ Page Fetch Error ({url}): {e}")
            return None

# =================================================================
# 7. ELO CACHE LOGIC
# =================================================================
ELO_CACHE = {"ATP": {}, "WTA": {}}

async def fetch_elo_ratings_optimized(bot: ScraperBot):
    """
    Fetches surface-specific Elo ratings.
    """
    logger.info("📊 Updating Elo Ratings...")
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
                        # Col 3: Hard, Col 4: Clay, Col 5: Grass
                        ELO_CACHE[tour][name] = {
                            'Hard': float(cols[3].get_text(strip=True) or 1500), 
                            'Clay': float(cols[4].get_text(strip=True) or 1500), 
                            'Grass': float(cols[5].get_text(strip=True) or 1500)
                        }
                        count += 1
                    except: continue
            logger.info(f"   ✅ Loaded {count} {tour} ratings.")

# =================================================================
# 8. MAIN PROCESSING LOGIC (FIXED DATE HANDLING)
# =================================================================
async def process_calendar_scan(bot: ScraperBot, start_date: datetime, players: List[Dict], skills_map: Dict, reports: List[Dict], resolver: ContextResolver):
    """
    Scrapes the match list and intelligently parses headers to determine the REAL match date.
    Fixes the issue where tomorrow's matches are listed under today's URL.
    """
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={start_date.year}&month={start_date.month}&day={start_date.day}"
    logger.info(f"📅 Scraper initialized for Base Date: {start_date.strftime('%Y-%m-%d')}")
    
    html = await bot.fetch_page(url)
    if not html: return

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    
    current_tour_name = "Unknown"
    
    # --- DATE CONTEXT TRACKER ---
    # We start with the base date, but update this as we encounter date headers.
    current_active_date = start_date 
    
    for table in tables:
        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            
            # A. DETECT DATE HEADERS (The Fix)
            # TennisExplorer puts dates in 'flags' or 't-name' rows
            if "flags" in row.get("class", []):
                header_text = row.get_text(separator=' ', strip=True)
                
                # Check for "Tomorrow"
                if "Tomorrow" in header_text:
                    current_active_date = start_date + timedelta(days=1)
                    logger.info(f"   🕒 Detected 'Tomorrow' header -> Switching date to {current_active_date.strftime('%Y-%m-%d')}")
                
                # Check for specific dates like "05.01."
                elif re.search(r'\d{1,2}\.\d{1,2}\.', header_text):
                    try:
                        day_month = re.search(r'(\d{1,2})\.(\d{1,2})\.', header_text)
                        new_day = int(day_month.group(1))
                        new_month = int(day_month.group(2))
                        
                        # Handle year wrap (e.g. scanning in Dec, match is in Jan)
                        year = start_date.year
                        if new_month < start_date.month and start_date.month == 12: 
                            year += 1
                        elif new_month > start_date.month and start_date.month == 1:
                            year -= 1 # Unlikely but safe

                        current_active_date = current_active_date.replace(year=year, month=new_month, day=new_day)
                        logger.info(f"   🕒 Detected Date header -> Switching date to {current_active_date.strftime('%Y-%m-%d')}")
                    except: pass
                continue

            # B. DETECT TOURNAMENT HEADER
            if "head" in row.get("class", []):
                current_tour_name = row.get_text(strip=True)
                continue

            # C. PARSE MATCH ROW
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            if i+1 >= len(rows): continue
            
            # Extract Time
            match_time_str = "12:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            # Extract Names
            p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
            
            # DB Lookup
            p1 = next((p for p in players if p['last_name'].lower() in p1_raw.lower()), None)
            p2 = next((p for p in players if p['last_name'].lower() in p2_raw.lower()), None)

            if p1 and p2:
                # Odds Extraction
                odds = []
                try:
                    txt = row_text + " " + rows[i+1].get_text()
                    nums = [float(x) for x in re.findall(r'\d+\.\d+', txt) if 1.0 < float(x) < 50.0]
                    if len(nums) >= 2: odds = nums[:2]
                except: pass
                
                m_odds1 = odds[0] if odds else 0.0
                m_odds2 = odds[1] if len(odds)>1 else 0.0
                
                if m_odds1 <= 1.0: continue

                # D. CHECK EXISTING MATCH (and update time if needed)
                existing = await db_manager.check_existing_match(p1['last_name'], p2['last_name'])
                
                # Construct ISO String using the CORRECT current_active_date
                iso_time = f"{current_active_date.strftime('%Y-%m-%d')}T{match_time_str}:00Z"
                
                if existing:
                    if existing[0].get('actual_winner_name'): continue
                    # Update existing match with corrected time/date
                    await db_manager.update_match(existing[0]['id'], {
                        "match_time": iso_time, "odds1": m_odds1, "odds2": m_odds2
                    })
                    continue

                # =====================================================
                # E. NEW MATCH DETECTED - ANALYZE
                # =====================================================
                logger.info(f"✨ Analyzing: {p1['last_name']} vs {p2['last_name']} @ {current_tour_name}")

                # 1. RESOLVE COURT (United Cup Logic)
                court_db, resolution_method = resolver.resolve_court(current_tour_name, p1.get('country'), p2.get('country'))
                
                if not court_db:
                    court_db = {'name': current_tour_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium', 'notes': 'Fallback Court'}
                    logger.warning(f"   ⚠️ Using Fallback Court for {current_tour_name}")
                else:
                    logger.info(f"   🏟️ Court Resolved: {court_db['name']} via {resolution_method}")
                
                # 2. GATHER AI DATA
                s1 = skills_map.get(p1['id'], {})
                s2 = skills_map.get(p2['id'], {})
                r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                
                # 3. CALL DEEP AI ANALYSIS
                ai_data = await ai_engine.analyze_matchup_deep(p1, p2, s1, s2, r1, r2, court_db)
                
                # 4. MATH CALCULATION
                # Detect Elo Surface
                elo_key = 'Hard'
                surf_lower = court_db.get('surface','').lower()
                if 'clay' in surf_lower: elo_key = 'Clay'
                elif 'grass' in surf_lower: elo_key = 'Grass'
                
                # Base Surface Factor
                surf_factor = 0.64 # Hard
                if elo_key == 'Clay': surf_factor = 0.60
                
                # Fetch Elo
                e1 = ELO_CACHE.get("ATP", {}).get(p1['last_name'].lower(), {}).get(elo_key, 1500)
                e2 = ELO_CACHE.get("ATP", {}).get(p2['last_name'].lower(), {}).get(elo_key, 1500)
                
                # Base P_Serve + AI Adjustments
                p_serve_1 = QuantumMathEngine.get_elo_based_p_serve(e1, e2, surf_factor) + ai_data.get('p1_serve_adjust', 0.0)
                p_serve_2 = QuantumMathEngine.get_elo_based_p_serve(e2, e1, surf_factor) + ai_data.get('p2_serve_adjust', 0.0)
                
                # Final Match Probability
                prob_match = QuantumMathEngine.calculate_match_probabilities(p_serve_1, p_serve_2)
                
                # 5. SAVE RESULT
                market_p1, _ = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                edge = prob_match - market_p1
                
                entry = {
                    "player1_name": p1['last_name'], "player2_name": p2['last_name'], 
                    "tournament": court_db['name'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    "ai_fair_odds1": round(1/prob_match, 2) if prob_match > 0.01 else 99,
                    "ai_fair_odds2": round(1/(1-prob_match), 2) if prob_match < 0.99 else 99,
                    # RICH JSON ANALYSIS STORAGE
                    "ai_analysis_text": json.dumps({
                        "edge": f"{edge*100:.1f}%",
                        "verdict": ai_data.get("final_verdict", "Pending"),
                        "physics": ai_data.get("physics_analysis", "Pending"),
                        "tactics": ai_data.get("tactical_analysis", "Pending"),
                        "mental": ai_data.get("mental_analysis", "Pending")
                    }),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_time
                }
                
                await db_manager.insert_match(entry)
                logger.info(f"   💾 Saved Match. Edge: {edge*100:.1f}% | Date: {current_active_date.strftime('%d.%m.')}")

# =================================================================
# 9. RUNNER
# =================================================================
async def run_pipeline():
    logger.info("🚀 Neural Scout v83.0 (Time-Aware Architect Edition) STARTING...")
    
    bot = ScraperBot()
    await bot.start()
    
    try:
        # 1. Update Elo Cache
        await fetch_elo_ratings_optimized(bot)
        
        # 2. Load DB Context
        logger.info("📥 Loading Database...")
        players, skills_list, reports, tournaments, _ = await db_manager.fetch_all_context_data()
        skills_map = {s['player_id']: s for s in skills_list if s.get('player_id')}
        resolver = ContextResolver(tournaments)
        
        if not players:
            logger.critical("❌ No players found in DB.")
            return

        # 3. Calendar Scan
        # We start scanning from TODAY.
        # The `process_calendar_scan` logic will handle +1, +2 days correctly by reading headers.
        today = datetime.now()
        
        # Optional: You can loop this if you want to scan next week's URL specifically, 
        # but usually Today's URL contains upcoming matches for the next 2-3 days.
        await process_calendar_scan(bot, today, players, skills_map, reports, resolver)

    except Exception as e:
        logger.critical(f"🔥 PIPELINE CRASH: {e}", exc_info=True)
    finally:
        await bot.stop()
        logger.info("🏁 Pipeline Finished.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
