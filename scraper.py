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
logger = logging.getLogger("NeuralScout_v91")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.5-pro'

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Prüfe Environment Variables.")
    sys.exit(1)

# Initialize Global DB Instance
db_manager = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. MATH CORE (WEIGHTED HYBRID MODEL)
# =================================================================
class QuantumMathEngine:
    """
    Implements the Weighted Hybrid Model:
    - AI Matchup (40%)
    - Skills (25%)
    - Court Physics (20%)
    - Elo (15%)
    """
    
    @staticmethod
    def sigmoid(x: float, sensitivity: float = 0.1) -> float:
        """Converts a difference (e.g. Skill Diff) into a 0-1 probability."""
        return 1 / (1 + math.exp(-sensitivity * x))

    @staticmethod
    def calculate_fair_probability(
        ai_tactical_p1: float, # 0.0 to 1.0 (from AI)
        ai_physics_p1: float,  # 0.0 to 1.0 (from AI based on BSI)
        skill_p1: float,       # 0-100
        skill_p2: float,       # 0-100
        elo_p1: float,
        elo_p2: float
    ) -> float:
        
        # 1. AI Matchup (40%)
        # Input is already probability-like from AI (e.g. 0.60 for P1)
        prob_ai = ai_tactical_p1
        
        # 2. Skills (25%)
        # Diff of overall ratings. Sensitivity 0.15 means 10 points diff ~ 80% win prob.
        skill_diff = skill_p1 - skill_p2
        prob_skills = QuantumMathEngine.sigmoid(skill_diff, sensitivity=0.12)
        
        # 3. Court Physics (20%)
        # AI determines who fits the court better (0.0 - 1.0)
        prob_physics = ai_physics_p1
        
        # 4. Elo (15%)
        # Standard Elo Formula
        prob_elo = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))
        
        # WEIGHTED SUM
        final_prob = (
            (prob_ai * 0.40) +
            (prob_skills * 0.25) +
            (prob_physics * 0.20) +
            (prob_elo * 0.15)
        )
        
        # Clamp to avoid 1.0 or 0.0 (Implies infinite odds)
        return max(0.05, min(0.95, final_prob))

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        """Removes Bookmaker Margin (Vig) to get True Market Prob."""
        if odds1 <= 1 or odds2 <= 1: return 0.5, 0.5
        inv1, inv2 = 1.0/odds1, 1.0/odds2
        margin = inv1 + inv2
        return inv1/margin, inv2/margin

# =================================================================
# 3. UTILITIES
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

# =================================================================
# 4. AI ENGINE (DEEP ANALYSIS & SCORING)
# =================================================================
class AIEngine:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(4) 

    async def _call_gemini(self, prompt: str, timeout: float = 40.0) -> Optional[Dict]:
        async with self.semaphore:
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
                    if resp.status_code != 200: return None
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(raw.replace("```json", "").replace("```", "").strip())
            except Exception as e:
                logger.error(f"AI Error: {e}")
                return None

    async def select_best_court(self, tour_name: str, p1: str, p2: str, candidates: List[Dict]) -> Optional[Dict]:
        """Resolves venue using RAG candidates."""
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
        """
        Generates Scores for the Weighted Model.
        """
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
            "tactical_score_p1": 55,  // P1 has slight tactical edge
            "physics_score_p1": 45,   // Court slightly favors P2
            "analysis_detail": "Detailed breakdown of why P1's forehand dominates P2's backhand despite the court speed..."
        }}
        """
        return await self._call_gemini(prompt, timeout=60.0) or {
            "tactical_score_p1": 50, "physics_score_p1": 50, "analysis_detail": "AI Timeout"
        }

ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)

# =================================================================
# 5. CONTEXT RESOLVER
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
# 6. SCRAPER & ELO
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

ELO_CACHE = {"ATP": {}, "WTA": {}}

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
# 7. MAIN LOGIC (WEIGHTED CALCULATION)
# =================================================================
async def process_day_url(bot, target_date, players, skills_map, reports, resolver):
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
            
            # 1. Header
            if "head" in row.get("class", []):
                link = row.find('a')
                current_tour_name = link.get_text(strip=True) if link else row.get_text(strip=True)
                continue

            # 2. Row
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            if i+1 >= len(rows): continue

            match_time_str = "12:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
            
            p1 = next((p for p in players if p['last_name'].lower() in p1_raw.lower()), None)
            p2 = next((p for p in players if p['last_name'].lower() in p2_raw.lower()), None)

            if p1 and p2:
                # 3. Check Result
                is_finished = False
                if re.search(r'\b[0-7]-[0-7]\b', row_text): is_finished = True

                # 4. Odds
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

                iso_time = f"{target_date.strftime('%Y-%m-%d')}T{match_time_str}:00Z"
                
                # Check Existing
                existing = db_manager.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1['last_name']},player2_name.eq.{p2['last_name']}),and(player1_name.eq.{p2['last_name']},player2_name.eq.{p1['last_name']})").execute()
                
                if existing.data:
                    if existing.data[0].get('actual_winner_name'): continue 
                    if is_finished: continue 
                    db_manager.table("market_odds").update({"match_time": iso_time, "odds1": m_odds1, "odds2": m_odds2}).eq("id", existing.data[0]['id']).execute()
                    continue

                if is_finished: continue

                # 5. NEW ANALYSIS
                logger.info(f"✨ Analyzing: {p1['last_name']} vs {p2['last_name']} @ {current_tour_name}")
                court_db, _ = await resolver.resolve_court_rag(current_tour_name, p1['last_name'], p2['last_name'])
                
                s1 = skills_map.get(p1['id'], {})
                s2 = skills_map.get(p2['id'], {})
                r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                
                # AI Step
                ai_data = await ai_engine.analyze_matchup_weighted(p1, p2, s1, s2, r1, r2, court_db)
                
                # Retrieve Inputs for Weights
                # A. Tactical Score (AI 40%) -> 0-100 to 0.0-1.0
                ai_tac_prob = ai_data.get('tactical_score_p1', 50) / 100.0
                
                # B. Physics Score (AI 20%)
                ai_phy_prob = ai_data.get('physics_score_p1', 50) / 100.0
                
                # C. Skills (25%)
                skill1 = to_float(s1.get('overall_rating', 50))
                skill2 = to_float(s2.get('overall_rating', 50))
                
                # D. Elo (15%)
                elo_key = 'Hard'
                if 'clay' in court_db.get('surface','').lower(): elo_key = 'Clay'
                elif 'grass' in court_db.get('surface','').lower(): elo_key = 'Grass'
                
                e1 = ELO_CACHE.get("ATP", {}).get(p1['last_name'].lower(), {}).get(elo_key)
                e2 = ELO_CACHE.get("ATP", {}).get(p2['last_name'].lower(), {}).get(elo_key)
                if not e1: e1 = skill1 * 15 + 500
                if not e2: e2 = skill2 * 15 + 500
                
                # CALCULATION (Weighted)
                prob_final = QuantumMathEngine.calculate_fair_probability(
                    ai_tac_prob, ai_phy_prob, skill1, skill2, e1, e2
                )
                
                market_p1, _ = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                
                entry = {
                    "player1_name": p1['last_name'], "player2_name": p2['last_name'], "tournament": court_db['name'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    "ai_fair_odds1": round(1/prob_final, 2) if prob_final > 0 else 99,
                    "ai_fair_odds2": round(1/(1-prob_final), 2) if prob_final < 1 else 99,
                    "ai_analysis_text": json.dumps({
                        "edge": f"{(prob_final-market_p1)*100:.1f}%",
                        "verdict": ai_data.get("analysis_detail"),
                        "scores": {
                            "tactical": f"{ai_data.get('tactical_score_p1')}/100",
                            "physics": f"{ai_data.get('physics_score_p1')}/100",
                            "skills": f"{skill1} vs {skill2}",
                            "elo": f"{e1:.0f} vs {e2:.0f}"
                        }
                    }),
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_time
                }
                
                db_manager.table("market_odds").insert(entry).execute()
                logger.info(f"   💾 Saved. Edge: {(prob-market_p1)*100:.1f}%") # Edge calc for log

# =================================================================
# 8. RUNNER
# =================================================================
async def run_pipeline():
    logger.info("🚀 Neural Scout v91.0 (Weighted Hybrid) STARTING...")
    
    bot = ScraperBot()
    await bot.start()
    
    try:
        await fetch_elo_ratings_optimized(bot)
        
        data = await db_manager.fetch_all_context_data()
        players, skills_list, reports, tournaments, _ = data
        
        if not players: return
        
        skills_map = {s['player_id']: s for s in skills_list}
        resolver = ContextResolver(tournaments)
        
        today = datetime.now()
        for i in range(14):
            await process_day_url(bot, today + timedelta(days=i), players, skills_map, reports, resolver)
            await asyncio.sleep(2)

    except Exception as e: logger.critical(f"🔥 CRASH: {e}", exc_info=True)
    finally: await bot.stop()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_pipeline())
