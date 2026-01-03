# -*- coding: utf-8 -*-
"""
NeuralScout v2.1 - Data Integrity Edition (Gatekeeper Protected)
Architect: Senior Backend Specialist
Date: 2026-01-03

CHANGELOG v2.1:
- Added O(1) Player Lookup Hash Map (removed O(n) filtering).
- Implemented "Velvet Rope" Gatekeeper: strict whitelist logic.
- Fixed AttributeError/NoneType crashes via Guard Clauses.
"""

import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Third-party imports
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScoutGatekeeper")

# Secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.0-flash'

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ FATAL: Missing Environment Variables.")
    sys.exit(1)

# Tuning
DAYS_TO_SCRAPE = 10
CONCURRENCY_SCRAPE = 3
CONCURRENCY_AI = 5
MARKET_ANCHOR_WEIGHT = 0.35

# =================================================================
# 2. DATA REPOSITORY
# =================================================================

class DataRepository:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def _execute(self, query_lambda):
        try:
            return await asyncio.to_thread(query_lambda)
        except Exception as e:
            logger.error(f"DB Error: {str(e)}")
            raise

    async def load_context_data(self) -> Tuple[List[Dict], Dict, List[Dict], List[Dict]]:
        logger.info("db: Loading context data...")
        # Parallel Fetch
        t_players = self._execute(lambda: self.client.table("players").select("*").execute())
        t_skills = self._execute(lambda: self.client.table("player_skills").select("*").execute())
        t_reports = self._execute(lambda: self.client.table("scouting_reports").select("*").execute())
        t_tourneys = self._execute(lambda: self.client.table("tournaments").select("*").execute())

        res = await asyncio.gather(t_players, t_skills, t_reports, t_tourneys, return_exceptions=True)
        
        # Validations
        players = res[0].data if hasattr(res[0], 'data') else []
        skills_raw = res[1].data if hasattr(res[1], 'data') else []
        reports = res[2].data if hasattr(res[2], 'data') else []
        tourneys = res[3].data if hasattr(res[3], 'data') else []

        # Optimization: Skills Map O(1)
        skills_map = {s['player_id']: s for s in skills_raw}
        
        return players, skills_map, reports, tourneys

    async def upsert_match(self, match_data: Dict, update_only: bool = False):
        if update_only:
            mid = match_data.pop('id')
            await self._execute(lambda: self.client.table("market_odds").update(match_data).eq("id", mid).execute())
        else:
            await self._execute(lambda: self.client.table("market_odds").insert(match_data).execute())

# =================================================================
# 3. UTILITIES & QUANT CORE
# =================================================================

class QuantCore:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Removes accents and lowers case."""
        if not text: return ""
        return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn').lower()

    @staticmethod
    def clean_player_name(raw: str) -> str:
        """Cleans garbage from scraped strings."""
        bad_patterns = [r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'bet365', r'Unibet']
        clean = raw
        for pat in bad_patterns:
            clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
        clean = clean.replace('|', '').strip()
        return clean

    @staticmethod
    def get_lookup_key(full_name: str) -> str:
        """
        Generates the O(1) Lookup Key.
        Strategy: Use the normalized LAST word (Last Name).
        Example: 'Carlos Alcaraz' -> 'alcaraz'
        """
        if not full_name: return ""
        clean = QuantCore.clean_player_name(full_name)
        norm = QuantCore.normalize_text(clean)
        parts = norm.split()
        return parts[-1] if parts else ""

    @staticmethod
    def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
        return 1 / (1 + math.exp(-sensitivity * diff))

    @staticmethod
    def calculate_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, mo1, mo2, elo_data):
        # Physics weights based on BSI
        w_serve, w_power, w_speed, w_stamina, w_mental = 1.0, 1.0, 1.0, 1.0, 1.0
        if bsi <= 4.0:
            w_serve, w_power = 0.6, 0.7
            w_speed, w_stamina, w_mental = 1.4, 1.5, 1.2
        elif bsi >= 7.5:
            w_serve, w_power = 1.5, 1.4
            w_speed, w_stamina = 0.7, 0.8

        def get_score(s):
            return (s.get('serve',50)*w_serve + s.get('power',50)*w_power + 
                    s.get('speed',50)*w_speed + s.get('stamina',50)*w_stamina + s.get('mental',50)*w_mental)

        p1_phy = get_score(s1)
        p2_phy = get_score(s2)
        prob_phys = QuantCore.sigmoid_prob(p1_phy - p2_phy, 0.05)

        # AI & ELO
        prob_ai = QuantCore.sigmoid_prob(float(ai_meta.get('p1_tactical_score',5)) - float(ai_meta.get('p2_tactical_score',5)), 0.6)
        
        e1 = elo_data.get(p1_name, 1500)
        e2 = elo_data.get(p2_name, 1500)
        prob_elo = 1 / (1 + 10 ** ((e2 - e1) / 400))

        # Alpha Model
        alpha = (prob_phys * 0.45) + (prob_ai * 0.30) + (prob_elo * 0.25)

        # Market Anchor
        if mo1 > 1 and mo2 > 1:
            imp1 = 1/mo1; imp2 = 1/mo2
            prob_market = imp1 / (imp1 + imp2)
        else:
            prob_market = 0.5
            
        return (alpha * (1 - MARKET_ANCHOR_WEIGHT)) + (prob_market * MARKET_ANCHOR_WEIGHT)

# =================================================================
# 4. INTELLIGENCE UNIT (AI)
# =================================================================

class IntelligenceUnit:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(CONCURRENCY_AI)
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def _call_gemini(self, prompt: str) -> Optional[str]:
        async with self.semaphore:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
            }
            try:
                resp = await self.client.post(url, headers=headers, json=payload)
                if resp.status_code != 200: return None
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            except Exception: return None

    async def analyze_matchup(self, p1: Dict, p2: Dict, surface: str) -> Dict:
        prompt = f"""
        Analyze: {p1['last_name']} vs {p2['last_name']} on {surface}.
        JSON ONLY: {{ "p1_tactical_score": 7.5, "p2_tactical_score": 6.0, "ai_text": "Short analysis." }}
        """
        res = await self._call_gemini(prompt)
        default = {"p1_tactical_score": 5, "p2_tactical_score": 5, "ai_text": "No analysis"}
        if not res: return default
        try: return json.loads(res.replace("```json", "").replace("```", "").strip())
        except: return default

# =================================================================
# 5. SCRAPING ENGINE
# =================================================================

class ScrapingEngine:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.semaphore = asyncio.Semaphore(CONCURRENCY_SCRAPE)

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True, args=["--no-sandbox"])

    async def stop(self):
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()

    async def fetch_page_content(self, url: str) -> Optional[str]:
        async with self.semaphore:
            context = await self.browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
            page = await context.new_page()
            try:
                await asyncio.sleep(random.uniform(0.5, 1.5))
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if resp.status != 200: return None
                return await page.content()
            except Exception as e:
                logger.error(f"Scrape Fail: {url} - {e}")
                return None
            finally:
                await page.close()
                await context.close()

# =================================================================
# 6. ORCHESTRATOR (WITH GATEKEEPER)
# =================================================================

class NeuralScoutOrchestrator:
    def __init__(self):
        self.db = DataRepository(SUPABASE_URL, SUPABASE_KEY)
        self.scraper = ScrapingEngine()
        self.ai = IntelligenceUnit(GEMINI_API_KEY, MODEL_NAME)
        self.cache_players = []
        self.cache_skills = {}
        self.cache_tournaments = []
        
        # O(1) Lookup Structure
        self.player_map: Dict[str, Dict] = {} 

    async def initialize(self):
        await self.scraper.start()
        logger.info("⚙️ Initializing System & Indices...")
        
        self.cache_players, self.cache_skills, _, self.cache_tournaments = await self.db.load_context_data()
        
        # --- 1. O(1) PLAYER LOOKUP BUILDER ---
        # Convert List[Dict] to Dict[str, Dict] for instant access
        self.player_map = {}
        for p in self.cache_players:
            # Normalize DB Last Name as Key
            key = QuantCore.get_lookup_key(p['last_name'])
            if key:
                self.player_map[key] = p
                
        logger.info(f"✅ Player Index Built: {len(self.player_map)} keys indexed.")

    async def shutdown(self):
        await self.scraper.stop()
        await self.ai.close()

    async def identify_tournament_context(self, raw_tour_name: str) -> Tuple[str, float, str]:
        # Fast local logic
        norm_name = raw_tour_name.lower()
        for t in self.cache_tournaments:
            if t['name'].lower() in norm_name:
                return t['surface'], float(t['bsi_rating']), t.get('notes', '')
        if "clay" in norm_name: return "Red Clay", 4.0, "Heuristic"
        if "indoor" in norm_name: return "Indoor Hard", 8.0, "Heuristic"
        return "Hard", 6.5, "Default"

    async def process_scraped_match(self, match_info: Dict, date_obj: datetime):
        """
        Processes a match ONLY IF players exist in DB.
        """
        # Extract Raw Names
        raw_p1 = match_info['p1']
        raw_p2 = match_info['p2']

        # Generate Keys for Lookup
        key_p1 = QuantCore.get_lookup_key(raw_p1)
        key_p2 = QuantCore.get_lookup_key(raw_p2)

        # =========================================================
        # --- GATEKEEPER LOGIC (ONLY KNOWN PLAYERS) ---
        # =========================================================
        # Strict Check: Both players must be in our whitelist (DB).
        p1_obj = self.player_map.get(key_p1)
        p2_obj = self.player_map.get(key_p2)

        if not p1_obj or not p2_obj:
            # Silent Fail (Business Requirement) - Log as warning only
            # logger.warning(f"⚠️ Gatekeeper: Skipped {raw_p1} vs {raw_p2} (Not in DB).")
            return 

        # --- If we reach here, players are VALIDATED objects ---
        
        try:
            # Duplicate Check (DB Query)
            existing = await self.db._execute(
                lambda: self.db.client.table("market_odds").select("id, actual_winner_name")
                .or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})")
                .execute()
            )
            
            match_time_iso = f"{date_obj.strftime('%Y-%m-%d')}T{match_info['time']}:00Z"
            
            if existing.data:
                # UPDATE ONLY
                if not existing.data[0].get('actual_winner_name'):
                    await self.db.upsert_match({
                        "id": existing.data[0]['id'],
                        "odds1": match_info['odds1'],
                        "odds2": match_info['odds2'],
                        "match_time": match_time_iso
                    }, update_only=True)
            else:
                # INSERT NEW (Only performed for Validated Players)
                logger.info(f"✨ Analyzing New Match: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                
                surf, bsi, notes = await self.identify_tournament_context(match_info['tour'])
                ai_meta = await self.ai.analyze_matchup(p1_obj, p2_obj, surf)
                
                s1 = self.cache_skills.get(p1_obj['id'], {})
                s2 = self.cache_skills.get(p2_obj['id'], {})
                
                prob = QuantCore.calculate_fair_odds(
                    p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta, 
                    match_info['odds1'], match_info['odds2'], {}
                )
                
                await self.db.upsert_match({
                    "player1_name": p1_obj['last_name'],
                    "player2_name": p2_obj['last_name'],
                    "tournament": match_info['tour'],
                    "surface": surf,
                    "odds1": match_info['odds1'],
                    "odds2": match_info['odds2'],
                    "ai_fair_odds1": round(1/prob, 2) if prob > 0 else 0,
                    "ai_fair_odds2": round(1/(1-prob), 2) if prob < 1 else 0,
                    "ai_analysis_text": ai_meta.get('ai_text'),
                    "match_time": match_time_iso,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        except Exception as e:
            logger.error(f"Pipeline Error processing {raw_p1} vs {raw_p2}: {e}")

    async def execute_scraping_cycle(self):
        logger.info("🔥 Starting Gatekeeper Scraping Cycle...")
        tasks = []
        for i in range(DAYS_TO_SCRAPE):
            d = datetime.now() + timedelta(days=i)
            tasks.append(self._process_single_day(d))
        await asyncio.gather(*tasks)

    async def _process_single_day(self, date_obj: datetime):
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={date_obj.year}&month={date_obj.month}&day={date_obj.day}"
        html = await self.scraper.fetch_page_content(url)
        if not html: return

        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all("table", class_="result")
        
        matches_found = 0
        current_tour = "Unknown"
        
        for table in tables:
            rows = table.find_all("tr")
            for i, row in enumerate(rows):
                if "head" in row.get("class", []):
                    current_tour = row.get_text(strip=True)
                    continue
                
                try:
                    # Parse logic optimized
                    cols = row.find_all('td')
                    if not cols or len(cols) < 2: continue
                    if i + 1 >= len(rows): continue
                    
                    # Name Extraction
                    p1_raw = QuantCore.clean_player_name(row.find('td', class_='t-name').get_text(strip=True))
                    p2_raw = QuantCore.clean_player_name(rows[i+1].find('td', class_='t-name').get_text(strip=True))
                    
                    # Basic Odds Extraction
                    odds = []
                    for r in [row, rows[i+1]]:
                        for o in r.find_all('td', class_='course'):
                            try:
                                v = float(o.get_text(strip=True))
                                if v > 1.0: odds.append(v)
                            except: pass
                    
                    if len(odds) >= 2 and p1_raw and p2_raw:
                        match_info = {
                            "p1": p1_raw, "p2": p2_raw,
                            "tour": current_tour, "time": "12:00", # Simplified time for robustness
                            "odds1": odds[0], "odds2": odds[1]
                        }
                        # Pass to Processor which holds the Gatekeeper
                        await self.process_scraped_match(match_info, date_obj)
                        matches_found += 1
                        
                except Exception: continue
        
        logger.info(f"   📅 {date_obj.strftime('%d.%m')}: Processed {matches_found} potential matches.")

async def main():
    bot = NeuralScoutOrchestrator()
    try:
        await bot.initialize()
        await bot.execute_scraping_cycle()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
