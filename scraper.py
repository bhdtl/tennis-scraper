# -*- coding: utf-8 -*-
"""
NeuralScout v3.1 - Fix: Smart Name Matching
Pattern: Token-based Lookup & Gatekeeper
Model: gemini-2.0-flash
Fixes: 100% Skip rate due to 'Surname Initial.' formatting (e.g. 'Sinner J.')
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
from typing import Dict, List, Any, Optional, Tuple

# Third-party imports
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# 1. SYSTEM CONFIGURATION
# =================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout")

# Secrets Management
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.0-flash' 

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ FATAL: Missing Environment Variables. Aborting.")
    sys.exit(1)

# Pipeline Tuning
DAYS_TO_SCRAPE = 10
CONCURRENCY_SCRAPE = 3
CONCURRENCY_AI = 5
MARKET_ANCHOR_WEIGHT = 0.35

# =================================================================
# 2. UTILITY BELT (Smart Matching Logic)
# =================================================================

class Utils:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Removes accents and converts to lowercase."""
        if not text: return ""
        return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn').lower()

    @staticmethod
    def clean_player_name(raw: str) -> str:
        """Removes betting ads and garbage."""
        bad_patterns = [r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'bet365', r'Unibet']
        clean = raw
        for pat in bad_patterns:
            clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
        # Remove trailing single letters often found in scores (e.g. "Name (S.)")
        clean = clean.replace('|', '').strip()
        return clean

    @staticmethod
    def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
        return 1 / (1 + math.exp(-sensitivity * diff))

# =================================================================
# 3. DATA REPOSITORY
# =================================================================

class DataRepository:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def _execute(self, query_lambda):
        try:
            return await asyncio.to_thread(query_lambda)
        except Exception as e:
            logger.error(f"DB IO Error: {e}")
            raise

    async def load_context(self) -> Tuple[List[Dict], Dict, List[Dict]]:
        """Loads static data."""
        logger.info("📡 DB: Fetching Players, Skills, Tournaments...")
        t_players = self._execute(lambda: self.client.table("players").select("*").execute())
        t_skills = self._execute(lambda: self.client.table("player_skills").select("*").execute())
        t_tourneys = self._execute(lambda: self.client.table("tournaments").select("*").execute())

        res = await asyncio.gather(t_players, t_skills, t_tourneys, return_exceptions=True)
        
        players = res[0].data if hasattr(res[0], 'data') else []
        skills_raw = res[1].data if hasattr(res[1], 'data') else []
        tourneys = res[2].data if hasattr(res[2], 'data') else []

        skills_map = {s['player_id']: s for s in skills_raw}
        
        return players, skills_map, tourneys

    async def check_match_exists(self, p1_name: str, p2_name: str) -> Optional[Dict]:
        """Checks if a match is already tracked."""
        # Note: Using strict equality check here might be risky if format differs, 
        # but usually we use the names FROM the DB for this query, which is safe.
        res = await self._execute(
            lambda: self.client.table("market_odds").select("id, actual_winner_name")
            .or_(f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})")
            .execute()
        )
        return res.data[0] if res.data else None

    async def upsert_match(self, payload: Dict, update_id: Optional[int] = None):
        if update_id:
            await self._execute(lambda: self.client.table("market_odds").update(payload).eq("id", update_id).execute())
        else:
            await self._execute(lambda: self.client.table("market_odds").insert(payload).execute())

# =================================================================
# 4. INTELLIGENCE UNIT (Gemini 2.0 Flash)
# =================================================================

class IntelligenceUnit:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(CONCURRENCY_AI)
        self.client = httpx.AsyncClient(timeout=25.0)

    async def close(self):
        await self.client.aclose()

    async def analyze_matchup(self, p1: Dict, p2: Dict, surface: str) -> Dict:
        prompt = f"""
        Role: Tennis Analyst. Match: {p1['last_name']} vs {p2['last_name']} on {surface}.
        JSON Response Only: {{ "p1_tactical_score": 7.5, "p2_tactical_score": 6.0, "ai_text": "Brief reason." }}
        """
        async with self.semaphore:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            try:
                # Randomized jitter to avoid hitting rate limits exactly at same ms
                await asyncio.sleep(random.uniform(0.1, 0.5))
                resp = await self.client.post(url, json=payload)
                if resp.status_code == 200:
                    txt = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(txt.replace("```json", "").replace("```", "").strip())
            except Exception as e:
                logger.warning(f"AI Error: {e}")
            
            return {"p1_tactical_score": 5, "p2_tactical_score": 5, "ai_text": "Analysis Unavailable"}

# =================================================================
# 5. CORE ORCHESTRATOR
# =================================================================

class NeuralScoutEngine:
    def __init__(self):
        self.db = DataRepository(SUPABASE_URL, SUPABASE_KEY)
        self.ai = IntelligenceUnit()
        self.scraper = None 
        
        self.player_map: Dict[str, Dict] = {} 
        self.skills_map: Dict[str, Dict] = {}
        self.tournaments: List[Dict] = []
        self.debug_skips_count = 0 

    async def initialize(self):
        logger.info("⚙️ Booting NeuralScout Engine v3.1...")
        raw_players, self.skills_map, self.tournaments = await self.db.load_context()
        
        # --- ROBUST INDEXING STRATEGY ---
        self.player_map = {}
        for p in raw_players:
            # We index by the normalized LAST NAME.
            # "Jannik Sinner" -> key "sinner"
            # "Carlos Alcaraz" -> key "alcaraz"
            last_name_key = Utils.normalize_text(p['last_name'])
            if last_name_key:
                self.player_map[last_name_key] = p
        
        logger.info(f"✅ Gatekeeper Index Built: {len(self.player_map)} players indexed by Last Name.")

    async def shutdown(self):
        if self.scraper: await self.scraper.close()
        await self.ai.close()
    
    def find_player_in_map(self, scraped_name: str) -> Optional[Dict]:
        """
        Smart Fuzzy Search:
        Input: "Sinner J." OR "J. Sinner" OR "Sinner"
        Logic: Split into tokens. Check if ANY token matches a Last Name in our DB.
        """
        cleaned = Utils.clean_player_name(scraped_name)
        norm = Utils.normalize_text(cleaned)
        
        # Split by space or dots
        tokens = re.split(r'[\s\.]+', norm)
        
        for token in tokens:
            if len(token) < 3: continue # Skip initials like "j" or "da" (too risky)
            
            if token in self.player_map:
                return self.player_map[token]
        
        return None

    def calculate_odds(self, p1_obj, p2_obj, bsi, ai_meta, mo1, mo2):
        s1 = self.skills_map.get(p1_obj['id'], {})
        s2 = self.skills_map.get(p2_obj['id'], {})
        
        # Physics
        p1_score = sum(s1.values()) if s1 else 350
        p2_score = sum(s2.values()) if s2 else 350
        prob_phys = Utils.sigmoid_prob(p1_score - p2_score, 0.05)
        
        # AI
        prob_ai = Utils.sigmoid_prob(ai_meta.get('p1_tactical_score', 5) - ai_meta.get('p2_tactical_score', 5), 0.6)
        
        alpha = (prob_phys * 0.60) + (prob_ai * 0.40)
        
        if mo1 > 1 and mo2 > 1:
            imp1 = 1/mo1
            prob_market = imp1 / (imp1 + (1/mo2))
        else:
            prob_market = 0.5
            
        return (alpha * (1 - MARKET_ANCHOR_WEIGHT)) + (prob_market * MARKET_ANCHOR_WEIGHT)

    async def process_day_scan(self, date_obj: datetime):
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={date_obj.year}&month={date_obj.month}&day={date_obj.day}"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                html = await page.content()
            except Exception as e:
                logger.error(f"Fetch Error: {e}")
                return
            finally:
                await browser.close()

        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all("table", class_="result")
        
        stats = {"total": 0, "processed": 0, "skipped": 0}

        for table in tables:
            rows = table.find_all("tr")
            current_tour = "Unknown"
            
            for i, row in enumerate(rows):
                if "head" in row.get("class", []):
                    current_tour = row.get_text(strip=True)
                    continue

                try:
                    cols = row.find_all('td')
                    if not cols or len(cols) < 2: continue
                    if i + 1 >= len(rows): continue
                    
                    raw_p1_node = row.find('td', class_='t-name')
                    raw_p2_node = rows[i+1].find('td', class_='t-name')
                    
                    if not raw_p1_node or not raw_p2_node: continue
                    
                    raw_p1 = raw_p1_node.get_text(strip=True)
                    raw_p2 = raw_p2_node.get_text(strip=True)
                    
                    stats["total"] += 1
                    
                    # =========================================================
                    # 🔍 NEW: SMART LOOKUP STRATEGY
                    # =========================================================
                    
                    p1_obj = self.find_player_in_map(raw_p1)
                    p2_obj = self.find_player_in_map(raw_p2)
                    
                    if not p1_obj or not p2_obj:
                        stats["skipped"] += 1
                        # DEBUG LOGGING for first 5 skips to verify the fix
                        if self.debug_skips_count < 5:
                            logger.warning(f"⚠️ SKIP MATCH (Debug): '{raw_p1}' OR '{raw_p2}' not found in DB index.")
                            self.debug_skips_count += 1
                        continue 
                    
                    # ✅ MATCH VALIDATED
                    stats["processed"] += 1
                    
                    # Odds Parsing
                    odds = []
                    for r in [row, rows[i+1]]:
                        for o in r.find_all('td', class_='course'):
                            try:
                                v = float(o.get_text(strip=True))
                                if v > 1.0: odds.append(v)
                            except: pass
                    
                    m_odds1 = odds[0] if len(odds) > 0 else 1.0
                    m_odds2 = odds[1] if len(odds) > 1 else 1.0
                    
                    time_cell = row.find('td', class_='first')
                    time_str = time_cell.get_text(strip=True) if time_cell else "12:00"
                    match_time_iso = f"{date_obj.strftime('%Y-%m-%d')}T{time_str}:00Z"
                    
                    existing = await self.db.check_match_exists(p1_obj['last_name'], p2_obj['last_name'])
                    
                    if existing:
                        if not existing.get('actual_winner_name'):
                             await self.db.upsert_match({
                                 "odds1": m_odds1, 
                                 "odds2": m_odds2, 
                                 "match_time": match_time_iso
                             }, update_id=existing['id'])
                    else:
                        logger.info(f"✨ NEW MATCH FOUND: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        
                        surf = "Hard"
                        bsi = 6.5
                        if "clay" in current_tour.lower(): surf, bsi = "Red Clay", 4.0
                        
                        ai_meta = await self.ai.analyze_matchup(p1_obj, p2_obj, surf)
                        prob = self.calculate_odds(p1_obj, p2_obj, bsi, ai_meta, m_odds1, m_odds2)
                        
                        await self.db.upsert_match({
                            "player1_name": p1_obj['last_name'],
                            "player2_name": p2_obj['last_name'],
                            "tournament": current_tour,
                            "surface": surf,
                            "odds1": m_odds1,
                            "odds2": m_odds2,
                            "ai_fair_odds1": round(1/prob, 2) if prob > 0 else 99,
                            "ai_fair_odds2": round(1/(1-prob), 2) if prob < 1 else 99,
                            "ai_analysis_text": ai_meta.get('ai_text'),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "match_time": match_time_iso
                        })

                except Exception as e:
                    logger.error(f"Row Parse Error: {e}")
                    continue

        logger.info(f"📊 {date_obj.strftime('%d.%m')}: Found {stats['total']} | PROCESSED {stats['processed']} | Skipped {stats['skipped']}")

    async def run(self):
        await self.initialize()
        
        tasks = []
        for i in range(DAYS_TO_SCRAPE):
            d = datetime.now() + timedelta(days=i)
            tasks.append(self.process_day_scan(d))
        
        # Batch Execution (Chunk size 2 to save resources)
        chunk_size = 2
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            logger.info(f"🚀 Processing Batch {i // chunk_size + 1}...")
            await asyncio.gather(*chunk)
            
        await self.shutdown()

if __name__ == "__main__":
    try:
        engine = NeuralScoutEngine()
        asyncio.run(engine.run())
    except KeyboardInterrupt: pass
    except Exception as e: logger.critical(f"System Crash: {e}", exc_info=True)
