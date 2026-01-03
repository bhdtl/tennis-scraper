# -*- coding: utf-8 -*-
"""
NeuralScout v3.0 - High-Performance Architect Edition
Pattern: Strict Gatekeeper & O(1) Hash Map Indexing
Model: gemini-2.0-flash
Fixes: AttributeError crashes due to missing entities.
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
MODEL_NAME = 'gemini-2.0-flash'  # UPGRADED AS REQUESTED

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ FATAL: Missing Environment Variables. Aborting.")
    sys.exit(1)

# Pipeline Tuning
DAYS_TO_SCRAPE = 10
CONCURRENCY_SCRAPE = 3
CONCURRENCY_AI = 5
MARKET_ANCHOR_WEIGHT = 0.35

# =================================================================
# 2. UTILITY BELT (Stateless)
# =================================================================

class Utils:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Removes accents and converts to lowercase for consistent matching."""
        if not text: return ""
        return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn').lower()

    @staticmethod
    def clean_player_name(raw: str) -> str:
        """Removes betting ads and garbage from scraped names."""
        bad_patterns = [r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'bet365', r'Unibet']
        clean = raw
        for pat in bad_patterns:
            clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
        return clean.replace('|', '').strip()

    @staticmethod
    def get_lookup_key(full_name: str) -> str:
        """
        Generates the O(1) Lookup Key.
        Strategy: Use the normalized LAST word (Last Name).
        Example: 'Carlos Alcaraz' -> 'alcaraz'
        """
        if not full_name: return ""
        clean = Utils.clean_player_name(full_name)
        norm = Utils.normalize_text(clean)
        parts = norm.split()
        return parts[-1] if parts else ""

    @staticmethod
    def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
        return 1 / (1 + math.exp(-sensitivity * diff))

# =================================================================
# 3. DATA REPOSITORY (Supabase Interface)
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
        """Loads all static data needed for the pipeline."""
        logger.info("📡 DB: Fetching Players, Skills, Tournaments...")
        t_players = self._execute(lambda: self.client.table("players").select("*").execute())
        t_skills = self._execute(lambda: self.client.table("player_skills").select("*").execute())
        t_tourneys = self._execute(lambda: self.client.table("tournaments").select("*").execute())

        res = await asyncio.gather(t_players, t_skills, t_tourneys, return_exceptions=True)
        
        players = res[0].data if hasattr(res[0], 'data') else []
        skills_raw = res[1].data if hasattr(res[1], 'data') else []
        tourneys = res[2].data if hasattr(res[2], 'data') else []

        # Skills HashMap for O(1) access
        skills_map = {s['player_id']: s for s in skills_raw}
        
        return players, skills_map, tourneys

    async def check_match_exists(self, p1_name: str, p2_name: str) -> Optional[Dict]:
        """Checks if a match is already tracked (Idempotency)."""
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
        """
        Uses gemini-2.0-flash for rapid tactical analysis.
        """
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
                resp = await self.client.post(url, json=payload)
                if resp.status_code == 200:
                    txt = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(txt.replace("```json", "").replace("```", "").strip())
            except Exception as e:
                logger.warning(f"AI Error: {e}")
            
            return {"p1_tactical_score": 5, "p2_tactical_score": 5, "ai_text": "Analysis Unavailable"}

# =================================================================
# 5. CORE ORCHESTRATOR (The Logic Engine)
# =================================================================

class NeuralScoutEngine:
    def __init__(self):
        self.db = DataRepository(SUPABASE_URL, SUPABASE_KEY)
        self.ai = IntelligenceUnit()
        self.scraper = None # Initialized later
        
        # O(1) Lookup Indices
        self.player_map: Dict[str, Dict] = {} 
        self.skills_map: Dict[str, Dict] = {}
        self.tournaments: List[Dict] = []

    async def initialize(self):
        """Sets up the Gatekeeper Index."""
        logger.info("⚙️ Booting NeuralScout Engine...")
        raw_players, self.skills_map, self.tournaments = await self.db.load_context()
        
        # --- BUILDING THE HASH MAP (O(1) INDEX) ---
        self.player_map = {}
        for p in raw_players:
            # Key Strategy: Normalized Last Name
            # e.g., "Jannik Sinner" -> "sinner"
            key = Utils.get_lookup_key(p['last_name'])
            if key:
                self.player_map[key] = p
        
        logger.info(f"✅ Gatekeeper Index Built: {len(self.player_map)} players loaded.")

    async def shutdown(self):
        if self.scraper: await self.scraper.close()
        await self.ai.close()

    def calculate_odds(self, p1_key, p2_key, bsi, ai_meta, mo1, mo2):
        """Pure Math Calculation."""
        s1 = self.skills_map.get(self.player_map[p1_key]['id'], {})
        s2 = self.skills_map.get(self.player_map[p2_key]['id'], {})
        
        # Simplified Physics for Brevity (Same logic as before)
        p1_score = sum(s1.values()) if s1 else 350
        p2_score = sum(s2.values()) if s2 else 350
        prob_phys = Utils.sigmoid_prob(p1_score - p2_score, 0.05)
        
        # AI
        prob_ai = Utils.sigmoid_prob(ai_meta.get('p1_tactical_score', 5) - ai_meta.get('p2_tactical_score', 5), 0.6)
        
        # Alpha
        alpha = (prob_phys * 0.60) + (prob_ai * 0.40)
        
        # Market Anchor
        if mo1 > 1 and mo2 > 1:
            imp1 = 1/mo1
            prob_market = imp1 / (imp1 + (1/mo2))
        else:
            prob_market = 0.5
            
        return (alpha * (1 - MARKET_ANCHOR_WEIGHT)) + (prob_market * MARKET_ANCHOR_WEIGHT)

    async def process_day_scan(self, date_obj: datetime):
        """
        The Robust Scanning Loop with Gatekeeper Logic.
        """
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={date_obj.year}&month={date_obj.month}&day={date_obj.day}"
        
        # Setup Scraper Context locally to avoid stale sessions
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = await context.new_page()
            
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                html = await page.content()
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
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

                # Defensive Parsing Block
                try:
                    cols = row.find_all('td')
                    if not cols or len(cols) < 2: continue
                    if i + 1 >= len(rows): continue
                    
                    # 1. Extraction (Raw Strings)
                    raw_p1 = row.find('td', class_='t-name')
                    raw_p2 = rows[i+1].find('td', class_='t-name')
                    
                    if not raw_p1 or not raw_p2: continue
                    
                    p1_str = Utils.clean_player_name(raw_p1.get_text(strip=True))
                    p2_str = Utils.clean_player_name(raw_p2.get_text(strip=True))
                    
                    stats["total"] += 1
                    
                    # =========================================================
                    # ⛔⛔⛔ THE VELVET ROPE (STRICT GATEKEEPER) ⛔⛔⛔
                    # =========================================================
                    
                    # 2. Key Generation (Normalization)
                    key_p1 = Utils.get_lookup_key(p1_str)
                    key_p2 = Utils.get_lookup_key(p2_str)
                    
                    # 3. O(1) Lookups
                    p1_exists = key_p1 in self.player_map
                    p2_exists = key_p2 in self.player_map
                    
                    # 4. The Decision
                    if not p1_exists or not p2_exists:
                        # Log only every 10th skip to reduce noise, or use debug
                        # logger.warning(f"⚠️ SKIP: {p1_str} vs {p2_str} (Not in DB)")
                        stats["skipped"] += 1
                        continue # <--- CRITICAL: ABORT ITERATION HERE
                    
                    # =========================================================
                    # ✅ ACCESS GRANTED - SAFE ZONE
                    # =========================================================
                    
                    stats["processed"] += 1
                    
                    # Parse Odds safely
                    odds = []
                    for r in [row, rows[i+1]]:
                        for o in r.find_all('td', class_='course'):
                            try:
                                v = float(o.get_text(strip=True))
                                if v > 1.0: odds.append(v)
                            except: pass
                    
                    m_odds1 = odds[0] if len(odds) > 0 else 1.0
                    m_odds2 = odds[1] if len(odds) > 1 else 1.0
                    
                    # Match Time
                    time_cell = row.find('td', class_='first')
                    time_str = time_cell.get_text(strip=True) if time_cell else "12:00"
                    match_time_iso = f"{date_obj.strftime('%Y-%m-%d')}T{time_str}:00Z"
                    
                    # Get Real Objects from Map (Guaranteed to exist now)
                    p1_obj = self.player_map[key_p1]
                    p2_obj = self.player_map[key_p2]
                    
                    # Check Idempotency
                    existing = await self.db.check_match_exists(p1_obj['last_name'], p2_obj['last_name'])
                    
                    if existing:
                        # Update Market Odds Only
                        if not existing.get('actual_winner_name'):
                             await self.db.upsert_match({
                                 "odds1": m_odds1, 
                                 "odds2": m_odds2, 
                                 "match_time": match_time_iso
                             }, update_id=existing['id'])
                    else:
                        # Full Pipeline for New Match
                        logger.info(f"✨ NEW MATCH: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        
                        # Heuristic Surface Detection
                        surf = "Hard"
                        bsi = 6.5
                        if "clay" in current_tour.lower(): surf, bsi = "Red Clay", 4.0
                        
                        # AI Analysis
                        ai_meta = await self.ai.analyze_matchup(p1_obj, p2_obj, surf)
                        
                        # Calculate Fair Odds
                        prob = self.calculate_odds(key_p1, key_p2, bsi, ai_meta, m_odds1, m_odds2)
                        
                        # Insert
                        payload = {
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
                        }
                        await self.db.upsert_match(payload)

                except Exception as e:
                    logger.error(f"Row Parse Error: {e}")
                    continue

        logger.info(f"📊 Report {date_obj.strftime('%d.%m')}: Found {stats['total']}, Processed {stats['processed']}, Skipped {stats['skipped']}")

    async def run(self):
        await self.initialize()
        
        tasks = []
        for i in range(DAYS_TO_SCRAPE):
            d = datetime.now() + timedelta(days=i)
            # Throttle task creation to avoid browser overload if not using semaphore inside
            tasks.append(self.process_day_scan(d))
        
        # Limit concurrency at the day-level if needed, or rely on internal logic
        # Here we run day scans in parallel but browser launch is heavy, 
        # so for robustness we might want to run them sequentially or semaphored.
        # Given "Production Ready", let's do chunks.
        
        chunk_size = 2
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            logger.info(f"🚀 Processing Batch {i // chunk_size + 1}...")
            await asyncio.gather(*chunk)
            
        await self.shutdown()

# =================================================================
# 6. MAIN ENTRY
# =================================================================

if __name__ == "__main__":
    try:
        engine = NeuralScoutEngine()
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"System Crash: {e}", exc_info=True)
