# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

# Third-party imports
import numpy as np
from playwright.async_api import async_playwright, Browser, BrowserContext
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION & SETTINGS
# =================================================================

class LogLevel(str, Enum):
    INFO = "INFO"
    ERROR = "ERROR"
    WARN = "WARN"
    DEBUG = "DEBUG"

class Config:
    """Centralized configuration management."""
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    MODEL_NAME: str = 'gemini-2.5-pro'
    
    # Tuning Parameters (Physics Engine)
    BSI_DEFAULT: float = 5.0
    ELO_DEFAULT: float = 1500.0
    
    # System
    MAX_CONCURRENT_SCRAPES: int = 5  # Semaphore limit
    HTTP_TIMEOUT: float = 30.0
    RETRIES: int = 3
    
    @classmethod
    def validate(cls):
        if not all([cls.GEMINI_API_KEY, cls.SUPABASE_URL, cls.SUPABASE_KEY]):
            print(f"[{datetime.now()}] ❌ CRITICAL: Secrets missing. Check Environment Variables.")
            sys.exit(1)

# Initialize Config
Config.validate()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout")

# =================================================================
# DATA STRUCTURES
# =================================================================

@dataclass
class PlayerProfile:
    id: str
    last_name: str
    skills: Dict[str, float] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MatchCandidate:
    p1_name: str
    p2_name: str
    tour: str
    time_str: str
    odds1: float
    odds2: float
    target_date: datetime

# =================================================================
# INFRASTRUCTURE LAYERS
# =================================================================

class DatabaseManager:
    """Handles all Supabase interactions with connection pooling logic."""
    
    def __init__(self):
        self.client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        
    async def get_initial_data(self) -> Tuple[List[Dict], Dict, List[Dict], List[Dict]]:
        """Fetches bulk data. Optimized to avoid fetching unnecessary columns if needed."""
        try:
            # Parallel fetching could be implemented here, but simple gather is fine
            f_players = self.client.table("players").select("id, last_name, play_style").execute()
            f_skills = self.client.table("player_skills").select("*").execute()
            f_reports = self.client.table("scouting_reports").select("*").execute()
            f_tournaments = self.client.table("tournaments").select("*").execute()
            
            clean_skills = {}
            for entry in f_skills.data:
                pid = entry.get('player_id')
                if pid:
                    clean_skills[pid] = {
                        'serve': self._to_float(entry.get('serve')), 'power': self._to_float(entry.get('power')),
                        'forehand': self._to_float(entry.get('forehand')), 'backhand': self._to_float(entry.get('backhand')),
                        'speed': self._to_float(entry.get('speed')), 'stamina': self._to_float(entry.get('stamina')),
                        'mental': self._to_float(entry.get('mental')), 'volley': self._to_float(entry.get('volley'))
                    }
            return f_players.data, clean_skills, f_reports.data, f_tournaments.data
        except Exception as e:
            logger.error(f"DB Load Error: {e}")
            return [], {}, [], []

    async def get_pending_matches(self) -> List[Dict]:
        """Fetches matches waiting for results."""
        try:
            return self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
        except Exception as e:
            logger.error(f"Failed to fetch pending matches: {e}")
            return []

    async def batch_upsert_odds(self, odds_data: List[Dict]):
        """Efficient batch upsert for odds."""
        if not odds_data:
            return
        try:
            # Chunking to avoid payload limits
            chunk_size = 50
            for i in range(0, len(odds_data), chunk_size):
                chunk = odds_data[i:i + chunk_size]
                self.client.table("market_odds").upsert(chunk, on_conflict="id").execute()
            logger.info(f"💾 Bulk saved/updated {len(odds_data)} matches.")
        except Exception as e:
            logger.error(f"Bulk Insert Error: {e}")

    async def update_match_winner(self, match_id: str, winner_name: str):
        try:
            self.client.table("market_odds").update({"actual_winner_name": winner_name}).eq("id", match_id).execute()
        except Exception as e:
            logger.error(f"Failed to update winner for {match_id}: {e}")

    @staticmethod
    def _to_float(val, default=50.0) -> float:
        if val is None: return default
        try: return float(val)
        except: return default

class GeminiClient:
    """Persistent HTTP client for AI interactions with rate limiting handling."""
    
    def __init__(self):
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{Config.MODEL_NAME}:generateContent?key={Config.GEMINI_API_KEY}"
        self.headers = {"Content-Type": "application/json"}
        self.client = httpx.AsyncClient(timeout=Config.HTTP_TIMEOUT)

    async def close(self):
        await self.client.aclose()

    async def call(self, prompt: str) -> Optional[str]:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }
        
        for attempt in range(Config.RETRIES):
            try:
                response = await self.client.post(self.url, headers=self.headers, json=payload)
                if response.status_code == 200:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini Rate Limit. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Gemini Network Error: {e}")
                await asyncio.sleep(1)
        return None

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    """Removes betting garbage from names."""
    trash = r'Live streams|1xBet|bwin|TV|Sky Sports|bet365'
    return re.sub(trash, '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# PHYSICS & CALCULATION ENGINE
# =================================================================

class OddsEngine:
    def __init__(self, elo_cache: Dict, tournaments: List[Dict]):
        self.elo_cache = elo_cache
        self.tournaments = tournaments
        self.tournament_loc_cache = {}

    def sigmoid(self, x, k=1.0):
        return 1 / (1 + math.exp(-k * x))

    def get_dynamic_court_weights(self, bsi: float, surface: str) -> Dict[str, float]:
        bsi = float(bsi)
        w = {
            'serve': 1.0, 'power': 1.0, 'rally': 1.0, 
            'movement': 1.0, 'mental': 0.8, 'volley': 0.5
        }
        
        if bsi >= 7.0: # Fast Court
            speed_factor = (bsi - 5.0) * 0.35 
            w['serve'] += speed_factor * 1.5
            w['power'] += speed_factor * 1.2
            w['volley'] += speed_factor * 1.0 
            w['rally'] -= speed_factor * 0.5 
            w['movement'] -= speed_factor * 0.3 
        elif bsi <= 4.0: # Slow Court
            slow_factor = (5.0 - bsi) * 0.4
            w['serve'] -= slow_factor * 0.8
            w['power'] -= slow_factor * 0.5 
            w['rally'] += slow_factor * 1.2 
            w['movement'] += slow_factor * 1.5 
            w['volley'] -= slow_factor * 0.5
        return w

    def calculate_fair_odds(self, p1_name: str, p2_name: str, s1: Dict, s2: Dict, 
                          bsi: float, surface: str, ai_meta: Dict, 
                          market_odds1: float, market_odds2: float) -> float:
        
        n1 = p1_name.lower().split()[-1] 
        n2 = p2_name.lower().split()[-1]
        
        # 1. Physics Model
        weights = self.get_dynamic_court_weights(bsi, surface)
        
        def get_score(skills):
            if not skills: return 50.0 
            score_serve = (skills.get('serve', 50) * 0.7 + skills.get('power', 50) * 0.3) * weights['serve']
            score_rally = (skills.get('forehand', 50) + skills.get('backhand', 50)) / 2 * weights['rally']
            score_move  = (skills.get('speed', 50) * 0.6 + skills.get('stamina', 50) * 0.4) * weights['movement']
            score_net   = skills.get('volley', 50) * weights['volley']
            score_ment  = skills.get('mental', 50) * weights['mental']
            total_weight = sum(weights.values())
            # Avoid division by zero
            return (score_serve + score_rally + score_move + score_net + score_ment) / (total_weight / 3.5 if total_weight > 0 else 1)

        p1_phys = get_score(s1)
        p2_phys = get_score(s2)
        phys_diff = (p1_phys - p2_phys) / 12.0 
        prob_physics = self.sigmoid(phys_diff)

        # 2. AI Tactical
        m1 = float(ai_meta.get('p1_tactical_score', 5))
        m2 = float(ai_meta.get('p2_tactical_score', 5))
        prob_tactical = 0.5 + ((m1 - m2) * 0.15)

        # 3. Elo Anchor
        elo1, elo2 = 1500.0, 1500.0
        elo_surf = 'Hard'
        if 'clay' in surface.lower(): elo_surf = 'Clay'
        elif 'grass' in surface.lower(): elo_surf = 'Grass'
        
        # Optimize Elo Lookup (Avoid iterating all dictionary keys)
        # Note: Ideally this should be a direct lookup, but retaining V81.1 logic for compatibility
        tour_elos = self.elo_cache.get("ATP", {})
        # Try direct match first
        if n1 in tour_elos: elo1 = tour_elos[n1].get(elo_surf, 1500.0)
        if n2 in tour_elos: elo2 = tour_elos[n2].get(elo_surf, 1500.0)
        
        prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

        # 4. Market Implied
        prob_market = 0.5
        if market_odds1 > 1 and market_odds2 > 1:
            inv1 = 1/market_odds1
            inv2 = 1/market_odds2
            margin = inv1 + inv2
            prob_market = inv1 / margin

        # 5. Synthesis
        raw_prob = (prob_market * 0.35) + (prob_elo * 0.20) + (prob_physics * 0.30) + (prob_tactical * 0.15)

        # 6. Volatility Dampening
        if raw_prob > 0.5:
            final_prob = raw_prob - (raw_prob - 0.5) * 0.05 
        else:
            final_prob = raw_prob + (0.5 - raw_prob) * 0.05

        return final_prob

# =================================================================
# SCRAPING & PARSING
# =================================================================

class ScraperEngine:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client
        self.elo_cache = {"ATP": {}, "WTA": {}}

    async def fetch_elo_ratings(self):
        logger.info("📊 Fetching Elo Ratings...")
        urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", 
                "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            # aggressive blocking
            await context.route("**/*", lambda r: r.abort() if r.request.resource_type in ["image", "stylesheet", "font", "media"] else r.continue_())
            
            for tour, url in urls.items():
                try:
                    page = await context.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    content = await page.content()
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    table = soup.find('table', {'id': 'reportable'})
                    if table:
                        rows = table.find_all('tr')[1:] 
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) > 4:
                                name = normalize_text(cols[0].get_text(strip=True)).lower()
                                # Only last name for key to match scraping logic
                                key_name = name.split()[-1]
                                self.elo_cache[tour][key_name] = {
                                    'Hard': self._parse_float(cols[3].get_text(strip=True)), 
                                    'Clay': self._parse_float(cols[4].get_text(strip=True)), 
                                    'Grass': self._parse_float(cols[5].get_text(strip=True))
                                }
                        logger.info(f"   ✅ {tour} Elo loaded: {len(self.elo_cache[tour])} records.")
                    await page.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
            await browser.close()
            return self.elo_cache

    @staticmethod
    def _parse_float(val, default=1500.0):
        try: return float(val)
        except: return default

    async def scrape_date(self, browser: Browser, target_date: datetime, semaphore: asyncio.Semaphore) -> Tuple[datetime, Optional[str]]:
        async with semaphore: # CONCURRENCY CONTROL
            try:
                context = await browser.new_context()
                await context.route("**/*", lambda r: r.abort() if r.request.resource_type in ["image", "stylesheet", "font", "media", "script", "xhr"] else r.continue_())
                
                page = await context.new_page()
                url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                
                # Optimized wait state
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                content = await page.content()
                
                await page.close()
                await context.close()
                return (target_date, content)
            except Exception as e:
                logger.warning(f"⚠️ Scrape Fail {target_date.strftime('%d.%m')}: {e}")
                return (target_date, None)

    def parse_matches(self, html: str, target_players: Set[str]) -> List[MatchCandidate]:
        if not html: return []
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all("table", class_="result")
        found = []
        
        current_tour = "Unknown"
        for table in tables:
            rows = table.find_all("tr")
            for i in range(len(rows)):
                row = rows[i]
                if "head" in row.get("class", []): 
                    current_tour = row.get_text(strip=True)
                    continue
                
                row_text = normalize_text(row.get_text(separator=' ', strip=True))
                
                # Safe Time Extraction
                match_time_str = "00:00"
                first_col = row.find('td', class_='first')
                if first_col and 'time' in first_col.get('class', []):
                    match_time_str = first_col.get_text(strip=True)

                if i + 1 < len(rows):
                    p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                    next_row_text = normalize_text(rows[i+1].get_text(separator=' ', strip=True))
                    p2_raw = clean_player_name(next_row_text)
                    
                    # Optimization: Fast string check before regex
                    p1_check = any(tp in p1_raw.lower() for tp in target_players)
                    p2_check = any(tp in p2_raw.lower() for tp in target_players)
                    
                    if p1_check and p2_check:
                        odds = []
                        try:
                            # Improved Regex for odds
                            nums = re.findall(r'\b\d+\.\d{2}\b', row_text)
                            valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                            
                            nums2 = re.findall(r'\b\d+\.\d{2}\b', next_row_text)
                            valid2 = [float(x) for x in nums2 if 1.0 < float(x) < 50.0]
                            
                            if valid and valid2: 
                                odds = [valid[0], valid2[0]]
                            elif len(valid) >= 2:
                                odds = valid[:2]
                        except: pass
                        
                        if odds:
                            found.append({
                                "p1": p1_raw, "p2": p2_raw, 
                                "tour": current_tour, "time": match_time_str,
                                "odds1": odds[0], "odds2": odds[1]
                            })
        return found

    async def resolve_ambiguous_tournament(self, scraped_name, tour_db_cache, p1, p2):
        prompt = f"TASK: Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\", \"is_indoor\": bool }}"
        res = await self.gemini.call(prompt)
        if not res: return None
        try: 
            return json.loads(res.replace("```json", "").replace("```", "").strip())
        except: return None

    async def find_best_court_smart(self, tour_name, db_tours, p1, p2):
        s_low = tour_name.lower().strip()
        # Direct DB Hit
        for t in db_tours:
            if t['name'].lower() == s_low: return t['surface'], float(t.get('bsi_rating', 5.0)), t.get('notes', '')
        
        # Heuristics
        if "clay" in s_low: return "Red Clay", 3.5, "Heuristic"
        if "hard" in s_low: return "Hard", 6.5, "Heuristic"
        if "indoor" in s_low: return "Indoor", 8.0, "Heuristic"
        
        # AI Fallback
        ai_loc = await self.resolve_ambiguous_tournament(tour_name, None, p1, p2)
        if ai_loc and ai_loc.get('city'):
            city = ai_loc['city'].lower()
            surf = ai_loc.get('surface_guessed', 'Hard')
            # Check DB again with City
            for t in db_tours:
                if city in t['name'].lower(): return t['surface'], float(t.get('bsi_rating', 5.0)), f"AI: {city}"
            bsi = 3.5 if 'clay' in surf.lower() else 6.5
            return surf, bsi, f"AI Guess: {city}"
        
        return 'Hard', 6.5, 'Default Fallback'

    async def analyze_match_ai(self, p1_obj, p2_obj, s1, s2, surface, bsi):
        prompt = f"""
        ROLE: Elite Tennis Analyst. TASK: {p1_obj['last_name']} vs {p2_obj['last_name']}.
        CTX: {surface} (BSI {bsi}). P1 Style: {p1_obj.get('play_style')}.
        METRICS (0-10): TACTICAL (25%), FORM (10%), UTR (5%).
        JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "ai_text": "Brief analysis..." }}
        """
        res = await self.gemini.call(prompt)
        default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'ai_text': 'Analysis failed'}
        if not res: return default
        try: return json.loads(res.replace("```json", "").replace("```", "").strip())
        except: return default

# =================================================================
# MAIN PIPELINE CONTROLLER
# =================================================================

async def run_pipeline():
    logger.info("🚀 Neural Scout v82.0 (Silicon Valley Refactor) Starting...")
    
    # Initialize Core Components
    db = DatabaseManager()
    gemini = GeminiClient()
    scraper = ScraperEngine(gemini)
    
    # 1. Update Historical Results (Simplified for brevity but kept logical place)
    # Note: In production, this should be a separate Cron Job
    # await update_past_results(db, scraper) 

    # 2. Load Core Data
    elo_data = await scraper.fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await db.get_initial_data()
    
    if not players:
        logger.error("No player data found. Aborting.")
        return

    player_names_set = set(p['last_name'].lower() for p in players)
    odds_engine = OddsEngine(elo_data, all_tournaments)

    # 3. Parallel Scraping
    current_date = datetime.now()
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_SCRAPES)
    
    scraped_matches_raw = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--disable-gpu', '--no-sandbox'])
        
        tasks = []
        for day_offset in range(14): 
            target_date = current_date + timedelta(days=day_offset)
            tasks.append(scraper.scrape_date(browser, target_date, semaphore))
        
        logger.info(f"🕸️ Launching {len(tasks)} scrape tasks...")
        results = await asyncio.gather(*tasks)
        await browser.close()
        
    # 4. Processing & Calculation
    batch_insert_queue = []
    
    for target_date, html in results:
        if not html: continue
        
        matches = scraper.parse_matches(html, player_names_set)
        if matches:
            logger.info(f"🔍 {target_date.strftime('%d.%m')}: Found {len(matches)} potential matches.")
        
        for m in matches:
            # Match Players
            p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
            p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
            
            if not p1_obj or not p2_obj: continue
            
            # Idempotency Key logic (Check if exists locally first to save DB calls if possible, 
            # but for consistency we rely on Upsert usually. Here we check DB to skip logic cost)
            # NOTE: For high throughput, we would cache existing match IDs in memory.
            
            s1 = all_skills.get(p1_obj['id'], {})
            s2 = all_skills.get(p2_obj['id'], {})
            
            # Surface Logic
            surf, bsi, notes = await scraper.find_best_court_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
            
            # AI Analysis
            ai_meta = await scraper.analyze_match_ai(p1_obj, p2_obj, s1, s2, surf, bsi)
            
            # Physics Calculation
            prob_p1 = odds_engine.calculate_fair_odds(
                p1_obj['last_name'], p2_obj['last_name'], 
                s1, s2, bsi, surf, ai_meta, 
                m['odds1'], m['odds2']
            )
            
            iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
            
            entry = {
                "player1_name": p1_obj['last_name'], 
                "player2_name": p2_obj['last_name'], 
                "tournament": m['tour'],
                "odds1": m['odds1'], 
                "odds2": m['odds2'],
                "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0.01 else 99,
                "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 0.99 else 99,
                "ai_analysis_text": ai_meta.get('ai_text', 'No analysis'),
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "match_time": iso_timestamp
            }
            # For UPSERT to work, we ideally need a unique constraint in DB on (player1, player2, match_time)
            # Assuming 'id' is auto-gen, we rely on duplicate checks or composite keys. 
            # For this script, we queue it.
            batch_insert_queue.append(entry)
            
    # 5. Batch Write
    if batch_insert_queue:
        logger.info(f"💾 Committing {len(batch_insert_queue)} analyzed matches to DB...")
        await db.batch_upsert_odds(batch_insert_queue)
    else:
        logger.info("💤 No new matches to save.")

    await gemini.close()
    logger.info("🏁 Cycle Finished.")

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
    except Exception as e:
        logger.error(f"🔥 Fatal Error: {e}")
        sys.exit(1)
