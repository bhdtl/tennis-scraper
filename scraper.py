# -- coding: utf-8 --
import asyncio
import json
import os
import re
import unicodedata
import math
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Third-party imports
import httpx
from playwright.async_api import async_playwright, Browser, Page
from supabase import create_client, Client
from pydantic import BaseModel, ValidationError

# =================================================================
# 0. SYSTEM CONFIGURATION & CONSTANTS
# =================================================================

# Precision settings for Financial Mathematics
getcontext().prec = 6

# Logging Configuration - Structured & Clean
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout")

class Config:
    """Centralized Configuration Management"""
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    MODEL_NAME: str = 'gemini-2.5-pro'
    CONCURRENCY_LIMIT: int = 3  # Reduced to 3 to prevent rate limits
    
    @classmethod
    def validate(cls):
        if not all([cls.GEMINI_API_KEY, cls.SUPABASE_URL, cls.SUPABASE_KEY]):
            logger.critical("❌ FATAL: Missing Environment Secrets. Check GitHub Secrets.")
            sys.exit(1)

Config.validate()

# =================================================================
# 1. DOMAIN MODELS (Type Safety & Validation)
# =================================================================

class PlayerSkills(BaseModel):
    serve: float = 50.0
    power: float = 50.0
    forehand: float = 50.0
    backhand: float = 50.0
    speed: float = 50.0
    stamina: float = 50.0
    mental: float = 50.0

class Player(BaseModel):
    id: str
    first_name: str
    last_name: str
    play_style: Optional[str] = "All-Rounder"
    skills: Optional[PlayerSkills] = None

class ScrapedMatch(BaseModel):
    p1_name: str
    p2_name: str
    tournament: str
    match_time_local: str
    odds1: Decimal
    odds2: Decimal

# =================================================================
# 2. UTILITY SERVICE (Helper Functions)
# =================================================================

class Utils:
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text: return ""
        text = text.replace('æ', 'ae').replace('ø', 'o')
        return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    @staticmethod
    def clean_player_name(raw: str) -> str:
        """Removes betting sponsor junk from names."""
        blacklist = [
            'Live streams', '1xBet', 'bwin', 'TV', 'Sky Sports', 'bet365', 
            'Unibet', 'William Hill'
        ]
        pattern = re.compile('|'.join(map(re.escape, blacklist)), re.IGNORECASE)
        clean = pattern.sub('', raw).replace('|', '').strip()
        return clean

    @staticmethod
    def get_last_name(full_name: str) -> str:
        if not full_name: return ""
        clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip()
        parts = clean.split()
        return parts[-1].lower() if parts else ""

# =================================================================
# 3. INFRASTRUCTURE SERVICES (External I/O)
# =================================================================

class DatabaseService:
    def __init__(self):
        self.client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

    def fetch_reference_data(self) -> Tuple[List[Player], List[Dict]]:
        try:
            players_raw = self.client.table("players").select("*").execute().data
            skills_raw = self.client.table("player_skills").select("*").execute().data
            tournaments_raw = self.client.table("tournaments").select("*").execute().data

            skills_map = {}
            for s in skills_raw:
                pid = s.get('player_id')
                if pid:
                    skills_map[pid] = PlayerSkills(**{k: float(v or 50) for k, v in s.items() if k in PlayerSkills.model_fields})

            players = []
            for p in players_raw:
                player = Player(
                    id=p['id'],
                    first_name=p.get('first_name', ''),
                    last_name=p.get('last_name', ''),
                    play_style=p.get('play_style'),
                    skills=skills_map.get(p['id'], PlayerSkills())
                )
                players.append(player)

            return players, tournaments_raw

        except Exception as e:
            logger.error(f"❌ DB Load Error: {e}")
            return [], []

    def get_existing_match(self, p1_last: str, p2_last: str):
        return self.client.table("market_odds").select("id, actual_winner_name").or_(
            f"and(player1_name.eq.{p1_last},player2_name.eq.{p2_last}),and(player1_name.eq.{p2_last},player2_name.eq.{p1_last})"
        ).execute()

    def update_match_odds(self, match_id: str, odds1: float, odds2: float, time_iso: str):
        self.client.table("market_odds").update({
            "odds1": odds1,
            "odds2": odds2,
            "match_time": time_iso
        }).eq("id", match_id).execute()

    def insert_match(self, data: Dict):
        self.client.table("market_odds").insert(data).execute()

    def update_winner(self, match_id: str, winner_name: str):
        self.client.table("market_odds").update({"actual_winner_name": winner_name}).eq("id", match_id).execute()

    def get_pending_matches(self):
        return self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data

class AIService:
    @staticmethod
    async def call_gemini(prompt: str) -> Optional[str]:
        await asyncio.sleep(0.5) 
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{Config.MODEL_NAME}:generateContent?key={Config.GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                if response.status_code != 200:
                    return None
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                logger.error(f"⚠️ AI Call Failed: {e}")
                return None

    @staticmethod
    async def analyze_match_meta(p1: Player, p2: Player, surface: str, bsi: float, notes: str) -> Dict:
        prompt = f"""
        ROLE: Elite Tennis Analyst.
        TASK: Analyze matchup {p1.last_name} vs {p2.last_name}.
        CONTEXT: Surface {surface} (Speed/BSI: {bsi}). Notes: {notes}.
        P1 PROFILE: Style: {p1.play_style}.
        P2 PROFILE: Style: {p2.play_style}.
        
        OUTPUT JSON ONLY:
        {{
            "p1_tactical_score": (int 1-10),
            "p2_tactical_score": (int 1-10),
            "ai_text": "Short concise reason (max 15 words)."
        }}
        """
        raw = await AIService.call_gemini(prompt)
        default = {"p1_tactical_score": 5, "p2_tactical_score": 5, "ai_text": "No AI analysis"}
        if not raw: return default
        try:
            cleaned = raw.replace("json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except:
            return default

# =================================================================
# 4. CORE ENGINES
# =================================================================

class MathEngine:
    @staticmethod
    def sigmoid(diff: float, sensitivity: float = 0.1) -> float:
        try:
            return 1 / (1 + math.exp(-sensitivity * diff))
        except OverflowError:
            return 0.0 if diff < 0 else 1.0

    @staticmethod
    def calculate_fair_odds(
        p1: Player, p2: Player, 
        surface: str, bsi: float, 
        ai_meta: Dict, 
        elo_data: Dict[str, Dict[str, float]],
        market_odds1: Decimal, market_odds2: Decimal
    ) -> Tuple[Decimal, Decimal, float]:
        
        bsi_val = float(bsi)
        s1 = p1.skills
        s2 = p2.skills
        
        # 1. AI TACTICAL
        m1 = float(ai_meta.get('p1_tactical_score', 5))
        m2 = float(ai_meta.get('p2_tactical_score', 5))
        prob_tactical = MathEngine.sigmoid(m1 - m2, sensitivity=0.8)

        # 2. SURFACE PHYSICS
        c1_phys, c2_phys = 0.0, 0.0
        if bsi_val <= 4.5: 
            c1_phys = s1.stamina + s1.speed + s1.mental + s1.backhand
            c2_phys = s2.stamina + s2.speed + s2.mental + s2.backhand
        elif bsi_val >= 7.0: 
            c1_phys = s1.serve + s1.power + s1.forehand
            c2_phys = s2.serve + s2.power + s2.forehand
        else:
            c1_phys = sum(s1.model_dump().values())
            c2_phys = sum(s2.model_dump().values())
            
        prob_physics = MathEngine.sigmoid(c1_phys - c2_phys, sensitivity=0.015)

        # 3. ELO
        elo_surf = 'Hard'
        if 'clay' in surface.lower(): elo_surf = 'Clay'
        elif 'grass' in surface.lower(): elo_surf = 'Grass'
        
        e1 = 1500.0; e2 = 1500.0
        for name, stats in elo_data.items():
            if p1.last_name.lower() in name: e1 = stats.get(elo_surf, 1500.0)
            if p2.last_name.lower() in name: e2 = stats.get(elo_surf, 1500.0)
            
        prob_elo = 1 / (1 + 10 ** ((e2 - e1) / 400))

        # 4. MARKET
        prob_market = 0.5
        if market_odds1 > 0 and market_odds2 > 0:
            inv1 = 1 / float(market_odds1)
            inv2 = 1 / float(market_odds2)
            prob_market = inv1 / (inv1 + inv2)

        prob_final = (prob_tactical * 0.35) + (prob_physics * 0.25) + (prob_elo * 0.20) + (prob_market * 0.20)
        prob_final = max(0.01, min(0.99, prob_final))
        
        fair_odds1 = Decimal(1 / prob_final).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        fair_odds2 = Decimal(1 / (1 - prob_final)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return fair_odds1, fair_odds2, prob_final

class ScraperEngine:
    def __init__(self):
        self.elo_cache = {"ATP": {}, "WTA": {}}
        self.combined_elo = {}
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY_LIMIT)
        # BROWSER CONFIG - STEALTH MODE
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async def _safe_goto(self, page: Page, url: str):
        """Wrapper to handle navigation errors gracefully."""
        try:
            # Clean URL to prevent Markdown injection
            clean_url = url.strip()
            if not clean_url.startswith('http'):
                logger.error(f"❌ Invalid URL format: {clean_url}")
                return False
                
            await page.goto(clean_url, wait_until="domcontentloaded", timeout=45000)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Nav Error ({clean_url}): {e}")
            return False

    async def fetch_elo(self):
        logger.info("📊 Loading Elo Ratings...")
        # RAW STRINGS - NO MARKDOWN
        urls = {
            "ATP": "[https://tennisabstract.com/reports/atp_elo_ratings.html](https://tennisabstract.com/reports/atp_elo_ratings.html)", 
            "WTA": "[https://tennisabstract.com/reports/wta_elo_ratings.html](https://tennisabstract.com/reports/wta_elo_ratings.html)"
        }
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            for tour, url in urls.items():
                try:
                    context = await browser.new_context(user_agent=self.user_agent)
                    page = await context.new_page()
                    
                    if await self._safe_goto(page, url):
                        content = await page.content()
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        table = soup.find('table', {'id': 'reportable'})
                        if table:
                            rows = table.find_all('tr')[1:] 
                            for row in rows:
                                cols = row.find_all('td')
                                if len(cols) > 4:
                                    name = Utils.normalize_text(cols[0].get_text(strip=True)).lower()
                                    try:
                                        self.elo_cache[tour][name] = {
                                            'Hard': float(cols[3].get_text(strip=True) or 1500), 
                                            'Clay': float(cols[4].get_text(strip=True) or 1500), 
                                            'Grass': float(cols[5].get_text(strip=True) or 1500)
                                        }
                                    except: continue
                        logger.info(f"   ✅ {tour} Elo loaded: {len(self.elo_cache[tour])}")
                    await context.close()
                except Exception as e:
                    logger.warning(f"   ⚠️ Elo Critical Error ({tour}): {e}")
            await browser.close()
        self.combined_elo = {**self.elo_cache["ATP"], **self.elo_cache["WTA"]}

    async def scrape_odds_day(self, date_obj: datetime, players: List[Player]) -> List[ScrapedMatch]:
        async with self.semaphore:
            target_players = set(p.last_name.lower() for p in players)
            found_matches = []
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=self.user_agent)
                page = await context.new_page()
                try:
                    # DYNAMIC URL GENERATION - CLEAN
                    url = f"[https://www.tennisexplorer.com/matches/?type=all&year=](https://www.tennisexplorer.com/matches/?type=all&year=){date_obj.year}&month={date_obj.month}&day={date_obj.day}"
                    
                    if await self._safe_goto(page, url):
                        # Wait for table explicitly to avoid empty scrapes
                        try:
                            await page.wait_for_selector("table.result", timeout=5000)
                        except: pass 
                        
                        content = await page.content()
                        found_matches = self._parse_explorer_html(content, target_players)
                except Exception as e:
                    logger.error(f"❌ Scrape Error {date_obj.date()}: {e}")
                finally:
                    await context.close()
                    await browser.close()
            return found_matches

    def _parse_explorer_html(self, html: str, target_players: set) -> List[ScrapedMatch]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        tables = soup.find_all("table", class_="result")
        current_tour = "Unknown"
        
        for table in tables:
            rows = table.find_all("tr")
            for i, row in enumerate(rows):
                if "head" in row.get("class", []):
                    current_tour = row.get_text(strip=True)
                    continue
                
                row_text = Utils.normalize_text(row.get_text(separator=' ', strip=True))
                if i + 1 >= len(rows): continue
                
                time_str = "00:00"
                first_col = row.find('td', class_='first')
                if first_col and 'time' in first_col.get('class', []):
                    time_str = first_col.get_text(strip=True)

                p1_raw = Utils.clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                row2_text = Utils.normalize_text(rows[i+1].get_text(separator=' ', strip=True))
                p2_raw = Utils.clean_player_name(row2_text)
                
                p1_ln = Utils.get_last_name(p1_raw)
                p2_ln = Utils.get_last_name(p2_raw)
                
                if (p1_ln in target_players) and (p2_ln in target_players):
                    odds = []
                    try:
                        nums = re.findall(r'\d+\.\d+', row_text)
                        valid = [Decimal(x) for x in nums if 1.0 < float(x) < 50.0]
                        if len(valid) >= 2: odds = valid[:2]
                        else:
                            nums2 = re.findall(r'\d+\.\d+', row2_text)
                            valid2 = [Decimal(x) for x in nums2 if 1.0 < float(x) < 50.0]
                            if valid and valid2: odds = [valid[0], valid2[0]]
                    except: pass
                    
                    if len(odds) >= 2:
                        try:
                            m = ScrapedMatch(
                                p1_name=p1_raw, p2_name=p2_raw, tournament=current_tour,
                                match_time_local=time_str, odds1=odds[0], odds2=odds[1]
                            )
                            results.append(m)
                        except ValidationError: pass
        return results

class VerificationService:
    def __init__(self, db: DatabaseService):
        self.db = db
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async def run_verification_cycle(self):
        logger.info("🏆 Checking Results...")
        pending = self.db.get_pending_matches()
        if not pending: return

        safe_matches = []
        now_utc = datetime.now(timezone.utc)
        for pm in pending:
            try:
                created_at = datetime.fromisoformat(pm['created_at'].replace('Z', '+00:00'))
                if (now_utc - created_at).total_seconds() > 3900: 
                    safe_matches.append(pm)
            except: continue

        if not safe_matches: return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=self.user_agent)
            
            for day_offset in range(3):
                target_date = datetime.now() - timedelta(days=day_offset)
                await self._process_results_page(context, target_date, safe_matches)
            await browser.close()

    async def _process_results_page(self, context: Browser, date_obj: datetime, pending_matches: List[Dict]):
        from bs4 import BeautifulSoup
        page = await context.new_page()
        try:
            # CLEAN URL
            url = f"[https://www.tennisexplorer.com/results/?type=all&year=](https://www.tennisexplorer.com/results/?type=all&year=){date_obj.year}&month={date_obj.month}&day={date_obj.day}"
            
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table', class_='result')
            if not table: return

            rows = table.find_all('tr')
            for i, row in enumerate(rows):
                if 'flags' in str(row) or 'head' in str(row): continue
                
                row_text = row.get_text(separator=" ", strip=True).lower()
                next_row_text = rows[i+1].get_text(separator=" ", strip=True).lower() if i+1 < len(rows) else ""

                for pm in pending_matches:
                    p1_last = Utils.get_last_name(pm['player1_name'])
                    p2_last = Utils.get_last_name(pm['player2_name'])

                    match_found = (p1_last in row_text and p2_last in next_row_text) or \
                                  (p2_last in row_text and p1_last in next_row_text) or \
                                  (p1_last in row_text and p2_last in row_text)

                    if match_found:
                        self._settle_match(row, rows[i+1] if i+1 < len(rows) else None, pm, p1_last, p2_last)
        except Exception: pass
        finally: await page.close()

    def _settle_match(self, row, next_row, pm, p1_last, p2_last):
        try:
            row_text = row.get_text(separator=" ", strip=True).lower()
            is_ret = "ret." in row_text or "w.o." in row_text
            
            def get_scores(cols):
                scores = []
                if not cols: return []
                for col in cols:
                    txt = col.get_text(strip=True)
                    if txt.isdigit() and len(txt) == 1 and int(txt) <= 7: scores.append(int(txt))
                return scores

            cols1 = row.find_all('td')
            cols2 = next_row.find_all('td') if next_row else []
            s1 = get_scores(cols1)
            s2 = get_scores(cols2)
            
            sets_p1_row = 0; sets_p2_row = 0
            for k in range(min(len(s1), len(s2))):
                if s1[k] > s2[k]: sets_p1_row += 1
                elif s2[k] > s1[k]: sets_p2_row += 1

            winner = None
            if sets_p1_row > sets_p2_row or (is_ret and sets_p1_row > sets_p2_row):
                if p1_last in row_text: winner = pm['player1_name']
                elif p2_last in row_text: winner = pm['player2_name']
            elif sets_p2_row > sets_p1_row or (is_ret and sets_p2_row > sets_p1_row):
                if next_row:
                    next_text = next_row.get_text(separator=" ", strip=True).lower()
                    if p1_last in next_text: winner = pm['player1_name']
                    elif p2_last in next_text: winner = pm['player2_name']

            if winner:
                self.db.update_winner(pm['id'], winner)
                logger.info(f"   ✅ WINNER SETTLED: {winner}")
        except Exception: pass

# =================================================================
# 5. ORCHESTRATOR
# =================================================================

class NeuralScoutPipeline:
    def __init__(self):
        self.db = DatabaseService()
        self.scraper = ScraperEngine()
        self.verifier = VerificationService(self.db)
        
    async def run(self):
        logger.info("🚀 Starting Neural Scout v2026...")
        await self.verifier.run_verification_cycle()
        await self.scraper.fetch_elo()
        players, db_tournaments = self.db.fetch_reference_data()
        
        if not players:
            logger.error("No players in DB.")
            return

        scrape_tasks = []
        today = datetime.now()
        for i in range(35):
            d = today + timedelta(days=i)
            scrape_tasks.append(self.scraper.scrape_odds_day(d, players))
        
        logger.info(f"⚡ Launching {len(scrape_tasks)} scan tasks...")
        results_matrix = await asyncio.gather(*scrape_tasks)
        all_matches = [m for day_res in results_matrix for m in day_res]
        logger.info(f"🔍 Found {len(all_matches)} potential matches.")

        for m in all_matches:
            await self._process_single_match(m, players, db_tournaments)
        logger.info("🏁 Pipeline Finished.")

    async def _process_single_match(self, m: ScrapedMatch, players: List[Player], tournaments: List[Dict]):
        p1 = next((p for p in players if p.last_name in m.p1_name), None)
        p2 = next((p for p in players if p.last_name in m.p2_name), None)
        if not p1 or not p2: return

        p1_last = p1.last_name
        p2_last = p2.last_name
        existing_res = self.db.get_existing_match(p1_last, p2_last)
        match_date_str = datetime.now().strftime('%Y-%m-%d')
        iso_timestamp = f"{match_date_str}T{m.match_time_local}:00Z"

        if existing_res.data:
            match_data = existing_res.data[0]
            if match_data.get('actual_winner_name'):
                logger.info(f"🔒 Locked: {p1_last} vs {p2_last}")
                return
            self.db.update_match_odds(match_data['id'], float(m.odds1), float(m.odds2), iso_timestamp)
            logger.info(f"🔄 Updated: {p1_last} vs {p2_last}")
            return

        if m.odds1 <= Decimal("1.0"): return
        logger.info(f"✨ Analyzing: {p1_last} vs {p2_last}")
        
        surf, bsi, notes = "Hard", 6.5, "Default"
        tour_lower = m.tournament.lower()
        found_tour = next((t for t in tournaments if t['name'].lower() in tour_lower), None)
        if found_tour:
            surf = found_tour['surface']; bsi = found_tour['bsi_rating']; notes = found_tour.get('notes', '')
        elif "clay" in tour_lower: surf, bsi = "Red Clay", 3.5
        elif "indoor" in tour_lower: surf, bsi = "Indoor Hard", 8.0
        
        ai_meta = await AIService.analyze_match_meta(p1, p2, surf, bsi, notes)
        fair1, fair2, prob = MathEngine.calculate_fair_odds(p1, p2, surf, bsi, ai_meta, self.scraper.combined_elo, m.odds1, m.odds2)

        entry = {
            "player1_name": p1_last, "player2_name": p2_last, "tournament": m.tournament,
            "odds1": float(m.odds1), "odds2": float(m.odds2),
            "ai_fair_odds1": float(fair1), "ai_fair_odds2": float(fair2),
            "ai_analysis_text": ai_meta.get('ai_text', ''),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "match_time": iso_timestamp
        }
        self.db.insert_match(entry)
        logger.info(f"💾 Saved: {p1_last} vs {p2_last} (AI Prob: {prob:.2f})")

if __name__ == "__main__":
    asyncio.run(NeuralScoutPipeline().run())
