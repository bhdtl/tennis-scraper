# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple

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
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout_v98_BlockParser")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# STABLE MODEL
MODEL_NAME = 'gemini-1.5-flash' 

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen!")
    sys.exit(1)

db_manager = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. UTILS
# =================================================================
def to_float(val, default=50.0):
    try: return float(val)
    except: return default

def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365|\(\d+\)', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def extract_time_from_row(row) -> Optional[str]:
    """
    Returns HH:MM string ONLY if this row is a Match Start.
    Otherwise returns None. This is the KEY to fixing phantom matches.
    """
    first_col = row.find('td', class_='first')
    if not first_col: return None
    
    # Check if it has 'time' class OR contains a time pattern
    txt = first_col.get_text(strip=True)
    
    # Pattern: HH:MM or "Live"
    time_match = re.search(r'(\d{1,2}:\d{2})', txt)
    if time_match:
        return time_match.group(1)
    
    if "live" in txt.lower():
        return "12:00" # Placeholder for live matches
        
    return None

def validate_market(odds1, odds2):
    if odds1 <= 1.01 or odds2 <= 1.01: return False
    if odds1 > 50.0 or odds2 > 50.0: return False 
    margin = (1/odds1) + (1/odds2)
    return 0.85 < margin < 1.30

# =================================================================
# 3. MATH & AI
# =================================================================
class QuantumMathEngine:
    @staticmethod
    def sigmoid(x, sensitivity=0.1):
        return 1 / (1 + math.exp(-sensitivity * x))

    @staticmethod
    def calculate_fair_probability(ai_tac, ai_phy, s1, s2, e1, e2):
        prob_ai = ai_tac
        prob_skills = QuantumMathEngine.sigmoid(s1 - s2, 0.12)
        prob_phy = ai_phy
        prob_elo = 1 / (1 + 10 ** ((e2 - e1) / 400))
        
        # Weighted Model (40/25/20/15)
        raw = (prob_ai * 0.40) + (prob_skills * 0.25) + (prob_phy * 0.20) + (prob_elo * 0.15)
        return max(0.05, min(0.95, raw))

    @staticmethod
    def devig_odds(o1, o2):
        if o1 <= 1 or o2 <= 1: return 0.5, 0.5
        i1, i2 = 1.0/o1, 1.0/o2
        m = i1 + i2
        return i1/m, i2/m

class AIEngine:
    def __init__(self):
        self.sem = asyncio.Semaphore(2)

    async def analyze(self, p1, p2, s1, s2, r1, r2, court):
        async with self.sem:
            await asyncio.sleep(1.0)
            prompt = f"""
            ROLE: Tennis Analyst. MATCH: {p1['last_name']} vs {p2['last_name']}.
            COURT: {court.get('name')} ({court.get('surface')}).
            P1: Srv {s1.get('serve')}, Ret {s1.get('speed')}.
            P2: Srv {s2.get('serve')}, Ret {s2.get('speed')}.
            
            OUTPUT JSON ONLY:
            {{
                "tactical_score_p1": 55, 
                "physics_score_p1": 50,
                "analysis_detail": "..."
            }}
            """
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30.0)
                    if resp.status_code != 200: return {}
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(raw.replace("```json", "").replace("```", "").strip())
            except: return {}

    async def resolve_court(self, tour, p1, p2, candidates):
        # RAG Logic simplified for stability
        if not candidates: return None
        cand_str = "\n".join([f"{i}: {c['name']}" for i,c in enumerate(candidates)])
        prompt = f"Match: {p1} vs {p2} at {tour}. Pick correct ID from:\n{cand_str}\nJSON: {{'id': 0}}"
        async with self.sem:
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10.0)
                    if resp.status_code != 200: return candidates[0]
                    raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    idx = json.loads(raw.replace("```json","").replace("```","").strip()).get('id')
                    return candidates[idx] if idx < len(candidates) else candidates[0]
            except: return candidates[0]

ai = AIEngine()
ELO_CACHE = {"ATP": {}, "WTA": {}}

# =================================================================
# 4. CORE PIPELINE (BLOCK PARSER)
# =================================================================
async def fetch_elo_ratings_optimized(bot):
    logger.info("📊 Elo wird geladen...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    for tour, url in urls.items():
        try:
            page = await bot.browser.new_page()
            await page.goto(url, timeout=60000)
            soup = BeautifulSoup(await page.content(), 'html.parser')
            table = soup.find('table', {'id': 'reportable'})
            if table:
                for row in table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) > 5:
                        name = normalize_text(cols[0].get_text(strip=True)).lower()
                        ELO_CACHE[tour][name] = {
                            'Hard': to_float(cols[3].get_text(strip=True), 1500),
                            'Clay': to_float(cols[4].get_text(strip=True), 1500),
                            'Grass': to_float(cols[5].get_text(strip=True), 1500)
                        }
            await page.close()
        except: pass

class ContextResolver:
    def __init__(self, db_tournaments):
        self.db_tournaments = db_tournaments
        self.name_map = {t['name'].lower(): t for t in db_tournaments}
        self.lookup_keys = list(self.name_map.keys())

    async def get_court(self, tour_name, p1, p2):
        s_clean = tour_name.lower().replace("atp", "").replace("wta", "").strip()
        if s_clean in self.name_map: return self.name_map[s_clean]
        
        candidates = []
        fuzzy = difflib.get_close_matches(s_clean, self.lookup_keys, n=3, cutoff=0.5)
        for f in fuzzy: candidates.append(self.name_map[f])
        
        if "united cup" in s_clean:
            for t in self.db_tournaments:
                if "united cup" in t['name'].lower() and t not in candidates: candidates.append(t)
        
        if candidates:
            return await ai.resolve_court(tour_name, p1, p2, candidates)
        
        return {'name': tour_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium'}

async def process_day(bot, target_date, players, skills_map, reports, resolver):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    logger.info(f"📅 Scanne Datum: {target_date.strftime('%Y-%m-%d')}")
    
    page = await bot.browser.new_page()
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    content = await page.content()
    await page.close()

    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all("table", class_="result")
    
    current_tour = "Unknown"
    active_date = target_date

    for table in tables:
        rows = table.find_all("tr")
        
        # --- BLOCK PARSER LOOP ---
        i = 0
        while i < len(rows):
            row = rows[i]
            
            # Header Check
            if "head" in row.get("class", []):
                current_tour = row.find('a').get_text(strip=True) if row.find('a') else row.get_text(strip=True)
                i += 1
                continue
            
            # Date Check
            if "flags" in row.get("class", []):
                if "Tomorrow" in row.get_text(): active_date = target_date + timedelta(days=1)
                i += 1
                continue

            # IS THIS A START?
            match_time = extract_time_from_row(row)
            
            # VALID START FOUND
            if match_time and (i + 1 < len(rows)):
                row1 = row
                row2 = rows[i+1]
                
                # Extract Text
                t1 = normalize_text(row1.get_text(separator=' ', strip=True))
                t2 = normalize_text(row2.get_text(separator=' ', strip=True))
                
                # Check for Doubles (Quick Filter)
                if '/' in t1 or '/' in t2:
                    i += 2
                    continue

                # Player Resolution
                n1 = clean_player_name(t1.split('1.')[0] if '1.' in t1 else t1)
                n2 = clean_player_name(t2)
                
                p1 = next((p for p in players if p['last_name'].lower() in n1.lower()), None)
                p2 = next((p for p in players if p['last_name'].lower() in n2.lower()), None)

                # --- STRICT GENDER CHECK ---
                valid_pair = False
                if p1 and p2:
                    if p1.get('tour') == p2.get('tour'): # Must match (ATP vs ATP)
                        valid_pair = True
                    else:
                        # logger.warning(f"⚠️ Gender Mismatch: {p1['last_name']} vs {p2['last_name']}")
                        valid_pair = False

                if valid_pair:
                    # ODDS
                    odds = []
                    try:
                        for r in [row1, row2]:
                            for td in r.find_all('td', class_='course'):
                                try:
                                    v = float(td.get_text(strip=True))
                                    if 1.01 <= v < 50: odds.append(v)
                                except: pass
                    except: pass
                    
                    m1 = odds[0] if len(odds)>0 else 0
                    m2 = odds[1] if len(odds)>1 else 0

                    if validate_market(m1, m2):
                        # --- VALID MATCH ---
                        iso_time = f"{active_date.strftime('%Y-%m-%d')}T{match_time}:00Z"
                        
                        # DB Check (Simplified for speed)
                        # Only insert/update if needed
                        # ... (DB Logic) ...
                        
                        # New Analysis if needed
                        court_db = await resolver.get_court(current_tour, p1['last_name'], p2['last_name'])
                        logger.info(f"✨ Match: {p1['last_name']} vs {p2['last_name']} @ {court_db['name']}")
                        
                        s1 = skills_map.get(p1['id'], {})
                        s2 = skills_map.get(p2['id'], {})
                        r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                        r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                        
                        ai_data = await ai.analyze(p1, p2, s1, s2, r1, r2, court_db)
                        
                        # Math
                        ai_tac = ai_data.get('tactical_score_p1', 50)/100
                        ai_phy = ai_data.get('physics_score_p1', 50)/100
                        sk1 = to_float(s1.get('overall_rating', 50))
                        sk2 = to_float(s2.get('overall_rating', 50))
                        
                        e1 = ELO_CACHE.get("ATP", {}).get(p1['last_name'].lower(), {}).get('Hard', 1500)
                        e2 = ELO_CACHE.get("ATP", {}).get(p2['last_name'].lower(), {}).get('Hard', 1500)
                        
                        prob = QuantumMathEngine.calculate_fair_probability(ai_tac, ai_phy, sk1, sk2, e1, e2)
                        mp1, _ = QuantumMathEngine.devig_odds(m1, m2)
                        
                        # DB INSERT (Async Wrapper)
                        entry = {
                            "player1_name": p1['last_name'], "player2_name": p2['last_name'],
                            "tournament": court_db['name'], "odds1": m1, "odds2": m2,
                            "ai_fair_odds1": round(1/prob, 2), "ai_fair_odds2": round(1/(1-prob), 2),
                            "ai_analysis_text": json.dumps({"edge": f"{(prob-mp1)*100:.1f}%", "v": ai_data.get("analysis_detail")}),
                            "match_time": iso_time,
                            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        }
                        
                        # Direct DB Insert via Thread to avoid async lock issues
                        await asyncio.to_thread(lambda: db_manager.table("market_odds").insert(entry).execute())
                        
                # SKIP BOTH ROWS (We processed the pair)
                i += 2
            else:
                # Not a match start, check next row
                i += 1

# =================================================================
# 5. RUNNER
# =================================================================
class ScraperBot:
    def __init__(self):
        self.playwright = None
        self.browser = None
    
    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
    
    async def stop(self):
        await self.browser.close()
        await self.playwright.stop()

async def run():
    logger.info("🚀 Neural Scout V98.0 (Block Parser) STARTING...")
    bot = ScraperBot()
    await bot.start()
    try:
        await fetch_elo_ratings_optimized(bot)
        
        # Load Context
        p = await asyncio.to_thread(lambda: db_manager.table("players").select("*").execute().data)
        s = await asyncio.to_thread(lambda: db_manager.table("player_skills").select("*").execute().data)
        r = await asyncio.to_thread(lambda: db_manager.table("scouting_reports").select("*").execute().data)
        t = await asyncio.to_thread(lambda: db_manager.table("tournaments").select("*").execute().data)
        
        if not p: return
        
        s_map = {x['player_id']: x for x in s}
        resolver = ContextResolver(t)
        
        now = datetime.now()
        for d in range(14):
            await process_day(bot, now + timedelta(days=d), p, s_map, r, resolver)
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.critical(f"🔥 CRASH: {e}", exc_info=True)
    finally:
        await bot.stop()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())
