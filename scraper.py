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
logger = logging.getLogger("NeuralScout_v102")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# --- STRICT USER REQUIREMENT ---
MODEL_NAME = 'gemini-2.5-flash-lite' 

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen!")
    sys.exit(1)

db_raw = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def clean_time_str(raw: str) -> str:
    if not raw: return "12:00"
    match = re.search(r'(\d{1,2}:\d{2})', raw)
    return match.group(1) if match else "12:00"

def extract_time_from_row(row) -> Optional[str]:
    """Returns HH:MM string ONLY if this row is a Match Start."""
    first_col = row.find('td', class_='first')
    if not first_col: return None
    txt = first_col.get_text(strip=True)
    if "result" in txt.lower(): return None
    
    time_match = re.search(r'(\d{1,2}:\d{2})', txt)
    if time_match: return time_match.group(1)
    if "live" in txt.lower(): return "12:00"
    return None

def validate_market(odds1, odds2):
    if odds1 <= 1.01 or odds2 <= 1.01: return False
    if odds1 > 50.0 or odds2 > 50.0: return False 
    margin = (1/odds1) + (1/odds2)
    return 0.85 < margin < 1.30

# =================================================================
# 3. DATABASE MANAGER
# =================================================================
class DatabaseManager:
    def __init__(self, client):
        self.client = client

    async def fetch_context(self):
        logger.info("📡 Fetching Database Context...")
        return await asyncio.gather(
            asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("market_odds").select("id, player1_name, player2_name, actual_winner_name").is_("actual_winner_name", "null").execute().data)
        )

    async def check_existing(self, p1, p2, active_matches):
        for m in active_matches:
            if (m['player1_name'] == p1 and m['player2_name'] == p2) or \
               (m['player1_name'] == p2 and m['player2_name'] == p1):
                return m
        return None

    async def upsert_match(self, match_data):
        await asyncio.to_thread(lambda: self.client.table("market_odds").insert(match_data).execute())

    async def update_odds(self, mid, time, o1, o2):
        await asyncio.to_thread(lambda: self.client.table("market_odds").update({
            "match_time": time, "odds1": o1, "odds2": o2
        }).eq("id", mid).execute())

db = DatabaseManager(db_raw)

# =================================================================
# 4. MATH & AI
# =================================================================
class QuantumMathEngine:
    @staticmethod
    def sigmoid(x, sensitivity=0.12):
        return 1 / (1 + math.exp(-sensitivity * x))

    @staticmethod
    def calc_fair_odds(ai_tac, ai_phy, s1, s2, e1, e2):
        p_ai = ai_tac
        p_skill = QuantumMathEngine.sigmoid(s1 - s2)
        p_phy = ai_phy
        p_elo = 1 / (1 + 10 ** ((e2 - e1) / 400))
        
        prob = (p_ai * 0.40) + (p_skill * 0.25) + (p_phy * 0.20) + (p_elo * 0.15)
        return max(0.05, min(0.95, prob))

    @staticmethod
    def devig(o1, o2):
        i1, i2 = 1/o1, 1/o2
        m = i1 + i2
        return i1/m, i2/m

class AIEngine:
    def __init__(self):
        self.sem = asyncio.Semaphore(1)

    async def analyze(self, p1, p2, s1, s2, r1, r2, court):
        async with self.sem:
            await asyncio.sleep(1.0)
            
            # --- DEEP DETAIL PROMPT ---
            prompt = f"""
            ROLE: Elite Tennis Analyst.
            TASK: Detailed Matchup Analysis.
            
            MATCH: {p1['last_name']} ({p1.get('play_style','Unknown')}) vs {p2['last_name']} ({p2.get('play_style','Unknown')}).
            VENUE: {court.get('name')} ({court.get('surface')}) | BSI: {court.get('bsi_rating')}.
            
            PLAYER 1 DATA ({p1['last_name']}): 
            - Skills: Serve {s1.get('serve')}, Return {s1.get('speed')}, Mental {s1.get('mental')}.
            - Report: {r1.get('strengths', 'N/A')}
            
            PLAYER 2 DATA ({p2['last_name']}):
            - Skills: Serve {s2.get('serve')}, Return {s2.get('speed')}, Mental {s2.get('mental')}.
            - Report: {r2.get('strengths', 'N/A')}

            INSTRUCTIONS:
            Analyze the specific tactical interaction. How does P1's strength target P2's weakness?
            Consider the court speed (BSI). Who does it favor and why?
            Provide a deep, multi-sentence verdict.

            OUTPUT JSON ONLY:
            {{
                "tactical_score_p1": 55, 
                "physics_score_p1": 50,
                "analysis_detail": "Detailed breakdown: P1's serve... while P2's backhand..."
            }}
            """
            
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient() as client:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
                        resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=35.0)
                        
                        if resp.status_code == 200:
                            raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                            return json.loads(raw.replace("```json","").replace("```","").strip())
                        
                        if resp.status_code == 429:
                            await asyncio.sleep(5)
                            continue
                        
                        # Fallback for 404/Other errors if user model name is wrong
                        if resp.status_code == 404:
                             logger.error(f"Model {MODEL_NAME} not found. Returning Default.")
                             return {"tactical_score_p1": 50, "physics_score_p1": 50, "analysis_detail": "Model Error"}
                            
                except Exception as e:
                    logger.error(f"AI Conn Error: {e}")
                    await asyncio.sleep(1)
            
            return {"tactical_score_p1": 50, "physics_score_p1": 50, "analysis_detail": "AI Timeout"}

    async def resolve_court(self, tour, p1, p2, candidates):
        if not candidates: return None
        cand_str = "\n".join([f"{i}: {c['name']} ({c.get('location')})" for i,c in enumerate(candidates)])
        prompt = f"Match: {p1} vs {p2} at {tour}. Pick ID from:\n{cand_str}\nJSON: {{'id': 0}}"
        async with self.sem:
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10.0)
                    if resp.status_code == 200:
                        raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                        idx = json.loads(raw.replace("```json","").replace("```","").strip()).get('id')
                        return candidates[idx] if idx < len(candidates) else candidates[0]
            except: pass
        return candidates[0]

ai_engine = AIEngine()
ELO_CACHE = {"ATP": {}, "WTA": {}}

# =================================================================
# 5. CORE LOGIC (HEADER FIX + BLOCK PARSER)
# =================================================================
async def fetch_elo_optimized(bot):
    logger.info("📊 Updating Elo...")
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
    def __init__(self, tours):
        self.tours = tours
        self.map = {t['name'].lower(): t for t in tours}
        self.keys = list(self.map.keys())

    async def resolve(self, tour_name, p1, p2):
        s = tour_name.lower().replace("atp", "").replace("wta", "").strip()
        if s in self.map: return self.map[s]
        
        cands = []
        fuzzy = difflib.get_close_matches(s, self.keys, n=3, cutoff=0.5)
        for f in fuzzy: cands.append(self.map[f])
        
        if "united cup" in s:
            for t in self.tours:
                if "united cup" in t['name'].lower() and t not in cands: cands.append(t)
        
        if cands: return await ai_engine.resolve_court(tour_name, p1, p2, cands)
        return {'name': tour_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium'}

async def process_day(bot, target_date, players, skills_map, reports, resolver, active_matches):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    logger.info(f"📅 Scanne: {target_date.strftime('%Y-%m-%d')}")
    
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
        i = 0
        while i < len(rows):
            row = rows[i]
            
            # --- 1. HEADERS & DATE (IMPROVED) ---
            # Date Flag
            if "flags" in row.get("class", []) and "head" not in row.get("class", []):
                txt = row.get_text()
                if "Tomorrow" in txt: active_date = target_date + timedelta(days=1)
                i += 1
                continue
            
            # Tournament Header (Aggressive Search)
            # 1. Classic Class Check
            is_head = "head" in row.get("class", [])
            # 2. Structure Check: Has 't-name' cell?
            has_tname = row.find('td', class_='t-name') is not None
            # 3. Link Check: Has link to /tennis/tournament/
            has_link = row.find('a', href=re.compile(r'/tennis/tournament/')) is not None
            
            if is_head or has_tname or has_link:
                link = row.find('a')
                if link: current_tour = link.get_text(strip=True)
                else: current_tour = row.get_text(strip=True)
                i += 1
                continue

            # --- 2. MATCH START DETECTION ---
            match_time = extract_time_from_row(row)
            
            if match_time and (i + 1 < len(rows)):
                row1, row2 = rows[i], rows[i+1]
                t1 = normalize_text(row1.get_text(separator=' ', strip=True))
                t2 = normalize_text(row2.get_text(separator=' ', strip=True))

                if '/' in t1 or '/' in t2:
                    i += 2
                    continue

                n1 = clean_player_name(t1.split('1.')[0] if '1.' in t1 else t1)
                n2 = clean_player_name(t2)
                
                p1 = next((p for p in players if p['last_name'].lower() in n1.lower()), None)
                p2 = next((p for p in players if p['last_name'].lower() in n2.lower()), None)

                valid = False
                if p1 and p2:
                    if p1.get('tour') == p2.get('tour'): valid = True
                
                if valid:
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
                        iso_time = f"{active_date.strftime('%Y-%m-%d')}T{clean_time_str(match_time)}:00Z"
                        existing = await db.check_existing(p1['last_name'], p2['last_name'], active_matches)
                        
                        if existing:
                            if not existing.get('actual_winner_name'):
                                await db.update_odds(existing['id'], iso_time, m1, m2)
                        else:
                            logger.info(f"✨ Analyzing: {p1['last_name']} vs {p2['last_name']} @ {current_tour}")
                            court = await resolver.resolve(current_tour, p1['last_name'], p2['last_name'])
                            
                            s1 = skills_map.get(p1['id'], {})
                            s2 = skills_map.get(p2['id'], {})
                            r1 = next((r for r in reports if r['player_id'] == p1['id']), {})
                            r2 = next((r for r in reports if r['player_id'] == p2['id']), {})
                            
                            ai_d = await ai_engine.analyze(p1, p2, s1, s2, r1, r2, court)
                            
                            ai_tac = ai_d.get('tactical_score_p1', 50)/100
                            ai_phy = ai_d.get('physics_score_p1', 50)/100
                            sk1 = to_float(s1.get('overall_rating', 50))
                            sk2 = to_float(s2.get('overall_rating', 50))
                            
                            ek = 'Hard'
                            if 'clay' in court.get('surface','').lower(): ek='Clay'
                            elif 'grass' in court.get('surface','').lower(): ek='Grass'
                            el1 = ELO_CACHE.get(p1['tour'], {}).get(p1['last_name'].lower(), {}).get(ek) or (sk1*15+500)
                            el2 = ELO_CACHE.get(p2['tour'], {}).get(p2['last_name'].lower(), {}).get(ek) or (sk2*15+500)
                            
                            prob = QuantumMathEngine.calc_fair_odds(ai_tac, ai_phy, sk1, sk2, el1, el2)
                            mp1, _ = QuantumMathEngine.devig(m1, m2)
                            
                            entry = {
                                "player1_name": p1['last_name'], "player2_name": p2['last_name'],
                                "tournament": court['name'], "odds1": m1, "odds2": m2,
                                "ai_fair_odds1": round(1/prob, 2), "ai_fair_odds2": round(1/(1-prob), 2),
                                "ai_analysis_text": json.dumps({"edge": f"{(prob-mp1)*100:.1f}%", "v": ai_d.get("analysis_detail")}),
                                "match_time": iso_time,
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            }
                            await db.upsert_match(entry)
                            logger.info(f"   💾 Saved. Edge: {(prob-mp1)*100:.1f}%")
                            
                i += 2 
            else:
                i += 1 

# =================================================================
# 6. RUNNER
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
    logger.info("🚀 Neural Scout v102.0 (Deep Analysis & Header Fix) STARTING...")
    bot = ScraperBot()
    await bot.start()
    try:
        await fetch_elo_optimized(bot)
        
        ctx = await db.fetch_context()
        p, s, r, t, matches = ctx
        
        if not p: return
        
        s_map = {x['player_id']: x for x in s}
        resolver = ContextResolver(t)
        
        now = datetime.now()
        for d in range(14):
            await process_day(bot, now + timedelta(days=d), p, s_map, r, resolver, matches)
            await asyncio.sleep(1)
            
    except Exception as e: logger.critical(f"🔥 CRASH: {e}", exc_info=True)
    finally: await bot.stop()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())
