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
from typing import Dict, List, Any, Optional

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
logger = logging.getLogger("NeuralScout_v200_Final")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Primary User Request
PRIMARY_MODEL = 'gemini-2.5-flash-lite'
# Fallback (Safety Net)
FALLBACK_MODEL = 'gemini-2.0-flash'

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
    # Remove betting spam and rank info
    raw = re.sub(r'\(\d+\)', '', raw) # Remove rank (5)
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def clean_time_str(raw: str) -> str:
    if not raw: return "12:00"
    # Find classic time HH:MM
    match = re.search(r'(\d{1,2}:\d{2})', raw)
    if match: return match.group(1)
    return "12:00"

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
        logger.info("📡 Loading Database Context...")
        return await asyncio.gather(
            asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data),
            asyncio.to_thread(lambda: self.client.table("market_odds").select("id, player1_name, player2_name, actual_winner_name").execute().data)
        )

    async def find_existing_match_id(self, p1, p2, active_matches):
        for m in active_matches:
            if (m['player1_name'] == p1 and m['player2_name'] == p2) or \
               (m['player1_name'] == p2 and m['player2_name'] == p1):
                return m
        return None

    async def upsert_match(self, match_data):
        await asyncio.to_thread(lambda: self.client.table("market_odds").insert(match_data).execute())

    async def update_match_data(self, mid, payload):
        await asyncio.to_thread(lambda: self.client.table("market_odds").update(payload).eq("id", mid).execute())

db = DatabaseManager(db_raw)

# =================================================================
# 4. AI & MATH ENGINE
# =================================================================
class QuantumMathEngine:
    @staticmethod
    def sigmoid(x, sensitivity=0.12): return 1 / (1 + math.exp(-sensitivity * x))
    
    @staticmethod
    def calc_fair_odds(ai_tac, ai_phy, s1, s2, e1, e2):
        prob = (ai_tac * 0.40) + (QuantumMathEngine.sigmoid(s1 - s2) * 0.25) + (ai_phy * 0.20) + ((1 / (1 + 10 ** ((e2 - e1) / 400))) * 0.15)
        return max(0.05, min(0.95, prob))
    
    @staticmethod
    def devig(o1, o2):
        i1, i2 = 1/o1, 1/o2
        m = i1 + i2
        return i1/m, i2/m

class AIEngine:
    def __init__(self):
        self.sem = asyncio.Semaphore(2)
        self.active_model = PRIMARY_MODEL

    async def call_gemini(self, prompt, timeout=30.0):
        async with self.sem:
            await asyncio.sleep(1.0)
            
            # Try Primary then Fallback
            models = [self.active_model, FALLBACK_MODEL]
            
            for m in models:
                try:
                    async with httpx.AsyncClient() as client:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
                        resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
                        
                        if resp.status_code == 200:
                            raw = resp.json()['candidates'][0]['content']['parts'][0]['text']
                            return json.loads(raw.replace("```json","").replace("```","").strip())
                        
                        elif resp.status_code == 404:
                            logger.warning(f"⚠️ Model {m} not found (404). Switching fallback.")
                            continue # Try next model
                        
                        elif resp.status_code == 429:
                            await asyncio.sleep(5)
                            continue
                            
                except Exception as e:
                    logger.error(f"AI Error ({m}): {e}")
                    await asyncio.sleep(1)
            
            return None

    async def analyze(self, p1, p2, s1, s2, r1, r2, court):
        prompt = f"""
        ROLE: Tennis Analyst. MATCH: {p1['last_name']} vs {p2['last_name']}.
        COURT: {court.get('name')} | BSI: {court.get('bsi_rating')} | Bounce: {court.get('bounce')}.
        P1 SKILL: Srv {s1.get('serve')}, Ret {s1.get('speed')}, Men {s1.get('mental')}. Report: {r1.get('strengths','')}
        P2 SKILL: Srv {s2.get('serve')}, Ret {s2.get('speed')}, Men {s2.get('mental')}. Report: {r2.get('strengths','')}
        
        TASK:
        1. TACTICAL (40%): Score P1 (0-100).
        2. PHYSICS (20%): Score P1 (0-100).
        3. VERDICT: Analysis.
        
        OUTPUT JSON: {{ "tactical_score_p1": 50, "physics_score_p1": 50, "analysis_detail": "..." }}
        """
        res = await self.call_gemini(prompt)
        return res or {"tactical_score_p1": 50, "physics_score_p1": 50, "analysis_detail": "AI Failed"}

    async def resolve_united_cup(self, p1, p2, candidates):
        cand_str = "\n".join([f"ID {i}: {c['name']} ({c.get('location')})" for i,c in enumerate(candidates)])
        prompt = f"TASK: Resolve United Cup Venue. Match: {p1} vs {p2}. Options:\n{cand_str}\nJSON: {{'id': 0}}"
        res = await self.call_gemini(prompt, timeout=15.0)
        if res and 'id' in res:
            idx = res['id']
            if idx < len(candidates): return candidates[idx]
        return candidates[0]

ai_engine = AIEngine()
ELO_CACHE = {"ATP": {}, "WTA": {}}

# =================================================================
# 5. CORE LOGIC (VETERAN STATE MACHINE PARSER)
# =================================================================
async def fetch_elo(bot):
    logger.info("📊 Updating Elo Ratings...")
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
        s_clean = tour_name.lower().replace("atp", "").replace("wta", "").strip()
        if "united cup" in s_clean:
            cands = [t for t in self.tours if "united cup" in t['name'].lower()]
            if len(cands) > 1: return await ai_engine.resolve_united_cup(p1, p2, cands)
            elif cands: return cands[0]
        
        matches = difflib.get_close_matches(s_clean, self.keys, n=1, cutoff=0.5)
        if matches: return self.map[matches[0]]
        return {'name': tour_name, 'surface': 'Hard', 'bsi_rating': 6.0, 'bounce': 'Medium'}

async def process_day(bot, target_date, players, skills_map, reports, resolver, active_matches):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    logger.info(f"📅 Scanning: {target_date.strftime('%Y-%m-%d')}")
    
    page = await bot.browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        content = await page.content()
    except:
        await page.close(); return
    await page.close()

    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all("table", class_="result")
    
    current_tour = "Unknown"
    active_date = target_date
    
    # --- STATE MACHINE VARIABLES ---
    pending_p1 = None
    pending_p1_odds = []
    pending_time = "12:00"
    pending_iso_time = None

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            # 1. HEADER CHECK (Updates State)
            if "flags" in row.get("class", []) and "head" not in row.get("class", []):
                if "Tomorrow" in row.get_text(): active_date = target_date + timedelta(days=1)
                pending_p1 = None # Reset
                continue
            
            link = row.find('a', href=re.compile(r'/tennis/tournament/'))
            if "head" in row.get("class", []) or link or "t-name" in str(row):
                if link: current_tour = link.get_text(strip=True)
                else: current_tour = row.get_text(strip=True)
                pending_p1 = None # Reset
                continue

            # 2. PARSE ROW CONTENT
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # Time Detection (New Match Indicator)
            first_col = row.find('td', class_='first')
            has_time = False
            match_time_str = "12:00"
            
            if first_col and 'time' in first_col.get('class', []):
                t_txt = first_col.get_text(strip=True).lower()
                tm = re.search(r'(\d{1,2}:\d{2})', t_txt)
                if tm: 
                    match_time_str = tm.group(1)
                    has_time = True
                elif any(x in t_txt for x in ["live", "after", "susp", "canc"]):
                    has_time = True # Active match

            # Player Extraction
            p_clean = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            p_obj = next((p for p in players if p['last_name'].lower() in p_clean.lower()), None)

            # Doubles Filter
            if '/' in row_text:
                pending_p1 = None
                continue

            # --- STATE LOGIC ---
            
            # CASE A: Start of a new match (Row with Time)
            if has_time and p_obj:
                pending_p1 = p_obj
                pending_time = match_time_str
                pending_iso_time = f"{active_date.strftime('%Y-%m-%d')}T{clean_time_str(match_time_str)}:00Z"
                
                # Get P1 Odds
                pending_p1_odds = []
                try:
                    for td in row.find_all('td', class_='course'):
                        v = float(td.get_text(strip=True))
                        if 1.01 <= v < 50: pending_p1_odds.append(v)
                except: pass
                continue # Wait for next row (P2)

            # CASE B: Second row (Player 2)
            if pending_p1 and p_obj and not has_time:
                p2_obj = p_obj
                
                # Safety Checks
                if p2_obj['id'] == pending_p1['id']: continue # Same player?
                if pending_p1.get('tour') != p2_obj.get('tour'): # Mixed gender?
                    pending_p1 = None
                    continue

                # Get P2 Odds
                p2_odds = []
                try:
                    for td in row.find_all('td', class_='course'):
                        v = float(td.get_text(strip=True))
                        if 1.01 <= v < 50: p2_odds.append(v)
                except: pass

                # Construct Odds Pair
                m1 = pending_p1_odds[0] if pending_p1_odds else 0
                m2 = p2_odds[0] if p2_odds else 0

                # --- RESULT CHECK (Actual Winner) ---
                actual_winner = None
                # Check for Bold tag in P1 row (we don't have P1 row obj here easily, but we can assume logic)
                # Better: Check scores in current row (P2) or logic
                # TennisExplorer usually bolds the winner name.
                is_p2_winner = row.find('b') is not None
                # P1 winner check is harder without P1 row object, but typically score format helps
                # Simple logic: If match finished and we have scores
                if "result" in str(row).lower() or re.search(r'\d-\d', row_text):
                    # Basic winner detection via odds or score not fully reliable here without parsing score string
                    # We will rely on DB 'actual_winner_name' update later if needed.
                    pass

                # --- DB PROCESS ---
                existing = await db.find_existing_match_id(pending_p1['last_name'], p2_obj['last_name'], active_matches)
                
                if validate_market(m1, m2):
                    if existing:
                        # Update
                        upd_data = {"match_time": pending_iso_time, "odds1": m1, "odds2": m2}
                        await db.update_match_data(existing['id'], upd_data)
                    else:
                        # New Analysis
                        court = await resolver.resolve(current_tour, pending_p1['last_name'], p2_obj['last_name'])
                        logger.info(f"✨ Match: {pending_p1['last_name']} vs {p2_obj['last_name']} @ {court['name']}")
                        
                        s1 = skills_map.get(pending_p1['id'], {})
                        s2 = skills_map.get(p2_obj['id'], {})
                        r1 = next((r for r in reports if r['player_id'] == pending_p1['id']), {})
                        r2 = next((r for r in reports if r['player_id'] == p2_obj['id']), {})
                        
                        ai_d = await ai_engine.analyze(pending_p1, p2_obj, s1, s2, r1, r2, court)
                        
                        ai_tac = ai_d.get('tactical_score_p1', 50)/100
                        ai_phy = ai_d.get('physics_score_p1', 50)/100
                        sk1 = to_float(s1.get('overall_rating', 50))
                        sk2 = to_float(s2.get('overall_rating', 50))
                        
                        ek = 'Hard'
                        if 'clay' in court.get('surface','').lower(): ek='Clay'
                        elif 'grass' in court.get('surface','').lower(): ek='Grass'
                        
                        el1 = ELO_CACHE.get(pending_p1['tour'], {}).get(pending_p1['last_name'].lower(), {}).get(ek) or (sk1*15+500)
                        el2 = ELO_CACHE.get(p2_obj['tour'], {}).get(p2_obj['last_name'].lower(), {}).get(ek) or (sk2*15+500)
                        
                        prob = QuantumMathEngine.calc_fair_odds(ai_tac, ai_phy, sk1, sk2, el1, el2)
                        mp1, _ = QuantumMathEngine.devig(m1, m2)
                        
                        entry = {
                            "player1_name": pending_p1['last_name'], "player2_name": p2_obj['last_name'],
                            "tournament": court['name'], "odds1": m1, "odds2": m2,
                            "ai_fair_odds1": round(1/prob, 2), "ai_fair_odds2": round(1/(1-prob), 2),
                            "ai_analysis_text": json.dumps({"edge": f"{(prob-mp1)*100:.1f}%", "v": ai_d.get("analysis_detail")}),
                            "match_time": pending_iso_time,
                            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        }
                        await db.upsert_match(entry)
                        logger.info(f"   💾 Saved. Edge: {(prob-mp1)*100:.1f}%")
                
                # Match processed, reset P1
                pending_p1 = None

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
    logger.info("🚀 Neural Scout v200.0 (Veteran State Machine) STARTING...")
    bot = ScraperBot()
    await bot.start()
    try:
        await fetch_elo(bot)
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
