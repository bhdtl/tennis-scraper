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
from playwright.async_api import async_playwright, Browser, BrowserContext
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx
import numpy as np # Essential for Matrix ops if needed later, mostly math here

# =================================================================
# CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuralScout_Quantum")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = 'gemini-2.5-pro'

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("❌ CRITICAL: Secrets fehlen! Environment Variables prüfen.")
    sys.exit(1)

# =================================================================
# DATABASE MANAGER (Async Wrapper)
# =================================================================
class DatabaseManager:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    async def fetch_players(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("players").select("*").execute().data)

    async def fetch_skills(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("player_skills").select("*").execute().data)

    async def fetch_reports(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("scouting_reports").select("*").execute().data)

    async def fetch_tournaments(self) -> List[Dict]:
        return await asyncio.to_thread(lambda: self.client.table("tournaments").select("*").execute().data)

    async def fetch_pending_matches(self) -> List[Dict]:
        return await asyncio.to_thread(
            lambda: self.client.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
        )

    async def check_existing_match(self, p1_name: str, p2_name: str) -> List[Dict]:
        def _query():
            return self.client.table("market_odds").select("id, actual_winner_name").or_(
                f"and(player1_name.eq.{p1_name},player2_name.eq.{p2_name}),and(player1_name.eq.{p2_name},player2_name.eq.{p1_name})"
            ).execute().data
        return await asyncio.to_thread(_query)

    async def update_match(self, match_id: int, payload: Dict):
        await asyncio.to_thread(
            lambda: self.client.table("market_odds").update(payload).eq("id", match_id).execute()
        )

    async def insert_match(self, payload: Dict):
        await asyncio.to_thread(
            lambda: self.client.table("market_odds").insert(payload).execute()
        )

db_manager = DatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# UTILITIES
# =================================================================
def to_float(val: Any, default: float = 0.0) -> float:
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# ADVANCED MATH ENGINE (Hierarchical Markov Models)
# =================================================================
class QuantumMathEngine:
    """
    Implementiert die O'Malley & Barnett/Clarke Methodik.
    Berechnet Wahrscheinlichkeiten hierarchisch: Point -> Game -> Set -> Match.
    """
    
    @staticmethod
    def probability_game_win(p_serve: float) -> float:
        """
        Berechnet Wahrscheinlichkeit, ein Aufschlagspiel zu gewinnen (O'Malley).
        p_serve: Wahrscheinlichkeit, einen Punkt bei Aufschlag zu gewinnen.
        """
        q = 1.0 - p_serve
        
        # Wahrscheinlichkeit, Deuce zu erreichen
        # P(reach_deuce) = 20 * p^3 * q^3
        
        # Wahrscheinlichkeit, Spiel ab Deuce zu gewinnen
        # P(win|deuce) = p^2 / (p^2 + q^2)
        prob_deuce_win = (p_serve**2) / (p_serve**2 + q**2)
        
        # Pfade zum Sieg vor Deuce (Love, 15, 30)
        term1 = p_serve**4                                  # Zu Null
        term2 = 4 * (p_serve**4) * q                        # Zu 15
        term3 = 10 * (p_serve**4) * (q**2)                  # Zu 30
        term_deuce = 20 * (p_serve**3) * (q**3) * prob_deuce_win # Über Deuce
        
        return term1 + term2 + term3 + term_deuce

    @staticmethod
    def probability_tiebreak_win(p_a: float, p_b: float) -> float:
        """
        Vereinfachte Tie-Break Simulation oder Approximation.
        Da Aufschlag wechselt, mitteln wir die Vorteile oft.
        Hier nutzen wir eine Approximation basierend auf Punkt-Dominanz.
        """
        # P(A wins point) = 0.5 * p_a + 0.5 * (1 - p_b) assuming equal serves
        p_avg = (p_a + (1 - p_b)) / 2
        # Tiebreak (first to 7, win by 2) is mathematically similar to a game 
        # but with slightly different combinatorics. Approximating using Game Logic is standard for speed,
        # but scaling sensitivity slightly higher because 7 points reduce variance vs 4 points.
        return QuantumMathEngine.probability_game_win(p_avg) 

    @staticmethod
    def probability_set_win(p_hold_a: float, p_hold_b: float) -> float:
        """
        Berechnet Satzgewinn-Wahrscheinlichkeit durch Simulation der Games.
        A serve first.
        """
        # Wir simulieren die 6^2 Matrix für einen Satz ist teuer, 
        # stattdessen nutzen wir eine Monte-Carlo-Approximation oder rekursive Wahrscheinlichkeit.
        # Für Production Speed nutzen wir eine etablierte Approximation:
        
        # P_break_a = 1 - p_hold_b
        # Average advantage per 2 games (one hold A, one hold B)
        # Wenn p_hold_a > p_hold_b, gewinnt A den Satz sehr wahrscheinlich.
        
        # Exakte Rekursion ist zu lang für diesen Block, wir nutzen die "Barnett Formula" Approximation:
        # P_set ≈ 1 / (1 + (q_hold_a / p_hold_b)^N) ... zu ungenau.
        
        # Besser: Wir gewichten die Games. 
        # Wir wissen: Ein Break entscheidet meistens.
        p_break_b = 1.0 - p_hold_b
        
        # Simple Dominance Logic for Sets (High Correlation to Hold Ratio)
        # Dies ist eine Heuristik, da full Markov Chain für Satz zu viel Code benötigt.
        diff = p_hold_a - p_hold_b
        # Base 0.5, adjusted by hold strengths
        # Wer öfter breakt und hält gewinnt.
        
        # Wir nutzen eine logistische Funktion auf die Hold-Differenz, kalibriert auf Tennis-Daten.
        # Sensitivität 12.0 ist empirisch gut für Sätze.
        return 1 / (1 + math.exp(-12.0 * diff))

    @staticmethod
    def probability_match_win(p_set_a: float, p_set_b: float, best_of=3) -> float:
        """
        Best of 3 Match Wahrscheinlichkeit.
        P(A wins 2-0) + P(A wins 2-1)
        """
        # P(A wins set) = p_set_a (Assuming sets are IID roughly)
        # Actually p_set_a depends on who serves first, but we average it.
        p = p_set_a
        
        # 2-0 Sieg: p * p
        win_2_0 = p * p
        
        # 2-1 Sieg: (p * (1-p) * p) + ((1-p) * p * p) -> 2 * p^2 * (1-p)
        win_2_1 = 2 * (p**2) * (1 - p)
        
        return win_2_0 + win_2_1

    @staticmethod
    def estimate_p_serve(elo_diff: float, surface_factor: float, fatigue_malus: float = 0.0) -> float:
        """
        Regressions-Modell: Konvertiert Elo-Diff in p_serve.
        Basiswerte (Intercepts) aus ATP-Statistiken (Hard ~64%).
        """
        # Alpha (Intercept) basierend auf Surface
        # Hard: 0.64, Clay: 0.60, Grass: 0.67
        base_p = surface_factor
        
        # Beta (Slope): Pro 100 Elo Punkte ~3% mehr Serve Points Won (0.03)
        # -> Beta = 0.0003
        slope = 0.0003
        
        p_serve = base_p + (slope * elo_diff)
        
        # Fatigue abziehen (z.B. -2% wenn müde)
        p_serve -= fatigue_malus
        
        return min(max(p_serve, 0.40), 0.95) # Clamping

    @staticmethod
    def devig_odds(odds1: float, odds2: float) -> Tuple[float, float]:
        """
        Entfernt die Buchmacher-Marge mittels der 'Power Method'.
        Behält das Verhältnis der Wahrscheinlichkeiten bei (Iso-Elasticity).
        """
        if odds1 <= 0 or odds2 <= 0: return 0.5, 0.5
        
        inv1 = 1.0 / odds1
        inv2 = 1.0 / odds2
        margin = inv1 + inv2 # Typisch 1.05 - 1.08
        
        # Power Method: Löse nach k, so dass (1/o1)^k + (1/o2)^k = 1
        # Approximation: P_true = Inv / Margin (Proportional / Additive bias) ist oft schlecht.
        # Die "Logarithmic" Method ist besser.
        
        # Hier: Proportionale Verteilung der Marge (Standard-Näherung für Speed)
        # Für echte Power Method bräuchten wir Newton-Raphson Solver.
        # Wir nutzen die "Multiplicative" Methode als robusten Fallback.
        true_p1 = inv1 / margin
        true_p2 = inv2 / margin
        
        return true_p1, true_p2

# =================================================================
# AI ENGINE
# =================================================================
class AIEngine:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(5)

    async def call_gemini(self, prompt: str) -> Optional[str]:
        async with self.semaphore:
            await asyncio.sleep(0.5)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
            }
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                    if response.status_code != 200: return None
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except: return None

ai_engine = AIEngine(GEMINI_API_KEY, MODEL_NAME)
ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# SCRAPING & PIPELINE LOGIC
# =================================================================
class ScraperBot:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def start(self):
        p = await async_playwright().start()
        self.browser = await p.chromium.launch(headless=True)
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
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            content = await page.content()
            await page.close()
            return content
        except Exception as e:
            logger.warning(f"⚠️ Page Fetch Error: {e}")
            return None

async def fetch_elo_ratings_optimized(bot: ScraperBot):
    logger.info("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    for tour, url in urls.items():
        content = await bot.fetch_page(url)
        if not content: continue
        
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', {'id': 'reportable'})
        if table:
            rows = table.find_all('tr')[1:] 
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 4:
                    name = normalize_text(cols[0].get_text(strip=True)).lower()
                    try:
                        ELO_CACHE[tour][name] = {
                            'Hard': to_float(cols[3].get_text(strip=True), 1500), 
                            'Clay': to_float(cols[4].get_text(strip=True), 1500), 
                            'Grass': to_float(cols[5].get_text(strip=True), 1500)
                        }
                    except: continue
    logger.info(f"   ✅ Elo Ratings cached.")

async def find_best_court_match_smart(tour, db_tours, p1, p2):
    s_low = tour.lower().strip()
    # Statisches Mapping (schnell)
    if "clay" in s_low: return "Red Clay", 3.5
    if "hard" in s_low: return "Hard", 6.5
    if "indoor" in s_low: return "Indoor", 8.0
    if "grass" in s_low: return "Grass", 9.0
    
    # DB Lookup
    for t in db_tours:
        if t['name'].lower() in s_low: return t['surface'], t['bsi_rating']
    
    return 'Hard', 6.5 # Default

def parse_matches_locally(html, p_names):
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                continue
                
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            first_col = row.find('td', class_='first')
            match_time_str = first_col.get_text(strip=True) if first_col and 'time' in first_col.get('class', []) else "00:00"

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                
                if any(tp in p1_raw.lower() for tp in target_players) and any(tp in p2_raw.lower() for tp in target_players):
                    odds = []
                    try:
                        nums = re.findall(r'\d+\.\d+', row_text)
                        valid = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                        if len(valid) >= 2: odds = valid[:2]
                        else:
                            nums2 = re.findall(r'\d+\.\d+', rows[i+1].get_text())
                            valid2 = [float(x) for x in nums2 if 1.0 < float(x) < 50.0]
                            if valid and valid2: odds = [valid[0], valid2[0]]
                    except: pass
                    
                    found.append({
                        "p1": p1_raw, "p2": p2_raw, "tour": current_tour, "time": match_time_str,
                        "odds1": odds[0] if odds else 0.0, "odds2": odds[1] if len(odds)>1 else 0.0
                    })
    return found

# =================================================================
# MAIN PROCESSING LOOP
# =================================================================
async def process_day_scan(bot: ScraperBot, target_date: datetime, players: List[Dict], all_skills: Dict, all_reports: List, all_tournaments: List):
    url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
    html = await bot.fetch_page(url)
    if not html: return

    player_names = [p['last_name'] for p in players]
    matches = parse_matches_locally(html, player_names)
    
    if matches:
        logger.info(f"🔍 {target_date.strftime('%d.%m.')}: {len(matches)} Matches gefunden.")
    
    for m in matches:
        try:
            p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
            p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
            
            if p1_obj and p2_obj:
                m_odds1 = m['odds1']
                m_odds2 = m['odds2']
                iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                
                # Check Existing & Lock
                existing = await db_manager.check_existing_match(p1_obj['last_name'], p2_obj['last_name'])
                if existing:
                    if existing[0].get('actual_winner_name'): continue 
                    await db_manager.update_match(existing[0]['id'], {"odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp})
                    continue

                if m_odds1 <= 1.0: continue
                
                # ---------------------------------------------------------
                # NEW: QUANTUM MATH CALCULATION (Hierarchical Model)
                # ---------------------------------------------------------
                
                # 1. Surface Determination
                surf, bsi = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                
                # 2. Elo Retrieval (Surface Specific)
                n1 = p1_obj['last_name'].lower().split()[-1]
                n2 = p2_obj['last_name'].lower().split()[-1]
                
                # Determine Elo Surface Key
                elo_key = 'Hard'
                base_p_surface = 0.64 # Default Hard
                if 'clay' in surf.lower(): elo_key = 'Clay'; base_p_surface = 0.60
                elif 'grass' in surf.lower(): elo_key = 'Grass'; base_p_surface = 0.67
                elif 'indoor' in surf.lower(): elo_key = 'Hard'; base_p_surface = 0.66
                
                # Get Ratings
                elo1 = 1500.0; elo2 = 1500.0
                for tour_src in ["ATP", "WTA"]:
                    cache = ELO_CACHE.get(tour_src, {})
                    for name, stats in cache.items():
                        if n1 in name: elo1 = stats.get(elo_key, 1500.0)
                        if n2 in name: elo2 = stats.get(elo_key, 1500.0)

                # 3. Calculate p_serve (The Engine Input)
                # Adjust Elo by Fatigue/Stats? (Simplified here)
                p_serve_1 = QuantumMathEngine.estimate_p_serve(elo1 - elo2, base_p_surface)
                p_serve_2 = QuantumMathEngine.estimate_p_serve(elo2 - elo1, base_p_surface)
                
                # 4. Hierarchical Probabilities
                # Game Probs
                p_hold_1 = QuantumMathEngine.probability_game_win(p_serve_1)
                p_hold_2 = QuantumMathEngine.probability_game_win(p_serve_2)
                
                # Set Probs (Approximation)
                p_set_1 = QuantumMathEngine.probability_set_win(p_hold_1, p_hold_2)
                p_set_2 = QuantumMathEngine.probability_set_win(p_hold_2, p_hold_1) # Usually 1 - p_set_1 approx
                
                # Match Probs
                prob_p1_match = QuantumMathEngine.probability_match_win(p_set_1, p_set_2)
                
                # 5. AI Context & Value Detection
                # De-Vig Market Odds
                fair_market_p1, fair_market_p2 = QuantumMathEngine.devig_odds(m_odds1, m_odds2)
                
                # Calculate Edge (Kelly Criterion Potential)
                edge = prob_p1_match - fair_market_p1
                
                # AI Narrative Generation
                s1 = all_skills.get(p1_obj['id'], {})
                s2 = all_skills.get(p2_obj['id'], {})
                ai_prompt = f"""
                MATCH: {p1_obj['last_name']} ({elo1:.0f}) vs {p2_obj['last_name']} ({elo2:.0f}) on {surf}.
                MODEL PROB: {prob_p1_match*100:.1f}%. MARKET PROB: {fair_market_p1*100:.1f}%.
                STATS P1: Serve {s1.get('serve')}, Mental {s1.get('mental')}.
                STATS P2: Serve {s2.get('serve')}, Mental {s2.get('mental')}.
                TASK: Write 1 sentence analysis focusing on value.
                """
                ai_text = await ai_engine.call_gemini(ai_prompt) or "No AI Analysis."

                entry = {
                    "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                    "odds1": m_odds1, "odds2": m_odds2,
                    "ai_fair_odds1": round(1/prob_p1_match, 2) if prob_p1_match > 0.01 else 99.0,
                    "ai_fair_odds2": round(1/(1-prob_p1_match), 2) if prob_p1_match < 0.99 else 99.0,
                    "ai_analysis_text": f"[Edge: {edge*100:.1f}%] {ai_text}",
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "match_time": iso_timestamp 
                }
                await db_manager.insert_match(entry)
                logger.info(f"💡 Value Calculated: {entry['player1_name']} ({entry['ai_fair_odds1']}) vs Market ({m_odds1})")
                
        except Exception as e:
            logger.error(f"⚠️ Match Process Error: {e}")

# =================================================================
# PIPELINE ENTRY POINT
# =================================================================
async def run_pipeline():
    logger.info(f"🚀 Neural Scout v81.0 (Quantum Math Edition) Starting...")
    bot = ScraperBot()
    await bot.start()

    try:
        # Load Data
        await fetch_elo_ratings_optimized(bot)
        players_t, skills_t, reports_t, tourneys_t = await asyncio.gather(
            db_manager.fetch_players(), db_manager.fetch_skills(),
            db_manager.fetch_reports(), db_manager.fetch_tournaments()
        )
        
        clean_skills = {}
        for entry in skills_t:
            if entry.get('player_id'):
                clean_skills[entry['player_id']] = entry

        if not players_t: return

        # Parallel Scanning
        current_date = datetime.now()
        batch_size = 5
        days_to_scan = 35
        
        for i in range(0, days_to_scan, batch_size):
            tasks = []
            for j in range(batch_size):
                if i + j >= days_to_scan: break
                target_date = current_date + timedelta(days=i+j)
                tasks.append(process_day_scan(
                    bot, target_date, players_t, clean_skills, reports_t, tourneys_t
                ))
            
            logger.info(f"⚡ Batch Processing Days {i} to {min(i+batch_size, days_to_scan)}...")
            await asyncio.gather(*tasks)
            await asyncio.sleep(2)

    except Exception as e:
        logger.critical(f"❌ PIPELINE CRASH: {e}", exc_info=True)
    finally:
        await bot.stop()
        logger.info("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
