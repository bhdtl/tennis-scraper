# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import random
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Set

import httpx
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_Architect")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V41.0 - Sticky AI / Dynamic Odds)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_NAME = 'gemini-2.0-flash'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena",
    "Brisbane": "Pat Rafter Arena",
    "Adelaide": "Memorial Drive Tennis Centre",
    "Melbourne": "Rod Laver Arena"
}
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

# =================================================================
# 2. HELPER FUNCTIONS & SAFETY
# =================================================================
def to_float(val: Any, default: float = 50.0) -> float:
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    if not raw: return ""
    # Remove standard garbage
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    # Remove seeding info like (1), (WC)
    clean = re.sub(r'\s*\(\d+\)', '', clean)
    clean = re.sub(r'\s*\(.*?\)', '', clean)
    return clean.replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = raw
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<.*?>', '', clean) # Remove stray tags
    clean = re.sub(r'S\d+.*$', '', clean) 
    clean = re.sub(r'H2H.*$', '', clean)
    clean = re.sub(r'\b(Challenger|Men|Women|Singles|Doubles)\b', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s\d+$', '', clean)
    return clean.strip()

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip()
    parts = clean.split()
    return parts[-1].lower() if parts else ""

def ensure_dict(data: Any) -> Dict:
    try:
        if isinstance(data, dict): return data
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict): return data[0]
        return {}
    except: return {}

def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str]) -> Optional[Dict]:
    """
    SOTA Matching: Handles 'Nadal R.' vs 'Rafael Nadal' and fuzzy matching.
    """
    if not scraped_name_raw or not db_players: return None
    
    clean_scrape = clean_player_name(scraped_name_raw).lower()
    last_name_scrape = clean_scrape.split()[0] if " " in clean_scrape else clean_scrape
    
    candidates = []
    for p in db_players:
        if not isinstance(p, dict): continue
        p_last = p.get('last_name', '').lower()
        
        # 1. Exact Last Name Match
        if p_last == last_name_scrape:
            candidates.append(p)
            continue
            
        # 2. Check if scrape is contained in DB full name (e.g. "Alcaraz" in "Carlos Alcaraz")
        if last_name_scrape in p_last:
             candidates.append(p)

    if not candidates: return None
    
    # Priority: Player with Scouting Report > First Match
    for cand in candidates:
        if cand['id'] in report_ids: return cand
    
    return candidates[0]

def calculate_fuzzy_score(scraped_name: str, db_name: str) -> int:
    s_norm = normalize_text(scraped_name).lower()
    d_norm = normalize_text(db_name).lower()
    if d_norm in s_norm and len(d_norm) > 3: return 100
    
    s_tokens = set(re.findall(r'\w+', s_norm))
    d_tokens = set(re.findall(r'\w+', d_norm))
    
    stop_words = {'atp', 'wta', 'open', 'tour', '2025', '2026', 'challenger', 'itf'}
    s_tokens -= stop_words
    d_tokens -= stop_words
    
    if not s_tokens or not d_tokens: return 0
    common = s_tokens.intersection(d_tokens)
    score = len(common) * 10
    
    # Bonus for specific locations often missed
    if "indoor" in s_tokens and "indoor" in d_tokens: score += 20
    if "canberra" in s_tokens and "canberra" in d_tokens: score += 30
    
    return score

# =================================================================
# 3. GEMINI ENGINE
# =================================================================
async def call_gemini(prompt: str, model: str = MODEL_NAME) -> Optional[str]:
    await asyncio.sleep(0.8) # Rate limit protection
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.4}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                log(f"   ⚠️ Gemini API Error: {response.status_code}")
                return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Gemini Call Failed: {e}")
            return None

# =================================================================
# 4. DATA FETCHING
# =================================================================
async def fetch_tennisexplorer_stats(browser: Browser, relative_url: str, surface: str) -> float:
    if not relative_url: return 0.5
    cache_key = f"{relative_url}_{surface}"
    if cache_key in SURFACE_STATS_CACHE: return SURFACE_STATS_CACHE[cache_key]
    
    if not relative_url.startswith("/"): relative_url = f"/{relative_url}"
    url = f"https://www.tennisexplorer.com{relative_url}?annual=all&t={int(time.time())}"
    
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        target_header = "Hard"
        if "clay" in surface.lower(): target_header = "Clay"
        elif "grass" in surface.lower(): target_header = "Grass"
        elif "indoor" in surface.lower(): target_header = "Indoors"
        
        tables = soup.find_all('table', class_='result')
        total_matches = 0; total_wins = 0
        
        for table in tables:
            headers = [h.get_text(strip=True) for h in table.find_all('th')]
            if "Summary" in headers and target_header in headers:
                try:
                    col_idx = headers.index(target_header)
                    for row in table.find_all('tr'):
                        cells = row.find_all(['td', 'th'])
                        if cells and "Summary" in cells[0].get_text():
                             if len(cells) > col_idx:
                                stats_text = cells[col_idx].get_text(strip=True)
                                if "/" in stats_text:
                                    w, l = map(int, stats_text.split('/'))
                                    total_matches = w + l
                                    total_wins = w
                                    break
                except: pass
                break
                
        if total_matches > 0:
            rate = total_wins / total_matches
            SURFACE_STATS_CACHE[cache_key] = rate
            return rate
    except: pass
    finally: await page.close()
    return 0.5

async def fetch_elo_ratings(browser: Browser):
    log("📊 Lade Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    for tour, url in urls.items():
        page = await browser.new_page()
        try:
            await page.goto(f"{url}?t={int(time.time())}", wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table', {'id': 'reportable'})
            if table:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) > 4:
                        name = normalize_text(cols[0].get_text(strip=True)).lower()
                        last_name = name.split()[-1] if " " in name else name
                        ELO_CACHE[tour][last_name] = {
                            'Hard': to_float(cols[3].get_text(strip=True), 1500),
                            'Clay': to_float(cols[4].get_text(strip=True), 1500),
                            'Grass': to_float(cols[5].get_text(strip=True), 1500)
                        }
                log(f"   ✅ {tour} Elo geladen: {len(ELO_CACHE[tour])}")
        except: pass
        finally: await page.close()

async def fetch_player_form_hybrid(browser: Browser, player_last_name: str) -> Dict[str, Any]:
    try:
        res = supabase.table("market_odds").select("actual_winner_name, match_time").or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%").not_.is_("actual_winner_name", "null").order("match_time", desc=True).limit(5).execute()
        matches = res.data
        if matches and isinstance(matches, list) and len(matches) >= 3: 
            wins = 0
            for m in matches:
                if isinstance(m, dict) and player_last_name.lower() in m.get('actual_winner_name', '').lower(): wins += 1
            trend = "Neutral"
            if wins >= 4: trend = "🔥 ON FIRE"
            elif wins >= 3: trend = "Good"
            elif len(matches) - wins >= 4: trend = "❄️ ICE COLD"
            return {"text": f"{trend} (DB: {wins}/{len(matches)} wins)"}
    except: pass
    return {"text": "No recent DB data."}

async def get_db_data():
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        
        clean_skills = {}
        if skills:
            for entry in skills:
                if not isinstance(entry, dict): continue
                pid = entry.get('player_id')
                if pid:
                    clean_skills[pid] = {
                        'serve': to_float(entry.get('serve')), 'power': to_float(entry.get('power')),
                        'forehand': to_float(entry.get('forehand')), 'backhand': to_float(entry.get('backhand')),
                        'speed': to_float(entry.get('speed')), 'stamina': to_float(entry.get('stamina')),
                        'mental': to_float(entry.get('mental'))
                    }
        return players or [], clean_skills, reports or [], tournaments or []
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# 5. MATH CORE (Dynamic Calculation)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2):
    ai_meta = ensure_dict(ai_meta)
    n1 = get_last_name(p1_name); n2 = get_last_name(p2_name)
    tour = "ATP"; bsi_val = to_float(bsi, 6.0)
    
    p1_stats = ELO_CACHE.get(tour, {}).get(n1, {})
    p2_stats = ELO_CACHE.get(tour, {}).get(n2, {})
    
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    elo1 = p1_stats.get(elo_surf, 1500)
    elo2 = p2_stats.get(elo_surf, 1500)
    
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8)
    
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    def get_defense(s): return s.get('speed', 50) + s.get('stamina', 50) + s.get('mental', 50)
    
    c1_score = get_offense(s1); c2_score = get_offense(s2)
    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.12)
    
    prob_skills = sigmoid_prob(sum(s1.values()) - sum(s2.values()), sensitivity=0.08)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    f1 = to_float(ai_meta.get('p1_form_score', 5)); f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    prob_surf_stat = surf_rate1 / (surf_rate1 + surf_rate2) if (surf_rate1 + surf_rate2) > 0 else 0.5
    
    weights = [0.20, 0.15, 0.10, 0.20, 0.15, 0.20] # Matchup, BSI, Skills, ELO, Form, SurfStats
    
    prob_alpha = (
        (prob_matchup * weights[0]) + 
        (prob_bsi * weights[1]) + 
        (prob_skills * weights[2]) + 
        (prob_elo * weights[3]) + 
        (prob_form * weights[4]) +
        (prob_surf_stat * weights[5])
    )
    
    if prob_alpha > 0.94: prob_alpha = 0.94
    elif prob_alpha < 0.06: prob_alpha = 0.06
    
    # --- Market Influence Integration ---
    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1; inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
    
    # 70% Alpha (AI/Stats), 30% Market
    final_prob = (prob_alpha * 0.70) + (prob_market * 0.30)
    return final_prob

def recalculate_fair_odds_with_new_market(old_fair_odds1: float, old_market_odds1: float, old_market_odds2: float, new_market_odds1: float, new_market_odds2: float) -> float:
    """
    Reverse engineers the AI Alpha component from the old calculation and applies it to new market odds.
    Formula: Final = (Alpha * 0.7) + (Market * 0.3)
    """
    try:
        # 1. Recover Old Market Probability
        old_prob_market = 0.5
        if old_market_odds1 > 1 and old_market_odds2 > 1:
            inv1 = 1/old_market_odds1; inv2 = 1/old_market_odds2
            old_prob_market = inv1 / (inv1 + inv2)
        
        # 2. Recover Old Final Probability
        if old_fair_odds1 <= 1.01: return 0.5 # Safety
        old_final_prob = 1 / old_fair_odds1
        
        # 3. Isolate Alpha (The AI/Stats Opinion)
        # Alpha * 0.7 = Final - (Market * 0.3)
        alpha_part = old_final_prob - (old_prob_market * 0.30)
        prob_alpha = alpha_part / 0.70
        
        # 4. Calculate New Market Probability
        new_prob_market = 0.5
        if new_market_odds1 > 1 and new_market_odds2 > 1:
            inv1 = 1/new_market_odds1; inv2 = 1/new_market_odds2
            new_prob_market = inv1 / (inv1 + inv2)
            
        # 5. Combine Alpha with New Market
        new_final_prob = (prob_alpha * 0.70) + (new_prob_market * 0.30)
        
        return new_final_prob
    except:
        return 0.5

# =================================================================
# 6. PIPELINE UTILS
# =================================================================
async def build_country_city_map(browser: Browser):
    if COUNTRY_TO_CITY_MAP: return
    COUNTRY_TO_CITY_MAP.update({
        "Australia": "Sydney", "United States": "Perth", "Poland": "Perth", 
        "Greece": "Sydney", "France": "Sydney" 
    })

async def resolve_united_cup_via_country(p1):
    return "Sydney"

async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    res = await call_gemini(f"Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\" }}")
    if res:
        try: 
            data = json.loads(res.replace("json", "").replace("```", "").strip())
            data = ensure_dict(data)
            TOURNAMENT_LOC_CACHE[scraped_name] = data
            return data
        except: pass
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2):
    s_low = clean_tournament_name(tour).lower().strip()
    best_match = None; best_score = 0
    for t in db_tours:
        score = calculate_fuzzy_score(s_low, t['name'])
        if score > best_score: best_score = score; best_match = t
    if best_match and best_score >= 30:
        return best_match['surface'], best_match['bsi_rating'], best_match.get('notes', '')

    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour)
    ai_loc = ensure_dict(ai_loc)
    if ai_loc and ai_loc.get('city'):
        surf = ai_loc.get('surface_guessed', 'Hard')
        return surf, (3.5 if 'clay' in surf.lower() else 6.5), f"AI Guess: {ai_loc['city']}"
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1, form2):
    log(f"   🤖 Asking AI for analysis on: {p1['last_name']} vs {p2['last_name']}")
    prompt = f"""
    ROLE: Elite Tennis Analyst.
    TASK: Analyze {p1['last_name']} vs {p2['last_name']} on {surface} (BSI {bsi}).
    DATA: ELO {elo1} vs {elo2}. FORM {form1['text']} vs {form2['text']}.
    COURT: {notes}
    OUTPUT JSON ONLY. FIELD 'ai_text' MUST BE 3 SENTENCES.
    JSON: {{ "p1_tactical_score": [0-10], "p2_tactical_score": [0-10], "p1_form_score": [0-10], "p2_form_score": [0-10], "ai_text": "Analysis string." }}
    """
    res = await call_gemini(prompt)
    data = ensure_dict(safe_get_ai_data(res))
    if not data.get('ai_text'):
        data['ai_text'] = f"Automated analysis: ELO {elo1} vs {elo2} suggests a competitive match on {surface}."
    return data

def safe_get_ai_data(res_text: Optional[str]) -> Dict[str, Any]:
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'ai_text': ''}
    if not res_text: return default
    try:
        cleaned = res_text.replace("json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return ensure_dict(data)
    except: return default

# =================================================================
# 7. PARSING & LOGIC (Updated for Sticky AI)
# =================================================================
async def scrape_tennis_explorer(browser: Browser, target_date):
    page = await browser.new_page()
    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}&t={int(time.time())}"
        log(f"📡 Scanning Date: {target_date.strftime('%Y-%m-%d')} -> {url}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        return await page.content()
    except Exception as e:
        log(f"   ⚠️ Scrape Error: {e}")
        return None
    finally: await page.close()

def parse_hybrid_tennis_explorer(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, 'html.parser')
    matches = []
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []):
                link = row.find("a")
                if link: current_tour = link.get_text(strip=True)
                i += 1; continue
            
            row_id = row.get("id", "")
            if not row_id or "bot" not in row.get("class", []): 
                i += 1; continue
            if i + 1 >= len(rows): break
            row2 = rows[i+1]
            
            try:
                p1_cell = row.find("td", class_="t-name")
                p1_a = p1_cell.find("a") if p1_cell else None
                p1_name = clean_player_name(p1_a.get_text(strip=True)) if p1_a else ""
                p1_href = p1_a['href'] if p1_a else None
                
                p2_cell = row2.find("td", class_="t-name")
                p2_a = p2_cell.find("a") if p2_cell else None
                p2_name = clean_player_name(p2_a.get_text(strip=True)) if p2_a else ""
                p2_href = p2_a['href'] if p2_a else None
                
                if not p1_name or not p2_name:
                    i += 2; continue
                
                # Time
                time_cell = row.find("td", class_="first")
                match_time = "00:00"
                if time_cell:
                    tm = re.search(r'(\d{1,2}:\d{2})', time_cell.get_text(strip=True))
                    if tm: match_time = tm.group(1).zfill(5)

                # Result
                winner_name = None; is_finished = False
                p1_res_cell = row.find("td", class_="result")
                p2_res_cell = row2.find("td", class_="result")
                if p1_res_cell and p2_res_cell:
                    try:
                        res1_txt = p1_res_cell.get_text(strip=True)
                        res2_txt = p2_res_cell.get_text(strip=True)
                        if res1_txt.isdigit() and res2_txt.isdigit():
                            s1 = int(res1_txt); s2 = int(res2_txt)
                            if s1 > s2 and s1 >= 2: winner_name = p1_name; is_finished = True
                            elif s2 > s1 and s2 >= 2: winner_name = p2_name; is_finished = True
                    except: pass
                
                # Fallback Result
                if not is_finished:
                    full_text = (row.get_text() + row2.get_text()).lower()
                    if "ret." in full_text or "w.o." in full_text:
                        if p1_res_cell and "ret." in p1_res_cell.get_text(): winner_name = p2_name; is_finished = True
                        if p2_res_cell and "ret." in p2_res_cell.get_text(): winner_name = p1_name; is_finished = True

                # Odds
                def extract_odd(r):
                    cells = r.find_all("td", class_=re.compile(r"course"))
                    for c in cells:
                        try:
                            val = float(c.get_text(strip=True))
                            if 1.0 < val < 100: return val
                        except: continue
                    return 0.0
                odds1 = extract_odd(row); odds2 = extract_odd(row2)
                
                matches.append({
                    "p1_raw": p1_name, "p2_raw": p2_name, "tour": clean_tournament_name(current_tour),
                    "time": match_time, "odds1": odds1, "odds2": odds2,
                    "p1_href": p1_href, "p2_href": p2_href, "actual_winner": winner_name, "is_finished": is_finished
                })
                i += 2 
            except: i += 1
    return matches

async def process_match_logic(browser, m, players, report_ids, all_skills, all_tournaments, target_date_str):
    p1_obj = find_player_smart(m['p1_raw'], players, report_ids)
    p2_obj = find_player_smart(m['p2_raw'], players, report_ids)
    if not p1_obj or not p2_obj: return 

    # 1. SETTLEMENT
    if m['is_finished'] and m['actual_winner']:
        try:
            pending = supabase.table("market_odds").select("id").eq("player1_name", p1_obj['last_name']).eq("player2_name", p2_obj['last_name']).is_("actual_winner_name", "null").execute()
            if pending.data:
                winner_last_name = p1_obj['last_name'] if m['actual_winner'] == m['p1_raw'] else p2_obj['last_name']
                for pm in pending.data:
                    supabase.table("market_odds").update({"actual_winner_name": winner_last_name}).eq("id", pm['id']).execute()
                    log(f"      🏆 SETTLED: {winner_last_name} won")
        except: pass
        return

    # 2. ODDS UPDATE (Sticky AI Logic)
    if m['odds1'] < 1.01 or m['odds2'] < 1.01: return
    
    existing_p1 = supabase.table("market_odds").select("id, odds1, odds2, ai_analysis_text, ai_fair_odds1, ai_fair_odds2").eq("player1_name", p1_obj['last_name']).eq("player2_name", p2_obj['last_name']).is_("actual_winner_name", "null").order("created_at", desc=True).limit(1).execute()
    
    db_match_id = None
    cached_ai = {}
    
    if existing_p1.data:
        rec = existing_p1.data[0]
        # SKIP if odds are identical
        if abs(rec.get('odds1', 0) - m['odds1']) < 0.02: return
            
        db_match_id = rec['id']
        if rec.get('ai_analysis_text'):
            cached_ai = {
                'ai_text': rec.get('ai_analysis_text'),
                'old_fair1': rec.get('ai_fair_odds1', 0),
                'old_odds1': rec.get('odds1', 0),
                'old_odds2': rec.get('odds2', 0)
            }

    ai_text_final = ""
    fair1, fair2 = 0, 0
    
    if db_match_id and cached_ai:
        log(f"   💰 Token Saver: Reusing AI Text, Recalculating Fair Odds due to Market Move")
        ai_text_final = cached_ai['ai_text']
        
        # RECALCULATION LOGIC: Use Old Fair Odds + New Market Odds
        new_prob = recalculate_fair_odds_with_new_market(
            old_fair_odds1=cached_ai['old_fair1'],
            old_market_odds1=cached_ai['old_odds1'],
            old_market_odds2=cached_ai['old_odds2'],
            new_market_odds1=m['odds1'],
            new_market_odds2=m['odds2']
        )
        fair1 = round(1/new_prob, 2) if new_prob > 0.01 else 99
        fair2 = round(1/(1-new_prob), 2) if new_prob < 0.99 else 99
        
    else:
        # FULL NEW ANALYSIS
        surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
        s1 = all_skills.get(p1_obj['id'], {}); s2 = all_skills.get(p2_obj['id'], {})
        surf_rate1 = await fetch_tennisexplorer_stats(browser, m['p1_href'], surf)
        surf_rate2 = await fetch_tennisexplorer_stats(browser, m['p2_href'], surf)
        
        f1_d = await fetch_player_form_hybrid(browser, p1_obj['last_name'])
        f2_d = await fetch_player_form_hybrid(browser, p2_obj['last_name'])
        elo_key = 'Clay' if 'clay' in surf.lower() else ('Grass' if 'grass' in surf.lower() else 'Hard')
        e1 = ELO_CACHE.get("ATP", {}).get(p1_obj['last_name'], {}).get(elo_key, 1500)
        e2 = ELO_CACHE.get("ATP", {}).get(p2_obj['last_name'], {}).get(elo_key, 1500)
        
        ai_data = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, {}, {}, surf, bsi, notes, e1, e2, f1_d, f2_d)
        prob = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_data, m['odds1'], m['odds2'], surf_rate1, surf_rate2)
        
        ai_text_final = ai_data.get('ai_text', 'Analysis pending.')
        fair1 = round(1/prob, 2) if prob > 0.01 else 99
        fair2 = round(1/(1-prob), 2) if prob < 0.99 else 99

    data = {
        "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
        "odds1": m['odds1'], "odds2": m['odds2'],
        "ai_fair_odds1": fair1, "ai_fair_odds2": fair2,
        "ai_analysis_text": ai_text_final,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "match_time": f"{target_date_str}T{m['time']}:00Z"
    }
    
    if db_match_id:
        supabase.table("market_odds").update(data).eq("id", db_match_id).execute()
        log(f"🔄 Updated Odds (AI Sticky): {p1_obj['last_name']} vs {p2_obj['last_name']}")
    else:
        supabase.table("market_odds").insert(data).execute()
        log(f"💾 New Analysis: {p1_obj['last_name']} vs {p2_obj['last_name']}")

async def run_pipeline():
    log(f"🚀 Neural Scout V41.0 Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, _, all_tournaments = await get_db_data()
            if not players: return
            
            report_ids = set()
            for day_offset in range(-1, 4): 
                target_date = datetime.now() + timedelta(days=day_offset)
                date_str = target_date.strftime('%Y-%m-%d')
                html = await scrape_tennis_explorer(browser, target_date)
                if not html: continue
                matches = parse_hybrid_tennis_explorer(html)
                log(f"🔍 Found {len(matches)} items for {date_str}")
                for m in matches:
                    await process_match_logic(browser, m, players, report_ids, all_skills, all_tournaments, date_str)
        finally: await browser.close()
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
