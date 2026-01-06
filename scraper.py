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
from typing import List, Dict, Optional, Any, Tuple

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
logger = logging.getLogger("NeuralScout")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout (V4.2 - Raw HTML Injection)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_NAME = 'gemini-2.5-pro'

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
FORM_CACHE: Dict[str, Dict[str, Any]] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 

# --- TOPOLOGY MAP ---
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena",
    "Sydney": "Ken Rosewall Arena"
}

# =================================================================
# 2. HELPER FUNCTIONS
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
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = re.sub(r'S\d+[A-Z0-9]*$', '', raw).strip()
    return clean

def get_last_name(full_name: str) -> str:
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip()
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# --- STANDARD SLUGIFY ---
def slugify_name(first: str, last: str) -> str:
    full = f"{first}-{last}".lower()
    return normalize_text(full).replace(' ', '-')

# --- IDENTITY FIX ---
def find_player_safe(scraped_name_raw: str, db_players: List[Dict]) -> Optional[Dict]:
    clean_scrape = clean_player_name(scraped_name_raw).lower()
    candidates = []
    for p in db_players:
        if p['last_name'].lower() in clean_scrape:
            candidates.append(p)
            
    if not candidates: return None
    if len(candidates) == 1: return candidates[0]
    
    for cand in candidates:
        first_name = cand.get('first_name', '').lower()
        if first_name:
            initial = first_name[0]
            if f"{initial}." in clean_scrape or f" {initial} " in clean_scrape or clean_scrape.startswith(f"{initial} "):
                return cand
    return candidates[0]

# =================================================================
# 3. GEMINI ENGINE
# =================================================================
async def call_gemini(prompt: str, model: str = MODEL_NAME) -> Optional[str]:
    await asyncio.sleep(1.0)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                log(f"⚠️ Gemini API Error: {response.status_code} - {response.text}")
                return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            log(f"⚠️ Gemini Connection Failed: {e}")
            return None

# =================================================================
# 4. CORE LOGIC & DATA FETCHING
# =================================================================

async def resolve_profile_url_via_search(browser: Browser, player_name: str, tour: str) -> Optional[str]:
    """
    Veteran Strategy: Nutze die interne Suche der Zielseite.
    """
    search_page = await browser.new_page()
    try:
        encoded_name = player_name.replace(" ", "+")
        target_url = f"https://www.tennisergebnisse.net/?s={encoded_name}"
        
        await search_page.goto(target_url, timeout=15000, wait_until="domcontentloaded")
        
        links = await search_page.locator("a[href*='/atp/'], a[href*='/wta/']").all()
        best_slug = None
        name_parts = normalize_text(player_name.lower()).split()
        
        for link in links:
            href = await link.get_attribute("href")
            if not href: continue
            
            clean_href = href.strip("/")
            slug = clean_href.split("/")[-1]
            
            if name_parts[-1] in slug:
                best_slug = slug
                break 
        
        if best_slug:
            log(f"   🎯 TARGET ACQUIRED: {player_name} -> {best_slug}")
            return best_slug
            
    except Exception as e:
        log(f"   ⚠️ Internal Search Failed: {e}")
    finally:
        await search_page.close()
    return None

async def fetch_surface_winrate_ai(browser: Browser, p_obj: Dict, surface: str) -> float:
    cache_key = f"{p_obj['id']}_{surface}"
    if cache_key in SURFACE_STATS_CACHE: return SURFACE_STATS_CACHE[cache_key]

    # 1. DB Check
    db_slug = p_obj.get('stat_url_slug') 
    slug = ""
    prefix = "wta" if p_obj.get('tour') == 'WTA' else "atp"

    if db_slug and len(db_slug) > 2:
        slug = db_slug
    else:
        slug = slugify_name(p_obj['first_name'], p_obj['last_name'])

    url = f"https://www.tennisergebnisse.net/{prefix}/{slug}/"
    
    page = await browser.new_page()
    valid_page = False
    
    try:
        response = await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        title = await page.title()
        
        if response.status == 404 or "nicht gefunden" in title.lower() or "suche" in page.url.lower():
            log(f"   ❌ Identity Mismatch for {p_obj['last_name']}. Initiating Search Protocol...")
            
            search_query = f"{p_obj['first_name']} {p_obj['last_name']}".strip()
            found_slug = await resolve_profile_url_via_search(browser, search_query, p_obj.get('tour', 'ATP'))
            
            if found_slug:
                try:
                    supabase.table("players").update({"stat_url_slug": found_slug}).eq("id", p_obj['id']).execute()
                    log(f"   💾 Identity Saved to DB: {found_slug}")
                    
                    p_obj['stat_url_slug'] = found_slug
                    url = f"https://www.tennisergebnisse.net/{prefix}/{found_slug}/"
                    await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                    valid_page = True
                except: pass
            else:
                if p_obj['first_name']:
                     found_slug_last = await resolve_profile_url_via_search(browser, p_obj['last_name'], p_obj.get('tour', 'ATP'))
                     if found_slug_last:
                        url = f"https://www.tennisergebnisse.net/{prefix}/{found_slug_last}/"
                        await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                        valid_page = True
                        supabase.table("players").update({"stat_url_slug": found_slug_last}).eq("id", p_obj['id']).execute()
        else:
            valid_page = True

        if valid_page:
            # --- VETERAN FIX: HTML EXTRACTION ---
            # Wir suchen gezielt nach Tabellen, die "Bilanz" oder Stats enthalten
            # Statt nur Text zu dumpen, holen wir das HTML der Tabellen
            
            # Suche nach Tabellen im Content
            tables_html = await page.evaluate("""() => {
                const tables = Array.from(document.querySelectorAll('table'));
                return tables.map(t => t.outerHTML).join('\\n\\n');
            }""")
            
            # Wenn keine Tabellen gefunden, fallback auf Body Text
            if not tables_html or len(tables_html) < 50:
                tables_html = await page.inner_text("body")
                
            relevant_data = tables_html[:25000] # HTML ist verbose, wir brauchen mehr buffer

            target_surf = "Hardcourt"
            if "clay" in surface.lower(): target_surf = "Clay/Sand"
            elif "grass" in surface.lower(): target_surf = "Grass"
            elif "indoor" in surface.lower(): target_surf = "Indoor Hard"

            prompt = f"""
            ANALYZE TENNIS STATS HTML for {p_obj['last_name']}. Target: {target_surf}.
            
            DATA (HTML Tables):
            {relevant_data}
            
            TASK:
            1. Look for rows containing "Hartplatz", "Sandplatz", "Rasen", "Teppich" or "Hard", "Clay", "Grass".
            2. Extract W/L (Wins/Losses). Example format might be "15/5" or "15 - 5".
            3. Prioritize "Career" (Karriere) row. If not found, sum up recent years (2025, 2024).
            
            OUTPUT JSON ONLY: {{ "win_rate": 0.65, "matches": 250 }}
            If NO DATA found, return {{ "win_rate": 0.5, "matches": 0 }}
            """
            
            res = await call_gemini(prompt)
            if res:
                data = json.loads(res.replace("json", "").replace("", "").strip())
                val = float(data.get('win_rate', 0.5))
                matches = int(data.get('matches', 0))
                
                if matches < 10: val = (val * matches + 0.5 * 10) / (matches + 10)
                val = max(0.05, min(0.95, val))
                
                log(f"   📊 {p_obj['last_name']}@{target_surf}: {val:.2f} ({matches} matches)")
                SURFACE_STATS_CACHE[cache_key] = val
                return val
             
    except Exception as e:
        log(f"   ⚠️ Stats Error: {e}")
        pass
    finally:
        await page.close()
    
    return 0.5

async def fetch_elo_ratings(browser: Browser):
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    
    for tour, url in urls.items():
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
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
                log(f"   ✅ {tour} Elo Ratings geladen: {len(ELO_CACHE[tour])} Spieler.")
        except Exception as e:
            log(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
        finally:
            await page.close()

async def fetch_player_form_hybrid(browser: Browser, player_last_name: str) -> Dict[str, Any]:
    try:
        res = supabase.table("market_odds")\
            .select("actual_winner_name, match_time")\
            .or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%")\
            .not_.is_("actual_winner_name", "null")\
            .order("match_time", desc=True)\
            .limit(5)\
            .execute()
            
        matches = res.data
        if matches and len(matches) >= 3: 
            wins = 0
            for m in matches:
                if player_last_name.lower() in m['actual_winner_name'].lower(): wins += 1
            
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
        for entry in skills:
            pid = entry.get('player_id')
            if pid:
                clean_skills[pid] = {
                    'serve': to_float(entry.get('serve')), 'power': to_float(entry.get('power')),
                    'forehand': to_float(entry.get('forehand')), 'backhand': to_float(entry.get('backhand')),
                    'speed': to_float(entry.get('speed')), 'stamina': to_float(entry.get('stamina')),
                    'mental': to_float(entry.get('mental'))
                }
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# 5. MATH CORE (SILICON VALLEY VETERAN EDITION)
# =================================================================
def sigmoid_prob(diff: float, sensitivity: float = 0.1) -> float:
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2, surf_rate1, surf_rate2):
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    bsi_val = to_float(bsi, 6.0)

    # 1. TACTICAL LAYER
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.8) 

    # 2. PHYSICS LAYER
    def get_offense(s): return s.get('serve', 50) + s.get('power', 50)
    def get_defense(s): return s.get('speed', 50) + s.get('stamina', 50) + s.get('mental', 50)
    def get_tech(s): return s.get('forehand', 50) + s.get('backhand', 50)

    off1 = get_offense(s1); def1 = get_defense(s1); tech1 = get_tech(s1)
    off2 = get_offense(s2); def2 = get_defense(s2); tech2 = get_tech(s2)

    c1_score = 0; c2_score = 0

    if bsi_val < 4.0: # THE MUD
        c1_score = (def1 * 0.7) + (tech1 * 0.3)
        c2_score = (def2 * 0.7) + (tech2 * 0.3)
    elif 4.0 <= bsi_val < 5.5: # GRINDER
        c1_score = (def1 * 0.5) + (tech1 * 0.4) + (off1 * 0.1)
        c2_score = (def2 * 0.5) + (tech2 * 0.4) + (off2 * 0.1)
    elif 5.5 <= bsi_val < 7.0: # NEUTRAL
        c1_score = def1 + tech1 + off1
        c2_score = def2 + tech2 + off2
    elif 7.0 <= bsi_val < 8.0: # FIRST STRIKE
        c1_score = (off1 * 0.5) + (tech1 * 0.4) + (def1 * 0.1)
        c2_score = (off2 * 0.5) + (tech2 * 0.4) + (def2 * 0.1)
    elif 8.0 <= bsi_val < 9.0: # SLICK
        c1_score = (off1 * 0.75) + (tech1 * 0.25)
        c2_score = (off2 * 0.75) + (tech2 * 0.25)
    else: # THE CASINO
        c1_score = off1
        c2_score = off2

    prob_bsi = sigmoid_prob(c1_score - c2_score, sensitivity=0.10)

    # 3. SKILL BASELINE
    score_p1 = sum(s1.values())
    score_p2 = sum(s2.values())
    prob_skills = sigmoid_prob(score_p1 - score_p2, sensitivity=0.08)

    # 4. ELO HISTORICAL LAYER
    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
        
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    # --- FORM COMPONENT ---
    f1 = to_float(ai_meta.get('p1_form_score', 5))
    f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.5)
    
    # --- SURFACE SPECIALIST COMPONENT ---
    surf_diff = surf_rate1 - surf_rate2
    prob_surface_stats = 0.5 + (surf_diff * 0.7) 
    prob_surface_stats = max(0.1, min(0.9, prob_surface_stats))

    # 5. WEIGHTED FUSION (Surface Heavy)
    physics_weight = 0.20
    if bsi_val >= 8.5 or bsi_val <= 3.5: physics_weight = 0.25
    
    elo_weight = 0.10
    matchup_weight = 0.25
    form_weight = 0.10
    surface_stats_weight = 0.25 
    
    skill_weight = 1.0 - (physics_weight + elo_weight + matchup_weight + form_weight + surface_stats_weight)

    prob_alpha = (prob_matchup * matchup_weight) + \
                 (prob_bsi * physics_weight) + \
                 (prob_skills * skill_weight) + \
                 (prob_elo * elo_weight) + \
                 (prob_form * form_weight) + \
                 (prob_surface_stats * surface_stats_weight)

    if prob_alpha > 0.60: prob_alpha = min(prob_alpha * 1.10, 0.94)
    elif prob_alpha < 0.40: prob_alpha = max(prob_alpha * 0.90, 0.06)

    prob_market = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        inv1 = 1/market_odds1
        inv2 = 1/market_odds2
        prob_market = inv1 / (inv1 + inv2)
        
    final_prob = (prob_alpha * 0.75) + (prob_market * 0.25)
    return final_prob

# =================================================================
# 6. RESULT VERIFICATION ENGINE
# =================================================================
async def update_past_results(browser: Browser):
    log("🏆 Checking for Match Results (Restored v95.0 Aggressive Logic)...")
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    
    if not pending_matches: return

    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending_matches:
        try:
            created_at_str = pm['created_at'].replace('Z', '+00:00')
            created_at = datetime.fromisoformat(created_at_str)
            if (now_utc - created_at).total_seconds() / 60 > 65:
                safe_matches.append(pm)
        except: continue

    if not safe_matches: return

    for day_offset in range(3):
        target_date = datetime.now() - timedelta(days=day_offset)
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            table = soup.find('table', class_='result')
            if not table: continue

            rows = table.find_all('tr')
            for i in range(len(rows)):
                row = rows[i]
                if 'flags' in str(row) or 'head' in str(row): continue
                
                for pm in safe_matches:
                    p1_last = get_last_name(pm['player1_name'])
                    p2_last = get_last_name(pm['player2_name'])
                    row_text = row.get_text(separator=" ", strip=True).lower()
                    next_row_text = rows[i+1].get_text(separator=" ", strip=True).lower() if i+1 < len(rows) else ""
                    
                    match_found = (p1_last in row_text and p2_last in next_row_text) or \
                                  (p2_last in row_text and p1_last in next_row_text) or \
                                  (p1_last in row_text and p2_last in row_text)
                    
                    if match_found:
                        try:
                            is_retirement = "ret." in row_text or "w.o." in row_text
                            cols1 = row.find_all('td')
                            cols2 = rows[i+1].find_all('td') if i+1 < len(rows) else []
                            
                            def extract_scores_aggressive(columns):
                                scores = []
                                for col in columns:
                                    txt = col.get_text(strip=True)
                                    if len(txt) > 4: continue
                                    if '(' in txt: txt = txt.split('(')[0]
                                    if txt.isdigit() and len(txt) == 1 and int(txt) <= 7: scores.append(int(txt))
                                return scores

                            p1_scores = extract_scores_aggressive(cols1)
                            p2_scores = extract_scores_aggressive(cols2)
                            
                            p1_sets = 0; p2_sets = 0
                            for k in range(min(len(p1_scores), len(p2_scores))):
                                if p1_scores[k] > p2_scores[k]: p1_sets += 1
                                elif p2_scores[k] > p1_scores[k]: p2_sets += 1
                            
                            winner_name = None
                            if (p1_sets >= 2 and p1_sets > p2_sets) or (is_retirement and p1_sets > p2_sets):
                                if p1_last in row_text: winner_name = pm['player1_name']
                                elif p2_last in row_text: winner_name = pm['player2_name']
                            elif (p2_sets >= 2 and p2_sets > p1_sets) or (is_retirement and p2_sets > p1_sets):
                                if p1_last in next_row_text: winner_name = pm['player1_name']
                                elif p2_last in next_row_text: winner_name = pm['player2_name']
                            
                            if winner_name:
                                supabase.table("market_odds").update({"actual_winner_name": winner_name}).eq("id", pm['id']).execute()
                                safe_matches = [x for x in safe_matches if x['id'] != pm['id']]
                                log(f"      ✅ Verified Winner: {winner_name}")
                        except: pass
        except: pass
        finally: await page.close()

# =================================================================
# 7. MAIN PIPELINE
# =================================================================
async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    prompt = f"TASK: Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\", \"is_indoor\": bool }}"
    res = await call_gemini(prompt)
    if not res: return None
    try:
        data = json.loads(res.replace("json", "").replace("", "").strip())
        TOURNAMENT_LOC_CACHE[scraped_name] = data
        return data
    except: return None

async def build_country_city_map(browser: Browser):
    if COUNTRY_TO_CITY_MAP: return
    url = "https://www.unitedcup.com/en/scores/group-standings"
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=20000, wait_until="networkidle")
        text_content = await page.inner_text("body")
        prompt = f"TASK: Map Country to City (United Cup). Text: {text_content[:20000]}. JSON ONLY."
        res = await call_gemini(prompt)
        if res:
            COUNTRY_TO_CITY_MAP.update(json.loads(res.replace("json", "").replace("", "").strip()))
    except: pass
    finally: await page.close()

async def resolve_united_cup_via_country(p1):
    if not COUNTRY_TO_CITY_MAP: return None
    cache_key = f"COUNTRY_{p1}"
    if cache_key in TOURNAMENT_LOC_CACHE: country = TOURNAMENT_LOC_CACHE[cache_key]
    else:
        res = await call_gemini(f"Country of player {p1}? JSON: {{'country': 'Name'}}")
        country = json.loads(res.replace("json", "").replace("", "").strip()).get("country", "Unknown") if res else "Unknown"
        TOURNAMENT_LOC_CACHE[cache_key] = country
            
    if country in COUNTRY_TO_CITY_MAP: return CITY_TO_DB_STRING.get(COUNTRY_TO_CITY_MAP[country])
    return None

async def find_best_court_match_smart(tour, db_tours, p1, p2):
    s_low = clean_tournament_name(tour).lower().strip()
    
    if "united cup" in s_low:
        arena_target = await resolve_united_cup_via_country(p1)
        if arena_target:
            for t in db_tours:
                if "united cup" in t['name'].lower() and arena_target.lower() in t.get('location', '').lower():
                    return t['surface'], t['bsi_rating'], f"United Cup ({arena_target})"
        return "Hard Court Outdoor", 8.3, "United Cup (Sydney Default)"

    for t in db_tours:
        if t['name'].lower() == s_low: return t['surface'], t['bsi_rating'], t.get('notes', '')
    
    if "clay" in s_low: return "Red Clay", 3.5, "Local"
    if "hard" in s_low: return "Hard", 6.5, "Local"
    if "indoor" in s_low: return "Indoor", 8.0, "Local"
    
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour)
    if ai_loc and ai_loc.get('city'):
        city = ai_loc['city'].lower()
        surf = ai_loc.get('surface_guessed', 'Hard')
        for t in db_tours:
            if city in t['name'].lower(): return t['surface'], t['bsi_rating'], f"AI: {city}"
        return surf, (3.5 if 'clay' in surf.lower() else 6.5), f"AI Guess: {city}"
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes, elo1, elo2, form1, form2):
    prompt = f"""
    ROLE: Elite Tennis Analyst (Silicon Valley Level).
    TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}).
    COURT INTEL: "{notes}"
    
    SURFACE ELO ({surface}):
    - {p1['last_name']}: {elo1}
    - {p2['last_name']}: {elo2}
    
    RECENT FORM (Last 5-10 matches):
    - {p1['last_name']}: {form1['text']}
    - {p2['last_name']}: {form2['text']}
    
    PLAYER 1: {p1['last_name']}
    - Style: {p1.get('play_style')}
    - Strengths: {r1.get('strengths', 'N/A')}
    - Weaknesses: {r1.get('weaknesses', 'N/A')}
    
    PLAYER 2: {p2['last_name']}
    - Style: {p2.get('play_style')}
    - Strengths: {r2.get('strengths', 'N/A')}
    - Weaknesses: {r2.get('weaknesses', 'N/A')}
    
    METRICS (0-10): TACTICAL (25%), FORM (10%), UTR (5%).
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "..." }}
    """
    res = await call_gemini(prompt)
    default_res = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return default_res
    try: 
        cleaned = res.replace("json", "").replace("", "").strip()
        return json.loads(cleaned)
    except: return default_res

async def scrape_tennis_odds_for_date(browser: Browser, target_date):
    page = await browser.new_page()
    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        return await page.content()
    except Exception as e:
        log(f"❌ Scrape Error: {e}")
        return None
    finally: await page.close()

def parse_matches_locally(html, p_names):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                i += 1; continue
            if "doubles" in current_tour.lower(): i += 1; continue
            if i + 1 >= len(rows): break

            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                raw_time = first_col.get_text(strip=True)
                time_match = re.search(r'(\d{1,2}:\d{2})', raw_time)
                if time_match: match_time_str = time_match.group(1).zfill(5) 

            p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
            p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))

            if '/' in p1_raw or '/' in p2_raw: i += 1; continue

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
                    "p1_raw": p1_raw, "p2_raw": p2_raw, "tour": clean_tournament_name(current_tour), 
                    "time": match_time_str, "odds1": odds[0] if odds else 0.0, "odds2": odds[1] if len(odds)>1 else 0.0
                })
                i += 2 
            else: i += 1 
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v4.1 (Internal Search Protocol) Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            await update_past_results(browser)
            await fetch_elo_ratings(browser)
            await build_country_city_map(browser)
            players, all_skills, all_reports, all_tournaments = await get_db_data()
            
            if not players: return

            current_date = datetime.now()
            player_names = [p['last_name'] for p in players]
            
            for day_offset in range(11): 
                target_date = current_date + timedelta(days=day_offset)
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue

                matches = parse_matches_locally(html, player_names)
                log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    try:
                        p1_obj = find_player_safe(m['p1_raw'], players)
                        p2_obj = find_player_safe(m['p2_raw'], players)
                        
                        if p1_obj and p2_obj:
                            m_odds1 = m['odds1']; m_odds2 = m['odds2']
                            iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                            existing = supabase.table("market_odds").select("id").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                            
                            if existing.data: continue 
                            if m_odds1 <= 1.0: continue
                            
                            log(f"✨ New Match: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                            s1 = all_skills.get(p1_obj['id'], {})
                            s2 = all_skills.get(p2_obj['id'], {})
                            r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                            r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                            
                            surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                            
                            # --- SMART SURFACE STATS (INTERNAL SEARCH) ---
                            surf_rate1 = await fetch_surface_winrate_ai(browser, p1_obj, surf)
                            surf_rate2 = await fetch_surface_winrate_ai(browser, p2_obj, surf)

                            # GET HYBRID FORM
                            f1_data = await fetch_player_form_hybrid(browser, p1_obj['last_name'])
                            f2_data = await fetch_player_form_hybrid(browser, p2_obj['last_name'])
                            
                            # GET ELO
                            elo_surf = 'Hard'
                            if 'clay' in surf.lower(): elo_surf = 'Clay'
                            elif 'grass' in surf.lower(): elo_surf = 'Grass'
                            elo1_val = ELO_CACHE.get("ATP", {}).get(p1_obj['last_name'].lower(), {}).get(elo_surf, 1500)
                            elo2_val = ELO_CACHE.get("ATP", {}).get(p2_obj['last_name'].lower(), {}).get(elo_surf, 1500)

                            # ANALYZE WITH FULL INTEL
                            ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes, elo1_val, elo2_val, f1_data, f2_data)
                            
                            # NEU: ÜBERGABE DER SURFACE RATES
                            prob_p1 = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta, m_odds1, m_odds2, surf_rate1, surf_rate2)
                            
                            entry = {
                                "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                                "odds1": m_odds1, "odds2": m_odds2,
                                "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0.01 else 99,
                                "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 0.99 else 99,
                                "ai_analysis_text": ai_meta.get('ai_text', 'No analysis'),
                                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "match_time": iso_timestamp 
                            }
                            supabase.table("market_odds").insert(entry).execute()
                            log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']} (BSI: {bsi})")

                    except Exception as e:
                        log(f"⚠️ Match Error: {e}")
        finally: await browser.close()
    
    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
