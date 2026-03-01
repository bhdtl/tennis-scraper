# -*- coding: utf-8 -*-

import asyncio
import os
import re
import random
import logging
import sys
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any

from playwright.async_api import async_playwright, Browser
from bs4 import BeautifulSoup
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TournamentOracle")

def log(msg: str):
    logger.info(msg)

log("🔮 Initializing The Tournament Oracle (Matchups + Deep Run Engine)...")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets missing!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. HELPERS (CLEANING & MATCHING)
# =================================================================
def normalize_text(text: str) -> str:
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw: str) -> str:
    if not raw: return ""
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    clean = re.sub(r'\s*\(\d+\)', '', clean) 
    clean = re.sub(r'\s*\(.*?\)', '', clean) 
    return clean.replace('|', '').strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = raw
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<.*?>', '', clean)
    clean = re.sub(r'S\d+.*$', '', clean) 
    clean = re.sub(r'H2H.*$', '', clean)
    clean = re.sub(r'\b(Challenger|Men|Women|Singles|Doubles)\b', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s\d+$', '', clean)
    return clean.strip()

def normalize_db_name(name: str) -> str:
    if not name: return ""
    n = name.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def calculate_fuzzy_score(scraped_name: str, db_name: str) -> int:
    s_norm = normalize_text(scraped_name).lower()
    d_norm = normalize_text(db_name).lower()
    if d_norm in s_norm and len(d_norm) > 3: return 100
    s_tokens = set(re.findall(r'\w+', s_norm))
    d_tokens = set(re.findall(r'\w+', d_norm))
    stop_words = {'atp', 'wta', 'open', 'tour', '2025', '2026', 'challenger'}
    s_tokens -= stop_words; d_tokens -= stop_words
    if not s_tokens or not d_tokens: return 0
    common = s_tokens.intersection(d_tokens)
    score = len(common) * 10
    if "indoor" in s_tokens and "indoor" in d_tokens: score += 20
    return score

def find_player_smart(scraped_name_raw: str, db_players: List[Dict]) -> Optional[Dict]:
    if not scraped_name_raw or not db_players: return None
    clean_scrape = clean_player_name(scraped_name_raw)
    parts = clean_scrape.split()
    scrape_last = ""
    scrape_initial = ""
    if len(parts) >= 2:
        last_token = parts[-1].replace('.', '')
        if len(last_token) == 1 and last_token.isalpha():
            scrape_initial = last_token.lower()
            scrape_last = " ".join(parts[:-1]) 
        else:
            scrape_last = parts[-1]
            scrape_initial = parts[0][0].lower() if parts[0] else ""
    else:
        scrape_last = clean_scrape

    target_last = normalize_db_name(scrape_last)
    candidates = []
    for p in db_players:
        db_last_raw = p.get('last_name', '')
        db_last = normalize_db_name(db_last_raw)
        match_score = 0
        if db_last == target_last: match_score = 100
        elif target_last in db_last or db_last in target_last: 
            if len(target_last) > 3 and len(db_last) > 3: match_score = 80
        if match_score > 0:
            db_first = p.get('first_name', '').lower()
            if scrape_initial and db_first:
                if db_first.startswith(scrape_initial): match_score += 20 
                else: match_score -= 50 
            if match_score > 50: candidates.append((p, match_score))
    if not candidates: return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def extract_rating(rating_obj: Any, default: float = 5.0) -> float:
    if not rating_obj: return default
    if isinstance(rating_obj, (int, float)): return float(rating_obj)
    if isinstance(rating_obj, dict) and 'score' in rating_obj: return float(rating_obj['score'])
    return default

def get_surface_rating(surface_ratings: Any, surface_type: str) -> float:
    if not surface_ratings or not isinstance(surface_ratings, dict): return 5.0
    surf_key = 'hard'
    if 'clay' in surface_type.lower(): surf_key = 'clay'
    elif 'grass' in surface_type.lower(): surf_key = 'grass'
    
    surf_data = surface_ratings.get(surf_key)
    if isinstance(surf_data, dict) and 'rating' in surf_data:
        return float(surf_data['rating'])
    return 5.0

# =================================================================
# 3. HTML PARSER
# =================================================================
def parse_draws_locally(html):
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        pending_p1_raw = None
        
        i = 0
        while i < len(rows):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                pending_p1_raw = None
                i += 1
                continue
            
            cols = row.find_all('td')
            if len(cols) < 2: 
                i += 1
                continue
            
            first_cell = row.find('td', class_='first')
            p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
            if not p_cell: 
                i += 1
                continue
                
            p_raw = clean_player_name(p_cell.get_text(strip=True))
            
            if pending_p1_raw:
                p2_raw = p_raw
                if '/' in pending_p1_raw or '/' in p2_raw: 
                    pending_p1_raw = None
                    i += 1
                    continue
                
                row_text = row.get_text(strip=True).lower()
                prev_row_text = rows[i-1].get_text(strip=True).lower()
                is_finished = False
                
                if 'ret' in row_text or 'w.o' in row_text or 'ret' in prev_row_text or 'w.o' in prev_row_text:
                    is_finished = True
                
                p_idx = cols.index(p_cell)
                if p_idx + 1 < len(cols):
                    if cols[p_idx+1].get_text(strip=True).isdigit():
                        is_finished = True

                if not is_finished:
                    found.append({
                        "tour": clean_tournament_name(current_tour),
                        "p1_raw": pending_p1_raw,
                        "p2_raw": p2_raw
                    })
                
                pending_p1_raw = None
            else:
                if first_cell and first_cell.get('rowspan') == '2': 
                    pending_p1_raw = p_raw
                else: 
                    pending_p1_raw = p_raw
            i += 1
            
    return found

# =================================================================
# 4. MATH ENGINES (MATCHUPS & DEEP RUNS)
# =================================================================
def get_player_base_power(skill: float, form: float, surface: float) -> float:
    """Berechnet die absolute Stärke (Power Index) eines Spielers auf einem Belag."""
    w_skill = 0.50
    w_form = 0.35
    w_surf = 0.15
    power = (skill / 10) * w_skill + form * w_form + surface * w_surf
    if surface < 4.5: power -= (4.5 - surface) * 1.5
    return max(1.0, power)

def run_monte_carlo_matchup(baseA: float, baseB: float, iterations: int = 1000) -> tuple:
    """Standard Matchup Simulation"""
    variance = 1.2
    winsA, winsB = 0, 0
    for _ in range(iterations):
        simA = baseA + (random.gauss(0, 1) * variance)
        simB = baseB + (random.gauss(0, 1) * variance)
        if simA > simB: winsA += 1
        else: winsB += 1
    return (winsA / iterations) * 100, (winsB / iterations) * 100

# =================================================================
# 5. MAIN PIPELINE
# =================================================================
async def scrape_tennis_oracle_for_date(browser: Browser, target_date: datetime, db_data: Dict):
    page = await browser.new_page()
    
    # 🚀 NEW: Dictionary to collect players per tournament for Deep Run calculation
    tournament_pools = {} 

    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        log(f"📡 Scanning draws for: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        html = await page.content()
        
        matches = parse_draws_locally(html)
        
        db_players = db_data['players']
        db_tournaments = db_data['tournaments']
        db_skills = db_data['skills']
        
        inserted_matches = 0
        
        for m in matches:
            s_low = m['tour'].lower().strip()
            best_tour = None
            best_score = 0
            for t in db_tournaments:
                score = calculate_fuzzy_score(s_low, t['name'])
                if score > best_score:
                    best_score = score
                    best_tour = t
            
            if best_score < 20 or not best_tour: continue
                
            p1_obj = find_player_smart(m['p1_raw'], db_players)
            p2_obj = find_player_smart(m['p2_raw'], db_players)
            
            if p1_obj and p2_obj and p1_obj['id'] != p2_obj['id']:
                n1 = p1_obj['last_name']
                n2 = p2_obj['last_name']
                tour_name = best_tour['name']
                tour_surface = best_tour['surface']
                
                # Fetch Stats
                s1 = db_skills.get(p1_obj['id'], {})
                s2 = db_skills.get(p2_obj['id'], {})
                
                baseA = get_player_base_power(
                    float(s1.get('overall_rating', 50)),
                    extract_rating(p1_obj.get('form_rating')),
                    get_surface_rating(p1_obj.get('surface_ratings'), tour_surface)
                )
                
                baseB = get_player_base_power(
                    float(s2.get('overall_rating', 50)),
                    extract_rating(p2_obj.get('form_rating')),
                    get_surface_rating(p2_obj.get('surface_ratings'), tour_surface)
                )
                
                # 1. MATCHUP ENGINE
                probA, probB = run_monte_carlo_matchup(baseA, baseB)
                predicted_winner = n1 if probA > probB else n2
                win_prob = max(probA, probB)
                
                payload = {
                    "tournament_name": tour_name,
                    "match_date": target_date.strftime('%Y-%m-%d'),
                    "player_a_name": n1,
                    "player_b_name": n2,
                    "predicted_winner": predicted_winner,
                    "win_probability": round(win_prob, 1),
                    "surface": tour_surface,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                supabase.table("tournament_oracle_draws").upsert(
                    payload, on_conflict="tournament_name,player_a_name,player_b_name"
                ).execute()
                inserted_matches += 1
                
                # 2. TOURNAMENT POOL COLLECTION (For Deep Runs)
                if tour_name not in tournament_pools:
                    tournament_pools[tour_name] = {"surface": tour_surface, "players": {}}
                
                # Add players to tournament pool with their absolute power
                tournament_pools[tour_name]["players"][n1] = baseA
                tournament_pools[tour_name]["players"][n2] = baseB

        log(f"✅ Saved {inserted_matches} match predictions.")

        # =================================================================
        # 🚀 THE DEEP RUN ENGINE (TOURNAMENT OUTRIGHTS)
        # =================================================================
        inserted_outrights = 0
        for tour_name, tour_data in tournament_pools.items():
            players_dict = tour_data["players"]
            surface = tour_data["surface"]
            
            # We need at least 4 known players to make a serious tournament prediction
            if len(players_dict) < 4: continue
            
            # Calculate Tournament Probabilities via Softmax-like Distribution
            # We square the power to amplify the difference between favorites and underdogs
            total_power_pool = sum(power**3 for power in players_dict.values())
            
            for player_name, power in players_dict.items():
                trophy_prob = ((power**3) / total_power_pool) * 100
                
                # Only save if they have at least a decent chance (> 2%)
                if trophy_prob >= 2.0:
                    outright_payload = {
                        "tournament_name": tour_name,
                        "player_name": player_name,
                        "trophy_probability": round(trophy_prob, 1),
                        "surface": surface,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }
                    supabase.table("tournament_outrights").upsert(
                        outright_payload, on_conflict="tournament_name,player_name"
                    ).execute()
                    inserted_outrights += 1
                    
        log(f"🏆 Calculated and saved {inserted_outrights} Deep Run Outrights.")

    except Exception as e: 
        log(f"⚠️ Scraping Error: {e}")
    finally: 
        await page.close()

async def run_pipeline():
    try:
        log("📥 Loading DB Context (Players, Skills, Tournaments)...")
        players = supabase.table("players").select("id, last_name, first_name, form_rating, surface_ratings").execute().data
        skills = supabase.table("player_skills").select("player_id, overall_rating").execute().data
        tournaments = supabase.table("tournaments").select("id, name, surface").execute().data
        
        if not players or not tournaments:
            log("❌ Error: Could not load DB reference data.")
            return

        skills_dict = {s['player_id']: s for s in (skills or [])}
        db_data = {'players': players, 'tournaments': tournaments, 'skills': skills_dict}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            for day_offset in range(0, 3): 
                target_date = datetime.now() + timedelta(days=day_offset)
                await scrape_tennis_oracle_for_date(browser, target_date, db_data)
                await asyncio.sleep(2) 
            await browser.close()
        log("🏁 Oracle Cycle Finished.")
        
    except Exception as e:
        log(f"❌ Pipeline crashed: {e}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
