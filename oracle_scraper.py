# -*- coding: utf-8 -*-

import asyncio
import os
import re
import random
import logging
import sys
import unicodedata
import httpx
import math
import difflib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Set

from playwright.async_api import async_playwright, Browser
from bs4 import BeautifulSoup
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Oracle_PreWarmer_SOTA")

def log(msg: str):
    logger.info(msg)

log("🔮 Initializing Oracle Pre-Warmer (V153.6 SOTA Parity - Markov Chain Integrated)...")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
FUNCTION_URL = os.environ.get("SUPABASE_FUNCTION_URL")

if not SUPABASE_URL or not SUPABASE_KEY or not FUNCTION_URL:
    log("❌ CRITICAL: Secrets missing (Need URL, KEY and FUNCTION_URL)!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# SOTA: DYNAMIC WEIGHTS ALIGNMENT
DYNAMIC_WEIGHTS = {
    "ATP": {"SKILL": 0.50, "FORM": 0.35, "SURFACE": 0.15, "MC_VARIANCE": 1.20},
    "WTA": {"SKILL": 0.50, "FORM": 0.35, "SURFACE": 0.15, "MC_VARIANCE": 1.20}
}

# =================================================================
# 2. HELPERS & SOTA PLAYER MATCHING
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
    n = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = n.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

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

# 🚀 SOTA FIX: Elite Player Matching (1:1 Parity with Hauptscraper)
def find_player_smart(scraped_name_raw: str, db_players: List[Dict], report_ids: Set[str] = None) -> Optional[Dict]:
    if report_ids is None: report_ids = set()
    if not scraped_name_raw or len(scraped_name_raw) < 3 or re.search(r'\d', scraped_name_raw): return None 
    
    bad_words = ['satz', 'set', 'game', 'über', 'unter', 'handicap', 'sieger', 'winner', 'tennis', 'live', 'stream', 'stats', 'tv']
    if any(w in scraped_name_raw.lower() for w in bad_words): return None

    clean_scrape = normalize_db_name(clean_player_name(scraped_name_raw))
    scrape_tokens = clean_scrape.split()
    if not scrape_tokens: return None

    candidates = []
    has_comma = "," in scraped_name_raw
    
    if has_comma:
        parts = scraped_name_raw.split(',')
        scrape_last_part = normalize_db_name(parts[0])
        scrape_first_part = normalize_db_name(parts[1]) if len(parts) > 1 else ""
    else:
        scrape_last_part = clean_scrape
        scrape_first_part = ""
        if len(scrape_tokens) > 1:
            scrape_first_part = scrape_tokens[0]
    
    for p in db_players:
        db_last = normalize_db_name(p.get('last_name', ''))
        db_first = normalize_db_name(p.get('first_name', ''))
        
        db_last_tokens = db_last.split()
        db_first_tokens = db_first.split()
        
        score = 0
        last_matched = False
        
        if has_comma:
            if db_last == scrape_last_part:
                score += 60
                last_matched = True
            elif len(db_last) >= 5 and get_similarity(db_last, scrape_last_part) >= 0.80:
                score += 60
                last_matched = True
                
            if last_matched and db_first and scrape_first_part:
                if db_first == scrape_first_part or scrape_first_part in db_first:
                    score += 80 
                elif len(db_first) >= 5 and get_similarity(db_first, scrape_first_part) >= 0.80:
                    score += 50
                else:
                    sf_tokens = scrape_first_part.split()
                    if sf_tokens and len(sf_tokens[0]) > 0 and sf_tokens[0][0] == db_first[0]:
                        score += 15 
                    elif sf_tokens and len(sf_tokens[0]) > 0 and db_first and sf_tokens[0][0] != db_first[0]:
                        score -= 100 
        else:
            if db_last_tokens and db_last_tokens[-1] in scrape_tokens:
                score += 60
                last_matched = True
            elif len(db_last) >= 6:
                for t in scrape_tokens:
                    if len(t) >= 6 and get_similarity(db_last, t) >= 0.85:
                        score += 60
                        last_matched = True
                        break
                        
            if last_matched:
                toxic_leftover = False
                for st in scrape_tokens:
                    if len(st) > 3: 
                        explained = False
                        for dt in db_last_tokens + db_first_tokens:
                            if st == dt or get_similarity(st, dt) > 0.80:
                                explained = True
                                break
                        if not explained:
                            toxic_leftover = True
                            break
                
                if toxic_leftover:
                    score -= 50 
                    
                if db_first and score >= 60:
                    if any(ft in scrape_tokens for ft in db_first_tokens) or (scrape_first_part and scrape_first_part in db_first):
                        score += 80 
                    else:
                        db_f_init = db_first[0]
                        has_contradicting = False
                        has_matching = False
                        for st in scrape_tokens:
                            c_st = st.replace('.', '')
                            if 0 < len(c_st) <= 2: 
                                if c_st[0] == db_f_init: 
                                    has_matching = True
                                else: 
                                    has_contradicting = True
                        if has_matching: 
                            score += 15
                        elif has_contradicting: 
                            score -= 100 
                            
        if score >= 60: 
            candidates.append((p, score))
                
    if not candidates: 
        return None
        
    # SOTA: Tie-Breaker basierend auf vorhandenem Scouting Report
    candidates.sort(key=lambda x: (x[1], x[0]['id'] in report_ids), reverse=True)
    
    if len(candidates) > 1:
        top_score = candidates[0][1]
        second_score = candidates[1][1]
        
        if top_score == second_score:
            p1_n = f"{candidates[0][0].get('first_name')} {candidates[0][0].get('last_name')}"
            p2_n = f"{candidates[1][0].get('first_name')} {candidates[1][0].get('last_name')}"
            log(f"🚨 TIE-BREAKER ALARM: '{scraped_name_raw}' ist mehrdeutig zwischen {p1_n} und {p2_n}. Match wird ignoriert!")
            return "TIE_BREAKER"
                
    return candidates[0][0]

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
# 3. SOTA MARKOV CHAIN ENGINE (Ported to Oracle)
# =================================================================
class MarkovChainEngine:
    @staticmethod
    def run_simulation(s1: Dict, s2: Dict, formA: float, formB: float, 
                       bsi: float, styleA: str, styleB: str, 
                       iterations: int = 1500) -> Dict[str, Any]:
        
        def get_serve_prob(serve_skill, power_skill):
            return 0.50 + (((serve_skill * 0.7) + (power_skill * 0.3)) / 100.0) * 0.25
            
        def get_return_def(speed_skill, backhand_skill, forehand_skill):
            return (((speed_skill * 0.4) + (backhand_skill * 0.3) + (forehand_skill * 0.3)) / 100.0)

        base_serve_win_A = get_serve_prob(s1.get('serve', 50), s1.get('power', 50))
        base_serve_win_B = get_serve_prob(s2.get('serve', 50), s2.get('power', 50))

        return_def_A = get_return_def(s1.get('speed', 50), s1.get('backhand', 50), s1.get('forehand', 50))
        return_def_B = get_return_def(s2.get('speed', 50), s2.get('backhand', 50), s2.get('forehand', 50))

        p_A_wins_on_serve = base_serve_win_A - (return_def_B * 0.12)
        p_B_wins_on_serve = base_serve_win_B - (return_def_A * 0.12)

        overall_A = s1.get('overall_rating', 50)
        overall_B = s2.get('overall_rating', 50)
        overall_gap_delta = (overall_A - overall_B) * 0.0035

        p_A_wins_on_serve += overall_gap_delta
        p_B_wins_on_serve -= overall_gap_delta

        bsi_modifier_A = (bsi - 6.5) * 0.015
        bsi_modifier_B = bsi_modifier_A

        safe_style_A = (styleA or "").lower()
        safe_style_B = (styleB or "").lower()

        if "big server" in safe_style_A or "first-strike" in safe_style_A:
            bsi_modifier_A *= (2.5 if bsi < 6.0 else 1.5)
        if "big server" in safe_style_B or "first-strike" in safe_style_B:
            bsi_modifier_B *= (2.5 if bsi < 6.0 else 1.5)

        if ("counter puncher" in safe_style_A or "grinder" in safe_style_A) and bsi < 6.0:
            return_def_A *= 1.20
            p_B_wins_on_serve -= 0.03
        if ("counter puncher" in safe_style_B or "grinder" in safe_style_B) and bsi < 6.0:
            return_def_B *= 1.20
            p_A_wins_on_serve -= 0.03

        p_A_wins_on_serve += bsi_modifier_A
        p_B_wins_on_serve += bsi_modifier_B

        p_A_wins_on_serve += ((formA - 5) * 0.008)
        p_B_wins_on_serve += ((formB - 5) * 0.008)

        p_A_wins_on_serve = max(0.40, min(0.92, p_A_wins_on_serve))
        p_B_wins_on_serve = max(0.40, min(0.92, p_B_wins_on_serve))

        def simulate_game(prob_serve_win):
            pts_srv, pts_ret = 0, 0
            while True:
                if random.random() < prob_serve_win: pts_srv += 1
                else: pts_ret += 1
                if pts_srv >= 4 and pts_srv - pts_ret >= 2: return True
                if pts_ret >= 4 and pts_ret - pts_srv >= 2: return False

        def simulate_tiebreak(prob_A, prob_B):
            pts_A, pts_B = 0, 0
            serves_A = True
            pts_played = 0
            while True:
                if serves_A:
                    if random.random() < prob_A: pts_A += 1
                    else: pts_B += 1
                else:
                    if random.random() < prob_B: pts_B += 1
                    else: pts_A += 1
                pts_played += 1
                if pts_played % 2 == 1: serves_A = not serves_A

                if pts_A >= 7 and pts_A - pts_B >= 2: return True
                if pts_B >= 7 and pts_B - pts_A >= 2: return False

        def simulate_set():
            games_A, games_B = 0, 0
            serves_A = True
            while True:
                if serves_A:
                    if simulate_game(p_A_wins_on_serve): games_A += 1
                    else: games_B += 1
                else:
                    if simulate_game(p_B_wins_on_serve): games_B += 1
                    else: games_A += 1
                serves_A = not serves_A

                if games_A == 6 and games_B == 6: return simulate_tiebreak(p_A_wins_on_serve, p_B_wins_on_serve)
                if games_A >= 6 and games_A - games_B >= 2: return True
                if games_B >= 6 and games_B - games_A >= 2: return False

        match_wins_A, match_wins_B = 0, 0
        for _ in range(iterations):
            sets_A, sets_B = 0, 0
            while sets_A < 2 and sets_B < 2:
                if simulate_set(): sets_A += 1
                else: sets_B += 1
            if sets_A == 2: match_wins_A += 1
            else: match_wins_B += 1

        prob_A = (match_wins_A / iterations) * 100
        prob_B = (match_wins_B / iterations) * 100

        return {
            "probA": round(prob_A, 1),
            "probB": round(prob_B, 1),
            "scoreA": overall_A,
            "scoreB": overall_B
        }

# 🚀 SOTA FIX: Dynamic Base Power Calculation (Using Neural Weights)
def get_player_base_power(skill: float, form: float, surface: float, tour: str = "ATP") -> float:
    weights = DYNAMIC_WEIGHTS.get(tour, DYNAMIC_WEIGHTS["ATP"])
    power = (skill / 10) * weights["SKILL"] + form * weights["FORM"] + surface * weights["SURFACE"]
    if surface < 4.5: power -= (4.5 - surface) * weights["MC_VARIANCE"]
    return max(1.0, power)

async def call_edge_function(payload: dict):
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(FUNCTION_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log(f"Edge Function Call Failed: {e}")
            return None

# =================================================================
# 4. ORACLE SCRAPER & PIPELINE
# =================================================================
async def scrape_tennis_oracle_for_date(browser: Browser, target_date: datetime, db_data: Dict):
    page = await browser.new_page()
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
        db_reports = db_data['reports']
        
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
            
            if p1_obj == "TIE_BREAKER" or p2_obj == "TIE_BREAKER":
                log(f"🚨 Tie-Breaker Alarm im Oracle! Überspringe Match ({m['p1_raw']} vs {m['p2_raw']}).")
                continue
            
            if p1_obj and p2_obj and p1_obj['id'] != p2_obj['id']:
                n1 = p1_obj['last_name']
                n2 = p2_obj['last_name']
                tour_name = best_tour['name']
                tour_surface = best_tour['surface']
                bsi_rating = best_tour.get('bsi_rating', 5.0)
                is_wta = "WTA" in tour_name.upper()
                tour_key = "WTA" if is_wta else "ATP"

                # 🚀 SOTA FIX: Calculate Markov Chain locally to feed the AI Edge Function
                s1_data = db_skills.get(p1_obj['id'], {})
                s2_data = db_skills.get(p2_obj['id'], {})
                f1_score = extract_rating(p1_obj.get('form_rating'))
                f2_score = extract_rating(p2_obj.get('form_rating'))
                
                mc_results = MarkovChainEngine.run_simulation(
                    s1=s1_data, s2=s2_data, 
                    formA=f1_score, formB=f2_score,
                    bsi=bsi_rating, 
                    styleA=p1_obj.get('play_style', ''), styleB=p2_obj.get('play_style', '')
                )
                
                log(f"   🧠 Sending {n1} vs {n2} to AI Edge Function (MC Context: {mc_results['probA']}%)...")
                
                # Payload Injection: We pass the MC prob to enforce Conviction Directives on the Edge
                payload = {
                    "playerAId": p1_obj['id'],
                    "playerBId": p2_obj['id'],
                    "surface": tour_surface,
                    "bsi": bsi_rating,
                    "location": tour_name,
                    "skillsA": s1_data,
                    "skillsB": s2_data,
                    "reportA": db_reports.get(p1_obj['id']),
                    "reportB": db_reports.get(p2_obj['id']),
                    "mc_prob_a": mc_results['probA'],
                    "mc_prob_b": mc_results['probB'],
                    "language": "en"
                }

                edge_result = await call_edge_function(payload)
                
                if edge_result and "winner_prediction" in edge_result:
                    predicted_winner = edge_result["winner_prediction"]
                    raw_prob = str(edge_result.get("win_probability", "50.0")).replace("%", "")
                    try: win_prob = float(raw_prob)
                    except: win_prob = 50.0

                    db_payload = {
                        "tournament_name": tour_name,
                        "match_date": target_date.strftime('%Y-%m-%d'),
                        "player_a_name": n1,
                        "player_b_name": n2,
                        "predicted_winner": predicted_winner,
                        "win_probability": win_prob,
                        "surface": tour_surface,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    supabase.table("tournament_oracle_draws").upsert(
                        db_payload, on_conflict="tournament_name,player_a_name,player_b_name"
                    ).execute()
                    
                    inserted_matches += 1
                    log(f"   ✅ AI Result: {predicted_winner} wins.")
                    await asyncio.sleep(2)
                else:
                    log(f"   ⚠️ Edge Function failed for {n1} vs {n2}. Skipping.")

                if tour_name not in tournament_pools:
                    tournament_pools[tour_name] = {"surface": tour_surface, "tour_key": tour_key, "players": {}}
                
                # 🚀 SOTA FIX: Dynamic Base Power for Outrights
                baseA = get_player_base_power(float(s1_data.get('overall_rating', 50)), f1_score, get_surface_rating(p1_obj.get('surface_ratings'), tour_surface), tour_key)
                baseB = get_player_base_power(float(s2_data.get('overall_rating', 50)), f2_score, get_surface_rating(p2_obj.get('surface_ratings'), tour_surface), tour_key)
                
                tournament_pools[tour_name]["players"][n1] = baseA
                tournament_pools[tour_name]["players"][n2] = baseB

        log(f"✅ Synced {inserted_matches} matches via Edge Function.")

        inserted_outrights = 0
        for tour_name, tour_data in tournament_pools.items():
            players_dict = tour_data["players"]
            surface = tour_data["surface"]
            if len(players_dict) < 4: continue
            
            total_power_pool = sum(power**3 for power in players_dict.values())
            for player_name, power in players_dict.items():
                trophy_prob = ((power**3) / total_power_pool) * 100
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
                    
        log(f"🏆 Calculated {inserted_outrights} Deep Run Outrights.")

    except Exception as e: 
        log(f"⚠️ Scraping Error: {e}")
    finally: 
        await page.close()

async def run_pipeline():
    try:
        # Load Weights first for SOTA alignment
        weights_res = supabase.table("ai_system_weights").select("*").execute()
        if weights_res.data:
            for w in weights_res.data:
                tour = w.get("tour", "ATP")
                DYNAMIC_WEIGHTS[tour] = {
                    "SKILL": float(w.get("weight_skill", 0.50)),
                    "FORM": float(w.get("weight_form", 0.35)),
                    "SURFACE": float(w.get("weight_surface", 0.15)),
                    "MC_VARIANCE": float(w.get("mc_variance", 1.20))
                }

        log("📥 Loading DB Context (Players)...")
        try:
            res = supabase.table("players").select("id, last_name, first_name, form_rating, surface_ratings").execute()
            players = res.data
        except Exception as e:
            log(f"🚨 CRITICAL: Database fetch failed. Error details: {str(e)}")
            log("🔄 Attempting Emergency Load (Safe Mode - No Ratings)...")
            res = supabase.table("players").select("id, last_name, first_name").execute()
            players = res.data

        log("📥 Loading DB Context (Tournaments)...")
        tournaments = supabase.table("tournaments").select("id, name, surface, bsi_rating").execute().data
        
        if not players or not tournaments:
            log("❌ Error: Could not load critical DB reference data.")
            return

        for p in players:
            if isinstance(p.get('form_rating'), str):
                try: import json; p['form_rating'] = json.loads(p['form_rating'])
                except: p['form_rating'] = None
            if isinstance(p.get('surface_ratings'), str):
                try: import json; p['surface_ratings'] = json.loads(p['surface_ratings'])
                except: p['surface_ratings'] = None

        skills = []
        try:
            skills = supabase.table("player_skills").select("*").execute().data
        except: log("⚠️ Skipping player_skills due to data error.")

        reports = []
        try:
            reports = supabase.table("scouting_reports").select("player_id, strengths, weaknesses").execute().data
        except: log("⚠️ Skipping scouting_reports due to data error.")

        skills_dict = {s['player_id']: s for s in (skills or [])}
        reports_dict = {r['player_id']: r for r in (reports or [])}
        db_data = {'players': players, 'tournaments': tournaments, 'skills': skills_dict, 'reports': reports_dict}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            for day_offset in range(0, 3): 
                target_date = datetime.now() + timedelta(days=day_offset)
                await scrape_tennis_oracle_for_date(browser, target_date, db_data)
            await browser.close()
        log("🏁 Oracle Cycle Finished.")
        
    except Exception as e:
        log(f"❌ Pipeline crashed: {e}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
