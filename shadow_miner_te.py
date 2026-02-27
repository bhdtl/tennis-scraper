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
import numpy as np
import difflib

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
logger = logging.getLogger("NeuralScout_ShadowMiner")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Neural Scout SHADOW MINER (Smart Background Ingestion)...")

# Secrets Load
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen! Prüfe GitHub/Groq Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global Caches
ELO_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE: Dict[str, Any] = {}
SURFACE_STATS_CACHE: Dict[str, float] = {} 
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 

# SELF LEARNING STATE MEMORY (Neu für V150 Parität)
DYNAMIC_WEIGHTS = {
    "ATP": {"SKILL": 0.50, "FORM": 0.35, "SURFACE": 0.15, "MC_VARIANCE": 1.20},
    "WTA": {"SKILL": 0.50, "FORM": 0.35, "SURFACE": 0.15, "MC_VARIANCE": 1.20}
}

CITY_TO_DB_STRING = {
    "Perth": "RAC Arena", "Sydney": "Ken Rosewall Arena",
    "Brisbane": "Pat Rafter Arena", "Adelaide": "Memorial Drive Tennis Centre",
    "Melbourne": "Rod Laver Arena"
}
COUNTRY_TO_CITY_MAP: Dict[str, str] = {}

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

def ensure_dict(data: Any) -> Dict:
    try:
        if isinstance(data, dict): return data
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict): return data[0]
        return {}
    except: return {}

def normalize_db_name(name: str) -> str:
    if not name: return ""
    n = name.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

# -----------------------------------------------------------------
# 🚀 SOTA FIX: THE BROTHER-RESOLUTION ENGINE (Exact Matching)
# -----------------------------------------------------------------
def parse_te_name(raw: str):
    """
    Extrahiert aus "Cerundolo F." -> Nachname: "cerundolo", Initiale: "f"
    """
    clean = re.sub(r'\s*\(\d+\)|\s*\(.*?\)', '', raw).strip()
    parts = clean.split()
    if not parts: return "", ""
    
    initial = ""
    last_name_parts = []
    
    for p in parts:
        if (len(p) <= 2 and p.endswith('.')) or (len(p) == 1 and p.isalpha()):
            if not initial:
                initial = p[0].lower()
        else:
            last_name_parts.append(p)
            
    last_name = " ".join(last_name_parts)
    
    # Fallback, falls der Name "Francisco Cerundolo" war (kein Punkt)
    if not initial and len(parts) > 1:
        initial = parts[0][0].lower()
        last_name = " ".join(parts[1:])
        
    return normalize_db_name(last_name), initial

def match_te_player(raw_name: str, db_players: List[Dict]) -> Any:
    """
    Findet den exakten Spieler und trennt Brüder anhand des Vornamens.
    Gibt "TIE_BREAKER" zurück, falls eine Zwilling/Bruder Situation besteht.
    """
    target_last, target_initial = parse_te_name(raw_name)
    if not target_last: return None
    
    best_match = None
    best_score = -1
    tie_flag = False 
    
    for p in db_players:
        db_last = normalize_db_name(p.get('last_name', ''))
        db_first = normalize_db_name(p.get('first_name', ''))
        
        score = 0
        
        # 1. Check Nachname
        if db_last == target_last or target_last in db_last or db_last in target_last:
            score += 50
            
            # 2. Check Vorname (Brüder trennen!)
            if target_initial and db_first:
                if db_first.startswith(target_initial):
                    score += 50 # Richtig!
                else:
                    score -= 100 # Falscher Bruder! -> Harter Abzug
            
            # 3. Score-Auswertung & Tie-Erkennung
            if score > 0:
                if score > best_score:
                    best_score = score
                    best_match = p
                    tie_flag = False # Klarer Favorit
                elif score == best_score:
                    tie_flag = True # 🚨 K. Pliskova Problem detektiert!
                
    # 🚨 SOTA FIX: Wenn Tie-Breaker zuschlägt, senden wir ein klares Abbruch-Signal!
    if tie_flag:
        return "TIE_BREAKER"
        
    return best_match

# =================================================================
# 3. CORE ENGINES (Form, Surface, Monte Carlo)
# =================================================================
class MomentumV2Engine:
    @staticmethod
    def calculate_rating(matches: List[Dict], player_name: str, max_matches: int = 15) -> Dict[str, Any]:
        if not matches: return {"score": 5.0, "text": "Neutral (No Data)", "history_summary": "", "color_hex": "#808080"}

        recent_matches = sorted(matches, key=lambda x: x.get('created_at', ''), reverse=True)[:max_matches]
        chrono_matches = recent_matches[::-1]

        rating = 5.5 
        momentum = 0.0
        history_log = []

        for idx, m in enumerate(chrono_matches):
            p_name_lower = player_name.lower()
            is_p1 = p_name_lower in m['player1_name'].lower()
            winner = m.get('actual_winner_name', "") or ""
            won = p_name_lower in winner.lower()

            odds = m.get('odds1', 1.50) if is_p1 else m.get('odds2', 1.50)
            if not odds or odds <= 1.0: odds = 1.50

            is_recent = idx >= (len(chrono_matches) - 5)
            weight = 1.5 if is_recent else 0.8
            impact = 0.0

            if won:
                if odds < 1.30: impact = 0.4      
                elif odds <= 2.00: impact = 0.8   
                else: impact = 1.8                
                score = str(m.get('score', ''))
                if score and "2-1" not in score and "1-2" not in score: impact += 0.3
                momentum += 0.2 
                history_log.append("W")
            else:
                if odds < 1.30: impact = -0.6     
                elif odds < 1.50: impact = -0.5
                elif odds <= 2.20: impact = -0.6  
                else: impact = -0.2                
                score = str(m.get('score', ''))
                if "2-1" in score or "1-2" in score: momentum = max(0.0, momentum - 0.1)
                else: momentum = 0.0 
                history_log.append("L")
            
            rating += (impact * weight)

        rating += momentum
        final_rating = max(1.0, min(10.0, rating))
        
        desc = "Average"
        if final_rating > 8.5: desc = "🔥 ELITE"
        elif final_rating > 7.0: desc = "📈 Strong"
        elif final_rating < 4.0: desc = "❄️ Cold"
        elif final_rating < 5.5: desc = "⚠️ Weak"
        
        color_hex = "#F0C808" 
        if final_rating >= 9.0: color_hex = "#FF00FF" 
        elif final_rating >= 8.0: color_hex = "#3366FF" 
        elif final_rating >= 7.0: color_hex = "#00B25B" 
        elif final_rating >= 6.0: color_hex = "#99CC33" 
        elif final_rating <= 4.0: color_hex = "#CC0000" 
        elif final_rating <= 5.5: color_hex = "#FF9933" 

        return {
            "score": round(final_rating, 2),
            "text": desc,
            "color_hex": color_hex,
            "history_summary": "".join(history_log[-5:])
        }

class SurfaceIntelligence:
    @staticmethod
    def normalize_surface_key(raw_surface: str) -> str:
        if not raw_surface: return "unknown"
        s = raw_surface.lower()
        if "grass" in s: return "grass"
        if "clay" in s or "sand" in s: return "clay"
        if "hard" in s or "carpet" in s or "acrylic" in s or "indoor" in s: return "hard"
        return "unknown"

    @staticmethod
    def get_matches_by_surface(all_matches: List[Dict], target_surface: str) -> List[Dict]:
        filtered = []
        target = SurfaceIntelligence.normalize_surface_key(target_surface)
        
        for m in all_matches:
            tour_name = str(m.get('tournament', '')).lower()
            ai_text = str(m.get('ai_analysis_text', '')).lower()
            found_surface = "unknown"
            
            match_hist = re.search(r'surface:\s*(hard|clay|grass)', ai_text)
            if match_hist: found_surface = match_hist.group(1)
            elif "hard court" in ai_text or "hard surface" in ai_text: found_surface = "hard"
            elif "red clay" in ai_text or "clay court" in ai_text: found_surface = "clay"
            elif "grass court" in ai_text: found_surface = "grass"
            elif "clay" in tour_name or "roland garros" in tour_name: found_surface = "clay"
            elif "grass" in tour_name or "wimbledon" in tour_name: found_surface = "grass"
            elif "hard" in tour_name or "us open" in tour_name or "australian open" in tour_name: found_surface = "hard"
            else:
                for db_key, db_surf in GLOBAL_SURFACE_MAP.items():
                    if db_key in tour_name or tour_name in db_key:
                        if len(db_key) > 3:
                            found_surface = db_surf
                            break
            
            if SurfaceIntelligence.normalize_surface_key(found_surface) == target:
                filtered.append(m)
        
        return filtered

    @staticmethod
    def compute_player_surface_profile(matches: List[Dict], player_name: str) -> Dict[str, Any]:
        profile = {}
        
        surfaces_data = {
            "hard": SurfaceIntelligence.get_matches_by_surface(matches, "hard"),
            "clay": SurfaceIntelligence.get_matches_by_surface(matches, "clay"),
            "grass": SurfaceIntelligence.get_matches_by_surface(matches, "grass")
        }
        
        for surf, surf_matches in surfaces_data.items():
            n_surf = len(surf_matches)
            
            if n_surf == 0:
                profile[surf] = {
                    "rating": 3.5, 
                    "color": "#808080", 
                    "matches_tracked": 0,
                    "text": "No Experience"
                }
                continue
                
            wins = 0
            for m in surf_matches:
                winner = m.get('actual_winner_name', "") or ""
                if player_name.lower() in winner.lower():
                    wins += 1
            win_rate = wins / n_surf
            
            vol_score = min(1.0, n_surf / 30.0) * 1.95
            win_score = win_rate * 4.55
            
            final_rating = 3.5 + vol_score + win_score
            final_rating = max(1.0, min(10.0, final_rating))
            
            desc = "Average"
            if final_rating >= 8.5: desc = "🔥 SPECIALIST"
            elif final_rating >= 7.0: desc = "📈 Strong"
            elif final_rating >= 5.5: desc = "Solid"
            elif final_rating >= 4.5: desc = "⚠️ Vulnerable"
            else: desc = "❄️ Weakness"
            
            color_hex = "#F0C808" 
            if final_rating >= 8.5: color_hex = "#FF00FF" 
            elif final_rating >= 7.5: color_hex = "#3366FF" 
            elif final_rating >= 6.5: color_hex = "#00B25B" 
            elif final_rating >= 5.5: color_hex = "#99CC33" 
            elif final_rating <= 4.5: color_hex = "#CC0000" 
            elif final_rating < 5.5: color_hex = "#FF9933" 

            profile[surf] = {
                "rating": round(final_rating, 2),
                "color": color_hex,
                "matches_tracked": n_surf,
                "text": desc
            }
            
        profile['_v95_mastery_applied'] = True
        return profile

def normal_cdf_prob(elo_diff: float, sigma: float = 280.0) -> float:
    z = elo_diff / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

# =================================================================
# 4. TE SCRAPING LOGIC (Background Mining WITH SOTA RESULTS)
# =================================================================
async def scrape_tennis_odds_for_date(browser: Browser, target_date):
    page = await browser.new_page()
    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}&t={int(time.time())}"
        log(f"📡 Scanning TE Background: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        return await page.content()
    except: return None
    finally: await page.close()

def parse_matches_locally_v5(html): 
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        pending_p1_raw = None; pending_p1_href = None; pending_time = "00:00"
        pending_p1_row = None
        
        i = 0
        while i < len(rows):
            row = rows[i]
            
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                pending_p1_raw = None
                i += 1; continue
            
            cols = row.find_all('td')
            if len(cols) < 2: i += 1; continue
            
            first_cell = row.find('td', class_='first')
            if first_cell and ('time' in first_cell.get('class', []) or 't-name' in first_cell.get('class', [])):
                tm = re.search(r'(\d{1,2}:\d{2})', first_cell.get_text(strip=True))
                if tm: pending_time = tm.group(1).zfill(5)
            
            p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
            if not p_cell: i += 1; continue
            p_raw = clean_player_name(p_cell.get_text(strip=True))
            p_href = p_cell.find('a')['href'] if p_cell.find('a') else ""
            
            raw_odds = []
            for c in row.find_all('td', class_=re.compile(r'course')):
                try:
                    val = float(c.get_text(strip=True))
                    if 1.01 <= val <= 100.0: raw_odds.append(val)
                except: pass

            if pending_p1_raw:
                p2_raw = p_raw; p2_href = p_href
                if '/' in pending_p1_raw or '/' in p2_raw: 
                    pending_p1_raw = None; i += 1; continue
                
                prev_row = pending_p1_row
                prev_odds = []
                for c in prev_row.find_all('td', class_=re.compile(r'course')):
                    try:
                        val = float(c.get_text(strip=True))
                        if 1.01 <= val <= 100.0: prev_odds.append(val)
                    except: pass
                
                all_odds = prev_odds + raw_odds
                
                # --- SOTA: RESULT (SCORE & WINNER) EXTRACTION ---
                def extract_row_data(r_row):
                    cells = r_row.find_all('td')
                    p_idx = -1
                    for idx, c in enumerate(cells):
                        if c.find('a') and 'time' not in c.get('class', []):
                            p_idx = idx
                            break
                    
                    if p_idx != -1 and p_idx + 1 < len(cells):
                        sets_cell = cells[p_idx + 1]
                        sets_val = sets_cell.get_text(strip=True)
                        
                        if sets_val.isdigit():
                            scores = []
                            for k in range(1, 6):
                                if p_idx + 1 + k >= len(cells): break
                                sc_cell = cells[p_idx + 1 + k]
                                
                                raw_txt = ""
                                for child in sc_cell.children:
                                    if child.name == 'sup': continue
                                    raw_txt += str(child).strip() if isinstance(child, str) else child.get_text(strip=True)
                                
                                raw_txt = re.sub(r'<[^>]+>', '', raw_txt).strip()
                                
                                if raw_txt.isdigit():
                                    scores.append(raw_txt)
                                else:
                                    break
                            return int(sets_val), scores
                    return -1, []

                s1, scores1 = extract_row_data(prev_row)
                s2, scores2 = extract_row_data(row)
                
                winner_found = None
                final_score = ""
                
                is_ret = "ret." in prev_row.get_text().lower() or "ret." in row.get_text().lower() or "w.o." in prev_row.get_text().lower() or "w.o." in row.get_text().lower()
                
                if s1 != -1 and s2 != -1:
                    if s1 > s2: winner_found = pending_p1_raw
                    elif s2 > s1: winner_found = p2_raw
                    
                    score_parts = []
                    min_len = min(len(scores1), len(scores2))
                    for k in range(min_len):
                        score_parts.append(f"{scores1[k]}-{scores2[k]}")
                    
                    if score_parts:
                        final_score = " ".join(score_parts)
                        if is_ret: final_score += " ret."

                final_o1 = all_odds[0] if len(all_odds) >= 2 else 0
                final_o2 = all_odds[1] if len(all_odds) >= 2 else 0
                
                # Wir nehmen das Match auf, wenn Quoten ODER ein Ergebnis existieren
                if (final_o1 > 0 and final_o2 > 0) or winner_found:
                    found.append({
                        "p1_raw": pending_p1_raw, "p2_raw": p2_raw, 
                        "tour": clean_tournament_name(current_tour), 
                        "time": pending_time, "odds1": final_o1, "odds2": final_o2,
                        "p1_href": pending_p1_href, "p2_href": p2_href,
                        "actual_winner": winner_found,
                        "score": final_score
                    })
                pending_p1_raw = None
            else:
                if first_cell and first_cell.get('rowspan') == '2': 
                    pending_p1_raw = p_raw; pending_p1_href = p_href; pending_p1_row = row
                else: 
                    pending_p1_raw = p_raw; pending_p1_href = p_href; pending_p1_row = row
            i += 1
    return found

# =================================================================
# 5. PIPELINE EXECUTION (THE SMART SHADOW PROCESS)
# =================================================================
async def run_pipeline():
    log(f"🚀 Neural Scout SMART SHADOW MINER V150.2 Starting...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            # Load DB state
            players = supabase.table("players").select("id, last_name, first_name").execute().data or []
            
            # Load Dynamic Weights
            weights_res = supabase.table("ai_system_weights").select("*").execute()
            if weights_res.data:
                for w in weights_res.data:
                    tour = w.get("tour", "ATP")
                    DYNAMIC_WEIGHTS[tour] = {
                        "SKILL": to_float(w.get("weight_skill"), 0.50),
                        "FORM": to_float(w.get("weight_form"), 0.35),
                        "SURFACE": to_float(w.get("weight_surface"), 0.15),
                        "MC_VARIANCE": to_float(w.get("mc_variance"), 1.20)
                    }

            for day_offset in range(-1, 2): 
                target_date = datetime.now() + timedelta(days=day_offset)
                html = await scrape_tennis_odds_for_date(browser, target_date)
                if not html: continue
                
                matches = parse_matches_locally_v5(html)
                log(f"🔍 TE Miner gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
                
                for m in matches:
                    await asyncio.sleep(0.1)
                    
                    # ---------------------------------------------------------
                    # 🚀 SOTA FIX 1: THE BROTHER-RESOLUTION ENGINE
                    # ---------------------------------------------------------
                    p1_db = match_te_player(m['p1_raw'], players)
                    p2_db = match_te_player(m['p2_raw'], players)

                    # 🚨 SOTA FIX: Hard Abort bei Zwillingen/Brüdern!
                    if p1_db == "TIE_BREAKER" or p2_db == "TIE_BREAKER":
                        log(f"🚨 Tie-Breaker Alarm! Überspringe Match ({m['p1_raw']} vs {m['p2_raw']}) um Ghost-Duplikate zu verhindern.")
                        continue

                    if not p1_db and not p2_db:
                        # BEIDE Spieler sind unbekannt -> Ignorieren! (Kein Müll in der DB)
                        continue 

                    # Wir nutzen den exakten DB-Namen, wenn gefunden!
                    full_n1 = p1_db['last_name'] if p1_db and isinstance(p1_db, dict) else clean_player_name(m['p1_raw'])
                    full_n2 = p2_db['last_name'] if p2_db and isinstance(p2_db, dict) else clean_player_name(m['p2_raw'])

                    if not full_n1 or not full_n2 or full_n1 == full_n2: continue

                    # ---------------------------------------------------------
                    # 🚀 SOTA FIX 2: DUPLICATE & RESULT-UPDATE LOGIC
                    # ---------------------------------------------------------
                    time_window = (datetime.now() - timedelta(days=4)).isoformat()
                    res1 = supabase.table("market_odds").select("id, actual_winner_name").eq("player1_name", full_n1).eq("player2_name", full_n2).gte("created_at", time_window).execute()
                    res2 = supabase.table("market_odds").select("id, actual_winner_name").eq("player1_name", full_n2).eq("player2_name", full_n1).gte("created_at", time_window).execute()
                    
                    existing_match = None
                    is_reversed = False
                    
                    if res1.data: 
                        existing_match = res1.data[0]
                    elif res2.data: 
                        existing_match = res2.data[0]
                        is_reversed = True

                    if existing_match:
                        # Match existiert bereits (Kein Duplikat anlegen!)
                        # ABER: Wenn TE ein Ergebnis hat, das in unserer DB noch fehlt, updaten wir es!
                        if m['actual_winner'] and not existing_match.get('actual_winner_name'):
                            
                            act_winner = full_n1 if m['actual_winner'] == m['p1_raw'] else full_n2
                            score_to_update = m['score']
                            
                            # Wenn die Reihenfolge in der DB umgekehrt ist, drehen wir den Score um
                            if is_reversed and m['score']:
                                parts = m['score'].split()
                                flipped_parts = []
                                for part in parts:
                                    if '-' in part:
                                        try:
                                            l, r = part.split('-')
                                            if 'ret' in r:
                                               r = r.replace('ret.', '').strip()
                                               flipped_parts.append(f"{r}-{l} ret.")
                                            else:
                                               flipped_parts.append(f"{r}-{l}")
                                        except:
                                            flipped_parts.append(part)
                                score_to_update = " ".join(flipped_parts)

                            log(f"🔄 Result-Lücke gefüllt: {full_n1} vs {full_n2} -> Winner: {act_winner}")
                            try:
                                supabase.table("market_odds").update({
                                    "actual_winner_name": act_winner,
                                    "score": score_to_update
                                }).eq("id", existing_match['id']).execute()
                            except Exception as e:
                                pass
                        
                        # Nach Update (oder wenn kein Update nötig) -> Gehe zum nächsten Match
                        continue

                    # ---------------------------------------------------------
                    # 3. 🤫 SILENT INSERTION (Für komplett neue Lücken-Matches)
                    # ---------------------------------------------------------
                    prob_market = 0.5
                    if m['odds1'] > 1 and m['odds2'] > 1:
                        inv1 = 1/m['odds1']; inv2 = 1/m['odds2']
                        prob_market = inv1 / (inv1 + inv2)
                    
                    fair1 = round(1/prob_market, 2) if prob_market > 0.01 else 99
                    fair2 = round(1/(1-prob_market), 2) if prob_market < 0.99 else 99

                    final_time_str = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                    
                    act_winner = None
                    if m['actual_winner']:
                        act_winner = full_n1 if m['actual_winner'] == m['p1_raw'] else full_n2

                    silent_data = {
                        "player1_name": full_n1, 
                        "player2_name": full_n2, 
                        "tournament": m['tour'],
                        "odds1": m['odds1'] if m['odds1'] > 0 else None, 
                        "odds2": m['odds2'] if m['odds2'] > 0 else None, 
                        "ai_fair_odds1": fair1, 
                        "ai_fair_odds2": fair2,
                        "actual_winner_name": act_winner,
                        "score": m['score'],
                        # ❌ HIER IST DER MAGISCHE SCHALTER:
                        "is_visible_in_scanner": False, 
                        "ai_analysis_text": "[BACKGROUND DATA] Captured for Elo & Form calculations only. Not eligible for Value Scanner.",
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "match_time": final_time_str
                    }
                    
                    try:
                        supabase.table("market_odds").insert(silent_data).execute()
                        log(f"💾 SILENT SAVE: {full_n1} vs {full_n2} (Mit Resultat? {bool(act_winner)})")
                    except Exception as ins_e:
                        pass

        finally: await browser.close()
    log("🏁 Smart Shadow Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
