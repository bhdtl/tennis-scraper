# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
import math
import random
import logging
import sys
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

log("🔮 Initializing The Tournament Oracle (Zero-Cost Math Engine)...")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets missing!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. HELPER FUNCTIONS (CLEANING & MATCHING)
# =================================================================
def clean_name(raw: str) -> str:
    if not raw: return ""
    clean = re.sub(r'\(.*?\)', '', raw) # Entfernt (Q), (WC), etc.
    return clean.strip().lower()

def match_tournament(scraped_tour_name: str, db_tournaments: List[Dict]) -> Optional[Dict]:
    """Findet ein Turnier nur, wenn es in der DB existiert."""
    s_name = scraped_tour_name.lower()
    # Säubere typische Scraping-Anhänge
    s_name = re.sub(r'\b(atp|wta|challenger|singles|doubles|men|women|-)\b', '', s_name).strip()
    
    for db_tour in db_tournaments:
        db_name = db_tour['name'].lower()
        # Check ob wesentlicher Teil des Namens übereinstimmt
        if db_name in s_name or s_name in db_name:
            return db_tour
    return None

def match_player(scraped_name: str, db_players: List[Dict]) -> Optional[Dict]:
    """Findet einen Spieler in der DB anhand des Nachnamens."""
    s_name = clean_name(scraped_name)
    parts = s_name.split()
    if not parts: return None
    
    # Letztes Wort ist meistens der Nachname
    target_last = parts[-1].replace('-', ' ').replace("'", "")
    
    for p in db_players:
        db_last = p.get('last_name', '').lower().replace('-', ' ').replace("'", "")
        if db_last == target_last:
            return p
        # Check für Doppelnamen etc.
        if len(target_last) > 4 and (target_last in db_last or db_last in target_last):
            return p
    return None

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
# 3. MONTE CARLO MATH ENGINE (Zero Cost)
# =================================================================
def run_monte_carlo(skillA: float, formA: float, surfaceA: float, 
                    skillB: float, formB: float, surfaceB: float, 
                    iterations: int = 1000) -> tuple:
    """Die exakte Mathe-Logik aus deiner Edge-Function, portiert auf Python."""
    
    # Weights
    w_skill = 0.50
    w_form = 0.35
    w_surf = 0.15
    variance = 1.2
    
    baseA = (skillA / 10) * w_skill + formA * w_form + surfaceA * w_surf
    baseB = (skillB / 10) * w_skill + formB * w_form + surfaceB * w_surf
    
    # "FISH OUT OF WATER" PENALTY 
    if surfaceA < 4.5: baseA -= (4.5 - surfaceA) * 1.5
    if surfaceB < 4.5: baseB -= (4.5 - surfaceB) * 1.5
    
    winsA = 0
    winsB = 0
    
    for _ in range(iterations):
        # random.gauss(0, 1) generiert normale Standardverteilung (wie z0/z1 in JS)
        simA = baseA + (random.gauss(0, 1) * variance)
        simB = baseB + (random.gauss(0, 1) * variance)
        
        if simA > simB: winsA += 1
        else: winsB += 1
        
    probA = (winsA / iterations) * 100
    probB = (winsB / iterations) * 100
    
    return probA, probB

# =================================================================
# 4. SCRAPING & PIPELINE
# =================================================================
async def scrape_draws_for_date(browser: Browser, target_date: datetime, db_data: Dict):
    page = await browser.new_page()
    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        log(f"📡 Scanning draws for: {target_date.strftime('%Y-%m-%d')}")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        db_players = db_data['players']
        db_tournaments = db_data['tournaments']
        db_skills = db_data['skills']
        
        inserted_count = 0
        
        for table in soup.find_all("table", class_="result"):
            rows = table.find_all("tr")
            current_tour_raw = "Unknown"
            matched_db_tour = None
            
            pending_p1_raw = None
            
            for row in rows:
                # 1. Is it a Tournament Header?
                if "head" in row.get("class", []): 
                    current_tour_raw = row.get_text(strip=True)
                    matched_db_tour = match_tournament(current_tour_raw, db_tournaments)
                    pending_p1_raw = None
                    continue
                
                # Wenn das Turnier NICHT in der Datenbank ist -> Überspringen!
                if not matched_db_tour:
                    continue
                    
                cols = row.find_all('td')
                if len(cols) < 2: continue
                
                # Check for Match Status (only upcoming matches, no results)
                # If there are scores, it's already played. We skip it.
                row_text = row.get_text(strip=True)
                if re.search(r'\d-\d', row_text) and not ":" in row_text:
                    continue
                
                # Extrahiere Spieler
                p_cell = next((c for c in cols if c.find('a') and 'time' not in c.get('class', [])), None)
                if not p_cell: continue
                p_raw = p_cell.get_text(strip=True)
                
                first_cell = row.find('td', class_='first')
                
                if pending_p1_raw:
                    p2_raw = p_raw
                    
                    # 2. Beide Spieler gegen DB abgleichen
                    p1_obj = match_player(pending_p1_raw, db_players)
                    p2_obj = match_player(p2_raw, db_players)
                    
                    # 3. NUR WENN BEIDE IN DER DB SIND -> Analysieren!
                    if p1_obj and p2_obj and p1_obj['id'] != p2_obj['id']:
                        n1 = p1_obj['last_name']
                        n2 = p2_obj['last_name']
                        
                        # --- DATA EXTRACTION ---
                        # Skills
                        s1 = db_skills.get(p1_obj['id'], {})
                        s2 = db_skills.get(p2_obj['id'], {})
                        skillA_overall = float(s1.get('overall_rating', 50))
                        skillB_overall = float(s2.get('overall_rating', 50))
                        
                        # Form
                        formA = extract_rating(p1_obj.get('form_rating'))
                        formB = extract_rating(p2_obj.get('form_rating'))
                        
                        # Surface
                        tour_surface = matched_db_tour['surface']
                        surfA = get_surface_rating(p1_obj.get('surface_ratings'), tour_surface)
                        surfB = get_surface_rating(p2_obj.get('surface_ratings'), tour_surface)
                        
                        # --- MONTE CARLO ENGINE ---
                        probA, probB = run_monte_carlo(skillA_overall, formA, surfA, skillB_overall, formB, surfB)
                        
                        predicted_winner = n1 if probA > probB else n2
                        win_prob = max(probA, probB)
                        
                        # --- SAVE TO SUPABASE ---
                        payload = {
                            "tournament_name": matched_db_tour['name'],
                            "match_date": target_date.strftime('%Y-%m-%d'),
                            "player_a_name": n1,
                            "player_b_name": n2,
                            "predicted_winner": predicted_winner,
                            "win_probability": round(win_prob, 1),
                            "surface": tour_surface,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Upsert (verhindert doppelte Einträge durch den UNIQUE constraint)
                        res = supabase.table("tournament_oracle_draws").upsert(
                            payload, on_conflict="tournament_name,player_a_name,player_b_name"
                        ).execute()
                        
                        inserted_count += 1
                        log(f"   🎯 Pred: {n1} vs {n2} -> {predicted_winner} ({round(win_prob,1)}%) @ {matched_db_tour['name']}")
                    
                    pending_p1_raw = None
                else:
                    if first_cell and first_cell.get('rowspan') == '2': 
                        pending_p1_raw = p_raw
                    else: 
                        pending_p1_raw = p_raw

        log(f"✅ Generated {inserted_count} predictions for {target_date.strftime('%Y-%m-%d')}.")
    except Exception as e: 
        log(f"⚠️ Scraping Error: {e}")
    finally: 
        await page.close()

async def run_pipeline():
    try:
        # Load Reference Data from DB
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
            
            # Scrape Heute + Morgen + Übermorgen (Die Draws)
            for day_offset in range(0, 3): 
                target_date = datetime.now() + timedelta(days=day_offset)
                await scrape_draws_for_date(browser, target_date, db_data)
                await asyncio.sleep(2) # Höflichkeits-Pause für TennisExplorer
                
            await browser.close()
        log("🏁 Oracle Cycle Finished.")
        
    except Exception as e:
        log(f"❌ Pipeline crashed: {e}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
