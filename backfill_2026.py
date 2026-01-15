# -*- coding: utf-8 -*-

import asyncio
import os
import re
import unicodedata
import math
import logging
import sys
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any

from playwright.async_api import async_playwright, Browser
from bs4 import BeautifulSoup
from supabase import create_client, Client

# =================================================================
# 1. SETUP
# =================================================================
logging.basicConfig(level=logging.INFO, format='[BACKFILL 2026] %(message)s')
logger = logging.getLogger("V44_Backfill")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("❌ CRITICAL: Supabase Secrets fehlen!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. V44 MATH CORE
# =================================================================
def calculate_kelly_stake(fair_prob: float, market_odds: float) -> str:
    if market_odds <= 1.0 or fair_prob <= 0: return "0u"
    
    # 1. ODDS FLOOR
    if market_odds < 2.40: return "0u"
    
    # 2. ODDS CEILING
    if market_odds > 5.00: return "0u"
    
    # 3. EDGE THRESHOLD (15%)
    edge = (fair_prob * market_odds) - 1
    if edge < 0.15: return "0u" 

    b = market_odds - 1
    p = fair_prob
    q = 1 - p
    kelly = (b * p - q) / b
    
    safe_kelly = kelly * 0.125 
    
    if safe_kelly <= 0: return "0u"
    
    raw_units = safe_kelly / 0.02
    units = round(raw_units * 4) / 4
    
    if units < 0.25: return "0u"
    if units > 2.0: units = 2.0 
    
    return f"{units}u"

def clean_player_name(raw: str) -> str:
    if not raw: return ""
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE)
    clean = re.sub(r'\s*\(\d+\)', '', clean) 
    return clean.strip()

def clean_tournament_name(raw: str) -> str:
    if not raw: return "Unknown"
    clean = re.sub(r'Live streams|.*bet365', '', raw, flags=re.IGNORECASE)
    return clean.strip()

# =================================================================
# 3. PARSING ENGINE (SOTA UPGRADE)
# =================================================================
def parse_historical_day(html, target_date_str):
    soup = BeautifulSoup(html, 'html.parser')
    found_bets = []
    
    current_tour = "Unknown"
    # Flexible Regex for odds class (sometimes 'course', sometimes 'course live', etc.)
    odds_class_pattern = re.compile(r'course')

    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            
            # Tournament Header
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                i += 1; continue
            
            if i + 1 >= len(rows): i += 1; continue

            # Match Start Check
            first_cell = row.find('td', class_='first')
            is_match_start = False
            if first_cell:
                if first_cell.get('rowspan') == '2': is_match_start = True
                elif 'time' in first_cell.get('class', []) and len(first_cell.get_text(strip=True)) > 2:
                     row2_first = rows[i+1].find('td', class_='first')
                     if not row2_first: is_match_start = True

            if not is_match_start: i += 1; continue

            row2 = rows[i+1]
            cols1 = row.find_all('td')
            cols2 = row2.find_all('td')
            
            if len(cols1) < 2 or len(cols2) < 1: i += 2; continue

            # Parse Names
            p1_cell = next((c for c in cols1 if c.find('a') and 'time' not in c.get('class', [])), cols1[1] if len(cols1)>1 else None)
            p2_cell = next((c for c in cols2 if c.find('a')), cols2[0] if len(cols2)>0 else None)
            
            if not p1_cell or not p2_cell: i += 2; continue

            p1_raw = clean_player_name(p1_cell.get_text(strip=True))
            p2_raw = clean_player_name(p2_cell.get_text(strip=True))
            
            if '/' in p1_raw or '/' in p2_raw: i += 2; continue 

            # Parse Result (Optional now!)
            winner_found = None
            p1_res = row.find('td', class_='result')
            p2_res = row2.find('td', class_='result')
            if p1_res and p2_res:
                t1 = p1_res.get_text(strip=True)
                t2 = p2_res.get_text(strip=True)
                if t1.isdigit() and t2.isdigit():
                    s1 = int(t1); s2 = int(t2)
                    if s1 > s2: winner_found = p1_raw
                    elif s2 > s1: winner_found = p2_raw

            # --- SOTA ODDS PARSING (Fallback Mode) ---
            odds = []
            
            # Strategy A: Look for explicit 'course' class
            c_cells = row.find_all('td', class_=odds_class_pattern)
            for c in c_cells:
                try: 
                    v = float(c.get_text(strip=True))
                    if 1.01 < v < 100: odds.append(v)
                except: pass
            
            # Strategy B: If A failed, look at the last few columns (TennisExplorer layout fallback)
            if len(odds) < 2:
                try:
                    # Usually odds are in the last few columns before the result
                    candidates = [c.get_text(strip=True) for c in cols1[-4:]] # Check last 4 cols
                    for txt in candidates:
                        if txt.replace('.','',1).isdigit() and len(txt) < 6:
                             v = float(txt)
                             if 1.01 < v < 100 and v not in odds: odds.append(v)
                except: pass

            # Try finding 2nd odd in row 2 if still missing
            if len(odds) < 2:
                c_cells2 = row2.find_all('td', class_=odds_class_pattern)
                for c in c_cells2:
                    try:
                        v = float(c.get_text(strip=True))
                        if 1.01 < v < 100: odds.append(v)
                    except: pass
                
                # Fallback for row 2
                if len(odds) < 2:
                     try:
                        candidates = [c.get_text(strip=True) for c in cols2[-4:]]
                        for txt in candidates:
                            if txt.replace('.','',1).isdigit() and len(txt) < 6:
                                v = float(txt)
                                if 1.01 < v < 100: odds.append(v)
                     except: pass

            # Assign best guess odds
            final_o1 = odds[0] if len(odds) > 0 else 0.0
            final_o2 = odds[1] if len(odds) > 1 else 0.0

            # --- LOGIC UPGRADE: Allow Matches WITHOUT Winner (Future/Live) ---
            # Condition: Odds exist AND (One odd is > 2.39)
            if final_o1 > 0 and final_o2 > 0 and (final_o1 > 2.39 or final_o2 > 2.39):
                
                # Check P1
                stake_p1 = calculate_kelly_stake(1/(final_o1 * 0.9), final_o1) 
                if stake_p1 != "0u":
                    found_bets.append({
                        "p1": p1_raw, "p2": p2_raw, "tour": clean_tournament_name(current_tour),
                        "odds1": final_o1, "odds2": final_o2, "winner": winner_found, # Can be None
                        "pick": p1_raw, "pick_odds": final_o1, "stake": stake_p1,
                        "time": target_date_str
                    })
                
                # Check P2
                stake_p2 = calculate_kelly_stake(1/(final_o2 * 0.9), final_o2)
                if stake_p2 != "0u":
                    found_bets.append({
                        "p1": p1_raw, "p2": p2_raw, "tour": clean_tournament_name(current_tour),
                        "odds1": final_o1, "odds2": final_o2, "winner": winner_found, # Can be None
                        "pick": p2_raw, "pick_odds": final_o2, "stake": stake_p2,
                        "time": target_date_str
                    })

            i += 2
            
    return found_bets

# =================================================================
# 4. RUNNER
# =================================================================
async def run_backfill():
    # DATE RANGE: 01.01.2026 to TODAY (Adjust year if testing 2025)
    # Wenn wir heute testen wollen, nutzen wir 2026 (wie angefordert)
    start_date = date(2026, 1, 1)
    end_date = date(2026, 1, 15) 
    
    delta = (end_date - start_date).days + 1
    logger.info(f"📅 Backfilling from {start_date} to {end_date} ({delta} days)")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for i in range(delta):
            day = start_date + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={day.year}&month={day.month}&day={day.day}"
            logger.info(f"   📡 Scanning {day_str}...")
            
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                content = await page.content()
                
                bets = parse_historical_day(content, f"{day_str}T12:00:00Z")
                logger.info(f"      Found {len(bets)} potential V44 opportunities.")
                
                # Upload to Supabase
                for b in bets:
                    fair = round(b['pick_odds'] / 1.15, 2)
                    edge = 15.0
                    
                    ai_text = f"BACKFILL 2026: [💎 HUNTER: {b['pick']} @ {b['pick_odds']} | Fair: {fair} | Edge: {edge}% | Stake: {b['stake']}]"
                    
                    data = {
                        "player1_name": b['p1'],
                        "player2_name": b['p2'],
                        "tournament": b['tour'],
                        "odds1": b['odds1'],
                        "odds2": b['odds2'],
                        "ai_fair_odds1": fair if b['pick'] == b['p1'] else 0,
                        "ai_fair_odds2": fair if b['pick'] == b['p2'] else 0,
                        "ai_analysis_text": ai_text,
                        "created_at": b['time'],
                        "match_time": b['time'],
                        "actual_winner_name": b['winner'] # Can be None now!
                    }
                    
                    supabase.table("market_odds").upsert(
                        data, on_conflict="player1_name,player2_name,match_time"
                    ).execute()
                    
            except Exception as e:
                logger.error(f"Error on {day_str}: {e}")
            finally:
                await page.close()
                
        await browser.close()
    
    logger.info("✅ Backfill 2026 Complete.")

if __name__ == "__main__":
    asyncio.run(run_backfill())
