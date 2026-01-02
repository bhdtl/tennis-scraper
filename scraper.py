# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
import sys
import numpy as np 
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION
# =================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("🔌 Initialisiere Neural Scout (V83.1 - The Score Master)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.5-pro' 
ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# HELPER FUNCTIONS
# =================================================================
def to_float(val, default=50.0):
    try: return float(val)
    except: return default

def normalize_text(text): 
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn') if text else ""

def clean_player_name(raw): 
    clean = re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365|\(.\)|ret\.', '', raw, flags=re.IGNORECASE)
    return clean.replace('|', '').strip()

def get_last_name(full_name):
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    return clean.split()[-1].lower() if clean else ""

# =================================================================
# ODDS & PHYSICS ENGINE (1:1 V81.0)
# =================================================================
async def call_gemini(prompt):
    await asyncio.sleep(0.5) 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            return response.json()['candidates'][0]['content']['parts'][0]['text'] if response.status_code == 200 else None
        except: return None

def sigmoid(x, k=1.0):
    return 1 / (1 + math.exp(-k * x))

def get_dynamic_court_weights(bsi, surface):
    bsi = float(bsi)
    w = {'serve': 1.0, 'power': 1.0, 'rally': 1.0, 'movement': 1.0, 'mental': 0.8, 'volley': 0.5}
    if bsi >= 7.0:
        f = (bsi - 5.0) * 0.35 
        w.update({'serve': 1.0+f*1.5, 'power': 1.0+f*1.2, 'volley': 0.5+f*1.0, 'rally': 1.0-f*0.5, 'movement': 1.0-f*0.3})
    elif bsi <= 4.0:
        f = (5.0 - bsi) * 0.4
        w.update({'serve': 1.0-f*0.8, 'power': 1.0-f*0.5, 'rally': 1.0+f*1.2, 'movement': 1.0+f*1.5, 'volley': 0.5-f*0.5})
    return w

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    n1, n2 = p1_name.lower().split()[-1], p2_name.lower().split()[-1]
    weights = get_dynamic_court_weights(to_float(bsi, 5.0), surface)
    
    def score(s): 
        if not s: return 50.0
        return (s.get('serve',50)*weights['serve'] + s.get('forehand',50)*weights['rally'] + s.get('speed',50)*weights['movement'] + s.get('mental',50)*weights['mental'])
    
    phys_prob = sigmoid((score(s1) - score(s2)) / 12.0)
    
    # ELO
    e1, e2 = 1500.0, 1500.0
    k = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    for n, d in ELO_CACHE.get("ATP", {}).items():
        if n1 in n: e1 = d.get(k, 1500)
        if n2 in n: e2 = d.get(k, 1500)
    elo_prob = 1 / (1 + 10 ** ((e2 - e1) / 400))

    # Market
    m_prob = 0.5
    if market_odds1 > 1 and market_odds2 > 1:
        m_prob = (1/market_odds1) / ((1/market_odds1) + (1/market_odds2))

    # AI
    ai_prob = 0.5 + (to_float(ai_meta.get('p1_tactical_score', 5)) - to_float(ai_meta.get('p2_tactical_score', 5))) * 0.05

    final = (m_prob * 0.35) + (elo_prob * 0.20) + (phys_prob * 0.30) + (ai_prob * 0.15)
    
    if final > 0.5: final -= (final - 0.5) * 0.05
    else: final += (0.5 - final) * 0.05
    return final

# =================================================================
# RESULT LOGIC (IMPORTED FROM OLD CODE)
# =================================================================
def extract_scores_aggressive(columns):
    """Extrahiert Tennis-Scores aus Tabellenzellen (Alter Code Logic)."""
    scores = []
    for col in columns:
        txt = col.get_text(strip=True)
        if len(txt) > 4: continue 
        if '(' in txt: txt = txt.split('(')[0]
        if txt.isdigit() and len(txt) == 1 and int(txt) <= 7:
            scores.append(int(txt))
    return scores

def determine_winner_from_scores(p1_name, p2_name, row1, row2):
    """Analysiert Scores, um den Gewinner zu finden."""
    try:
        cols1 = row1.find_all('td')
        cols2 = row2.find_all('td') if row2 else []
        
        s1 = extract_scores_aggressive(cols1)
        s2 = extract_scores_aggressive(cols2)
        
        if not s1 or not s2: return None

        p1_sets = 0
        p2_sets = 0
        for k in range(min(len(s1), len(s2))):
            if s1[k] > s2[k]: p1_sets += 1
            elif s2[k] > s1[k]: p2_sets += 1
        
        # Check Retirement
        row_text = row1.get_text(strip=True).lower() + (row2.get_text(strip=True).lower() if row2 else "")
        is_ret = "ret." in row_text or "w.o." in row_text

        if (p1_sets >= 2 and p1_sets > p2_sets) or (is_ret and p1_sets > p2_sets):
            return p1_name
        elif (p2_sets >= 2 and p2_sets > p1_sets) or (is_ret and p2_sets > p1_sets):
            return p2_name
    except: pass
    return None

# =================================================================
# SCRAPING ENGINE (V83.1 - HYBRID)
# =================================================================
async def fetch_elo_ratings():
    log("📊 Lade ELO...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto("https://tennisabstract.com/reports/atp_elo_ratings.html", timeout=60000)
            soup = BeautifulSoup(await page.content(), 'html.parser')
            rows = soup.find('table', {'id': 'reportable'}).find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 4:
                    ELO_CACHE["ATP"][normalize_text(cols[0].get_text(strip=True)).lower()] = {
                        'Hard': to_float(cols[3].get_text()), 'Clay': to_float(cols[4].get_text()), 'Grass': to_float(cols[5].get_text())
                    }
        except: pass
        await browser.close()

async def scrape_single_date(browser, target_date):
    # Robust Context
    context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media", "font"] else route.continue_())
    
    page = await context.new_page()
    try:
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        try: await page.wait_for_selector(".result", timeout=5000)
        except: pass
        
        content = await page.content()
        await context.close()
        return (target_date, content)
    except:
        await context.close()
        return (target_date, None)

def parse_matches_locally(html, p_names):
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    target_players = set(p.lower() for p in p_names)
    
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tour = "Unknown"
        
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                continue
            
            p1_col = row.find('td', class_='t-name')
            if not p1_col: continue
            if i + 1 >= len(rows): continue
            
            row2 = rows[i+1]
            p2_col = row2.find('td', class_='t-name')
            if not p2_col: continue

            p1_name = clean_player_name(p1_col.get_text(strip=True))
            p2_name = clean_player_name(p2_col.get_text(strip=True))

            if not (any(tp in p1_name.lower() for tp in target_players) and any(tp in p2_name.lower() for tp in target_players)):
                continue

            # --- RESULT DETECTION (THE FIX) ---
            # Wir nutzen die alte Score-Logic HIER, direkt beim Parsen.
            winner_detected = determine_winner_from_scores(p1_name, p2_name, row, row2)

            # --- ODDS EXTRACTION ---
            o1, o2 = 0.0, 0.0
            c1 = row.find('td', class_='course')
            c2 = row2.find('td', class_='course')
            
            if c1: 
                try: o1 = float(c1.get_text(strip=True))
                except: pass
            if c2: 
                try: o2 = float(c2.get_text(strip=True))
                except: pass

            time_col = row.find('td', class_='first')
            m_time = time_col.get_text(strip=True) if time_col and 'time' in time_col.get('class', []) else "00:00"

            if (o1 > 1.0 and o2 > 1.0) or winner_detected:
                found.append({
                    "p1": p1_name, "p2": p2_name, "tour": current_tour, 
                    "time": m_time, "odds1": o1, "odds2": o2,
                    "winner": winner_detected
                })
    return found

# =================================================================
# MAIN PIPELINE
# =================================================================
async def get_db_data():
    players = supabase.table("players").select("*").execute().data
    skills = supabase.table("player_skills").select("*").execute().data
    tournaments = supabase.table("tournaments").select("*").execute().data
    reports = supabase.table("scouting_reports").select("*").execute().data
    c_skills = {s['player_id']: s for s in skills}
    return players, c_skills, reports, tournaments

async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    if scraped_name in TOURNAMENT_LOC_CACHE: return TOURNAMENT_LOC_CACHE[scraped_name]
    prompt = f"TASK: Locate Match {p1} vs {p2} | SOURCE: '{scraped_name}' JSON: {{ \"city\": \"City\", \"surface_guessed\": \"Hard/Clay\", \"is_indoor\": bool }}"
    res = await call_gemini(prompt)
    if not res: return None
    try: 
        data = json.loads(res.replace("```json", "").replace("```", "").strip())
        TOURNAMENT_LOC_CACHE[scraped_name] = data
        return data
    except: return None

async def find_best_court_match_smart(tour, db_tours, p1, p2):
    s_low = tour.lower().strip()
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

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    prompt = f"ROLE: Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}. CTX: {surface} (BSI {bsi}). JSON: {{ \"p1_tactical_score\": 7, \"p2_tactical_score\": 5, \"ai_text\": \"...\" }}"
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5}
    if not res: return d
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return d

async def run_pipeline():
    log(f"🚀 Neural Scout v83.1 (The Score Master) Starting...")
    await fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    p_names = [p['last_name'] for p in players]
    
    # 1. PARALLEL SCRAPING
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--disable-gpu', '--no-sandbox'])
        tasks = [scrape_single_date(browser, current_date + timedelta(days=i)) for i in range(14)]
        results = await asyncio.gather(*tasks)
        await browser.close()

    # 2. PROCESSING
    for date, html in results:
        if not html: continue
        matches = parse_matches_locally(html, p_names)
        if matches: log(f"🔍 {date.strftime('%d.%m')}: Found {len(matches)} Matches")
        
        for m in matches:
            try:
                p1 = next((p for p in players if p['last_name'] in m['p1']), None)
                p2 = next((p for p in players if p['last_name'] in m['p2']), None)
                if not p1 or not p2: continue

                iso_time = f"{date.strftime('%Y-%m-%d')}T{m['time']}:00Z"
                
                # DB Check
                existing = supabase.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1['last_name']},player2_name.eq.{p2['last_name']}),and(player1_name.eq.{p2['last_name']},player2_name.eq.{p1['last_name']})").execute()
                
                if existing.data:
                    row = existing.data[0]
                    
                    # CASE A: WINNER DETECTED? -> CLOSE MATCH!
                    if m['winner']:
                        if row.get('actual_winner_name') != m['winner']:
                            log(f"🏆 RESULT FOUND: {m['winner']} won! Updating DB.")
                            supabase.table("market_odds").update({
                                "actual_winner_name": m['winner']
                            }).eq("id", row['id']).execute()
                        continue # Done.

                    # CASE B: MATCH LOCKED? -> SKIP
                    if row.get('actual_winner_name'): 
                        continue
                    
                    # CASE C: UPDATE ODDS (If plausible)
                    if m['odds1'] > 1.0 and m['odds2'] > 1.0:
                        supabase.table("market_odds").update({
                            "odds1": m['odds1'], "odds2": m['odds2'], "match_time": iso_time
                        }).eq("id", row['id']).execute()
                        log(f"🔄 Updated Odds: {p1['last_name']} vs {p2['last_name']}")
                else:
                    # NEW INSERT
                    if m['odds1'] <= 1.0: continue

                    s1, s2 = all_skills.get(p1['id'], {}), all_skills.get(p2['id'], {})
                    surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1['last_name'], p2['last_name'])
                    ai_meta = await analyze_match_with_ai(p1, p2, s1, s2, {}, {}, surf, bsi, notes)
                    
                    prob = calculate_physics_fair_odds(p1['last_name'], p2['last_name'], s1, s2, bsi, surf, ai_meta, m['odds1'], m['odds2'])
                    
                    supabase.table("market_odds").insert({
                        "player1_name": p1['last_name'], "player2_name": p2['last_name'],
                        "tournament": m['tour'], "odds1": m['odds1'], "odds2": m['odds2'],
                        "ai_fair_odds1": round(1/prob, 2) if prob > 0.01 else 99,
                        "ai_fair_odds2": round(1/(1-prob), 2) if prob < 0.99 else 99,
                        "ai_analysis_text": "Live Analysis", "created_at": datetime.now(timezone.utc).isoformat(),
                        "match_time": iso_time,
                        "actual_winner_name": m['winner'] # If scraped finished
                    }).execute()
                    log(f"💾 Saved: {p1['last_name']} vs {p2['last_name']}")

            except Exception as e: log(f"⚠️ Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
