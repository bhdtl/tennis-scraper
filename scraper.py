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
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION & LOGGING
# =================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("🔌 Initialisiere Neural Scout (V61.0 - Surface Physics Value Scanner)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.5-pro' 

# Globaler Cache für Elo Ratings (um unnötige Requests zu vermeiden)
ELO_CACHE = {"ATP": {}, "WTA": {}}

# =================================================================
# ELO DATA INJECTION (TennisAbstract Scraper)
# =================================================================
async def fetch_elo_ratings():
    """
    Holt die aktuellen Surface-Specific Elo Ratings von TennisAbstract.
    Dies ist der 'Realitäts-Check' für die KI.
    """
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {
        "ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html",
        "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for tour, url in urls.items():
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', {'id': 'reportable'})
                
                if table:
                    rows = table.find_all('tr')[1:] # Skip Header
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) > 4:
                            name = normalize_text(cols[0].get_text(strip=True)).lower()
                            # TennisAbstract Struktur: Name, Age, Elo, HardRaw, ClayRaw, GrassRaw
                            # Wir versuchen Hard (col 3), Clay (col 4), Grass (col 5) zu extrahieren
                            try:
                                elo_hard = float(cols[3].get_text(strip=True) or 1500)
                                elo_clay = float(cols[4].get_text(strip=True) or 1500)
                                elo_grass = float(cols[5].get_text(strip=True) or 1500)
                                ELO_CACHE[tour][name] = {'Hard': elo_hard, 'Clay': elo_clay, 'Grass': elo_grass}
                            except: continue
                    log(f"   ✅ {tour} Elo Ratings geladen: {len(ELO_CACHE[tour])} Spieler.")
                await page.close()
            except Exception as e:
                log(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
        
        await browser.close()

# =================================================================
# GEMINI ENGINE
# =================================================================
async def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json", 
            "temperature": 0.2
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=90.0)
            if response.status_code != 200:
                return None
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except: return None

# =================================================================
# MATH CORE V2 (The Silicon Valley Upgrade - FULL WEIGHTING INTEGRATED)
# =================================================================
def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta):
    """
    Die neue, hochkomplexe Berechnungs-Engine.
    Kombiniert: Skills + Surface Physics + Elo + AI Meta-Data (Fatigue).
    """
    # 1. SETUP
    # Normalisieren für Matching
    n1 = p1_name.lower().split()[-1] # Nachname
    n2 = p2_name.lower().split()[-1]
    
    # Tour ermitteln (einfache Heuristik, sonst ATP default)
    tour = "ATP" 
    
    # 2. ELO LOOKUP (Surface Specific)
    elo1 = 1500; elo2 = 1500
    
    # Mappe unseren Surface Namen auf TennisAbstract Spalten
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    # Suche in Cache (Fuzzy Search wäre besser, hier simpel)
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500)
        if n2 in name: elo2 = stats.get(elo_surf, 1500)
        
    # Elo Delta Probability
    # Die Elo-Wahrscheinlichkeit ist der "Anker"
    elo_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    # 3. PHYSICS & SKILL ENGINE
    is_fast = bsi >= 7.0
    is_slow = bsi <= 4.0
    
    # Dynamic Weighting (CSI Multiplier - CRITICAL UPDATE)
    # Auf schnellem Belag zählt der Aufschlag exponentiell mehr
    w_serve = 1.0 * (1.6 if is_fast else (0.4 if is_slow else 1.0))
    w_base  = 1.0 * (0.5 if is_fast else (1.5 if is_slow else 1.0))
    w_move  = 1.0 * (0.4 if is_fast else (1.4 if is_slow else 1.0)) # Movement auf Sand extrem wichtig
    
    # Skill Differentials (Skala 0-100)
    serve_diff = ((s1.get('serve', 50) + s1.get('power', 50)) - (s2.get('serve', 50) + s2.get('power', 50))) * w_serve
    base_diff  = ((s1.get('forehand', 50) + s1.get('backhand', 50)) - (s2.get('forehand', 50) + s2.get('backhand', 50))) * w_base
    phys_diff  = ((s1.get('speed', 50) + s1.get('stamina', 50)) - (s2.get('speed', 50) + s2.get('stamina', 50))) * w_move
    ment_diff  = (s1.get('mental', 50) - s2.get('mental', 50)) * 0.9 # Mental leicht erhöht
    
    # Aggregation der Skills
    # Wir teilen durch einen Faktor, um den Score in einen Bereich für die Sigmoid-Funktion zu bringen
    raw_skill_score = (serve_diff + base_diff + phys_diff + ment_diff) / 160.0
    
    # 4. SPECIALIST PENALTY & FATIGUE (Data from AI Meta)
    # ai_meta kommt aus der Gemini Analyse
    fatigue_p1 = ai_meta.get('p1_fatigue_score', 0) # 0-10 (10 = Dead)
    fatigue_p2 = ai_meta.get('p2_fatigue_score', 0)
    surface_comfort_p1 = ai_meta.get('p1_surface_comfort', 5) # 0-10
    surface_comfort_p2 = ai_meta.get('p2_surface_comfort', 5)
    
    # Penalty Calculation
    # Fatigue bestraft stark (exponentiell bei hohen Werten wäre besser, hier linear genähert)
    fatigue_malus = (fatigue_p1 - fatigue_p2) * 0.25 
    # Comfort Bonus
    comfort_bonus = (surface_comfort_p1 - surface_comfort_p2) * 0.30
    
    # 5. FINAL FUSION
    # Skill Probability via Sigmoid Funktion
    skill_prob = 1 / (1 + math.exp(-1.0 * (raw_skill_score - fatigue_malus + comfort_bonus)))
    
    # Weighted Average:
    # 45% Elo (Markt-Basis) vs 55% Physics/AI (Edge)
    final_prob = (elo_prob * 0.45) + (skill_prob * 0.55)
    
    return final_prob

# =================================================================
# DATA LOADING
# =================================================================
async def get_db_data():
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        return players, skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], [], [], []

def find_best_court_match(scraped_tour_name, db_tournaments):
    scraped_lower = scraped_tour_name.lower().strip()
    for t in db_tournaments:
        if t['name'].lower() == scraped_lower: return t['surface'], t['bsi_rating'], t.get('notes', '')
    for t in db_tournaments:
        if t['name'].lower() in scraped_lower or scraped_lower in t['name'].lower(): return t['surface'], t['bsi_rating'], t.get('notes', '')
    if 'indoor' in scraped_lower: return 'Indoor', 8.2, 'Fast Indoor fallback'
    if any(x in scraped_lower for x in ['clay', 'sand']): return 'Red Clay', 3.5, 'Slow Clay fallback'
    return 'Hard', 6.5, 'Standard Hard fallback'

# =================================================================
# AI ANALYSIS (With Meta-Data Extraction)
# =================================================================
async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    # Wir nutzen Gemini nicht nur für Text, sondern als Sensor für "Soft Factors"
    prompt = f"""
    ROLE: Elite Tennis Scout.
    TASK: Deep Analysis of {p1['last_name']} vs {p2['last_name']}.
    
    CONTEXT:
    - Surface: {surface} (BSI {bsi}/10).
    - Notes: {notes}
    
    DATA P1: {r1.get('strengths')}, {r1.get('weaknesses')}. Style: {p1.get('play_style')}.
    DATA P2: {r2.get('strengths')}, {r2.get('weaknesses')}. Style: {p2.get('play_style')}.
    
    MISSION:
    1. Estimate "Surface Comfort" (0-10) for both. Is P1 a clay specialist on hard?
    2. Estimate "Fatigue Risk" (0-10) based on typical schedules for these players.
    3. Write a 3-paragraph tactical breakdown.
    
    OUTPUT JSON ONLY:
    {{
        "p1_surface_comfort": 8,
        "p2_surface_comfort": 4,
        "p1_fatigue_score": 2,
        "p2_fatigue_score": 1,
        "ai_text": "Paragraph 1: P1 Analysis... Paragraph 2: P2 Analysis... Paragraph 3: Clash..."
    }}
    """
    
    res = await call_gemini(prompt)
    if not res: return {'p1_surface_comfort': 5, 'p2_surface_comfort': 5, 'ai_text': "Analysis Pending."}
    
    try:
        clean_json = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {'p1_surface_comfort': 5, 'p2_surface_comfort': 5, 'ai_text': "Parsing Error."}

# =================================================================
# SCRAPER CORE
# =================================================================
def normalize_text(text):
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw):
    noise = [r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'bet365', r'Unibet', r'William Hill']
    for pat in noise: raw = re.sub(pat, '', raw, flags=re.IGNORECASE)
    return raw.replace('|', '').strip()

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try: await page.wait_for_selector(".result", timeout=10000)
            except: 
                await browser.close()
                return None
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            log(f"❌ Scrape Error: {e}")
            await browser.close()
            return None

def clean_html_for_extraction(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(["script", "style", "nav", "footer"]): tag.extract()
    txt = ""
    tables = soup.find_all("table", class_="result")
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []):
                current_tour = row.get_text(strip=True)
                continue
            row_text = normalize_text(row.get_text(separator=' | ', strip=True))
            if re.search(r'\d{2}:\d{2}', row_text) and i+1 < len(rows):
                p1 = clean_player_name(row_text)
                p2 = clean_player_name(normalize_text(rows[i+1].get_text(separator=' | ', strip=True)))
                txt += f"TOURNAMENT: {current_tour} | {p1} VS {p2}\n"
    return txt

# =================================================================
# MAIN PIPELINE
# =================================================================
async def run_pipeline():
    log(f"🚀 Neural Scout v61 (Elo + Physics Engine) Starting...")
    
    # 1. Elo Ratings laden (One-Time Fetch)
    await fetch_elo_ratings()
    
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    
    for day_offset in range(35): 
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        cleaned_text = clean_html_for_extraction(html)
        if not cleaned_text: continue

        player_names = [p['last_name'] for p in players]
        
        extract_prompt = f"""
        Extract matches where BOTH players are in this list: {json.dumps(player_names)}
        Input Text: {cleaned_text[:25000]}
        OUTPUT JSON: {{ "matches": [ {{ "p1": "Lastname", "p2": "Lastname", "tour": "Tour Name (Full)", "odds1": 1.5, "odds2": 2.5 }} ] }}
        If odds missing, set to 0.
        """
        
        extract_res = await call_gemini(extract_prompt)
        if not extract_res: continue

        try:
            clean_json = extract_res.replace("```json", "").replace("```", "").strip()
            matches = json.loads(clean_json).get("matches", [])
            log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
            
            for m in matches:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    s1 = next((s for s in all_skills if s['player_id'] == p1_obj['id']), {})
                    s2 = next((s for s in all_skills if s['player_id'] == p2_obj['id']), {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    # 1. COURT & BSI
                    surf, bsi, notes = find_best_court_match(m['tour'], all_tournaments)
                    
                    # 2. AI META-ANALYSIS (Fatigue & Comfort)
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    # 3. PHYSICS & ELO CALCULATION
                    prob_p1 = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta)
                    
                    fair_odds1 = round(1 / prob_p1, 2) if prob_p1 > 0.01 else 99.0
                    fair_odds2 = round(1 / (1 - prob_p1), 2) if prob_p1 < 0.99 else 99.0
                    
                    # Value Check
                    market_odds1 = float(m.get('odds1', 0))
                    market_odds2 = float(m.get('odds2', 0))
                    
                    match_entry = {
                        "player1_name": p1_obj['last_name'],
                        "player2_name": p2_obj['last_name'],
                        "tournament": m['tour'],
                        "odds1": market_odds1,
                        "odds2": market_odds2,
                        "ai_fair_odds1": fair_odds1,
                        "ai_fair_odds2": fair_odds2,
                        "ai_analysis_text": ai_meta.get('ai_text', 'No analysis'),
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    
                    log(f"💾 Saving: {p1_obj['last_name']} vs {p2_obj['last_name']} (Fair: {fair_odds1} vs Market: {market_odds1})")
                    supabase.table("market_odds").upsert(match_entry, on_conflict="player1_name, player2_name, tournament").execute()

        except Exception as e:
            log(f"⚠️ Process Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
