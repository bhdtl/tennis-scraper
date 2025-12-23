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

log("🔌 Initialisiere Neural Scout (V70.0 - Efficiency Engine)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.5-pro' 

# Caches um API Calls zu sparen
ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} # Merkt sich KI-Ergebnisse für Orte

# =================================================================
# DATA NORMALIZATION LAYER
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def normalize_player_skills(raw_skills_list):
    normalized = {}
    for entry in raw_skills_list:
        pid = entry.get('player_id')
        if not pid: continue
        normalized[pid] = {
            'serve': to_float(entry.get('serve')),
            'power': to_float(entry.get('power')),
            'forehand': to_float(entry.get('forehand')),
            'backhand': to_float(entry.get('backhand')),
            'speed': to_float(entry.get('speed')),
            'stamina': to_float(entry.get('stamina')),
            'mental': to_float(entry.get('mental'))
        }
    return normalized

# =================================================================
# ELO ENGINE
# =================================================================
async def fetch_elo_ratings():
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
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
                await page.close()
            except Exception as e:
                log(f"   ⚠️ Elo Fetch Warning ({tour}): {e}")
        await browser.close()

# =================================================================
# GEMINI ENGINE (Optimized)
# =================================================================
async def call_gemini(prompt):
    # RPM Drossel: Kurze Pause vor jedem Call, schont das Limit
    await asyncio.sleep(1.0) 
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0) # Timeout reduziert
            if response.status_code != 200: return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except: return None

# =================================================================
# MATH CORE V3 (35/25/15/10/10/5)
# =================================================================
def sigmoid_prob(diff, sensitivity=0.1):
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1_clean, s2_clean, bsi, surface, ai_meta):
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    
    # 1. SKILLS (35%)
    sum_p1 = sum(s1_clean.values())
    sum_p2 = sum(s2_clean.values())
    prob_skills = sigmoid_prob(sum_p1 - sum_p2, sensitivity=0.08)

    # 2. MATCHUP AI (25%)
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.4)

    # 3. ELO (15%)
    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    # 4. COURT BSI (10%)
    bsi_val = to_float(bsi, 6.0)
    is_fast = bsi_val >= 7.0
    is_slow = bsi_val <= 4.0
    c_p1 = 0.0; c_p2 = 0.0
    
    if is_fast:
        c_p1 = (s1_clean['serve'] + s1_clean['power']) * 1.5
        c_p2 = (s2_clean['serve'] + s2_clean['power']) * 1.5
    elif is_slow:
        c_p1 = (s1_clean['speed'] + s1_clean['stamina']) * 1.5
        c_p2 = (s2_clean['speed'] + s2_clean['stamina']) * 1.5
    else:
        c_p1 = sum_p1 / 7.0
        c_p2 = sum_p2 / 7.0
    prob_court = sigmoid_prob(c_p1 - c_p2, sensitivity=0.05)

    # 5. FORM (10%)
    f1 = to_float(ai_meta.get('p1_form_score', 5))
    f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.4)

    # 6. UTR (5%)
    utr1 = to_float(ai_meta.get('p1_utr', 10.0))
    utr2 = to_float(ai_meta.get('p2_utr', 10.0))
    prob_utr = sigmoid_prob(utr1 - utr2, sensitivity=0.8)

    final_prob = (
        (prob_skills  * 0.35) +
        (prob_matchup * 0.25) +
        (prob_elo     * 0.15) +
        (prob_court   * 0.10) +
        (prob_form    * 0.10) +
        (prob_utr     * 0.05)
    )
    
    return final_prob

async def get_db_data():
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        clean_skills = normalize_player_skills(skills)
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# SMART HELPERS (Optimized with Local Cache)
# =================================================================
async def resolve_ambiguous_tournament(p1, p2, scraped_name):
    # CACHE CHECK: Haben wir diesen Turniernamen schon mal gelöst?
    if scraped_name in TOURNAMENT_LOC_CACHE:
        return TOURNAMENT_LOC_CACHE[scraped_name]

    # Komprimierter Prompt spart Tokens
    prompt = f"""
    TASK: Locate Tennis Match (City/Surface).
    MATCH: {p1} vs {p2} | SOURCE: "{scraped_name}"
    JSON: {{ "city": "CityName", "surface_guessed": "Hard/Clay/Grass", "is_indoor": bool }}
    """
    res = await call_gemini(prompt)
    if not res: return None
    try: 
        data = json.loads(res.replace("```json", "").replace("```", "").strip())
        TOURNAMENT_LOC_CACHE[scraped_name] = data # Speichern im Cache
        return data
    except: return None

async def find_best_court_match_smart(scraped_tour_name, db_tournaments, p1_name, p2_name):
    scraped_lower = scraped_tour_name.lower().strip()
    for t in db_tournaments:
        if t['name'].lower() == scraped_lower: return t['surface'], t['bsi_rating'], t.get('notes', '')
    
    # Lokale Fallbacks (Sparen API Calls für Standard-Namen)
    if "clay" in scraped_lower: return "Red Clay", 3.5, "Local Keyword Match"
    if "hard" in scraped_lower: return "Hard", 6.5, "Local Keyword Match"
    if "indoor" in scraped_lower: return "Indoor", 8.0, "Local Keyword Match"

    if "futures" not in scraped_lower or len(scraped_lower) > 15:
        for t in db_tournaments:
            if t['name'].lower() in scraped_lower or scraped_lower in t['name'].lower(): 
                return t['surface'], t['bsi_rating'], t.get('notes', '')

    log(f"   🤖 AI resolving location for {p1_name} vs {p2_name}...")
    ai_loc = await resolve_ambiguous_tournament(p1_name, p2_name, scraped_tour_name)
    
    if ai_loc and ai_loc.get('city'):
        city = ai_loc['city'].lower()
        surf = ai_loc.get('surface_guessed', 'Hard')
        for t in db_tournaments:
            if city in t['name'].lower():
                return t['surface'], t['bsi_rating'], f"AI inferred: {city}"
        guessed_bsi = 3.5 if 'clay' in surf.lower() else 6.5
        return surf, guessed_bsi, f"AI Guess: {city}"
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    # Komprimierter Prompt (Gleicher Inhalt, weniger Worte = weniger Tokens)
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: Analyze {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}). P1 Style: {p1.get('play_style')}. P2 Style: {p2.get('play_style')}.
    
    OUTPUT METRICS (0-10):
    1. TACTICAL (25%): Style matchup?
    2. FORM (10%): Recent wins vs odds. High score = beating favorites.
    3. UTR (5%): Estimate UTR (e.g. 14.2).
    
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "Brief analysis..." }}
    """
    res = await call_gemini(prompt)
    default = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return default
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return default

# =================================================================
# LOCAL PARSING (The Huge Token Saver)
# =================================================================
def normalize_text(text): 
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn') if text else ""

def clean_player_name(raw): 
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def parse_matches_locally(html_content, player_names):
    """
    Ersetzt den teuren AI-Call durch schnelles Python.
    Sucht nach Zeilen mit Spielern aus unserer DB und extrahiert Odds via Regex.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all("table", class_="result")
    
    found_matches = []
    # Set für schnellen Lookup (O(1)) statt Liste (O(n))
    target_players = set(p.lower() for p in player_names)
    
    current_tour = "Unknown"
    
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            # Turnier Header finden
            if "head" in row.get("class", []): 
                current_tour = row.get_text(strip=True)
                continue
            
            # Zeilen Text holen
            cols = row.find_all("td")
            if not cols: continue
            
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # Prüfen ob Match-Zeile (hat oft eine Uhrzeit wie 14:00 oder ist Teil eines Blocks)
            # Wir schauen auf i+1 für den Gegner
            if i + 1 < len(rows):
                # Namen extrahieren
                raw_p1 = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text) # grober split
                raw_p2 = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
                
                # Check: Sind diese Namen in unserer DB?
                p1_found = any(tp in raw_p1.lower() for tp in target_players)
                p2_found = any(tp in raw_p2.lower() for tp in target_players)
                
                if p1_found and p2_found:
                    # ODDS Extraction via Regex (Sucht nach dezimalzahlen am Ende)
                    # Suche in der Zelle, die Quoten enthält (meistens die letzten TDs)
                    odds = []
                    try:
                        # TennisExplorer hat Quoten oft in spezifischen Spalten. 
                        # Wir suchen alle Zahlen > 1.0 in der Zeile
                        nums = re.findall(r'\d+\.\d+', row_text)
                        # Filtern auf plausible Quoten (1.01 bis 50.0)
                        valid_nums = [float(x) for x in nums if 1.0 < float(x) < 50.0]
                        if len(valid_nums) >= 2:
                            odds = valid_nums[:2] # Nehmen wir an erste ist 1, zweite ist 2
                        else:
                            # Versuch Zeile 2 (Gegner Zeile hat oft Quote 2)
                            nums2 = re.findall(r'\d+\.\d+', rows[i+1].get_text())
                            valid_nums2 = [float(x) for x in nums2 if 1.0 < float(x) < 50.0]
                            if valid_nums and valid_nums2:
                                odds = [valid_nums[0], valid_nums2[0]]
                    except: pass
                    
                    odd1 = odds[0] if len(odds) > 0 else 0.0
                    odd2 = odds[1] if len(odds) > 1 else 0.0
                    
                    match_data = {
                        "p1": raw_p1, "p2": raw_p2, 
                        "tour": current_tour, 
                        "odds1": odd1, "odds2": odd2
                    }
                    found_matches.append(match_data)
                    
    return found_matches

async def run_pipeline():
    log(f"🚀 Neural Scout v70 (Efficiency Engine) Starting...")
    await fetch_elo_ratings()
    
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    player_names = [p['last_name'] for p in players]
    
    for day_offset in range(35): 
        target_date = current_date + timedelta(days=day_offset)
        
        # 1. SCRAPE HTML
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        # 2. LOCAL PARSING (Spart Token!)
        matches = parse_matches_locally(html, player_names)
        log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
        
        for m in matches:
            try:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    market_odds1 = m['odds1']
                    market_odds2 = m['odds2']
                    
                    # Check Existing
                    existing = supabase.table("market_odds").select("id").eq("player1_name", p1_obj['last_name']).eq("player2_name", p2_obj['last_name']).execute()
                    if existing.data:
                        match_id = existing.data[0]['id']
                        supabase.table("market_odds").update({"odds1": market_odds1, "odds2": market_odds2}).eq("id", match_id).execute()
                        log(f"🔄 Update: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        continue

                    if market_odds1 <= 1.0: 
                        log(f"⏩ Skip New (No Odds): {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        continue
                    
                    # Analyze New (Hier nutzt er noch AI, aber nur für relevante Matches!)
                    log(f"✨ Analyzing: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                    s1 = all_skills.get(p1_obj['id'], {})
                    s2 = all_skills.get(p2_obj['id'], {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    prob_p1 = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta)
                    
                    entry = {
                        "player1_name": p1_obj['last_name'], "player2_name": p2_obj['last_name'], "tournament": m['tour'],
                        "odds1": market_odds1, "odds2": market_odds2,
                        "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0 else 99,
                        "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 1 else 99,
                        "ai_analysis_text": ai_meta.get('ai_text', 'No analysis'),
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    supabase.table("market_odds").insert(entry).execute()
                    log(f"💾 Saved: {p1_obj['last_name']} vs {p2_obj['last_name']} (Fair: {entry['ai_fair_odds1']})")

            except Exception as e:
                log(f"⚠️ Match Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
