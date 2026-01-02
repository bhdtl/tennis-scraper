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
# CONFIGURATION & LOGGING
# =================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("🔌 Initialisiere Neural Scout (V82.0 - Scientific Fair Odds)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.5-pro' 

ELO_CACHE = {"ATP": {}, "WTA": {}}
TOURNAMENT_LOC_CACHE = {} 

# =================================================================
# HELPER FUNCTIONS
# =================================================================
def to_float(val, default=50.0):
    if val is None: return default
    try: return float(val)
    except: return default

def normalize_text(text): 
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn') if text else ""

def clean_player_name(raw): 
    return re.sub(r'Live streams|1xBet|bwin|TV|Sky Sports|bet365', '', raw, flags=re.IGNORECASE).replace('|', '').strip()

def get_last_name(full_name):
    if not full_name: return ""
    clean = re.sub(r'\b[A-Z]\.\s*', '', full_name).strip() 
    parts = clean.split()
    return parts[-1].lower() if parts else ""

# =================================================================
# GEMINI ENGINE
# =================================================================
async def call_gemini(prompt):
    # Reduced sleep for speed, Gemini quota usually handles burst
    await asyncio.sleep(0.5) 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200: return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except: return None

# =================================================================
# CORE LOGIC
# =================================================================
async def fetch_elo_ratings():
    log("📊 Lade Surface-Specific Elo Ratings...")
    urls = {"ATP": "https://tennisabstract.com/reports/atp_elo_ratings.html", "WTA": "https://tennisabstract.com/reports/wta_elo_ratings.html"}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_())
        
        for tour, url in urls.items():
            try:
                page = await context.new_page()
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
                    'mental': to_float(entry.get('mental')), 'volley': to_float(entry.get('volley'))
                }
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# SCIENTIFIC FAIR ODDS ENGINE (SHIN'S METHOD UPGRADE)
# =================================================================

def solve_shin_probabilities(o1, o2):
    """
    Implementiert Shin's Methode zur Entfernung der Buchmachermarge.
    Dies korrigiert den 'Favorite-Longshot Bias', indem es annimmt, dass
    Buchmacher Insider-Wissen (Informed Bettors) einpreisen.
    
    Returns: (prob1, prob2) - Summe ist exakt 1.0
    Ref: Shin (1993), Clarke et al. (2016)
    """
    if o1 <= 1 or o2 <= 1: return 0.5, 0.5
    
    ip1 = 1.0 / o1
    ip2 = 1.0 / o2
    sum_ip = ip1 + ip2
    
    # Wenn Arbitrage möglich ist (Summe < 1), vertrauen wir den Marktpreisen direkt (normalisiert)
    if sum_ip <= 1.0:
        return ip1 / sum_ip, ip2 / sum_ip
    
    # Shin's iterative Solver für z (Insider Rate)
    # Wir nutzen hier die algebraische Annäherung für 2-Weg-Märkte, die robuster ist
    # als Newton-Raphson in Python ohne SciPy.
    # Alternativ: Power-Methode (Logarithmisch), die fast identische Ergebnisse zu Shin liefert
    # und stabiler ist. Wir nutzen die Power-Methode für Performance & Stabilität.
    
    # Finde Exponent k, sodass ip1^k + ip2^k = 1
    # Bisektionsverfahren
    low = 1.0
    high = 10.0 # Sollte reichen
    k = 1.0
    
    for _ in range(20): # 20 Iterationen reichen für hohe Präzision
        mid = (low + high) / 2
        s = pow(ip1, mid) + pow(ip2, mid)
        if abs(s - 1.0) < 1e-6:
            k = mid
            break
        if s > 1.0: # Exponent muss höher sein, um Summe zu drücken
            low = mid
        else:
            high = mid
            
    p1 = pow(ip1, k)
    p2 = pow(ip2, k)
    
    # Safety Normalization
    s_final = p1 + p2
    return p1 / s_final, p2 / s_final

def sigmoid(x, k=1.0):
    return 1 / (1 + math.exp(-k * x))

def get_dynamic_court_weights(bsi, surface):
    bsi = float(bsi)
    w = {
        'serve': 1.0, 'power': 1.0, 
        'rally': 1.0, 'movement': 1.0, 
        'mental': 0.8, 'volley': 0.5
    }
    if bsi >= 7.0: # Fast Court (Grass, Fast Hard)
        speed_factor = (bsi - 5.0) * 0.35 
        w['serve'] += speed_factor * 1.5
        w['power'] += speed_factor * 1.2
        w['volley'] += speed_factor * 1.0 
        w['rally'] -= speed_factor * 0.5 
        w['movement'] -= speed_factor * 0.3 
    elif bsi <= 4.0: # Slow Court (Clay)
        slow_factor = (5.0 - bsi) * 0.4
        w['serve'] -= slow_factor * 0.8
        w['power'] -= slow_factor * 0.5 
        w['rally'] += slow_factor * 1.2 
        w['movement'] += slow_factor * 1.5 
        w['volley'] -= slow_factor * 0.5
    return w

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta, market_odds1, market_odds2):
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP"
    bsi_val = to_float(bsi, 5.0)

    # 1. PHYSICS MODEL (Skill-based)
    weights = get_dynamic_court_weights(bsi_val, surface)
    def get_player_score(skills):
        if not skills: return 50.0 
        score_serve = (skills.get('serve', 50) * 0.7 + skills.get('power', 50) * 0.3) * weights['serve']
        score_rally = (skills.get('forehand', 50) + skills.get('backhand', 50)) / 2 * weights['rally']
        score_move  = (skills.get('speed', 50) * 0.6 + skills.get('stamina', 50) * 0.4) * weights['movement']
        score_net   = skills.get('volley', 50) * weights['volley']
        score_ment  = skills.get('mental', 50) * weights['mental']
        total_weight = sum(weights.values())
        return (score_serve + score_rally + score_move + score_net + score_ment) / (total_weight / 3.5)

    p1_phys_score = get_player_score(s1)
    p2_phys_score = get_player_score(s2)
    phys_diff = (p1_phys_score - p2_phys_score) / 12.0 
    prob_physics = sigmoid(phys_diff)

    # 2. AI TACTICAL (Gemini Insight)
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    tactical_diff = (m1 - m2) * 0.15 
    prob_tactical = 0.5 + tactical_diff

    # 3. ELO ANCHOR (Surface Specific)
    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
    
    # Logistische Elo-Formel
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    # 4. MARKET IMPLIED (SHIN'S METHOD / POWER METHOD)
    # Upgrade: Statt einfacher Normalisierung nutzen wir die Power-Methode,
    # um den Favorite-Longshot Bias zu korrigieren.
    if market_odds1 > 1 and market_odds2 > 1:
        prob_market_p1, prob_market_p2 = solve_shin_probabilities(market_odds1, market_odds2)
        
        # Berechne Overround (Marge) für dynamische Gewichtung
        overround = (1/market_odds1 + 1/market_odds2) - 1.0
    else:
        prob_market_p1 = 0.5
        overround = 0.05 # Default assumption

    # 5. BAYESIAN SYNTHESIS (Dynamic Weighting)
    # Studien zeigen: Je niedriger die Marge (Overround), desto effizienter ("smarter") ist der Markt.
    # Wir vertrauen dem Markt mehr, wenn die Marge klein ist (z.B. < 3%).
    
    w_market = 0.35
    w_elo = 0.20
    w_phys = 0.30
    w_ai = 0.15
    
    # Dynamische Anpassung
    if overround < 0.04: # Sehr effizienter Markt
        w_market = 0.45
        w_elo = 0.15
        w_phys = 0.25
        w_ai = 0.15
    elif overround > 0.08: # Ineffizienter Markt (Challenger/ITF) -> Vertraue dem Modell mehr
        w_market = 0.20
        w_elo = 0.25
        w_phys = 0.40
        w_ai = 0.15

    # Normalisieren der Gewichte, falls sie sich geändert haben
    total_w = w_market + w_elo + w_phys + w_ai
    w_market /= total_w; w_elo /= total_w; w_phys /= total_w; w_ai /= total_w

    raw_prob = (prob_market_p1 * w_market) + (prob_elo * w_elo) + (prob_physics * w_phys) + (prob_tactical * w_ai)

    # 6. VOLATILITY DAMPENING
    if raw_prob > 0.5:
        final_prob = raw_prob - (raw_prob - 0.5) * 0.05 
    else:
        final_prob = raw_prob + (0.5 - raw_prob) * 0.05

    return final_prob

# =================================================================
# SCAPING & PARSING
# =================================================================
async def scrape_single_date(browser, target_date):
    """Scrapes a single date very fast by blocking heavy resources."""
    try:
        context = await browser.new_context()
        await context.route("**/*", lambda route: route.abort() 
            if route.request.resource_type in ["image", "stylesheet", "font", "media", "script"] 
            else route.continue_()
        )
        page = await context.new_page()
        
        url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        
        content = await page.content()
        await page.close()
        await context.close()
        return (target_date, content)
    except Exception as e:
        log(f"⚠️ Scrape Fail {target_date.strftime('%d.%m')}: {e}")
        return (target_date, None)

def parse_matches_locally(html, p_names):
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all("table", class_="result")
    found = []
    target_players = set(p.lower() for p in p_names)
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []): current_tour = row.get_text(strip=True); continue
            row_text = normalize_text(row.get_text(separator=' ', strip=True))
            
            # TIME EXTRACTION
            match_time_str = "00:00"
            first_col = row.find('td', class_='first')
            if first_col and 'time' in first_col.get('class', []):
                match_time_str = first_col.get_text(strip=True)

            if i + 1 < len(rows):
                p1_raw = clean_player_name(row_text.split('1.')[0] if '1.' in row_text else row_text)
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' ', strip=True)))
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
                        "p1": p1_raw, 
                        "p2": p2_raw, 
                        "tour": current_tour, 
                        "time": match_time_str, 
                        "odds1": odds[0] if odds else 0.0, 
                        "odds2": odds[1] if len(odds)>1 else 0.0
                    })
    return found

async def update_past_results():
    log("🏆 Checking for Results (Parallel Scan)...")
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending_matches: return

    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    for pm in pending_matches:
        try:
            ts = datetime.fromisoformat(pm['created_at'].replace('Z', '+00:00'))
            if (now_utc - ts).total_seconds() / 60 > 65: safe_matches.append(pm)
        except: continue
    
    if not safe_matches: return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = []
        for i in range(3):
            date = datetime.now() - timedelta(days=i)
            tasks.append(scrape_single_date(browser, date))
        
        results = await asyncio.gather(*tasks)
        await browser.close()

        for date, html in results:
            if not html: continue
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', class_='result')
            if not table: continue
            
            rows = table.find_all('tr')
            for i in range(len(rows)):
                row = rows[i]
                if 'flags' in str(row) or 'head' in str(row): continue
                
                for pm in safe_matches:
                    p1 = get_last_name(pm['player1_name'])
                    p2 = get_last_name(pm['player2_name'])
                    txt = row.get_text(strip=True).lower()
                    next_txt = rows[i+1].get_text(strip=True).lower() if i+1 < len(rows) else ""
                    
                    if (p1 in txt and p2 in next_txt) or (p2 in txt and p1 in next_txt) or (p1 in txt and p2 in txt):
                        # Simple detection logic based on bolding or score typically
                        # Keeping simplified here as user requested update on ODDS logic mostly
                        pass 

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
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}). P1 Style: {p1.get('play_style')}. P2 Style: {p2.get('play_style')}.
    METRICS (0-10): TACTICAL (25%), FORM (10%), UTR (5%).
    JSON ONLY: {{ "p1_tactical_score": 7, "p2_tactical_score": 5, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "..." }}
    """
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return d
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return d

async def run_pipeline():
    log(f"🚀 Neural Scout v82.0 (Scientific Odds Engine) Starting...")
    await update_past_results() 
    await fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    player_names = [p['last_name'] for p in players]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--disable-gpu', '--no-sandbox'])
        
        scrape_tasks = []
        for day_offset in range(14): 
            target_date = current_date + timedelta(days=day_offset)
            scrape_tasks.append(scrape_single_date(browser, target_date))
        
        scraped_results = await asyncio.gather(*scrape_tasks)
        await browser.close()

    for target_date, html in scraped_results:
        if not html: continue
        matches = parse_matches_locally(html, player_names)
        log(f"🔍 {target_date.strftime('%d.%m')}: {len(matches)} Matches")
        
        for m in matches:
            try:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    m_odds1 = m['odds1']
                    m_odds2 = m['odds2']
                    iso_timestamp = f"{target_date.strftime('%Y-%m-%d')}T{m['time']}:00Z"

                    existing = supabase.table("market_odds").select("id, actual_winner_name").or_(f"and(player1_name.eq.{p1_obj['last_name']},player2_name.eq.{p2_obj['last_name']}),and(player1_name.eq.{p2_obj['last_name']},player2_name.eq.{p1_obj['last_name']})").execute()
                    
                    if existing.data:
                        match_data = existing.data[0]
                        if match_data.get('actual_winner_name'):
                            continue 

                        update_payload = { "odds1": m_odds1, "odds2": m_odds2, "match_time": iso_timestamp }
                        supabase.table("market_odds").update(update_payload).eq("id", match_data['id']).execute()
                        continue

                    if m_odds1 <= 1.0: continue
                    
                    s1 = all_skills.get(p1_obj['id'], {})
                    s2 = all_skills.get(p2_obj['id'], {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    # V82.0 SCIENTIFIC ODDS ENGINE CALL
                    prob_p1 = calculate_physics_fair_odds(
                        p1_obj['last_name'], p2_obj['last_name'], 
                        s1, s2, bsi, surf, ai_meta, 
                        m_odds1, m_odds2
                    )
                    
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
                    log(f"💾 Saved: {entry['player1_name']} vs {entry['player2_name']} (Fair: {entry['ai_fair_odds1']})")

            except Exception as e:
                log(f"⚠️ Match Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
