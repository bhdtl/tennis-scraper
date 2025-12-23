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

log("🔌 Initialisiere Neural Scout (V67.0 - Precision Weighting Protocol)...")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'gemini-2.5-pro' 

ELO_CACHE = {"ATP": {}, "WTA": {}}

# =================================================================
# ELO DATA INJECTION
# =================================================================
async def fetch_elo_ratings():
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
                    rows = table.find_all('tr')[1:] 
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) > 4:
                            name = normalize_text(cols[0].get_text(strip=True)).lower()
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
            "temperature": 0.1 
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
# MATH CORE V3 - EXACT WEIGHTING
# =================================================================
def sigmoid_prob(diff, sensitivity=0.1):
    """Converts a score difference into a probability (0-1)."""
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta):
    """
    V3 Engine: 6-Component Weighted Model
    """
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    
    # ---------------------------------------------------------
    # 1. SKILLS (35%) - Raw Attributes from DB
    # ---------------------------------------------------------
    # Wir nehmen hier reine Skills OHNE Surface-Anpassung (die kommt bei Court)
    skill_diff_p1 = (s1.get('serve', 50) + s1.get('power', 50) + s1.get('forehand', 50) + s1.get('backhand', 50) + s1.get('speed', 50) + s1.get('mental', 50))
    skill_diff_p2 = (s2.get('serve', 50) + s2.get('power', 50) + s2.get('forehand', 50) + s2.get('backhand', 50) + s2.get('speed', 50) + s2.get('mental', 50))
    
    # Skill Diff skaliert durch 100 für Sigmoid
    prob_skills = sigmoid_prob(skill_diff_p1 - skill_diff_p2, sensitivity=0.08)

    # ---------------------------------------------------------
    # 2. MATCHUP AI (25%) - Tactical Fit from Gemini
    # ---------------------------------------------------------
    matchup_p1 = ai_meta.get('p1_tactical_score', 5) # 0-10
    matchup_p2 = ai_meta.get('p2_tactical_score', 5)
    prob_matchup = sigmoid_prob(matchup_p1 - matchup_p2, sensitivity=0.4)

    # ---------------------------------------------------------
    # 3. ELO (15%) - Surface Specific from TennisAbstract
    # ---------------------------------------------------------
    elo1 = 1500; elo2 = 1500
    elo_surf = 'Hard'
    if 'clay' in surface.lower(): elo_surf = 'Clay'
    elif 'grass' in surface.lower(): elo_surf = 'Grass'
    
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500)
        if n2 in name: elo2 = stats.get(elo_surf, 1500)
    
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    # ---------------------------------------------------------
    # 4. COURT BSI (10%) - Physics Fit (BSI Logic)
    # ---------------------------------------------------------
    is_fast = bsi >= 7.0
    is_slow = bsi <= 4.0
    
    # Wer profitiert vom Court?
    # Fast Court: Bonus für Serve/Power
    # Slow Court: Bonus für Speed/Stamina
    c_score_p1 = 0
    c_score_p2 = 0
    
    if is_fast:
        c_score_p1 = (s1.get('serve', 50) + s1.get('power', 50)) * 1.5
        c_score_p2 = (s2.get('serve', 50) + s2.get('power', 50)) * 1.5
    elif is_slow:
        c_score_p1 = (s1.get('speed', 50) + s1.get('stamina', 50)) * 1.5
        c_score_p2 = (s2.get('speed', 50) + s2.get('stamina', 50)) * 1.5
    else:
        # Balanced Court: Allround Bonus
        c_score_p1 = sum(s1.values()) / len(s1) if s1 else 50
        c_score_p2 = sum(s2.values()) / len(s2) if s2 else 50
        
    prob_court = sigmoid_prob(c_score_p1 - c_score_p2, sensitivity=0.05)

    # ---------------------------------------------------------
    # 5. FORM (10%) - Calculated via Opponent Odds (AI Proxy)
    # ---------------------------------------------------------
    # Gemini liefert hier einen Score (0-10), der berücksichtigt:
    # "Hat er gegen den Favoriten (1.20 Quote) gewonnen?" -> Hoher Form Score
    form_p1 = ai_meta.get('p1_form_score', 5)
    form_p2 = ai_meta.get('p2_form_score', 5)
    prob_form = sigmoid_prob(form_p1 - form_p2, sensitivity=0.4)

    # ---------------------------------------------------------
    # 6. UTR (5%) - Universal Tennis Rating from Sofascore/AI
    # ---------------------------------------------------------
    utr_p1 = ai_meta.get('p1_utr', 10.0)
    utr_p2 = ai_meta.get('p2_utr', 10.0)
    # UTR Diff von 1.0 ist signifikant -> Sensitivity hoch
    prob_utr = sigmoid_prob(utr_p1 - utr_p2, sensitivity=0.8)

    # ---------------------------------------------------------
    # FINAL WEIGHTED SUM
    # ---------------------------------------------------------
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
        return players, skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], [], [], []

# =================================================================
# SMART HELPERS
# =================================================================
async def resolve_ambiguous_tournament(p1_name, p2_name, scraped_tour_name):
    prompt = f"""
    TASK: Identify the specific Tennis Tournament Location & Surface.
    MATCH: {p1_name} vs {p2_name}
    SOURCE LABEL: "{scraped_tour_name}"
    DATE: Today/Upcoming.
    INSTRUCTION: Return the CITY NAME and the SURFACE type (Hard/Clay/Grass/Carpet).
    OUTPUT JSON ONLY: {{ "city": "Monastir", "surface_guessed": "Hard", "is_indoor": false }}
    """
    res = await call_gemini(prompt)
    if not res: return None
    try:
        clean = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except: return None

async def find_best_court_match_smart(scraped_tour_name, db_tournaments, p1_name, p2_name):
    scraped_lower = scraped_tour_name.lower().strip()
    for t in db_tournaments:
        if t['name'].lower() == scraped_lower: 
            return t['surface'], t['bsi_rating'], t.get('notes', '')
    if "futures" not in scraped_lower or len(scraped_lower) > 15:
        for t in db_tournaments:
            if t['name'].lower() in scraped_lower or scraped_lower in t['name'].lower(): 
                return t['surface'], t['bsi_rating'], t.get('notes', '')

    log(f"   🤖 KI recherchiert Turnierort für: {p1_name} vs {p2_name} ({scraped_tour_name[:20]}...)...")
    ai_loc = await resolve_ambiguous_tournament(p1_name, p2_name, scraped_tour_name)
    
    if ai_loc and ai_loc.get('city'):
        city = ai_loc['city'].lower()
        surface_guess = ai_loc.get('surface_guessed', 'Hard')
        for t in db_tournaments:
            if city in t['name'].lower():
                log(f"   ✅ KI Location Match: {t['name']} (BSI: {t['bsi_rating']})")
                return t['surface'], t['bsi_rating'], f"AI inferred location: {city}"
        guessed_bsi = 3.5 if 'clay' in surface_guess.lower() else (8.0 if ai_loc.get('is_indoor') else 6.5)
        log(f"   ⚠️ Location '{city}' nicht in DB. Nutze KI Surface: {surface_guess}")
        return surface_guess, guessed_bsi, f"AI Guess: {city}"

    if 'indoor' in scraped_lower: return 'Indoor', 8.2, 'Fast Indoor fallback'
    if any(x in scraped_lower for x in ['clay', 'sand']): return 'Red Clay', 3.5, 'Slow Clay fallback'
    return 'Hard', 6.5, 'Standard Hard fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    # PROMPT UPDATED FOR UTR AND FORM-VIA-ODDS
    prompt = f"""
    ROLE: Elite Tennis Scout & Data Analyst (SofaScore Integration).
    TASK: Deep Analysis of {p1['last_name']} vs {p2['last_name']}.
    
    CONTEXT:
    - Surface: {surface} (BSI {bsi}/10).
    - Notes: {notes}
    
    DATA P1: {r1.get('strengths')}, {r1.get('weaknesses')}. Style: {p1.get('play_style')}.
    DATA P2: {r2.get('strengths')}, {r2.get('weaknesses')}. Style: {p2.get('play_style')}.
    
    REQUIRED METRICS (0-10 Scale):
    1. TACTICAL SCORE (25% Weight): How well does P1's style match P2's weaknesses? (e.g. Lefty vs weak BH).
    2. FORM SCORE (10% Weight): Analyze recent matches. Did they beat favorites (high odds) or lose to underdogs? 
       - If P1 beat a 1.20 favorite -> High Form Score (8-9).
       - If P1 lost to a 5.00 underdog -> Low Form Score (1-2).
    3. UTR ESTIMATE (5% Weight): Search or estimate their Universal Tennis Rating (e.g., 14.5 vs 13.8).
    
    OUTPUT JSON ONLY:
    {{
        "p1_tactical_score": 7,
        "p2_tactical_score": 5,
        "p1_form_score": 8,
        "p2_form_score": 4,
        "p1_utr": 14.2,
        "p2_utr": 13.8,
        "ai_text": "Tactical analysis..."
    }}
    """
    res = await call_gemini(prompt)
    if not res: return {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    try:
        clean_json = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_utr': 10, 'p2_utr': 10}

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
            await page.goto(url, wait_until="networkidle", timeout=60000)
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
                txt += f"TOUR: {current_tour} | MATCH: {p1} VS {p2} | RAW_ROW_1: {row_text} | RAW_ROW_2: {rows[i+1].get_text()} \n"
    return txt

async def run_pipeline():
    log(f"🚀 Neural Scout v67 (Precision Weighting Protocol) Starting...")
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
        ANALYZE THIS TENNIS DATA.
        Target Players (Filter): {json.dumps(player_names)}
        
        INSTRUCTIONS:
        1. Find matches where BOTH players are in the Target List.
        2. Extract Market Odds (Decimal) from the "RAW_ROW" text.
           - Look for numbers like 1.57, 2.30, 10.00.
           - CAUTION: If the row is empty or contains only "-" or "info", ODDS ARE 0.0.
        3. Return JSON.
        
        INPUT TEXT:
        {cleaned_text[:30000]}
        
        OUTPUT JSON: {{ "matches": [ {{ "p1": "Lastname", "p2": "Lastname", "tour": "Full Tour Name", "odds1": 1.55, "odds2": 2.45 }} ] }}
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
                    market_odds1 = float(m.get('odds1', 0.0))
                    market_odds2 = float(m.get('odds2', 0.0))
                    
                    # 1. Update Existing
                    existing_match = supabase.table("market_odds").select("id").eq("player1_name", p1_obj['last_name']).eq("player2_name", p2_obj['last_name']).execute()
                    
                    if existing_match.data and len(existing_match.data) > 0:
                        match_id = existing_match.data[0]['id']
                        update_payload = { "odds1": market_odds1, "odds2": market_odds2, "tournament": m['tour'] }
                        supabase.table("market_odds").update(update_payload).eq("id", match_id).execute()
                        log(f"🔄 Updated Odds for: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        continue

                    # 2. Skip No Odds
                    if market_odds1 <= 1.0 or market_odds2 <= 1.0:
                        log(f"⏩ Skipping New Match {p1_obj['last_name']} vs {p2_obj['last_name']} - No Odds.")
                        continue
                    
                    # 3. Analyze New
                    log(f"✨ New Match Detected: {p1_obj['last_name']} vs {p2_obj['last_name']} -> Deep Analysis...")
                    s1 = next((s for s in all_skills if s['player_id'] == p1_obj['id']), {})
                    s2 = next((s for s in all_skills if s['player_id'] == p2_obj['id']), {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    surf, bsi, notes = await find_best_court_match_smart(m['tour'], all_tournaments, p1_obj['last_name'], p2_obj['last_name'])
                    ai_meta = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    prob_p1 = calculate_physics_fair_odds(p1_obj['last_name'], p2_obj['last_name'], s1, s2, bsi, surf, ai_meta)
                    
                    fair_odds1 = round(1 / prob_p1, 2) if prob_p1 > 0.01 else 99.0
                    fair_odds2 = round(1 / (1 - prob_p1), 2) if prob_p1 < 0.99 else 99.0
                    
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
                    
                    log(f"💾 Saving NEW: {p1_obj['last_name']} vs {p2_obj['last_name']} (Fair: {fair_odds1})")
                    supabase.table("market_odds").insert(match_entry).execute()

        except Exception as e:
            log(f"⚠️ Process Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
