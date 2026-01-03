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

log("🔌 Initialisiere Neural Scout (V73.0 - Dynamic AI Authority)...")

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

# =================================================================
# GEMINI ENGINE
# =================================================================
async def call_gemini(prompt):
    await asyncio.sleep(1.0) # Rate Limit Schutz
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200: return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except: return None

# =================================================================
# CORE LOGIC: ELO & DB
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

async def get_db_data():
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        
        # Normalize Skills
        clean_skills = {}
        for entry in skills:
            pid = entry.get('player_id')
            if pid:
                clean_skills[pid] = {
                    'serve': to_float(entry.get('serve')), 'power': to_float(entry.get('power')),
                    'forehand': to_float(entry.get('forehand')), 'backhand': to_float(entry.get('backhand')),
                    'speed': to_float(entry.get('speed')), 'stamina': to_float(entry.get('stamina')),
                    'mental': to_float(entry.get('mental'))
                }
        return players, clean_skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], {}, [], []

# =================================================================
# MATH CORE V5 (AGENTIC WEIGHTING)
# =================================================================
def sigmoid_prob(diff, sensitivity=0.1):
    return 1 / (1 + math.exp(-sensitivity * diff))

def calculate_physics_fair_odds(p1_name, p2_name, s1, s2, bsi, surface, ai_meta):
    """
    V5 Engine: Dynamic Authority.
    Wenn die AI einen 'Decisive Advantage' sieht, zählt ihre Meinung doppelt so viel wie die DB-Skills.
    """
    n1 = p1_name.lower().split()[-1] 
    n2 = p2_name.lower().split()[-1]
    tour = "ATP" 
    
    # --- 1. AI TACTICAL SCORE (THE BOSS METRIC) ---
    m1 = to_float(ai_meta.get('p1_tactical_score', 5))
    m2 = to_float(ai_meta.get('p2_tactical_score', 5))
    
    # Berechne die "Dominanz" der AI-Meinung (Wie sicher ist sie sich?)
    # Differenz z.B. |8 - 3| = 5.
    ai_confidence_gap = abs(m1 - m2)
    
    # DYNAMIC WEIGHTING LOGIC:
    # Hoher Gap -> AI hat Recht, DB Skills sind unwichtig.
    # Niedriger Gap -> Wir vertrauen den langfristigen Stats.
    if ai_confidence_gap >= 3.0:
        weight_matchup = 0.50  # AI entscheidet zu 50%
        weight_skills = 0.15   # DB Skills fast irrelevant
        log(f"   🤖 AI Dominance Override: Trusting Text Analysis over Stats for {p1_name} vs {p2_name}")
    else:
        weight_matchup = 0.25  # Standard
        weight_skills = 0.35   # Standard
    
    prob_matchup = sigmoid_prob(m1 - m2, sensitivity=0.55)

    # --- 2. CONTEXTUAL SKILLS ---
    bsi_val = to_float(bsi, 6.0)
    
    def get_contextual_score(skills):
        if not skills: return 300.0
        # Einfache Anpassung: Auf Clay (BSI<4) zählt Speed/Stamina doppelt, Power halb.
        if bsi_val <= 4.0:
            return (skills.get('speed',50)*1.5 + skills.get('stamina',50)*1.5 + skills.get('mental',50)*1.2 + skills.get('serve',50)*0.5)
        elif bsi_val >= 7.5:
            return (skills.get('serve',50)*1.5 + skills.get('power',50)*1.5 + skills.get('mental',50)*1.0)
        return sum(skills.values()) # Balanced

    score1 = get_contextual_score(s1)
    score2 = get_contextual_score(s2)
    prob_skills = sigmoid_prob(score1 - score2, sensitivity=0.10)

    # --- 3. ELO & REST ---
    elo1 = 1500.0; elo2 = 1500.0
    elo_surf = 'Clay' if 'clay' in surface.lower() else ('Grass' if 'grass' in surface.lower() else 'Hard')
    for name, stats in ELO_CACHE.get(tour, {}).items():
        if n1 in name: elo1 = stats.get(elo_surf, 1500.0)
        if n2 in name: elo2 = stats.get(elo_surf, 1500.0)
    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    f1 = to_float(ai_meta.get('p1_form_score', 5))
    f2 = to_float(ai_meta.get('p2_form_score', 5))
    prob_form = sigmoid_prob(f1 - f2, sensitivity=0.45)

    u1 = to_float(ai_meta.get('p1_utr', 10.0))
    u2 = to_float(ai_meta.get('p2_utr', 10.0))
    prob_utr = sigmoid_prob(u1 - u2, sensitivity=0.9)

    # --- 4. COURT FIT (Simple) ---
    # Bonus für den, dessen Stil zur BSI passt (wird oft durch Skills abgedeckt, aber hier als kleiner Bonus)
    prob_court = 0.5 
    if bsi_val <= 4.0 and "grinder" in str(ai_meta).lower(): # Wenn AI sagt "Grinder" und Clay
         # Wir müssten wissen WER der Grinder ist. Vereinfacht: wir nutzen die Scores von oben.
         # Da prob_skills schon BSI enthält, lassen wir prob_court neutraler oder nutzen es als "Tie-Breaker"
         pass

    # --- FINAL MIX ---
    # Wir nutzen die dynamischen Gewichte von oben
    raw_prob = (
        (prob_matchup * weight_matchup) + 
        (prob_skills * weight_skills) + 
        (prob_elo * 0.15) +
        (prob_form * 0.10) +
        (prob_utr * 0.05) +
        (prob_skills * 0.05) # Rest auf Skills als Proxy für Court
    )
    
    # --- 5. THE "PEACEFUL ODDS" KILLER ---
    # Wenn die Wahrscheinlichkeit klar ist, drück sie weiter auseinander.
    # Simuliert, dass ein 80% Favorit in echt oft eine 1.10 Quote hat (90%).
    if raw_prob > 0.65:
        adjusted_prob = raw_prob + (raw_prob * 0.10) # Boost Favorit
        adjusted_prob = min(adjusted_prob, 0.95)
    elif raw_prob < 0.35:
        adjusted_prob = raw_prob - (raw_prob * 0.10) # Crush Underdog
        adjusted_prob = max(adjusted_prob, 0.05)
    else:
        adjusted_prob = raw_prob

    return adjusted_prob

# =================================================================
# RESULT VERIFICATION ENGINE
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Yesterday & Today)...")
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending_matches:
        log("   ✅ No pending matches to verify.")
        return

    for day_offset in range(2): 
        target_date = datetime.now() - timedelta(days=day_offset)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                await page.goto(url, wait_until="networkidle", timeout=60000)
                content = await page.content()
                await browser.close()
                soup = BeautifulSoup(content, 'html.parser')
                relevant_rows = []
                for row in soup.find_all("tr"):
                    row_text = row.get_text(separator=" ", strip=True)
                    for pm in pending_matches:
                        if pm['player1_name'] in row_text and pm['player2_name'] in row_text:
                            relevant_rows.append(row_text)
                if not relevant_rows: continue

                prompt = f"""
                ANALYZE TENNIS RESULTS. Input: {json.dumps(relevant_rows)}
                Task: Identify WINNER. Output JSON: [ {{ "p1": "Name", "p2": "Name", "winner_lastname": "Name" }} ]
                """
                res = await call_gemini(prompt)
                if res:
                    try:
                        results = json.loads(res.replace("```json", "").replace("```", "").strip())
                        for r in results:
                            winner = r.get('winner_lastname')
                            if winner:
                                for pm in pending_matches:
                                    if winner in pm['player1_name']:
                                        supabase.table("market_odds").update({"actual_winner_name": pm['player1_name']}).eq("id", pm['id']).execute()
                                        log(f"   🎉 Result: {pm['player1_name']} won")
                                    elif winner in pm['player2_name']:
                                        supabase.table("market_odds").update({"actual_winner_name": pm['player2_name']}).eq("id", pm['id']).execute()
                                        log(f"   🎉 Result: {pm['player2_name']} won")
                    except: pass
            except: await browser.close()

# =================================================================
# MAIN PIPELINE
# =================================================================
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
    
    log(f"   🤖 AI resolving location for {p1} vs {p2}...")
    ai_loc = await resolve_ambiguous_tournament(p1, p2, tour)
    if ai_loc and ai_loc.get('city'):
        city = ai_loc['city'].lower()
        surf = ai_loc.get('surface_guessed', 'Hard')
        for t in db_tours:
            if city in t['name'].lower(): return t['surface'], t['bsi_rating'], f"AI: {city}"
        return surf, (3.5 if 'clay' in surf.lower() else 6.5), f"AI Guess: {city}"
    return 'Hard', 6.5, 'Fallback'

async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    # PROMPT UPGRADE: Enforce DECISIVE scoring for mismatches
    prompt = f"""
    ROLE: Elite Tennis Analyst. TASK: {p1['last_name']} vs {p2['last_name']}.
    CTX: {surface} (BSI {bsi}). 
    P1 ({p1.get('play_style')}) vs P2 ({p2.get('play_style')}).
    
    INSTRUCTION: If one player's style is a PERFECT match for the surface/opponent (e.g. Grinder on Clay vs Big Server), you MUST give them a high Tactical Score (8-9) and the opponent low (3-4).
    
    METRICS (0-10): 
    1. TACTICAL: Style clash efficiency.
    2. FORM: Recent momentum.
    3. UTR: Class difference.
    
    JSON ONLY: {{ "p1_tactical_score": 9, "p2_tactical_score": 3, "p1_form_score": 8, "p2_form_score": 4, "p1_utr": 14.2, "p2_utr": 13.8, "ai_text": "..." }}
    """
    res = await call_gemini(prompt)
    d = {'p1_tactical_score': 5, 'p2_tactical_score': 5, 'p1_form_score': 5, 'p2_form_score': 5, 'p1_utr': 10, 'p2_utr': 10}
    if not res: return d
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return d

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            log(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            log(f"❌ Scrape Error: {e}")
            await browser.close()
            return None

def parse_matches_locally(html, p_names):
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
                    found.append({"p1": p1_raw, "p2": p2_raw, "tour": current_tour, "odds1": odds[0] if odds else 0.0, "odds2": odds[1] if len(odds)>1 else 0.0})
    return found

async def run_pipeline():
    log(f"🚀 Neural Scout v73.0 (Dynamic AI Authority) Starting...")
    await update_past_results()
    await fetch_elo_ratings()
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    player_names = [p['last_name'] for p in players]
    
    for day_offset in range(35): 
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        matches = parse_matches_locally(html, player_names)
        log(f"🔍 Gefunden: {len(matches)} Matches am {target_date.strftime('%d.%m.')}")
        
        for m in matches:
            try:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    m_odds1 = m['odds1']
                    m_odds2 = m['odds2']
                    
                    # LOGIC: Check Existing & UPDATE AI ANALYSIS if needed
                    existing = supabase.table("market_odds").select("id").eq("player1_name", p1_obj['last_name']).eq("player2_name", p2_obj['last_name']).execute()
                    if existing.data:
                        # Optional: Force re-analysis if odds changed significantly? For now just update odds.
                        supabase.table("market_odds").update({"odds1": m_odds1, "odds2": m_odds2}).eq("id", existing.data[0]['id']).execute()
                        log(f"🔄 Update: {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        continue

                    if m_odds1 <= 1.0: 
                        log(f"⏩ Skip New (No Odds): {p1_obj['last_name']} vs {p2_obj['last_name']}")
                        continue
                    
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
                        "odds1": m_odds1, "odds2": m_odds2,
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
