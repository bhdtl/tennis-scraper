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
# CONFIGURATION
# =================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("🔌 Initialisiere Neural Scout (Gemini 2.5 Pro Core)...")

# 1. Keys laden
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

log(f"🔑 Key Check: URL={'OK' if SUPABASE_URL else 'MISSING'}, DB_KEY={'OK' if SUPABASE_KEY else 'MISSING'}, GEMINI={'OK' if GEMINI_API_KEY else 'MISSING'}")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY in GitHub.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- DEIN WUNSCH-MODELL ---
MODEL_NAME = 'gemini-2.5-pro' 

# =================================================================
# GEMINI API ENGINE (REST)
# =================================================================
async def call_gemini(prompt):
    # API URL für das spezifische Modell
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
            # 60s Timeout für das große Pro Modell
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                log(f"⚠️ Gemini API Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
            
        except Exception as e:
            log(f"⚠️ Gemini Network Error: {e}")
            return None

# =================================================================
# DATA LOADING (Deine DB Logik)
# =================================================================
async def get_db_data():
    log("📥 Lade Datenbank-Daten...")
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        log(f"✅ Geladen: {len(players)} Spieler, {len(skills)} Skills, {len(tournaments)} Turniere.")
        return players, skills, reports, tournaments
    except Exception as e:
        log(f"❌ DB Load Error: {e}")
        return [], [], [], []

def find_best_court_match(scraped_tour_name, db_tournaments):
    scraped_lower = scraped_tour_name.lower().strip()
    
    # 1. Exakter Match
    for t in db_tournaments:
        if t['name'].lower() == scraped_lower:
            return t['surface'], t['bsi_rating'], t.get('notes', '')

    # 2. Fuzzy Match
    best_candidate = None
    for t in db_tournaments:
        db_name = t['name'].lower()
        if db_name in scraped_lower or scraped_lower in db_name:
            if best_candidate is None or len(db_name) < len(best_candidate['name']):
                best_candidate = t
    
    if best_candidate:
        return best_candidate['surface'], best_candidate['bsi_rating'], best_candidate.get('notes', '')

    # 3. Fallback
    if 'indoor' in scraped_lower: return 'Indoor', 8.2, 'Fast Indoor fallback'
    if any(x in scraped_lower for x in ['clay', 'sand', 'roland']): return 'Red Clay', 3.5, 'Slow Clay fallback'
    return 'Hard', 6.5, 'Standard Hard fallback'

# =================================================================
# MATH CORE
# =================================================================
def calculate_math_odds(s1, s2, bsi):
    is_fast = bsi >= 7
    is_slow = bsi <= 4
    w_serve = 2.2 if is_fast else (0.6 if is_slow else 1.0)
    w_baseline = 0.7 if is_fast else (1.4 if is_slow else 1.0)
    w_mental = 1.2
    
    serve_val1 = s1.get('serve', 50) + s1.get('power', 50)
    serve_val2 = s2.get('serve', 50) + s2.get('power', 50)
    serve_diff = (serve_val1 - serve_val2) * w_serve
    
    base_val1 = s1.get('forehand', 50) + s1.get('backhand', 50)
    base_val2 = s2.get('forehand', 50) + s2.get('backhand', 50)
    base_diff = (base_val1 - base_val2) * w_baseline
    
    mental_diff = (s1.get('mental', 50) - s2.get('mental', 50)) * w_mental
    total_score = (serve_diff + base_diff + mental_diff) / 200
    
    return 1 / (1 + math.exp(-0.7 * (6.0 + total_score - 6.0)))

# =================================================================
# AI ANALYSIS
# =================================================================
async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    prompt = f"""
    ROLE: Elite Tennis Analyst.
    MATCH: {p1['last_name']} vs {p2['last_name']}
    COURT: {surface} (Speed BSI: {bsi}/10). Notes: {notes}
    
    P1 STATS: Srv {s1.get('serve')}, FH {s1.get('forehand')}, BH {s1.get('backhand')}, Men {s1.get('mental')}.
    P1 SCOUT: {r1.get('strengths', 'N/A')} (Pros), {r1.get('weaknesses', 'N/A')} (Cons).
    
    P2 STATS: Srv {s2.get('serve')}, FH {s2.get('forehand')}, BH {s2.get('backhand')}, Men {s2.get('mental')}.
    P2 SCOUT: {r2.get('strengths', 'N/A')} (Pros), {r2.get('weaknesses', 'N/A')} (Cons).
    
    TASK: Analyze matchup based on court speed.
    OUTPUT JSON: {{"analysis_brief": "One sharp tactical sentence.", "p1_win_probability": 0.XX}}
    """
    res = await call_gemini(prompt)
    if not res: return 0.5, "AI Timeout"
    try:
        data = json.loads(res.replace("```json", "").replace("```", "").strip())
        return data.get('p1_win_probability', 0.5), data.get('analysis_brief', 'No analysis')
    except: return 0.5, "AI Parse Error"

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
    log(f"🚀 Neural Scout v57 (Model: {MODEL_NAME}) Starting...")
    
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: 
        log("⚠️ Keine Spieler in DB gefunden.")
        return

    current_date = datetime.now()
    
    # 35 Tage Future Scan
    for day_offset in range(35): 
        target_date = current_date + timedelta(days=day_offset)
        
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        cleaned_text = clean_html_for_extraction(html)
        if not cleaned_text: continue

        player_names = [p['last_name'] for p in players]
        
        extract_prompt = f"""
        Extract matches where BOTH players are in this list: {json.dumps(player_names)}
        Input Text: {cleaned_text[:20000]}
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
                    
                    # 1. COURT MATCHING
                    surf, bsi, notes = find_best_court_match(m['tour'], all_tournaments)
                    log(f"   🎾 Match: {m['tour']} -> BSI {bsi} ({surf})")
                    
                    # 2. CALCULATION
                    math_prob = calculate_math_odds(s1, s2, bsi)
                    ai_prob, ai_reason = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    final_prob_p1 = (math_prob * 0.5) + (ai_prob * 0.5)
                    fair_odds1 = round(1 / final_prob_p1, 2) if final_prob_p1 > 0.01 else 99.0
                    fair_odds2 = round(1 / (1 - final_prob_p1), 2) if final_prob_p1 < 0.99 else 99.0
                    
                    match_entry = {
                        "player1_name": p1_obj['last_name'],
                        "player2_name": p2_obj['last_name'],
                        "tournament": m['tour'],
                        "odds1": m['odds1'],
                        "odds2": m['odds2'],
                        "ai_fair_odds1": fair_odds1,
                        "ai_fair_odds2": fair_odds2,
                        "ai_analysis_text": f"[{surf}, BSI {bsi}] {ai_reason}",
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    
                    supabase.table("market_odds").upsert(match_entry, on_conflict="player1_name, player2_name, tournament").execute()

        except Exception as e:
            log(f"⚠️ Verarbeitungsfehler: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
