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

log("🔌 Initialisiere Neural Scout (Gemini 2.5 Deep Sync)...")

# 1. Keys laden
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MODEL: HIGH END ---
MODEL_NAME = 'gemini-2.5-pro' 

# =================================================================
# GEMINI API ENGINE
# =================================================================
async def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json", 
            "temperature": 0.3 # Präzise aber kreativ genug für Text
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=90.0) # Mehr Zeit für Deep Analysis
            if response.status_code != 200:
                log(f"⚠️ Gemini API Error {response.status_code}: {response.text}")
                return None
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            log(f"⚠️ Gemini Network Error: {e}")
            return None

# =================================================================
# DATA LOADING
# =================================================================
async def get_db_data():
    log("📥 Lade Deep Data (Skills & Reports)...")
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
    # 1. Exakt
    for t in db_tournaments:
        if t['name'].lower() == scraped_lower:
            return t['surface'], t['bsi_rating'], t.get('notes', '')
    # 2. Fuzzy
    best_candidate = None
    for t in db_tournaments:
        db_name = t['name'].lower()
        if db_name in scraped_lower or scraped_lower in db_name:
            if best_candidate is None or len(db_name) < len(best_candidate['name']):
                best_candidate = t
    if best_candidate:
        return best_candidate['surface'], best_candidate['bsi_rating'], best_candidate.get('notes', '')
    # 3. Fallback
    if 'indoor' in scraped_lower: return 'Indoor', 8.2, 'Fast Indoor'
    if any(x in scraped_lower for x in ['clay', 'sand', 'roland']): return 'Red Clay', 3.5, 'Slow Clay'
    return 'Hard', 6.5, 'Standard Hard'

# =================================================================
# MATH CORE (Synchronisiert mit Frontend V60)
# =================================================================
def calculate_sophisticated_fair_odds(s1, s2, bsi):
    # VETERAN MATH: Identisch zum Frontend Code für Konsistenz
    is_fast = bsi >= 7
    is_slow = bsi <= 4
    
    # Gewichtung je nach Court Speed
    w_serve = 1.0; w_baseline = 1.0; w_mental = 0.8; w_physical = 0.8
    
    if is_fast:
        w_serve = 2.4      # Aufschlag dominiert
        w_baseline = 0.6
        w_mental = 0.7
        w_physical = 0.3
    elif is_slow:
        w_serve = 0.5
        w_baseline = 1.5   # Grind dominiert
        w_mental = 1.3     # Geduld ist wichtig
        w_physical = 1.6
        
    # Safe Gets mit Defaults
    s1_serve = s1.get('serve', 50); s1_power = s1.get('power', 50)
    s2_serve = s2.get('serve', 50); s2_power = s2.get('power', 50)
    
    s1_base = s1.get('forehand', 50) + s1.get('backhand', 50)
    s2_base = s2.get('forehand', 50) + s2.get('backhand', 50)
    
    s1_phys = s1.get('speed', 50) + s1.get('stamina', 50)
    s2_phys = s2.get('speed', 50) + s2.get('stamina', 50)

    # Differenzen berechnen
    serve_diff = ((s1_serve + s1_power) - (s2_serve + s2_power)) * w_serve
    baseline_diff = (s1_base - s2_base) * w_baseline
    physical_diff = (s1_phys - s2_phys) * w_physical
    mental_diff = (s1.get('mental', 50) - s2.get('mental', 50)) * w_mental

    # Total Score (Divisor 160 für etwas höhere Scores -> Green Zone)
    total_advantage = (serve_diff + baseline_diff + physical_diff + mental_diff) / 160
    
    # Klassen-Unterschied (Overall Rating Bonus)
    class_diff = (s1.get('overall_rating', 50) - s2.get('overall_rating', 50)) / 18
    
    total_score = total_advantage + class_diff
    
    # Sigmoid für Wahrscheinlichkeit
    try:
        prob_p1 = 1 / (1 + math.exp(-0.75 * total_score))
    except OverflowError:
        prob_p1 = 0.99 if total_score > 0 else 0.01
        
    return prob_p1

# =================================================================
# AI ANALYSIS (Deep Structure)
# =================================================================
async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    # Prompt synced mit Frontend-Logik
    prompt = f"""
    ROLE: Elite Tennis Tactical Analyst.
    TASK: Deep Match Analysis: {p1['last_name']} vs {p2['last_name']}.
    
    ### CONTEXT
    - Surface: {surface}
    - Speed Index: {bsi}/10 (1=Slow, 10=Fast).
    - Court Notes: {notes}
    
    ### PLAYER A: {p1['last_name']}
    - Stats: Srv {s1.get('serve')}, FH {s1.get('forehand')}, BH {s1.get('backhand')}, Men {s1.get('mental')}.
    - Report: {r1.get('strengths', 'N/A')} (Pros), {r1.get('weaknesses', 'N/A')} (Cons).
    
    ### PLAYER B: {p2['last_name']}
    - Stats: Srv {s2.get('serve')}, FH {s2.get('forehand')}, BH {s2.get('backhand')}, Men {s2.get('mental')}.
    - Report: {r2.get('strengths', 'N/A')} (Pros), {r2.get('weaknesses', 'N/A')} (Cons).
    
    ### OUTPUT REQUIREMENT
    Write a sophisticated 3-paragraph analysis (approx 150 words):
    1. Analysis of Player A's weapons on this specific surface.
    2. Analysis of Player B's counter-play and vulnerabilities.
    3. The Clash: How the court speed (BSI {bsi}) decides the winner. Mention specific shots.
    
    OUTPUT JSON ONLY:
    {{
        "deep_analysis": "The full 3-paragraph text here...",
        "win_probability_p1": 0.XX
    }}
    """
    
    res = await call_gemini(prompt)
    if not res: return 0.5, "AI Timeout - Defaulting to Statistical Analysis."
    
    try:
        clean_json = res.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data.get('win_probability_p1', 0.5), data.get('deep_analysis', 'Analysis unavailable.')
    except:
        return 0.5, "AI Parse Error."

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
    log("🚀 Neural Scout v60 (Deep Sync Edition) Starting...")
    
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

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
                    
                    # 2. MATH ODDS (Synchronisiert)
                    prob_p1 = calculate_sophisticated_fair_odds(s1, s2, bsi)
                    
                    # 3. AI DEEP ANALYSIS
                    ai_prob_raw, ai_text = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    # 4. HYBRID CALCULATION (50/50 Split)
                    final_prob_p1 = (prob_p1 * 0.5) + (ai_prob_raw * 0.5)
                    
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
                        "ai_analysis_text": ai_text, # Speichert jetzt den langen 3-Absatz Text
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    
                    log(f"💾 Saving: {p1_obj['last_name']} vs {p2_obj['last_name']} ({fair_odds1} vs {fair_odds2})")
                    supabase.table("market_odds").upsert(match_entry, on_conflict="player1_name, player2_name, tournament").execute()

        except Exception as e:
            log(f"⚠️ Process Error: {e}")

    log("🏁 Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
