# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import math
import logging
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx

# =================================================================
# CONFIGURATION
# =================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SWITCH: GEMINI KEYS
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: API Keys fehlen (Prüfe GEMINI_API_KEY in Secrets)!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Wir nutzen das stärkste verfügbare Modell für die Analyse
MODEL_NAME = 'gemini-1.5-pro' 

# =================================================================
# GEMINI ENGINE (The New Motor)
# =================================================================
async def call_gemini(prompt):
    """Sendet Anfragen an die Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    
    # Gemini Payload Struktur
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "response_mime_type": "application/json", # Zwingt Gemini zu sauberem JSON
            "temperature": 0.2
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            # Gemini Pro kann bei komplexen Analysen etwas dauern, daher 60s Timeout
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            # Extrahiere die Antwort
            return data['candidates'][0]['content']['parts'][0]['text']
            
        except Exception as e:
            logger.error(f"Gemini Network Error: {e}")
            return None

# =================================================================
# DATA LOADING & MATCHING LOGIC (Preserved)
# =================================================================
async def get_db_data():
    try:
        players = supabase.table("players").select("*").execute().data
        skills = supabase.table("player_skills").select("*").execute().data
        reports = supabase.table("scouting_reports").select("*").execute().data
        tournaments = supabase.table("tournaments").select("*").execute().data
        return players, skills, reports, tournaments
    except Exception as e:
        logger.error(f"DB Load Error: {e}")
        return [], [], [], []

def find_best_court_match(scraped_tour_name, db_tournaments):
    """
    Silicon Valley Logic für United Cup & Co.
    """
    scraped_lower = scraped_tour_name.lower().strip()
    
    # 1. Priorität: Exakter Match (z.B. "United Cup (Sydney)")
    for t in db_tournaments:
        if t['name'].lower() == scraped_lower:
            return t['surface'], t['bsi_rating'], t.get('notes', '')

    # 2. Priorität: "Contains" Match (Fuzzy)
    best_candidate = None
    for t in db_tournaments:
        db_name = t['name'].lower()
        # Findet "United Cup" in "United Cup (Perth)" oder umgekehrt
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
# MATH CORE (Preserved)
# =================================================================
def calculate_math_odds(s1, s2, bsi):
    is_fast = bsi >= 7
    is_slow = bsi <= 4
    w_serve = 2.2 if is_fast else (0.6 if is_slow else 1.0)
    w_baseline = 0.7 if is_fast else (1.4 if is_slow else 1.0)
    w_mental = 1.2
    
    # Safe Get mit Default 50
    serve_val1 = s1.get('serve', 50) + s1.get('power', 50)
    serve_val2 = s2.get('serve', 50) + s2.get('power', 50)
    serve_diff = (serve_val1 - serve_val2) * w_serve
    
    base_val1 = s1.get('forehand', 50) + s1.get('backhand', 50)
    base_val2 = s2.get('forehand', 50) + s2.get('backhand', 50)
    base_diff = (base_val1 - base_val2) * w_baseline
    
    mental_diff = (s1.get('mental', 50) - s2.get('mental', 50)) * w_mental
    
    total_score = (serve_diff + base_diff + mental_diff) / 200
    
    # Sigmoid Wahrscheinlichkeit
    return 1 / (1 + math.exp(-0.7 * (6.0 + total_score - 6.0)))

# =================================================================
# AI ANALYSIS (Updated for Gemini)
# =================================================================
async def analyze_match_with_ai(p1, p2, s1, s2, r1, r2, surface, bsi, notes):
    prompt = f"""
    ROLE: Elite Tennis Analyst.
    MATCH: {p1['last_name']} vs {p2['last_name']}.
    COURT: {surface} (Speed BSI: {bsi}/10). Notes: {notes}
    
    PLAYER A ({p1['last_name']}):
    - Skills: Srv {s1.get('serve')}, FH {s1.get('forehand')}, BH {s1.get('backhand')}, Men {s1.get('mental')}.
    - Scout: {r1.get('strengths', 'N/A')} (Pros), {r1.get('weaknesses', 'N/A')} (Cons).
    
    PLAYER B ({p2['last_name']}):
    - Skills: Srv {s2.get('serve')}, FH {s2.get('forehand')}, BH {s2.get('backhand')}, Men {s2.get('mental')}.
    - Scout: {r2.get('strengths', 'N/A')} (Pros), {r2.get('weaknesses', 'N/A')} (Cons).
    
    TASK: Analyze based on the specific court speed (BSI {bsi}). Does the surface favor the big server or the grinder?
    
    OUTPUT JSON ONLY:
    {{
        "analysis_brief": "One sharp tactical sentence focusing on court speed vs weapons.",
        "p1_win_probability": 0.XX
    }}
    """
    
    res = await call_gemini(prompt)
    if not res: return 0.5, "AI Timeout"
    
    try:
        # Gemini packt JSON manchmal in Markdown Blöcke, wir reinigen das
        clean_json = res.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data.get('p1_win_probability', 0.5), data.get('analysis_brief', 'No analysis')
    except:
        return 0.5, "AI Parse Error"

# =================================================================
# SCRAPER CORE (Preserved)
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
            logger.info(f"📡 Scanning: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try: await page.wait_for_selector(".result", timeout=10000)
            except: 
                await browser.close()
                return None
            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            logger.error(f"Scrape Error: {e}")
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
    logger.info("🚀 Neural Scout v53 (Gemini Pro - 35 Day Future Scan) Starting...")
    
    # 1. Daten laden
    players, all_skills, all_reports, all_tournaments = await get_db_data()
    if not players: return

    current_date = datetime.now()
    
    # 35 Tage Loop für United Cup und AO Vorbereitung
    for day_offset in range(35): 
        target_date = current_date + timedelta(days=day_offset)
        
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        cleaned_text = clean_html_for_extraction(html)
        if not cleaned_text: continue

        # Filter für Gemini
        player_names = [p['last_name'] for p in players]
        
        extract_prompt = f"""
        Extract matches where BOTH players are in this list: {json.dumps(player_names)}
        Input Text: {cleaned_text[:20000]}
        
        OUTPUT JSON: 
        {{ 
            "matches": [ 
                {{ "p1": "Lastname", "p2": "Lastname", "tour": "Tour Name (Full)", "odds1": 1.5, "odds2": 2.5 }} 
            ] 
        }}
        If odds missing, set to 0.
        """
        
        # 2. Extrahieren mit Gemini
        extract_res = await call_gemini(extract_prompt)
        if not extract_res: continue

        try:
            clean_json = extract_res.replace("```json", "").replace("```", "").strip()
            matches = json.loads(clean_json).get("matches", [])
            
            for m in matches:
                p1_obj = next((p for p in players if p['last_name'] in m['p1']), None)
                p2_obj = next((p for p in players if p['last_name'] in m['p2']), None)
                
                if p1_obj and p2_obj:
                    s1 = next((s for s in all_skills if s['player_id'] == p1_obj['id']), {})
                    s2 = next((s for s in all_skills if s['player_id'] == p2_obj['id']), {})
                    r1 = next((r for r in all_reports if r['player_id'] == p1_obj['id']), {})
                    r2 = next((r for r in all_reports if r['player_id'] == p2_obj['id']), {})
                    
                    # 3. SMART COURT MATCHING
                    surf, bsi, notes = find_best_court_match(m['tour'], all_tournaments)
                    
                    # 4. MATH PROB
                    math_prob = calculate_math_odds(s1, s2, bsi)
                    
                    # 5. GEMINI DEEP ANALYSIS
                    ai_prob, ai_reason = await analyze_match_with_ai(p1_obj, p2_obj, s1, s2, r1, r2, surf, bsi, notes)
                    
                    # 6. HYBRID ODDS CALCULATION
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
                    
                    logger.info(f"✅ Analyzed: {p1_obj['last_name']} vs {p2_obj['last_name']} @ {m['tour']} | AI: {ai_prob}")
                    supabase.table("market_odds").upsert(match_entry, on_conflict="player1_name, player2_name, tournament").execute()

        except Exception as e:
            logger.error(f"Processing Error: {e}")

    logger.info("🏁 35-Day Gemini Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
