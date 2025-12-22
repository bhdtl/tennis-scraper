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
# CONFIGURATION & DEBUGGING
# =================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Keys aus den Environment Variables laden
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Liest entweder SUPABASE_KEY oder SUPABASE_SERVICE_ROLE_KEY (Sicherheitsnetz)
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# --- DEBUG DIAGNOSE START (Das ist neu, um den Fehler zu finden) ---
print("\n" + "="*40)
print("🔍 DEBUG: PRÜFUNG DER GITHUB SECRETS")
print(f"1. GROQ_API_KEY:   {'✅ VORHANDEN' if GROQ_API_KEY else '❌ FEHLT'}")
print(f"2. SUPABASE_URL:   {'✅ VORHANDEN' if SUPABASE_URL else '❌ FEHLT'}")
print(f"3. SUPABASE_KEY:   {'✅ VORHANDEN' if SUPABASE_KEY else '❌ FEHLT'}")
print("="*40 + "\n")
# --- DEBUG DIAGNOSE ENDE ---

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    # Wir sagen dem Fehler genau, was fehlt
    missing = []
    if not GROQ_API_KEY: missing.append("GROQ_API_KEY")
    if not SUPABASE_URL: missing.append("SUPABASE_URL")
    if not SUPABASE_KEY: missing.append("SUPABASE_KEY")
    raise ValueError(f"CRITICAL: Abbruch! Folgende Keys fehlen: {', '.join(missing)}")

# Client Initialisierung (nur wenn Keys da sind)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MODEL_NAME = 'llama-3.3-70b-versatile'

# =================================================================
# MATH CORE (Unverändert)
# =================================================================
def calculate_sophisticated_fair_odds(s1, s2, bsi):
    is_fast = bsi >= 7
    is_slow = bsi <= 4
    w_serve = 2.2 if is_fast else (0.6 if is_slow else 1.0)
    w_baseline = 0.7 if is_fast else (1.4 if is_slow else 1.0)
    w_mental = 0.6 if is_fast else (1.2 if is_slow else 0.8)
    w_physical = 0.4 if is_fast else (1.5 if is_slow else 0.8)
    
    # Safe Get mit Default 50, falls Daten fehlen
    serve_diff = ((s1.get('serve', 50) + s1.get('power', 50)) - (s2.get('serve', 50) + s2.get('power', 50))) * w_serve
    baseline_diff = ((s1.get('forehand', 50) + s1.get('backhand', 50)) - (s2.get('forehand', 50) + s2.get('backhand', 50))) * w_baseline
    physical_diff = ((s1.get('speed', 50) + s1.get('stamina', 50)) - (s2.get('speed', 50) + s2.get('stamina', 50))) * w_physical
    mental_diff = (s1.get('mental', 50) - s2.get('mental', 50)) * w_mental
    
    total_advantage_a = (serve_diff + baseline_diff + physical_diff + mental_diff) / 200
    class_diff = (s1.get('overall_rating', 50) - s2.get('overall_rating', 50)) / 25
    
    rating_a = 6.0 + total_advantage_a + class_diff
    rating_b = 6.0 - total_advantage_a - class_diff
    
    try:
        return 1 / (1 + math.exp(-0.7 * (rating_a - rating_b)))
    except OverflowError:
        return 0.99 if rating_a > rating_b else 0.01

def detect_surface_config(tournament_name):
    name = (tournament_name or '').lower()
    if 'indoor' in name: return 'Indoor', 8.2
    if any(x in name for x in ['clay', 'sand', 'roland garros']): return 'Red Clay', 3.5
    if any(x in name for x in ['hard', 'australian', 'us open']): return 'Hard', 6.5
    return 'Hard', 5.0

# =================================================================
# API & SCRAPING HELPERS
# =================================================================
async def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            if response.status_code != 200:
                logger.error(f"Groq Error {response.status_code}: {response.text}")
                return None
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Groq Connection Error: {e}")
            return None

def normalize_text(text):
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw):
    noise = [r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'beIN Sports', r'bet365', r'Unibet', r'William Hill']
    for pat in noise: raw = re.sub(pat, '', raw, flags=re.IGNORECASE)
    return raw.replace('|', '').strip()

async def get_known_players():
    try:
        response = supabase.table("players").select("*").execute()
        return response.data
    except Exception as e:
        logger.error(f"DB Error (get_known_players): {e}")
        return []

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        # Headless mode für Server-Umgebung
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            logger.info(f"📡 Scanning Date: {target_date.strftime('%Y-%m-%d')}")
            
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # Kurz warten, ob Tabelle geladen wird
            try:
                await page.wait_for_selector(".result", timeout=10000)
            except:
                logger.warning("Keine Ergebnisse auf der Seite gefunden.")
                await browser.close()
                return None

            content = await page.content()
            await browser.close()
            return content
        except Exception as e:
            logger.error(f"Playwright Error: {e}")
            await browser.close()
            return None

def clean_html_for_ai(html_content):
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
            # Einfacher Check: Ist es eine Match-Zeile? (Hat Zeitformat HH:MM)
            if re.search(r'\d{2}:\d{2}', row_text) and i+1 < len(rows):
                p1_raw = clean_player_name(row_text)
                # Der Gegner steht oft in der nächsten Zeile bei TennisExplorer
                p2_raw = clean_player_name(normalize_text(rows[i+1].get_text(separator=' | ', strip=True)))
                
                txt += f"TOURNAMENT: {current_tour} | MATCH: {p1_raw} VS {p2_raw}\n"
    return txt

# =================================================================
# MAIN PIPELINE
# =================================================================
async def run_pipeline():
    logger.info("🚀 Sync Cycle Start (Groq Future Scout)...")
    
    known_players = await get_known_players()
    if not known_players:
        logger.error("⚠️ Abbruch: Keine Spieler in der Datenbank gefunden.")
        return

    # Lade Skills für Berechnung
    skills_res = supabase.table("player_skills").select("*").execute()
    all_skills = skills_res.data if skills_res.data else []

    current_date = datetime.now()
    
    # Scanne die nächsten 7 Tage (Future Scan)
    for day_offset in range(7):
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue

        cleaned_text = clean_html_for_ai(html)
        if not cleaned_text: continue

        # Filter: Wir suchen nur Matches mit unseren bekannten Spielern
        # Das spart AI-Tokens und verhindert Halluzinationen
        player_names = [p['last_name'] for p in known_players]
        
        prompt = f"""
        ### ROLE: Data Auditor
        Extract matches from the text below where BOTH players exist in this list: {json.dumps(player_names)}
        
        TEXT DATA:
        {cleaned_text[:15000]} 
        
        RULES: 
        1. If odds are missing or results like '6-4', set odds1 and odds2 to 0.00.
        2. Format match_time as HH:MM
        
        OUTPUT JSON ONLY:
        {{
            "matches": [
                {{
                    "player1_last_name": "Name",
                    "player2_last_name": "Name",
                    "tournament": "Tournament Name",
                    "match_time": "HH:MM",
                    "odds1": 1.50,
                    "odds2": 2.50
                }}
            ]
        }}
        """

        ai_res = await call_groq(prompt)
        if not ai_res: continue

        try:
            data = json.loads(ai_res)
            matches = data.get("matches", [])
            
            for m in matches:
                # Finde Spieler IDs aus unserer DB
                p1 = next((p for p in known_players if p['last_name'] in m['player1_last_name']), None)
                p2 = next((p for p in known_players if p['last_name'] in m['player2_last_name']), None)
                
                if p1 and p2:
                    # Berechne unsere eigenen "Fair Odds"
                    s1 = next((s for s in all_skills if s['player_id'] == p1['id']), {})
                    s2 = next((s for s in all_skills if s['player_id'] == p2['id']), {})
                    surf, bsi = detect_surface_config(m['tournament'])
                    
                    prob_p1 = calculate_sophisticated_fair_odds(s1, s2, bsi)
                    
                    # Upsert in DB
                    match_data = {
                        "player1_name": p1['last_name'],
                        "player2_name": p2['last_name'],
                        "tournament": m['tournament'],
                        "odds1": m['odds1'],
                        "odds2": m['odds2'],
                        "ai_fair_odds1": round(1/prob_p1, 2) if prob_p1 > 0 else 0,
                        "ai_fair_odds2": round(1/(1-prob_p1), 2) if prob_p1 < 1 else 0,
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    
                    logger.info(f"💾 Saving Match: {p1['last_name']} vs {p2['last_name']} ({m['tournament']})")
                    supabase.table("market_odds").upsert(match_data, on_conflict="player1_name, player2_name, tournament").execute()

        except json.JSONDecodeError:
            logger.error("Failed to parse AI JSON response")
        except Exception as e:
            logger.error(f"Processing Error: {e}")

    logger.info("🏁 Cycle Done.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
