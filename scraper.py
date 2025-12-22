# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
import unicodedata
import time
import math
import logging
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from supabase import create_client, Client

# =================================================================
# CONFIGURATION - UNIFIED NEURAL SCOUT (V25.1 - HARDENED SECURITY)
# =================================================================
# Silicon Valley Standard: STRIKTE Nutzung von Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = 'gemini-2.5-pro'

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisierung der Clients (stürzt ab, wenn Keys fehlen - gewollt für Sicherheit)
if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("❌ CRITICAL ERROR: API Keys missing in Environment Secrets!")
    raise ValueError("Missing Security Credentials")

client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MATH CORE (SYNCED WITH FRONTEND V25.2) ---
def calculate_sophisticated_fair_odds(s1, s2, bsi):
    is_fast = bsi >= 7
    is_slow = bsi <= 4
    w_serve = 2.2 if is_fast else (0.6 if is_slow else 1.0)
    w_baseline = 0.7 if is_fast else (1.4 if is_slow else 1.0)
    w_mental = 0.6 if is_fast else (1.2 if is_slow else 0.8)
    w_physical = 0.4 if is_fast else (1.5 if is_slow else 0.8)
    serve_diff = ((s1['serve'] + s1['power']) - (s2['serve'] + s2['power'])) * w_serve
    baseline_diff = ((s1['forehand'] + s1['backhand']) - (s2['forehand'] + s2['backhand'])) * w_baseline
    physical_diff = ((s1['speed'] + s1['stamina']) - (s2['speed'] + s2['stamina'])) * w_physical
    mental_diff = (s1['mental'] - s2['mental']) * w_mental
    total_advantage_a = (serve_diff + baseline_diff + physical_diff + mental_diff) / 200
    class_diff = (s1['overall_rating'] - s2['overall_rating']) / 25
    rating_a = 6.0 + total_advantage_a + class_diff
    rating_b = 6.0 - total_advantage_a - class_diff
    prob_a = 1 / (1 + math.exp(-0.7 * (rating_a - rating_b)))
    return prob_a

def detect_surface_config(tournament_name):
    name = (tournament_name or '').lower()
    if 'indoor' in name: return 'Indoor', 8.2
    if any(x in name for x in ['clay', 'sand', 'roland garros']): return 'Red Clay', 3.5
    if any(x in name for x in ['hard', 'australian', 'us open']): return 'Hard', 6.5
    if 'grass' in name: return 'Grass', 9.2
    return 'Hard', 5.0

# --- SCRAPER FUNCTIONS ---
def normalize_text(text):
    if not text: return ""
    text = text.replace('æ', 'ae').replace('Æ', 'Ae').replace('ø', 'o')
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw_name_segment):
    noise_patterns = [
        r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'beIN Sports', 
        r'bet365', r'Unibet', r'William Hill', r'\(Aus\)', r'\(Gbr/Irl\)'
    ]
    clean = raw_name_segment
    for pat in noise_patterns:
        clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
    return clean.replace('|', '').strip()

async def get_known_players():
    logger.info("📥 Synchronisiere Player-Matrix...")
    try:
        response = supabase.table("players").select("*").execute()
        return response.data
    except Exception as e:
        logger.error(f"❌ DB-Fehler beim Laden der Spieler: {e}")
        return []

async def apply_stealth(page):
    await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = {runtime: {}};
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    """)

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        page = await browser.new_page()
        await apply_stealth(page)
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            logger.info(f"📡 Scanning Date: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            try:
                await page.wait_for_selector(".result", timeout=15000)
                content = await page.content()
            except:
                content = None
            await browser.close()
            return content
        except Exception as e:
            logger.error(f"❌ Error at {target_date.strftime('%Y-%m-%d')}: {e}")
            await browser.close()
            return None

def clean_html_for_ai(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(["script", "style", "nav", "aside", "footer", "header", "svg"]):
        tag.extract()
    tables = soup.find_all("table", class_="result")
    if not tables: return ""
    structured_text = ""
    current_tournament = "Unknown Tournament"
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            row = rows[i]
            if "head" in row.get("class", []) or "hdr" in row.get("class", []):
                current_tournament = re.sub(r'[A-Z0-9]{5,}.*$', '', row.get_text(strip=True)).strip()
                continue
            row_text = normalize_text(row.get_text(separator=' | ', strip=True))
            if re.search(r'\d{2}:\d{2}', row_text):
                cleaned_p1 = clean_player_name(row_text)
                if i + 1 < len(rows):
                    next_row = normalize_text(rows[i+1].get_text(separator=' | ', strip=True))
                    cleaned_p2 = clean_player_name(next_row)
                    cleaned_p1 = re.sub(r'[-\s]+$', '', cleaned_p1)
                    cleaned_p2 = re.sub(r'\(\d+\).*$', '', cleaned_p2)
                    structured_text += f"TOURNAMENT: {current_tournament} | MATCH: {cleaned_p1} VS {cleaned_p2}\n"
    return structured_text

def parse_with_gemini_pro(text_data, known_players):
    logger.info(f"🧠 {MODEL_NAME}: Neural Sync...")
    ref_list = [f"{p['first_name']} {p['last_name']}" for p in known_players]
    prompt = f"Extract matches where BOTH players appear in {ref_list}\nInput: {text_data}\nRules: If odds missing, set to 0. Output JSON Array."
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=types.GenerateContentConfig(temperature=0.0))
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

async def process_neural_analysis(p1, p2, tournament, s1, s2):
    prompt = f"Tactical analysis for {p1['last_name']} vs {p2['last_name']} ({tournament}). JSON: {{'briefing': '...', 'ai_prob': 0.XX}}"
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=types.GenerateContentConfig(temperature=0.0))
        return json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group(0))
    except: return None

# --- HYGIENE ENGINE ---
def perform_database_hygiene(matches_to_upsert, matches_to_delete):
    if matches_to_delete:
        for m in matches_to_delete:
            try:
                supabase.table("market_odds").delete().match({"player1_name": m['player1_name'], "player2_name": m['player2_name'], "tournament": m['tournament']}).execute()
            except: pass
    if matches_to_upsert:
        current_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for m in matches_to_upsert: m["created_at"] = current_ts
        supabase.table("market_odds").upsert(matches_to_upsert, on_conflict="player1_name, player2_name, tournament").execute()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        supabase.table("market_odds").delete().lt("created_at", cutoff).execute()
        supabase.table("market_odds").delete().or_("odds1.eq.0,odds2.eq.0").execute()
    except: pass

async def run_pipeline():
    logger.info("🚀 Starting Multi-Day Cloud Scout...")
    known_players = await get_known_players()
    if not known_players: return
    skills_res = supabase.table("player_skills").select("*").execute()
    all_skills = skills_res.data
    existing_odds_res = supabase.table("market_odds").select("*").execute()
    existing_map = {f"{r['player1_name']}-{r['player2_name']}-{r['tournament']}": r for r in existing_odds_res.data}

    current_date = datetime.now()
    for day_offset in range(14):
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue
        cleaned = clean_html_for_ai(html)
        raw = parse_with_gemini_pro(cleaned, known_players)
        valid_upserts, invalid_deletes = [], []
        for m in raw:
            if float(m.get('odds1', 0)) < 1.01:
                invalid_deletes.append(m)
                continue
            key = f"{m['player1_name']}-{m['player2_name']}-{m['tournament']}"
            if key in existing_map and existing_map[key]['ai_analysis_text']:
                m.update({"ai_fair_odds1": existing_map[key]['ai_fair_odds1'], "ai_fair_odds2": existing_map[key]['ai_fair_odds2'], "ai_analysis_text": existing_map[key]['ai_analysis_text']})
            else:
                try:
                    p1_db = next(p for p in known_players if p['last_name'] in m['player1_name'])
                    p2_db = next(p for p in known_players if p['last_name'] in m['player2_name'])
                    s1 = next(s for s in all_skills if s['player_id'] == p1_db['id'])
                    s2 = next(s for s in all_skills if s['player_id'] == p2_db['id'])
                    surf, bsi = detect_surface_config(m['tournament'])
                    skill_p = calculate_sophisticated_fair_odds(s1, s2, bsi)
                    analysis = await process_neural_analysis(p1_db, p2_db, m['tournament'], s1, s2)
                    if analysis:
                        final_p1 = (skill_p * 0.5) + (analysis['ai_prob'] * 0.5)
                        m.update({"ai_fair_odds1": round(1/final_p1, 2), "ai_fair_odds2": round(1/(1-final_p1), 2), "ai_analysis_text": analysis['briefing']})
                except: continue
            valid_upserts.append(m)
        perform_database_hygiene(valid_upserts, invalid_deletes)
    logger.info("🏁 Multi-Day Cycle Complete.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
