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

# VETERAN FIX: Lokale .env Unterstützung für einfacheres Debugging
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # In GitHub Actions ist dotenv oft nicht nötig, da Secrets injiziert werden

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from supabase import create_client, Client
import httpx 

# =================================================================
# CONFIGURATION - UNIFIED NEURAL SCOUT (V27.0 - ROBUST EDITION)
# =================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# VETERAN FIX: Granularer Check, damit wir wissen, WAS fehlt
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Fallback für Key-Namen

missing_keys = []
if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")
if not SUPABASE_URL: missing_keys.append("SUPABASE_URL")
if not SUPABASE_KEY: missing_keys.append("SUPABASE_KEY")

if missing_keys:
    logger.error(f"❌ CRITICAL: The following keys are missing in Secrets: {', '.join(missing_keys)}")
    logger.error("👉 Action Required: Go to GitHub Repo -> Settings -> Secrets -> Actions and add them.")
    raise ValueError(f"Missing Security Credentials: {missing_keys}")

MODEL_NAME = 'llama-3.3-70b-versatile'
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MATH CORE (V24.0 SYNCED - UNTOUCHED) ---
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
    return 1 / (1 + math.exp(-0.7 * (rating_a - rating_b)))

def detect_surface_config(tournament_name):
    name = (tournament_name or '').lower()
    if 'indoor' in name: return 'Indoor', 8.2
    if any(x in name for x in ['clay', 'sand', 'roland garros']): return 'Red Clay', 3.5
    if any(x in name for x in ['hard', 'australian', 'us open']): return 'Hard', 6.5
    return 'Hard', 5.0

# --- GROQ API HELPER ---
async def call_groq(prompt):
    """Führt eine KI-Anfrage über die Groq API aus."""
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
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return None

# --- SCRAPER LOGIC ---
def normalize_text(text):
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text.replace('æ', 'ae').replace('ø', 'o')) if unicodedata.category(c) != 'Mn')

def clean_player_name(raw):
    noise = [r'Live streams', r'1xBet', r'bwin', r'TV', r'Sky Sports', r'beIN Sports', r'bet365', r'Unibet', r'William Hill']
    for pat in noise: raw = re.sub(pat, '', raw, flags=re.IGNORECASE)
    return raw.replace('|', '').strip()

async def get_known_players():
    try: return supabase.table("players").select("*").execute().data
    except: return []

async def scrape_tennis_odds_for_date(target_date):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        page = await browser.new_page()
        try:
            url = f"https://www.tennisexplorer.com/matches/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
            logger.info(f"📡 Scanning Date: {target_date.strftime('%Y-%m-%d')}")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            try:
                await page.wait_for_selector(".result", timeout=15000)
                content = await page.content()
            except: content = None
            await browser.close()
            return content
        except Exception as e:
            logger.error(f"❌ Scrape Error: {e}")
            await browser.close()
            return None

def clean_html_for_ai(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(["script", "style", "nav", "footer"]): tag.extract()
    tables = soup.find_all("table", class_="result")
    txt = ""
    current_tour = "Unknown"
    for table in tables:
        rows = table.find_all("tr")
        for i in range(len(rows)):
            if "head" in rows[i].get("class", []):
                current_tour = rows[i].get_text(strip=True)
                continue
            row_text = normalize_text(rows[i].get_text(separator=' | ', strip=True))
            if re.search(r'\d{2}:\d{2}', row_text) and i+1 < len(rows):
                p1 = clean_player_name(row_text)
                p2 = clean_player_name(normalize_text(rows[i+1].get_text(separator=' | ', strip=True)))
                txt += f"TOURNAMENT: {current_tour} | MATCH: {p1} VS {p2}\n"
    return txt

# --- HYGIENE ---
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
    logger.info("🚀 Sync Cycle Start (Groq Future Scout)...")
    known_players = await get_known_players()
    if not known_players: 
        logger.warning("⚠️ No known players found in DB.")
        return

    skills_res = supabase.table("player_skills").select("*").execute()
    all_skills = skills_res.data
    existing_odds_res = supabase.table("market_odds").select("*").execute()
    existing_map = {f"{r['player1_name']}-{r['player2_name']}-{r['tournament']}": r for r in existing_odds_res.data}

    current_date = datetime.now()
    for day_offset in range(14): # 14 Tage Future-Scan
        target_date = current_date + timedelta(days=day_offset)
        html = await scrape_tennis_odds_for_date(target_date)
        if not html: continue
        cleaned = clean_html_for_ai(html)
        ref_list = [f"{p['first_name']} {p['last_name']}" for p in known_players]
        
        prompt = f"""
        ### ROLE: Data Auditor
        Extract matches where BOTH players appear in {ref_list}
        Input: {cleaned}
        RULES: If odds are missing or results like '6-4', set odds1 and odds2 to 0.00.
        OUTPUT JSON ONLY: {{"matches": [{{"player1_name": "...", "player2_name": "...", "odds1": 1.XX, "odds2": 1.XX, "tournament": "...", "match_time": "HH:MM"}}]}}
        """
        
        try:
            raw_response = await call_groq(prompt)
            if not raw_response: continue
            data = json.loads(raw_response)
            matches = data.get('matches', [])
            
            valid_upserts, invalid_deletes = [], []
            for m in matches:
                if float(m.get('odds1', 0)) < 1.01:
                    invalid_deletes.append(m)
                    continue
                key = f"{m['player1_name']}-{m['player2_name']}-{m['tournament']}"
                if key in existing_map:
                    m.update({"ai_fair_odds1": existing_map[key]['ai_fair_odds1'], "ai_fair_odds2": existing_map[key]['ai_fair_odds2'], "ai_analysis_text": existing_map[key]['ai_analysis_text']})
                else:
                    try:
                        p1 = next(p for p in known_players if p['last_name'] in m['player1_name'])
                        p2 = next(p for p in known_players if p['last_name'] in m['player2_name'])
                        s1 = next(s for s in all_skills if s['player_id'] == p1['id'])
                        s2 = next(s for s in all_skills if s['player_id'] == p2['id'])
                        surf, bsi = detect_surface_config(m['tournament'])
                        skill_p = calculate_sophisticated_fair_odds(s1, s2, bsi)
                        
                        # Neural Analysis via Groq
                        analysis_prompt = f"Analyze matchup: {p1['last_name']} vs {p2['last_name']} on {surf}. Skills: {s1} vs {s2}. Output JSON only: {{'briefing': '...', 'ai_prob': 0.XX}}"
                        analysis_raw = await call_groq(analysis_prompt)
                        if analysis_raw:
                            analysis = json.loads(analysis_raw)
                            final_p1 = (skill_p * 0.5) + (analysis['ai_prob'] * 0.5)
                            m.update({"ai_fair_odds1": round(1/final_p1, 2), "ai_fair_odds2": round(1/(1-final_p1), 2), "ai_analysis_text": analysis['briefing']})
                    except Exception as e:
                        logger.warning(f"Skipping analysis for match {m['player1_name']} vs {m['player2_name']}: {e}")
                        continue
                valid_upserts.append(m)
            perform_database_hygiene(valid_upserts, invalid_deletes)
        except Exception as e:
            logger.error(f"Pipeline Step Error: {e}")
            continue
    logger.info("🏁 Cycle Done.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
