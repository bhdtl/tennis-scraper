# -- coding: utf-8 --

import asyncio
import os
import sys
import logging
import unicodedata
import difflib
from typing import List, Dict

import httpx
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("Matchstat_Mapper")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere Matchstat ID Mapper (GITHUB ACTIONS EDITION)...")

# Secrets Load
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY") 

if not SUPABASE_URL or not SUPABASE_KEY or not RAPIDAPI_KEY:
    log("❌ CRITICAL: Secrets fehlen! Prüfe die GitHub Action Environment Variablen.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def normalize_name(name: str) -> str:
    if not name: 
        return ""
    n = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = n.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    return n

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def fetch_all_supabase_players() -> List[Dict]:
    """Holt alle Spieler aus der Supabase (mit Pagination-Bypass)."""
    log("📡 Lade Spieler aus Supabase...")
    data = []
    offset = 0
    limit = 1000
    while True:
        try:
            res = supabase.table("players").select("id, first_name, last_name, matchstat_id").range(offset, offset + limit - 1).execute()
            chunk = res.data or []
            data.extend(chunk)
            if len(chunk) < limit:
                break
            offset += limit
        except Exception as e:
            log(f"⚠️ Pagination error auf Supabase: {e}")
            break
    log(f"✅ {len(data)} Spieler aus Supabase geladen.")
    return data

# =================================================================
# 3. MATCHSTAT API CLIENT
# =================================================================
async def fetch_all_matchstat_players(api_key: str, tour: str) -> List[Dict]:
    """
    Lädt alle Spieler einer Tour von Matchstat. 
    Nutzt eine große pageSize, um API-Calls extrem gering zu halten.
    """
    tour_str = "wta" if tour.lower() == "wta" else "atp"
    base_url = f"https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/{tour_str}/player"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "tennis-api-atp-wta-itf.p.rapidapi.com"
    }
    
    all_players = []
    page = 1
    page_size = 500 
    
    async with httpx.AsyncClient() as client:
        while True:
            log(f"📡 Lade Matchstat {tour.upper()} Spieler (Seite {page})...")
            url = f"{base_url}?pageNo={page}&pageSize={page_size}"
            
            try:
                res = await client.get(url, headers=headers, timeout=20.0)
                if res.status_code == 429:
                    log("⚠️ Rate Limit erreicht! Warte 5 Sekunden...")
                    await asyncio.sleep(5)
                    continue
                
                if res.status_code != 200:
                    log(f"❌ API Error: {res.status_code} - {res.text}")
                    break
                    
                data = res.json()
                
                player_list = data.get('data', data) if isinstance(data, dict) else data
                
                if not player_list or not isinstance(player_list, list) or len(player_list) == 0:
                    break 
                    
                all_players.extend(player_list)
                
                if len(player_list) < page_size:
                    break
                    
                page += 1
                await asyncio.sleep(0.5) 
                
            except Exception as e:
                log(f"❌ Request Fehler: {e}")
                break
                
    log(f"✅ {len(all_players)} {tour.upper()} Spieler von Matchstat geladen.")
    return all_players

# =================================================================
# 4. CORE MAPPING LOGIC
# =================================================================
async def run_mapper():
    log("🚀 Starte ID-Mapping Prozess (Supabase <-> Matchstat)...")
    
    db_players = fetch_all_supabase_players()
    players_to_map = [p for p in db_players if not p.get('matchstat_id')]
    
    if not players_to_map:
        log("🏆 Alle Spieler in Supabase haben bereits eine Matchstat ID. Nichts zu tun!")
        return
        
    log(f"🔍 Es müssen {len(players_to_map)} Spieler gemappt werden.")
    
    ms_atp_players = await fetch_all_matchstat_players(RAPIDAPI_KEY, "atp")
    ms_wta_players = await fetch_all_matchstat_players(RAPIDAPI_KEY, "wta")
    all_ms_players = ms_atp_players + ms_wta_players
    
    updated_count = 0
    batch_updates = []
    
    log("🧠 Starte Fuzzy-Matching Engine...")
    
    for db_p in players_to_map:
        db_first = normalize_name(db_p.get('first_name', ''))
        db_last = normalize_name(db_p.get('last_name', ''))
        db_full = f"{db_first} {db_last}".strip()
        
        best_match_id = None
        best_score = 0.0
        
        for ms_p in all_ms_players:
            ms_name = normalize_name(ms_p.get('name', ''))
            
            if db_full == ms_name:
                best_match_id = ms_p['id']
                best_score = 1.0
                break
                
            db_reversed = f"{db_last} {db_first}".strip()
            if db_reversed == ms_name:
                best_match_id = ms_p['id']
                best_score = 1.0
                break
                
            sim = get_similarity(db_full, ms_name)
            if sim > best_score:
                best_score = sim
                best_match_id = ms_p['id']
                
        if best_score > 0.85 and best_match_id:
            batch_updates.append({
                "id": db_p['id'],
                "matchstat_id": best_match_id
            })
            updated_count += 1
            
            if len(batch_updates) >= 100:
                try:
                    supabase.table("players").upsert(batch_updates).execute()
                    log(f"💾 100 IDs in Supabase gespeichert...")
                    batch_updates = []
                except Exception as e:
                    log(f"❌ DB Upsert Error: {e}")
                    
    if batch_updates:
        try:
            supabase.table("players").upsert(batch_updates).execute()
        except Exception as e:
            log(f"❌ DB Upsert Error (Final Batch): {e}")
            
    log(f"🏁 Mapping abgeschlossen! {updated_count} von {len(players_to_map)} Spielern wurden erfolgreich mit Matchstat verknüpft.")
    log("💡 Du kannst das Skript nun jederzeit über GitHub Actions neu starten, wenn neue Spieler hinzukommen.")

if __name__ == "__main__":
    asyncio.run(run_mapper())
