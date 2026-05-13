# -- coding: utf-8 --

import asyncio
import csv
import io
import os
import re
import unicodedata
import logging
import sys
import difflib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

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
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("Sackmann_DataLake")

def log(msg: str):
    logger.info(msg)

log("⚡ Initialisiere Elite Data Lake Engine (Raw Extraction V2.1 - POSTGRES CASE FIX)...")

# Secrets Load
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def normalize_name(name: str) -> str:
    if not name: return ""
    n = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = n.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

async def fetch_csv_from_github(url: str) -> List[Dict[str, str]]:
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(url, timeout=60.0)
            if res.status_code == 200:
                reader = csv.DictReader(io.StringIO(res.text))
                return list(reader)
            else:
                log(f"⚠️ GitHub Request fehlgeschlagen: {res.status_code}")
        except Exception as e:
            log(f"❌ Netzwerkfehler beim Laden der CSV: {e}")
    return []

def to_int(val: Any) -> Optional[int]:
    try: return int(float(val))
    except: return None

def to_float(val: Any) -> Optional[float]:
    try: return float(val)
    except: return None

# =================================================================
# 3. PHASE 1: THE PLAYER MAPPER
# =================================================================
async def sync_player_ids():
    log("🔍 PHASE 1: Synchronisiere interne Spieler mit Sackmann IDs...")
    
    players_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
    sackmann_players = await fetch_csv_from_github(players_url)
    if not sackmann_players:
        log("❌ Konnte Sackmann Player File nicht laden.")
        return

    db_players = []
    offset = 0
    limit = 1000
    while True:
        res = supabase.table("players").select("id, first_name, last_name, sackmann_id").range(offset, offset + limit - 1).execute()
        chunk = res.data or []
        db_players.extend(chunk)
        if len(chunk) < limit: break
        offset += limit
        
    players_to_match = [p for p in db_players if p.get('sackmann_id') is None]
    log(f"🎯 {len(players_to_match)} Spieler ohne Sackmann-ID. Starte Fuzzy Matching...")

    updates = []
    for db_p in players_to_match:
        db_full = normalize_name(f"{db_p.get('first_name', '')} {db_p.get('last_name', '')}")
        best_match_id = None
        best_score = 0.0
        db_last = normalize_name(db_p.get('last_name', ''))
        
        for sp in sackmann_players:
            sp_last = normalize_name(sp.get('name_last', ''))
            if sp_last != db_last: continue 
            sp_full = normalize_name(f"{sp.get('name_first', '')} {sp.get('name_last', '')}")
            score = get_similarity(db_full, sp_full)
            if score > best_score:
                best_score = score
                best_match_id = sp['player_id']
            if score == 1.0: break 
                
        if best_score > 0.88 and best_match_id:
            updates.append({
                "id": db_p['id'],
                "sackmann_id": int(best_match_id)
            })

    if updates:
        log(f"💾 Speichere {len(updates)} neu gematchte IDs (Sequential Update)...")
        for u_data in updates:
            try:
                supabase.table("players").update({"sackmann_id": u_data["sackmann_id"]}).eq("id", u_data["id"]).execute()
            except Exception as e:
                pass
    else:
        log("✅ Keine neuen Spieler-Matches gefunden.")

# =================================================================
# 4. PHASE 2: THE OMNI-DATA INGESTION (All fields!)
# =================================================================
async def build_historical_lake():
    log("🌊 PHASE 2: Extrahiere ALLE Rohdaten aus GitHub in Supabase...")
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    
    # Wir laden die volle moderne Ära. 
    years = [str(y) for y in range(2015, 2027)]
    total_inserted = 0
    
    for y in years:
        log(f"📡 Verarbeite Jahr {y}...")
        t_url = f"{base_url}atp_matches_{y}.csv"
        c_url = f"{base_url}atp_matches_qual_chall_{y}.csv"
        f_url = f"{base_url}atp_matches_futures_{y}.csv"
        
        t_rows = await fetch_csv_from_github(t_url)
        c_rows = await fetch_csv_from_github(c_url)
        f_rows = await fetch_csv_from_github(f_url) 
        
        year_total = t_rows + c_rows + f_rows
        if not year_total: continue
        
        db_inserts = []
        for m in year_total:
            try:
                w_id = to_int(m.get('winner_id'))
                l_id = to_int(m.get('loser_id'))
                m_num = to_int(m.get('match_num'))
                t_id = m.get('tourney_id')
                
                if not w_id or not l_id or not t_id or m_num is None: continue
                
                raw_date = m.get('tourney_date', '')
                if len(raw_date) == 8:
                    fmt_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                else:
                    fmt_date = "2015-01-01"
                
                db_inserts.append({
                    "tourney_id": t_id,
                    "tourney_name": m.get('tourney_name'),
                    "surface": m.get('surface'),
                    "draw_size": to_int(m.get('draw_size')),
                    "tourney_level": m.get('tourney_level'),
                    "match_date": fmt_date,
                    "match_num": m_num,
                    
                    "winner_sackmann_id": w_id,
                    "winner_seed": m.get('winner_seed'),
                    "winner_entry": m.get('winner_entry'),
                    "winner_name": m.get('winner_name'),
                    "winner_hand": m.get('winner_hand'),
                    "winner_ht": to_int(m.get('winner_ht')),
                    "winner_ioc": m.get('winner_ioc'),
                    "winner_age": to_float(m.get('winner_age')),
                    "winner_rank": to_int(m.get('winner_rank')),
                    "winner_rank_points": to_int(m.get('winner_rank_points')),
                    
                    "loser_sackmann_id": l_id,
                    "loser_seed": m.get('loser_seed'),
                    "loser_entry": m.get('loser_entry'),
                    "loser_name": m.get('loser_name'),
                    "loser_hand": m.get('loser_hand'),
                    "loser_ht": to_int(m.get('loser_ht')),
                    "loser_ioc": m.get('loser_ioc'),
                    "loser_age": to_float(m.get('loser_age')),
                    "loser_rank": to_int(m.get('loser_rank')),
                    "loser_rank_points": to_int(m.get('loser_rank_points')),
                    
                    "score": m.get('score'),
                    "best_of": to_int(m.get('best_of')),
                    "round": m.get('round'),
                    "minutes": to_int(m.get('minutes')),
                    
                    # 🚀 THE FIX: Alle Spalten-Keys strikt in Kleinbuchstaben, passend zu PostgreSQL!
                    "w_ace": to_int(m.get('w_ace')),
                    "w_df": to_int(m.get('w_df')),
                    "w_svpt": to_int(m.get('w_svpt')),
                    "w_1stin": to_int(m.get('w_1stIn')),     # Vorher: w_1stIn
                    "w_1stwon": to_int(m.get('w_1stWon')),   # Vorher: w_1stWon
                    "w_2ndwon": to_int(m.get('w_2ndWon')),   # Vorher: w_2ndWon
                    "w_svgms": to_int(m.get('w_SvGms')),     # Vorher: w_SvGms
                    "w_bpsaved": to_int(m.get('w_bpSaved')), # Vorher: w_bpSaved
                    "w_bpfaced": to_int(m.get('w_bpFaced')), # Vorher: w_bpFaced
                    
                    "l_ace": to_int(m.get('l_ace')),
                    "l_df": to_int(m.get('l_df')),
                    "l_svpt": to_int(m.get('l_svpt')),
                    "l_1stin": to_int(m.get('l_1stIn')),     # Vorher: l_1stIn
                    "l_1stwon": to_int(m.get('l_1stWon')),   # Vorher: l_1stWon
                    "l_2ndwon": to_int(m.get('l_2ndWon')),   # Vorher: l_2ndWon
                    "l_svgms": to_int(m.get('l_SvGms')),     # Vorher: l_SvGms
                    "l_bpsaved": to_int(m.get('l_bpSaved')), # Vorher: l_bpSaved
                    "l_bpfaced": to_int(m.get('l_bpFaced'))  # Vorher: l_bpFaced
                })
            except Exception as loop_e:
                continue

        log(f"💾 Pushe {len(db_inserts)} Matches aus {y} in Supabase Data Lake...")
        chunk_size = 500
        for i in range(0, len(db_inserts), chunk_size):
            try:
                supabase.table("historical_matches").insert(db_inserts[i:i+chunk_size]).execute()
            except Exception as e:
                if "duplicate key value" not in str(e): 
                    log(f"⚠️ Insert Error bei Chunk {i}: {str(e)}")
                    
        total_inserted += len(db_inserts)
            
    log(f"✅ DATA LAKE GEBAUT: Insgesamt {total_inserted} Matchesrohdaten verarbeitet.")

# =================================================================
# EXECUTION
# =================================================================
async def main():
    await sync_player_ids()
    await build_historical_lake()
    log("🏁 SACKMANN ENGINE FINISHED. Dein Data Lake ist bereit für die Matrix.")

if __name__ == "__main__":
    asyncio.run(main())
