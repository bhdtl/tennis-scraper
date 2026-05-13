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
logger = logging.getLogger("Sackmann_Engine")

def log(msg: str):
    logger.info(msg)

log("⚡ Initialisiere Elite Quant Engine (Sackmann Integration V1.3 - LOGGING FIX)...")

# Secrets Load
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen! Bitte exportiere SUPABASE_URL und SUPABASE_SERVICE_ROLE_KEY.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. HELPER FUNCTIONS (The Normalizers)
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

# =================================================================
# 3. PHASE 1: THE PLAYER MAPPER
# =================================================================
async def sync_player_ids():
    log("🔍 PHASE 1: Synchronisiere interne Spieler mit Sackmann IDs...")
    
    players_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
    sackmann_players = await fetch_csv_from_github(players_url)
    if not sackmann_players:
        log("❌ Konnte Sackmann Player File nicht laden. Überspringe Phase 1.")
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
# 4. PHASE 2: HISTORICAL DATA LAKE BUILDER
# =================================================================
async def build_historical_lake() -> List[Dict]:
    log("🌊 PHASE 2: Baue historischen Data Lake auf (ATP & Challenger ab 2015)...")
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    years = [str(y) for y in range(2015, 2027)]
    
    all_matches = []
    
    # 🚀 VORBEREITUNG: Falls Reste in der Tabelle sind, ignorieren wir sie oder schreiben einfach neu
    for y in years:
        log(f"📡 Lade Match-Daten für {y}...")
        t_url = f"{base_url}atp_matches_{y}.csv"
        c_url = f"{base_url}atp_matches_qual_chall_{y}.csv"
        
        t_rows = await fetch_csv_from_github(t_url)
        c_rows = await fetch_csv_from_github(c_url)
        
        year_total = t_rows + c_rows
        if not year_total: continue
        
        all_matches.extend(year_total)
        
        db_inserts = []
        for m in year_total:
            try:
                if not m.get('winner_id') or not m.get('loser_id'): continue
                
                raw_date = m.get('tourney_date', '20150101')
                if len(raw_date) == 8:
                    fmt_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                else:
                    fmt_date = "2015-01-01"
                
                db_inserts.append({
                    "tourney_id": m.get('tourney_id', 'unknown'),
                    "tourney_name": m.get('tourney_name', 'unknown'),
                    "surface": m.get('surface', 'Hard'),
                    "match_date": fmt_date,
                    "match_num": int(m.get('match_num', 0)),
                    "winner_sackmann_id": int(m.get('winner_id')),
                    "loser_sackmann_id": int(m.get('loser_id')),
                    "score": m.get('score', ''),
                    "minutes": int(m.get('minutes') or 0),
                    "w_svpt": int(m.get('w_svpt') or 0),
                    "w_1stWon": int(m.get('w_1stWon') or 0),
                    "w_2ndWon": int(m.get('w_2ndWon') or 0),
                    "w_bpSaved": int(m.get('w_bpSaved') or 0),
                    "w_bpFaced": int(m.get('w_bpFaced') or 0),
                    "l_svpt": int(m.get('l_svpt') or 0),
                    "l_1stWon": int(m.get('l_1stWon') or 0),
                    "l_2ndWon": int(m.get('l_2ndWon') or 0),
                    "l_bpSaved": int(m.get('l_bpSaved') or 0),
                    "l_bpFaced": int(m.get('l_bpFaced') or 0)
                })
            except: continue

        # 🚀 FIX: Echter Insert-Block mit Error-Logging, damit wir sehen, was Supabase stört
        log(f"💾 Sende {len(db_inserts)} Matches aus {y} an Supabase...")
        for i in range(0, len(db_inserts), 500):
            try:
                # Wir nutzen insert (ohne on_conflict). Falls es knallt, sehen wir es!
                supabase.table("historical_matches").insert(db_inserts[i:i+500]).execute()
            except Exception as e:
                error_msg = str(e)
                if "duplicate key value" not in error_msg: # Duplikate sind uns egal, echte Fehler wollen wir sehen
                    log(f"⚠️ DB Insert Error in Jahr {y}: {error_msg}")
            
    log(f"✅ Data Lake stabilisiert. {len(all_matches)} historische Matches im Speicher.")
    return all_matches

# =================================================================
# 5. PHASE 3: THE ALCHEMIST (Sharp Metrics Generation)
# =================================================================
def calculate_and_push_sharp_metrics(matches: List[Dict]):
    log("🧮 PHASE 3: Berechne mathematische Spieler-Skills (Hold%, Break%, Fatigue)...")
    
    player_data = {}
    def init_stats(sid: int):
        if sid not in player_data:
            player_data[sid] = {
                "s_played": {"Hard": 0, "Clay": 0, "Grass": 0},
                "s_won": {"Hard": 0, "Clay": 0, "Grass": 0},
                "r_played": {"Hard": 0, "Clay": 0, "Grass": 0},
                "r_won": {"Hard": 0, "Clay": 0, "Grass": 0},
                "bp_faced": 0, "bp_saved": 0,
                "bp_opps": 0, "bp_conv": 0,
                "recent_min": 0, "total_matches": 0
            }

    now_ts = datetime.now().timestamp()

    for row in matches:
        try:
            w_id = int(row.get('winner_id'))
            l_id = int(row.get('loser_id'))
            surf = row.get('surface', 'Hard')
            if surf not in ['Hard', 'Clay', 'Grass']: surf = 'Hard'
            
            init_stats(w_id); init_stats(l_id)
            
            w_svpt = int(row.get('w_svpt') or 0)
            w_won = int(row.get('w_1stWon') or 0) + int(row.get('w_2ndWon') or 0)
            w_bpS = int(row.get('w_bpSaved') or 0)
            w_bpF = int(row.get('w_bpFaced') or 0)
            
            l_svpt = int(row.get('l_svpt') or 0)
            l_won = int(row.get('l_1stWon') or 0) + int(row.get('l_2ndWon') or 0)
            l_bpS = int(row.get('l_bpSaved') or 0)
            l_bpF = int(row.get('l_bpFaced') or 0)
            
            if w_svpt == 0 or l_svpt == 0: continue
            
            player_data[w_id]["s_played"][surf] += w_svpt
            player_data[w_id]["s_won"][surf] += w_won
            player_data[w_id]["r_played"][surf] += l_svpt
            player_data[w_id]["r_won"][surf] += (l_svpt - l_won)
            player_data[w_id]["bp_faced"] += w_bpF
            player_data[w_id]["bp_saved"] += w_bpS
            player_data[w_id]["bp_opps"] += l_bpF
            player_data[w_id]["bp_conv"] += (l_bpF - l_bpS)
            player_data[w_id]["total_matches"] += 1
            
            player_data[l_id]["s_played"][surf] += l_svpt
            player_data[l_id]["s_won"][surf] += l_won
            player_data[l_id]["r_played"][surf] += w_svpt
            player_data[l_id]["r_won"][surf] += (w_svpt - w_won)
            player_data[l_id]["bp_faced"] += l_bpF
            player_data[l_id]["bp_saved"] += l_bpS
            player_data[l_id]["bp_opps"] += w_bpF
            player_data[l_id]["bp_conv"] += (w_bpF - w_bpS)
            player_data[l_id]["total_matches"] += 1
            
            raw_date = row.get('tourney_date', '20150101')
            m_ts = datetime.strptime(raw_date, "%Y%m%d").timestamp()
            if (now_ts - m_ts) <= (14 * 24 * 3600):
                player_data[w_id]["recent_min"] += int(row.get('minutes') or 0)
                player_data[l_id]["recent_min"] += int(row.get('minutes') or 0)
                
        except: continue

    db_players = []
    off = 0; lim = 1000
    while True:
        r = supabase.table("players").select("id, sackmann_id").not_.is_("sackmann_id", "null").range(off, off + lim - 1).execute()
        c = r.data or []
        db_players.extend(c)
        if len(c) < lim: break
        off += lim

    updates = []
    for p in db_players:
        sid = p.get('sackmann_id')
        if sid in player_data:
            d = player_data[sid]
            metrics = {
                "total_matches_analyzed": d["total_matches"],
                "clutch": {
                    "bp_saved_pct": round(d["bp_saved"]/d["bp_faced"]*100, 1) if d["bp_faced"] > 5 else None,
                    "bp_conv_pct": round(d["bp_conv"]/d["bp_opps"]*100, 1) if d["bp_opps"] > 5 else None
                },
                "fatigue": {"recent_14d_minutes": d["recent_min"]}
            }
            for s in ["hard", "clay", "grass"]:
                s_cap = s.capitalize()
                sp = d["s_played"][s_cap]; sw = d["s_won"][s_cap]
                rp = d["r_played"][s_cap]; rw = d["r_won"][s_cap]
                metrics[f"serve_{s}"] = round(sw/sp*100, 1) if sp > 20 else None
                metrics[f"return_{s}"] = round(rw/rp*100, 1) if rp > 20 else None
            
            updates.append({
                "player_id": p['id'],
                "sackmann_metrics": metrics,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })

    for u_data in updates:
        try: 
            supabase.table("player_skills").update({
                "sackmann_metrics": u_data["sackmann_metrics"],
                "updated_at": u_data["updated_at"]
            }).eq("player_id", u_data["player_id"]).execute()
        except: pass

    log("🏁 ENGINE CYCLE FINISHED. Deine Datenbasis ist jetzt Weltklasse.")

# =================================================================
# EXECUTION
# =================================================================
async def main():
    await sync_player_ids()
    matches = await build_historical_lake()
    calculate_and_push_sharp_metrics(matches)

if __name__ == "__main__":
    asyncio.run(main())
