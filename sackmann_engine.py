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
from datetime import datetime, timezone, timedelta
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

log("⚡ Initialisiere Elite Data Lake Engine (Dual-Core DELTA SYNC V3.3 - RAM SORT EDITION)...")

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
            elif res.status_code == 404:
                log(f"ℹ️ Datei nicht gefunden (404), überspringe: {url.split('/')[-1]}")
            else:
                log(f"⚠️ GitHub Request fehlgeschlagen: {res.status_code} für {url}")
        except Exception as e:
            log(f"❌ Netzwerkfehler beim Laden der CSV: {e}")
    return []

# SOTA Fix: Striktes Parsing für IDs
def to_int(val: Any) -> Optional[int]:
    try: return int(float(val))
    except: return None

# SOTA Fix: Striktes Parsing für Mathematik (Verhindert "None + None" Crashes)
def to_int_safe(val: Any) -> int:
    try: return int(float(val))
    except: return 0

def to_float(val: Any) -> Optional[float]:
    try: return float(val)
    except: return None

def parse_date(date_str: str) -> datetime:
    if not date_str: return datetime(2015, 1, 1, tzinfo=timezone.utc)
    if "T" in date_str:
        try: return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except: pass
    date_str = str(date_str).replace("-", "")
    if len(date_str) >= 8:
        try: return datetime.strptime(date_str[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except: pass
    return datetime(2015, 1, 1, tzinfo=timezone.utc)

def get_total_games(score_str: str) -> int:
    if not isinstance(score_str, str): return 0
    clean_score = re.sub(r'\.\d+', '', score_str.lower().replace(":", "-").strip())
    if "ret" in clean_score or "w.o" in clean_score: return 0
    sets = re.findall(r'\b(\d+)\s*-\s*(\d+)\b', clean_score)
    try:
        return sum(int(s[0]) + int(s[1]) for s in sets)
    except:
        return 0

# --- ELO MATH ---
def calculate_k_factor(matches_played: int) -> float:
    return 250.0 / ((matches_played + 5.0) ** 0.4)

def calc_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))

def update_elo(old_rating: float, k_factor: float, actual_score: float, expected_score: float) -> float:
    return old_rating + k_factor * (actual_score - expected_score)

def aggregate_stats(matches_list: List[Dict]) -> Optional[Dict]:
    valid_matches = [m for m in matches_list if m.get('svpt', 0) > 0]
    count = len(valid_matches)
    if count == 0: return None
    
    aces = sum(m['aces'] for m in valid_matches)
    dfs = sum(m['dfs'] for m in valid_matches)
    svpt = sum(m['svpt'] for m in valid_matches)
    first_in = sum(m['1stin'] for m in valid_matches)
    first_won = sum(m['1stwon'] for m in valid_matches)
    second_won = sum(m['2ndwon'] for m in valid_matches)
    bpsaved = sum(m['bpsaved'] for m in valid_matches)
    bpfaced = sum(m['bpfaced'] for m in valid_matches)
    ret_pts = sum(m['ret_pts'] for m in valid_matches)
    ret_won = sum(m['ret_won'] for m in valid_matches)
    bp_opps = sum(m['bp_opps'] for m in valid_matches)
    bp_conv = sum(m['bp_conv'] for m in valid_matches)
    
    return {
        "matches_with_stats": count,
        "aces_per_match": round(aces / count, 1),
        "df_per_match": round(dfs / count, 1),
        "first_in_pct": round((first_in / svpt) * 100, 1) if svpt > 0 else 0,
        "first_win_pct": round((first_won / first_in) * 100, 1) if first_in > 0 else 0,
        "second_win_pct": round((second_won / (svpt - first_in)) * 100, 1) if (svpt - first_in) > 0 else 0,
        "bp_saved_pct": round((bpsaved / bpfaced) * 100, 1) if bpfaced > 0 else 0,
        "ret_win_pct": round((ret_won / ret_pts) * 100, 1) if ret_pts > 0 else 0,
        "bp_conv_pct": round((bp_conv / bp_opps) * 100, 1) if bp_opps > 0 else 0
    }

# =================================================================
# 3. PHASE 1: THE PLAYER MAPPER (DUAL-CORE)
# =================================================================
async def sync_player_ids():
    log("🔍 PHASE 1: Synchronisiere interne Spieler mit Sackmann IDs (ATP & WTA)...")
    
    atp_players_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
    wta_players_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv"
    
    atp_players = await fetch_csv_from_github(atp_players_url)
    wta_players = await fetch_csv_from_github(wta_players_url)
    
    sackmann_players = atp_players + wta_players
    
    if not sackmann_players:
        log("❌ Konnte Sackmann Player Files nicht laden.")
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
    
    if not players_to_match:
        log("✅ Keine neuen Spieler-Matches benötigt.")
        return
        
    log(f"🎯 {len(players_to_match)} Spieler ohne Sackmann-ID im System. Starte Fuzzy Matching...")

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
                best_match_id = sp.get('player_id')
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

# =================================================================
# 4. PHASE 2: THE OMNI-DATA INGESTION (DELTA SYNC)
# =================================================================
async def build_historical_lake():
    log("🌊 PHASE 2: Extrahiere NEUE Rohdaten aus GitHub (ATP & WTA) in Supabase...")
    
    tours = [
        {"prefix": "atp", "repo": "tennis_atp", "label": "ATP"},
        {"prefix": "wta", "repo": "tennis_wta", "label": "WTA"}
    ]
    
    total_inserted = 0
    
    for tour_info in tours:
        prefix = tour_info["prefix"]
        repo = tour_info["repo"]
        tour_label = tour_info["label"]
        base_url = f"https://raw.githubusercontent.com/JeffSackmann/{repo}/master/"
        
        res = supabase.table("historical_matches").select("match_date").eq("tour", tour_label).order("match_date", desc=True).limit(1).execute()
        
        last_date_str = "2015-01-01"
        if res.data and len(res.data) > 0 and res.data[0]['match_date']:
            last_date_str = res.data[0]['match_date']
            
        start_year = int(last_date_str[:4])
        years = [str(y) for y in range(start_year, 2027)]
        
        log(f"=== Starte Delta-Sync für {tour_label} ab {last_date_str} (Starte Download ab Jahr {start_year}) ===")
        
        for y in years:
            log(f"📡 Lade {tour_label} Jahr {y}...")
            urls_to_fetch = []
            
            if prefix == "atp":
                urls_to_fetch.append(f"{base_url}atp_matches_{y}.csv")
                urls_to_fetch.append(f"{base_url}atp_matches_qual_chall_{y}.csv")
                urls_to_fetch.append(f"{base_url}atp_matches_futures_{y}.csv")
            else:
                urls_to_fetch.append(f"{base_url}wta_matches_{y}.csv")
                urls_to_fetch.append(f"{base_url}wta_matches_qual_itf_{y}.csv")
                
            year_total = []
            for url in urls_to_fetch:
                rows = await fetch_csv_from_github(url)
                year_total.extend(rows)
                
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
                        
                    if fmt_date <= last_date_str:
                        continue
                    
                    db_inserts.append({
                        "tourney_id": t_id,
                        "tourney_name": m.get('tourney_name'),
                        "surface": m.get('surface'),
                        "draw_size": to_int(m.get('draw_size')),
                        "tourney_level": m.get('tourney_level'),
                        "match_date": fmt_date,
                        "match_num": m_num,
                        "tour": tour_label, 
                        
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
                        
                        "w_ace": to_int_safe(m.get('w_ace')),
                        "w_df": to_int_safe(m.get('w_df')),
                        "w_svpt": to_int_safe(m.get('w_svpt')),
                        "w_1stin": to_int_safe(m.get('w_1stIn')),     
                        "w_1stwon": to_int_safe(m.get('w_1stWon')),   
                        "w_2ndwon": to_int_safe(m.get('w_2ndWon')),   
                        "w_svgms": to_int_safe(m.get('w_SvGms')),     
                        "w_bpsaved": to_int_safe(m.get('w_bpSaved')), 
                        "w_bpfaced": to_int_safe(m.get('w_bpFaced')), 
                        
                        "l_ace": to_int_safe(m.get('l_ace')),
                        "l_df": to_int_safe(m.get('l_df')),
                        "l_svpt": to_int_safe(m.get('l_svpt')),
                        "l_1stin": to_int_safe(m.get('l_1stIn')),     
                        "l_1stwon": to_int_safe(m.get('l_1stWon')),   
                        "l_2ndwon": to_int_safe(m.get('l_2ndWon')),   
                        "l_svgms": to_int_safe(m.get('l_SvGms')),     
                        "l_bpsaved": to_int_safe(m.get('l_bpSaved')), 
                        "l_bpfaced": to_int_safe(m.get('l_bpFaced'))  
                    })
                except Exception as loop_e:
                    continue

            if db_inserts:
                log(f"💾 Pushe {len(db_inserts)} NEUE {tour_label} Matches aus {y} in Supabase Data Lake...")
                chunk_size = 500
                for i in range(0, len(db_inserts), chunk_size):
                    try:
                        supabase.table("historical_matches").insert(db_inserts[i:i+chunk_size]).execute()
                    except Exception as e:
                        if "duplicate key value" not in str(e): 
                            log(f"⚠️ Insert Error bei Chunk {i}: {str(e)}")
                            
                total_inserted += len(db_inserts)
            else:
                log(f"⏭️ Keine neuen Matches für {y} gefunden.")
            
    log(f"✅ DUAL-CORE DELTA SYNC ABGESCHLOSSEN: {total_inserted} neue Matches hinzugefügt.")


# =================================================================
# 5. PHASE 3: ELO, QUANT MATRIX & DURABILITY COMPILATION
# =================================================================
class AlchemistEngine:
    def __init__(self):
        self.players: Dict[int, Dict] = {}

    def init_player(self, p_id: int):
        if p_id not in self.players:
            self.players[p_id] = {
                "elo": {"overall": 1500.0, "hard": 1500.0, "clay": 1500.0, "grass": 1500.0},
                "matches_played": {"overall": 0, "hard": 0, "clay": 0, "grass": 0},
                "raw_stats": [],
                "history_q": [], 
                "overall_tracking": {"played": 0, "won": 0},
                "fatigue_tracking": {"played": 0, "won": 0}
            }

    def process_match(self, row: Dict):
        w_id = to_int(row.get('winner_sackmann_id'))
        l_id = to_int(row.get('loser_sackmann_id'))
        if not w_id or not l_id: return

        self.init_player(w_id)
        self.init_player(l_id)

        raw_surf = str(row.get('surface', 'hard')).lower()
        surf = raw_surf if raw_surf in ['hard', 'clay', 'grass'] else 'hard'
        m_date = parse_date(row.get('match_date'))

        w_svpt = to_int_safe(row.get('w_svpt'))
        l_svpt = to_int_safe(row.get('l_svpt'))
        total_pts = w_svpt + l_svpt
        
        if total_pts > 0:
            match_minutes = int(total_pts * (0.85 if surf == 'clay' else 0.75))
        else:
            match_minutes = 100

        def check_fatigue_state(p_id):
            self.players[p_id]["history_q"] = [d for d in self.players[p_id]["history_q"] if (m_date - d['date']).days <= 14]
            acute_mins = sum(d['minutes'] for d in self.players[p_id]["history_q"] if (m_date - d['date']).days <= 3)
            chronic_mins = sum(d['minutes'] for d in self.players[p_id]["history_q"])
            return acute_mins >= 200 or chronic_mins >= 600

        w_is_fatigued = check_fatigue_state(w_id)
        l_is_fatigued = check_fatigue_state(l_id)

        self.players[w_id]["overall_tracking"]["played"] += 1
        self.players[w_id]["overall_tracking"]["won"] += 1
        self.players[l_id]["overall_tracking"]["played"] += 1

        if w_is_fatigued:
            self.players[w_id]["fatigue_tracking"]["played"] += 1
            self.players[w_id]["fatigue_tracking"]["won"] += 1
        if l_is_fatigued:
            self.players[l_id]["fatigue_tracking"]["played"] += 1

        self.players[w_id]["history_q"].append({"date": m_date, "minutes": match_minutes})
        self.players[l_id]["history_q"].append({"date": m_date, "minutes": match_minutes})

        kw_overall = calculate_k_factor(self.players[w_id]["matches_played"]["overall"])
        kl_overall = calculate_k_factor(self.players[l_id]["matches_played"]["overall"])
        kw_surf = calculate_k_factor(self.players[w_id]["matches_played"][surf])
        kl_surf = calculate_k_factor(self.players[l_id]["matches_played"][surf])

        exp_w_over = calc_expected_score(self.players[w_id]["elo"]["overall"], self.players[l_id]["elo"]["overall"])
        exp_w_surf = calc_expected_score(self.players[w_id]["elo"][surf], self.players[l_id]["elo"][surf])

        self.players[w_id]["elo"]["overall"] = update_elo(self.players[w_id]["elo"]["overall"], kw_overall, 1.0, exp_w_over)
        self.players[l_id]["elo"]["overall"] = update_elo(self.players[l_id]["elo"]["overall"], kl_overall, 0.0, 1.0 - exp_w_over)
        self.players[w_id]["elo"][surf] = update_elo(self.players[w_id]["elo"][surf], kw_surf, 1.0, exp_w_surf)
        self.players[l_id]["elo"][surf] = update_elo(self.players[l_id]["elo"][surf], kl_surf, 0.0, 1.0 - exp_w_surf)

        self.players[w_id]["matches_played"]["overall"] += 1
        self.players[l_id]["matches_played"]["overall"] += 1
        self.players[w_id]["matches_played"][surf] += 1
        self.players[l_id]["matches_played"][surf] += 1

        def extract_stats(p_id, prefix, opp_prefix):
            svpt = to_int_safe(row.get(f'{prefix}svpt'))
            if svpt == 0: return 
            
            opp_svpt = to_int_safe(row.get(f'{opp_prefix}svpt'))
            opp_1stwon = to_int_safe(row.get(f'{opp_prefix}1stwon'))
            opp_2ndwon = to_int_safe(row.get(f'{opp_prefix}2ndwon'))
            
            self.players[p_id]["raw_stats"].append({
                "date": m_date,
                "surface": surf,
                "aces": to_int_safe(row.get(f'{prefix}ace')),
                "dfs": to_int_safe(row.get(f'{prefix}df')),
                "svpt": svpt,
                "1stin": to_int_safe(row.get(f'{prefix}1stin')),
                "1stwon": to_int_safe(row.get(f'{prefix}1stwon')),
                "2ndwon": to_int_safe(row.get(f'{prefix}2ndwon')),
                "bpsaved": to_int_safe(row.get(f'{prefix}bpsaved')),
                "bpfaced": to_int_safe(row.get(f'{prefix}bpfaced')),
                "ret_pts": opp_svpt,
                "ret_won": (opp_svpt - opp_1stwon - opp_2ndwon),
                "bp_opps": to_int_safe(row.get(f'{opp_prefix}bpfaced')),
                "bp_conv": (to_int_safe(row.get(f'{opp_prefix}bpfaced')) - to_int_safe(row.get(f'{opp_prefix}bpsaved')))
            })

        extract_stats(w_id, "w_", "l_")
        extract_stats(l_id, "l_", "w_")

    def compile_final_profiles(self) -> Dict[int, Dict]:
        final_data = {}
        now = datetime.now(timezone.utc)
        current_year = now.year
        thirty_days_ago = now - timedelta(days=30)
        
        for p_id, data in self.players.items():
            elo_metrics = {
                "overall": round(data["elo"]["overall"], 1),
                "hard": round(data["elo"]["hard"], 1),
                "clay": round(data["elo"]["clay"], 1),
                "grass": round(data["elo"]["grass"], 1),
                "matches_tracked": data["matches_played"]["overall"],
                "matches_hard": data["matches_played"]["hard"],
                "matches_clay": data["matches_played"]["clay"],
                "matches_grass": data["matches_played"]["grass"]
            }

            sorted_stats = sorted(data["raw_stats"], key=lambda x: x['date'], reverse=True)
            adv_stats = {"all": {}, "ytd": {}, "1m": {}, "l7": {}}
            
            for surf_key in ["overall", "hard", "clay", "grass"]:
                s_matches = sorted_stats if surf_key == "overall" else [m for m in sorted_stats if m['surface'] == surf_key]
                
                adv_stats["all"][surf_key] = aggregate_stats(s_matches)
                ytd_matches = [m for m in s_matches if m['date'].year == current_year]
                adv_stats["ytd"][surf_key] = aggregate_stats(ytd_matches)
                m1_matches = [m for m in s_matches if m['date'] >= thirty_days_ago]
                adv_stats["1m"][surf_key] = aggregate_stats(m1_matches)
                l7_matches = s_matches[:7]
                adv_stats["l7"][surf_key] = aggregate_stats(l7_matches)

            live_history = [d for d in data["history_q"] if (now - d['date']).days <= 14]
            chronic_minutes = sum(d['minutes'] for d in live_history)
            acute_minutes = sum(d['minutes'] for d in live_history if (now - d['date']).total_seconds() <= (72 * 3600))
            
            o_played = data["overall_tracking"]["played"]
            o_won = data["overall_tracking"]["won"]
            f_played = data["fatigue_tracking"]["played"]
            f_won = data["fatigue_tracking"]["won"]

            base_win_rate = (o_won + 5) / (o_played + 10)
            fatigue_win_rate = (f_won + (base_win_rate * 5)) / (f_played + 5)
            drop_off = base_win_rate - fatigue_win_rate

            durability_index = round(max(10.0, min(99.0, 80.0 - (drop_off * 300))))

            surface_ui = {}
            for surf in ['hard', 'clay', 'grass']:
                e_val = elo_metrics[surf]
                rating = ((e_val - 1400) / 700.0) * 9.0 + 1.0
                rating = max(1.0, min(10.0, rating))
                
                if rating >= 8.5: text, color = "🔥 ELITE", "#FF00FF"
                elif rating >= 7.0: text, color = "📈 STRONG", "#3366FF"
                elif rating >= 5.5: text, color = "✅ SOLID", "#00B25B"
                elif rating >= 4.0: text, color = "⚠️ VULNERABLE", "#F0C808"
                else: text, color = "❄️ WEAKNESS", "#CC0000"

                win_pct = round((1 / (1 + math.pow(10, (1500 - e_val)/400))) * 100, 1)
                surface_ui[surf] = {
                    "rating": round(rating, 1),
                    "color": color,
                    "matches_tracked": elo_metrics[f"matches_{surf}"],
                    "text": text,
                    "win_rate": f"{win_pct}% (True Elo)"
                }
            surface_ui['_v95_mastery_applied'] = True

            final_data[p_id] = {
                "elo_metrics": elo_metrics,
                "advanced_stats": adv_stats,
                "surface_ratings": surface_ui,
                "sackmann_metrics": {
                    "fatigue": {
                        "recent_14d_minutes": int(chronic_minutes),
                        "acute_72h_minutes": int(acute_minutes),
                        "durability_index": durability_index
                    }
                }
            }
        return final_data


async def compile_intelligence_matrix():
    log("🌊 Lade ALLE Matches aus Supabase (High-Speed Block Fetching)...")
    matches = []
    offset = 0
    
    # 🚀 SOTA FIX: Komplett ohne .order(), reines Offset-Streaming um Supabase Timeouts zu umgehen
    while True:
        res = supabase.table("historical_matches") \
            .select("winner_sackmann_id,loser_sackmann_id,surface,match_date,w_ace,w_df,w_svpt,w_1stin,w_1stwon,w_2ndwon,w_bpsaved,w_bpfaced,l_ace,l_df,l_svpt,l_1stin,l_1stwon,l_2ndwon,l_bpsaved,l_bpfaced") \
            .range(offset, offset + 999) \
            .execute()
            
        chunk = res.data or []
        matches.extend(chunk)
        
        if len(chunk) < 1000: 
            break
            
        offset += 1000
        
    log(f"✅ {len(matches)} Matches erfolgreich per Block-Fetch geladen.")
    
    log("⏱️ Sortiere Matches chronologisch in RAM...")
    # Wir sortieren extrem schnell lokal mit Python, statt die Datenbank zu belasten
    matches.sort(key=lambda x: str(x.get("match_date", "2015-01-01")))
    
    log("🧮 Simuliere Elo, Advanced Stats & True Match Load (Points-to-Minutes)...")
    engine = AlchemistEngine()
    for m in matches: engine.process_match(m)

    log("💾 Lade Supabase UUIDs & Player Names...")
    db_players = []
    offset = 0
    while True:
        res = supabase.table("players").select("id, sackmann_id, first_name, last_name").not_.is_("sackmann_id", "null").range(offset, offset + 999).execute()
        chunk = res.data or []
        db_players.extend(chunk)
        if len(chunk) < 1000: break
        offset += 1000

    name_to_id = {}
    for p in db_players:
        sid = p.get('sackmann_id')
        if sid:
            full_name = f"{p.get('first_name','')} {p.get('last_name','')}".strip().lower()
            last_name = str(p.get('last_name','')).strip().lower()
            name_to_id[full_name] = sid
            name_to_id[last_name] = sid

    log("🌊 Injeziere Live-Scanner Matches (Games-to-Minutes)...")
    fourteen_days_ago = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    res_live = supabase.table("market_odds").select("player1_name, player2_name, match_time, created_at, score").gte("created_at", fourteen_days_ago).execute()
    
    for lm in (res_live.data or []):
        date_str = lm.get('match_time') or lm.get('created_at')
        m_date = parse_date(date_str)
        
        score_str = str(lm.get('score', ''))
        total_games = get_total_games(score_str)
        calc_minutes = int(total_games * 4.5) if total_games > 0 else 100
        
        for p_col in ['player1_name', 'player2_name']:
            p_name = str(lm.get(p_col, '')).strip().lower()
            sid = name_to_id.get(p_name)
            if not sid:
                last = p_name.split()[-1] if p_name.split() else ''
                sid = name_to_id.get(last)
                
            if sid:
                engine.init_player(sid)
                engine.players[sid]["history_q"].append({"date": m_date, "minutes": calc_minutes})

    log("🧮 Kompiliere endgültige Quant-Profile...")
    compiled_data = engine.compile_final_profiles()

    log("🚀 Injeziere Matrix in Supabase (Player Skills & Players UI)...")
    success = 0
    for p in db_players:
        sid = p.get('sackmann_id')
        if sid in compiled_data:
            d = compiled_data[sid]
            try:
                supabase.table("player_skills").update({
                    "elo_metrics": d["elo_metrics"],
                    "advanced_stats": d["advanced_stats"],
                    "sackmann_metrics": d["sackmann_metrics"], 
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("player_id", p["id"]).execute()
                
                supabase.table("players").update({
                    "surface_ratings": d["surface_ratings"]
                }).eq("id", p["id"]).execute()
                success += 1
            except: pass
            
    log(f"🏁 ALCHEMIST FINISHED. {success} Spieler besitzen nun Gott-Level-Statistiken!")

# =================================================================
# EXECUTION
# =================================================================
async def main():
    await sync_player_ids()
    await build_historical_lake()
    await compile_intelligence_matrix()

if __name__ == "__main__":
    asyncio.run(main())
