# -- coding: utf-8 --

import asyncio
import os
import math
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("Quant_Alchemist")

def log(msg: str): logger.info(msg)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def to_int(val: Any) -> int:
    try: return int(float(val))
    except: return 0

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

# --- ELO MATH ---
def calculate_k_factor(matches_played: int) -> float:
    return 250.0 / ((matches_played + 5.0) ** 0.4)

def calc_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))

def update_elo(old_rating: float, k_factor: float, actual_score: float, expected_score: float) -> float:
    return old_rating + k_factor * (actual_score - expected_score)

# 🚀 SOTA FIX: Zählt nur Matches, die auch echte Stats haben (verhindert Dilution durch Live-Matches)
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

# --- THE AGGREGATOR ---
class AlchemistEngine:
    def __init__(self):
        self.players: Dict[int, Dict] = {}

    def init_player(self, p_id: int):
        if p_id not in self.players:
            self.players[p_id] = {
                "elo": {"overall": 1500.0, "hard": 1500.0, "clay": 1500.0, "grass": 1500.0},
                "matches_played": {"overall": 0, "hard": 0, "clay": 0, "grass": 0},
                "raw_stats": [] 
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

        # 1. ELO CALCULATION
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

        # 2. TIME-SERIES STATS COLLECTION
        def extract_stats(p_id, prefix, opp_prefix):
            svpt = to_int(row.get(f'{prefix}svpt'))
            if svpt == 0: return # Skip missing stats
            
            opp_svpt = to_int(row.get(f'{opp_prefix}svpt'))
            opp_1stwon = to_int(row.get(f'{opp_prefix}1stwon'))
            opp_2ndwon = to_int(row.get(f'{opp_prefix}2ndwon'))
            
            self.players[p_id]["raw_stats"].append({
                "date": m_date,
                "surface": surf,
                "aces": to_int(row.get(f'{prefix}ace')),
                "dfs": to_int(row.get(f'{prefix}df')),
                "svpt": svpt,
                "1stin": to_int(row.get(f'{prefix}1stin')),
                "1stwon": to_int(row.get(f'{prefix}1stwon')),
                "2ndwon": to_int(row.get(f'{prefix}2ndwon')),
                "bpsaved": to_int(row.get(f'{prefix}bpsaved')),
                "bpfaced": to_int(row.get(f'{prefix}bpfaced')),
                "ret_pts": opp_svpt,
                "ret_won": (opp_svpt - opp_1stwon - opp_2ndwon),
                "bp_opps": to_int(row.get(f'{opp_prefix}bpfaced')),
                "bp_conv": (to_int(row.get(f'{opp_prefix}bpfaced')) - to_int(row.get(f'{opp_prefix}bpsaved')))
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

            # 🚀 TIME SERIES BUCKETING
            sorted_stats = sorted(data["raw_stats"], key=lambda x: x['date'], reverse=True)
            adv_stats = {"all": {}, "ytd": {}, "1m": {}, "l7": {}}
            
            for surf_key in ["overall", "hard", "clay", "grass"]:
                s_matches = sorted_stats if surf_key == "overall" else [m for m in sorted_stats if m['surface'] == surf_key]
                
                # ALL
                adv_stats["all"][surf_key] = aggregate_stats(s_matches)
                # YTD
                ytd_matches = [m for m in s_matches if m['date'].year == current_year]
                adv_stats["ytd"][surf_key] = aggregate_stats(ytd_matches)
                # 1M
                m1_matches = [m for m in s_matches if m['date'] >= thirty_days_ago]
                adv_stats["1m"][surf_key] = aggregate_stats(m1_matches)
                # L7 (Die letzten 7 dieses Belags)
                l7_matches = s_matches[:7]
                adv_stats["l7"][surf_key] = aggregate_stats(l7_matches)

            # 🚀 SOTA: FATIGUE / LOAD MANAGEMENT (ACWR & Durability Index)
            recent_14d_matches = [m for m in sorted_stats if (now - m['date']).days <= 14]
            recent_72h_matches = [m for m in sorted_stats if (now - m['date']).total_seconds() <= (72 * 3600)]
            
            # Basis-Minuten (~100 pro Match)
            base_minutes = len(recent_14d_matches) * 100
            acute_minutes = len(recent_72h_matches) * 100
            
            # ACWR Multiplikator: Wenn ein Spieler einen extremen Spike in 72h hat
            fatigue_multiplier = 1.0
            if acute_minutes >= 200: # Mehr als ~2 Matches in 3 Tagen
                fatigue_multiplier = 1.5 + ((acute_minutes - 200) / 100.0) * 0.25
                
            estimated_minutes = int(base_minutes * fatigue_multiplier)
            
            # 🧬 Durability Index (Robustheits-Faktor V1)
            # Spieler mit massiver Karriere-Erfahrung (viel Volume) haben in der Regel eine bessere Basis-Robustheit
            total_history = data["matches_played"]["overall"]
            durability_index = round(min(95.0, max(40.0, 55.0 + (total_history / 25.0))), 1)

            # UI Surface Ratings (SOTA Lineare Interpolation)
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
                        "recent_14d_minutes": estimated_minutes,
                        "acute_72h_minutes": acute_minutes,
                        "durability_index": durability_index
                    }
                }
            }
        return final_data


async def main():
    log("🌊 Lade 135.000+ Matches aus dem Data Lake...")
    matches = []
    offset = 0
    while True:
        res = supabase.table("historical_matches").select("winner_sackmann_id,loser_sackmann_id,surface,match_date,w_ace,w_df,w_svpt,w_1stin,w_1stwon,w_2ndwon,w_bpsaved,w_bpfaced,l_ace,l_df,l_svpt,l_1stin,l_1stwon,l_2ndwon,l_bpsaved,l_bpfaced").order("match_date", desc=False).range(offset, offset + 999).execute()
        chunk = res.data or []
        matches.extend(chunk)
        if len(chunk) < 1000: break
        offset += 1000
    
    log("🧮 Simuliere Elo & Aggregiere Time-Series Advanced Stats & Fatigue...")
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

    # 🚀 SOTA FIX: Namens-Mapping für die Live-Fatigue Injection
    name_to_id = {}
    for p in db_players:
        sid = p.get('sackmann_id')
        if sid:
            full_name = f"{p.get('first_name','')} {p.get('last_name','')}".strip().lower()
            last_name = str(p.get('last_name','')).strip().lower()
            name_to_id[full_name] = sid
            name_to_id[last_name] = sid

    log("🌊 Lade aktuelle Scanner-Matches für exaktes Fatigue-Tracking...")
    fourteen_days_ago = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    res_live = supabase.table("market_odds").select("player1_name, player2_name, match_time, created_at").gte("created_at", fourteen_days_ago).execute()
    
    for lm in (res_live.data or []):
        date_str = lm.get('match_time') or lm.get('created_at')
        m_date = parse_date(date_str)
        
        for p_col in ['player1_name', 'player2_name']:
            p_name = str(lm.get(p_col, '')).strip().lower()
            sid = name_to_id.get(p_name)
            if not sid:
                last = p_name.split()[-1] if p_name.split() else ''
                sid = name_to_id.get(last)
                
            if sid:
                # Füge das Live-Match als Geister-Statistik hinzu (nur für Minuten-Zählung)
                engine.init_player(sid)
                engine.players[sid]["raw_stats"].append({
                    "date": m_date,
                    "surface": "hard", 
                    "aces": 0, "dfs": 0, "svpt": 0, "1stin": 0, "1stwon": 0, "2ndwon": 0,
                    "bpsaved": 0, "bpfaced": 0, "ret_pts": 0, "ret_won": 0, "bp_opps": 0, "bp_conv": 0
                })

    log("🧮 Kompiliere endgültige Profile...")
    compiled_data = engine.compile_final_profiles()

    log("🚀 Injeziere Matrix in Supabase (Player Skills & Players UI)...")
    success = 0
    for p in db_players:
        sid = p.get('sackmann_id')
        if sid in compiled_data:
            d = compiled_data[sid]
            try:
                # Update Skills (Backend)
                supabase.table("player_skills").update({
                    "elo_metrics": d["elo_metrics"],
                    "advanced_stats": d["advanced_stats"],
                    "sackmann_metrics": d["sackmann_metrics"], 
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("player_id", p["id"]).execute()
                
                # Update UI (Frontend Fix)
                supabase.table("players").update({
                    "surface_ratings": d["surface_ratings"]
                }).eq("id", p["id"]).execute()
                success += 1
            except: pass
            
    log(f"🏁 ALCHEMIST FINISHED. {success} Spieler besitzen nun Gott-Level-Statistiken (inkl. Live-Load Management)!")

if __name__ == "__main__":
    asyncio.run(main())
