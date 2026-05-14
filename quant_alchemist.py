# -- coding: utf-8 --

import asyncio
import os
import math
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

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

# --- ELO MATH ---
def calculate_k_factor(matches_played: int) -> float:
    return 250.0 / ((matches_played + 5.0) ** 0.4)

def calc_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))

def update_elo(old_rating: float, k_factor: float, actual_score: float, expected_score: float) -> float:
    return old_rating + k_factor * (actual_score - expected_score)

# --- THE AGGREGATOR ---
class AlchemistEngine:
    def __init__(self):
        self.players: Dict[int, Dict] = {}

    def init_player(self, p_id: int):
        if p_id not in self.players:
            self.players[p_id] = {
                "elo": {"overall": 1500.0, "hard": 1500.0, "clay": 1500.0, "grass": 1500.0},
                "matches_played": {"overall": 0, "hard": 0, "clay": 0, "grass": 0},
                "stats": {
                    "hard": {"matches":0,"aces":0,"dfs":0,"svpt":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0,"ret_pts":0,"ret_won":0,"bp_opps":0,"bp_conv":0},
                    "clay": {"matches":0,"aces":0,"dfs":0,"svpt":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0,"ret_pts":0,"ret_won":0,"bp_opps":0,"bp_conv":0},
                    "grass": {"matches":0,"aces":0,"dfs":0,"svpt":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0,"ret_pts":0,"ret_won":0,"bp_opps":0,"bp_conv":0},
                    "overall": {"matches":0,"aces":0,"dfs":0,"svpt":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0,"ret_pts":0,"ret_won":0,"bp_opps":0,"bp_conv":0}
                }
            }

    def process_match(self, row: Dict):
        w_id = to_int(row.get('winner_sackmann_id'))
        l_id = to_int(row.get('loser_sackmann_id'))
        if not w_id or not l_id: return

        self.init_player(w_id)
        self.init_player(l_id)

        raw_surf = str(row.get('surface', 'hard')).lower()
        surf = raw_surf if raw_surf in ['hard', 'clay', 'grass'] else 'hard'

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

        # 2. STATS AGGREGATION
        def add_stats(p_id, s_type, prefix, opp_prefix):
            if to_int(row.get(f'{prefix}svpt')) == 0: return # Skip if no detailed stats
            for target_surf in [surf, "overall"]:
                s = self.players[p_id]["stats"][target_surf]
                s["matches"] += 1
                s["aces"] += to_int(row.get(f'{prefix}ace'))
                s["dfs"] += to_int(row.get(f'{prefix}df'))
                s["svpt"] += to_int(row.get(f'{prefix}svpt'))
                s["1stin"] += to_int(row.get(f'{prefix}1stin'))
                s["1stwon"] += to_int(row.get(f'{prefix}1stwon'))
                s["2ndwon"] += to_int(row.get(f'{prefix}2ndwon'))
                s["bpsaved"] += to_int(row.get(f'{prefix}bpsaved'))
                s["bpfaced"] += to_int(row.get(f'{prefix}bpfaced'))

                opp_svpt = to_int(row.get(f'{opp_prefix}svpt'))
                opp_1stwon = to_int(row.get(f'{opp_prefix}1stwon'))
                opp_2ndwon = to_int(row.get(f'{opp_prefix}2ndwon'))
                s["ret_pts"] += opp_svpt
                s["ret_won"] += (opp_svpt - opp_1stwon - opp_2ndwon)

                s["bp_opps"] += to_int(row.get(f'{opp_prefix}bpfaced'))
                s["bp_conv"] += (to_int(row.get(f'{opp_prefix}bpfaced')) - to_int(row.get(f'{opp_prefix}bpsaved')))

        add_stats(w_id, surf, "w_", "l_")
        add_stats(l_id, surf, "l_", "w_")

    def compile_final_profiles(self) -> Dict[int, Dict]:
        final_data = {}
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

            adv_stats = {}
            for s_key in ["hard", "clay", "grass", "overall"]:
                s = data["stats"][s_key]
                m_count = s["matches"]
                svpt = s["svpt"]
                if m_count == 0 or svpt == 0:
                    adv_stats[s_key] = None
                    continue

                adv_stats[s_key] = {
                    "matches_with_stats": m_count,
                    "aces_per_match": round(s["aces"] / m_count, 1),
                    "df_per_match": round(s["dfs"] / m_count, 1),
                    "first_in_pct": round((s["1stin"] / svpt) * 100, 1),
                    "first_win_pct": round((s["1stwon"] / s["1stin"]) * 100, 1) if s["1stin"] > 0 else 0,
                    "second_win_pct": round((s["2ndwon"] / (svpt - s["1stin"])) * 100, 1) if (svpt - s["1stin"]) > 0 else 0,
                    "bp_saved_pct": round((s["bpsaved"] / s["bpfaced"]) * 100, 1) if s["bpfaced"] > 0 else 0,
                    "ret_win_pct": round((s["ret_won"] / s["ret_pts"]) * 100, 1) if s["ret_pts"] > 0 else 0,
                    "bp_conv_pct": round((s["bp_conv"] / s["bp_opps"]) * 100, 1) if s["bp_opps"] > 0 else 0
                }

            # UI Surface Ratings (Pre-Calculated for Frontend)
            def get_rating_info(elo_val):
                if elo_val >= 1850: return 9.5, "🔥 SPECIALIST", "#FF00FF"
                elif elo_val >= 1700: return 8.0, "📈 Strong", "#3366FF"
                elif elo_val >= 1550: return 6.5, "Solid", "#00B25B"
                elif elo_val >= 1400: return 5.0, "Average", "#F0C808"
                else: return 3.5, "❄️ Weakness", "#CC0000"

            surface_ui = {}
            for surf in ['hard', 'clay', 'grass']:
                e_val = elo_metrics[surf]
                rating, text, color = get_rating_info(e_val)
                win_pct = round((1 / (1 + math.pow(10, (1500 - e_val)/400))) * 100, 1)
                surface_ui[surf] = {
                    "rating": rating,
                    "color": color,
                    "matches_tracked": elo_metrics[f"matches_{surf}"],
                    "text": text,
                    "win_rate": f"{win_pct}% (True Elo)"
                }
            surface_ui['_v95_mastery_applied'] = True

            final_data[p_id] = {
                "elo_metrics": elo_metrics,
                "advanced_stats": adv_stats,
                "surface_ratings": surface_ui
            }
        return final_data


async def main():
    log("🌊 Lade 135.000+ Matches aus dem Data Lake...")
    matches = []
    offset = 0
    while True:
        res = supabase.table("historical_matches").select("winner_sackmann_id,loser_sackmann_id,surface,w_ace,w_df,w_svpt,w_1stin,w_1stwon,w_2ndwon,w_bpsaved,w_bpfaced,l_ace,l_df,l_svpt,l_1stin,l_1stwon,l_2ndwon,l_bpsaved,l_bpfaced").order("match_date", desc=False).range(offset, offset + 999).execute()
        chunk = res.data or []
        matches.extend(chunk)
        if len(chunk) < 1000: break
        offset += 1000
    
    log("🧮 Simuliere Elo & Aggregiere Advanced Stats...")
    engine = AlchemistEngine()
    for m in matches: engine.process_match(m)
    compiled_data = engine.compile_final_profiles()

    log("💾 Lade Supabase UUIDs...")
    db_players = []
    offset = 0
    while True:
        res = supabase.table("players").select("id, sackmann_id").not_.is_("sackmann_id", "null").range(offset, offset + 999).execute()
        chunk = res.data or []
        db_players.extend(chunk)
        if len(chunk) < 1000: break
        offset += 1000

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
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("player_id", p["id"]).execute()
                # Update UI (Frontend Fix)
                supabase.table("players").update({
                    "surface_ratings": d["surface_ratings"]
                }).eq("id", p["id"]).execute()
                success += 1
            except: pass
            
    log(f"🏁 ALCHEMIST FINISHED. {success} Spieler besitzen nun Gott-Level-Statistiken!")

if __name__ == "__main__":
    asyncio.run(main())
