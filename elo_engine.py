# -- coding: utf-8 --

import asyncio
import os
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Elo_Engine")

def log(msg: str):
    logger.info(msg)

log("⚡ Initialisiere Surface Elo Engine (Quantum Dynamics V1.0)...")

# Secrets
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# 2. ELO MATHEMATICS (The Quant Core)
# =================================================================
# K-Faktor: Wie stark reagiert das Rating?
# Wir nutzen eine exponentielle Decay-Funktion (wie im Original-Skript, aber optimiert)
def calculate_k_factor(matches_played: int) -> float:
    K_BASE = 250.0
    OFFSET = 5.0
    SHAPE = 0.4
    return K_BASE / ((matches_played + OFFSET) ** SHAPE)

def calc_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))

def update_elo(old_rating: float, k_factor: float, actual_score: float, expected_score: float) -> float:
    return old_rating + k_factor * (actual_score - expected_score)

# =================================================================
# 3. THE ENGINE
# =================================================================
class EloTracker:
    def __init__(self):
        # Struktur: player_sackmann_id -> {"overall": 1500, "hard": 1500, "matches_played": 0...}
        self.players: Dict[int, Dict[str, Any]] = {}
        
    def init_player(self, p_id: int):
        if p_id not in self.players:
            self.players[p_id] = {
                "overall": 1500.0,
                "hard": 1500.0,
                "clay": 1500.0,
                "grass": 1500.0,
                "matches_played_overall": 0,
                "matches_played_hard": 0,
                "matches_played_clay": 0,
                "matches_played_grass": 0,
                "peak_overall": 1500.0
            }

    def process_match(self, winner_id: int, loser_id: int, surface: str):
        if not winner_id or not loser_id: return
        
        self.init_player(winner_id)
        self.init_player(loser_id)
        
        surf_key = surface.lower() if surface else "hard"
        if surf_key not in ["hard", "clay", "grass"]:
            surf_key = "hard"

        # 1. Berechne K-Faktoren (Basierend auf Erfahrung)
        k_w_overall = calculate_k_factor(self.players[winner_id]["matches_played_overall"])
        k_l_overall = calculate_k_factor(self.players[loser_id]["matches_played_overall"])
        
        k_w_surf = calculate_k_factor(self.players[winner_id][f"matches_played_{surf_key}"])
        k_l_surf = calculate_k_factor(self.players[loser_id][f"matches_played_{surf_key}"])

        # 2. Hole aktuelle Elos
        elo_w_overall = self.players[winner_id]["overall"]
        elo_l_overall = self.players[loser_id]["overall"]
        
        elo_w_surf = self.players[winner_id][surf_key]
        elo_l_surf = self.players[loser_id][surf_key]

        # 3. Berechne Expected Scores
        exp_w_overall = calc_expected_score(elo_w_overall, elo_l_overall)
        exp_l_overall = 1.0 - exp_w_overall
        
        exp_w_surf = calc_expected_score(elo_w_surf, elo_l_surf)
        exp_l_surf = 1.0 - exp_w_surf

        # 4. Update Ratings (Winner Score = 1.0, Loser Score = 0.0)
        new_w_overall = update_elo(elo_w_overall, k_w_overall, 1.0, exp_w_overall)
        new_l_overall = update_elo(elo_l_overall, k_l_overall, 0.0, exp_l_overall)
        
        new_w_surf = update_elo(elo_w_surf, k_w_surf, 1.0, exp_w_surf)
        new_l_surf = update_elo(elo_l_surf, k_l_surf, 0.0, exp_l_surf)

        # 5. Speichere neue Werte
        self.players[winner_id]["overall"] = new_w_overall
        self.players[loser_id]["overall"] = new_l_overall
        
        self.players[winner_id][surf_key] = new_w_surf
        self.players[loser_id][surf_key] = new_l_surf

        # 6. Erhöhe Match-Counter
        self.players[winner_id]["matches_played_overall"] += 1
        self.players[loser_id]["matches_played_overall"] += 1
        
        self.players[winner_id][f"matches_played_{surf_key}"] += 1
        self.players[loser_id][f"matches_played_{surf_key}"] += 1

        # 7. Peak Tracking
        if new_w_overall > self.players[winner_id]["peak_overall"]:
            self.players[winner_id]["peak_overall"] = new_w_overall


async def fetch_historical_matches() -> List[Dict]:
    log("📡 Lade den gesamten historischen Data Lake aus Supabase...")
    all_matches = []
    offset = 0
    limit = 1000
    
    # 🚀 Wir laden strikt aufsteigend nach Datum, damit das Elo chronologisch wächst!
    while True:
        res = supabase.table("historical_matches").select("winner_sackmann_id, loser_sackmann_id, surface, match_date").order("match_date", desc=False).range(offset, offset + limit - 1).execute()
        chunk = res.data or []
        all_matches.extend(chunk)
        if len(chunk) < limit: break
        offset += limit
        
    log(f"✅ {len(all_matches)} historische Matches geladen.")
    return all_matches


async def push_elo_to_db(tracker: EloTracker):
    log("💾 Hole Spieler-Mappings aus Supabase für finales Elo-Update...")
    db_players = []
    offset = 0
    limit = 1000
    while True:
        res = supabase.table("players").select("id, sackmann_id").not_.is_("sackmann_id", "null").range(offset, offset + limit - 1).execute()
        chunk = res.data or []
        db_players.extend(chunk)
        if len(chunk) < limit: break
        offset += limit
        
    updates = []
    for p in db_players:
        sid = p.get('sackmann_id')
        if sid and sid in tracker.players:
            d = tracker.players[sid]
            
            # Formatiere das JSON für die Datenbank
            elo_metrics = {
                "overall": round(d["overall"], 1),
                "hard": round(d["hard"], 1),
                "clay": round(d["clay"], 1),
                "grass": round(d["grass"], 1),
                "peak_overall": round(d["peak_overall"], 1),
                "matches_tracked": d["matches_played_overall"]
            }
            
            updates.append({
                "player_id": p['id'],
                "elo_metrics": elo_metrics
            })

    log(f"🚀 Pushe {len(updates)} Elo-Profile in die player_skills Tabelle...")
    
    success_count = 0
    for u_data in updates:
        try:
            supabase.table("player_skills").update({
                "elo_metrics": u_data["elo_metrics"],
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("player_id", u_data["player_id"]).execute()
            success_count += 1
        except Exception as e:
            pass

    log(f"🏆 ELO ENGINE FINISHED. {success_count} Spieler haben nun tagesaktuelle Surface-Elos.")

# =================================================================
# MAIN EXECUTION
# =================================================================
async def main():
    # 1. Lade Daten
    historical_matches = await fetch_historical_matches()
    
    # 2. Initialisiere den Quant-Rechner
    tracker = EloTracker()
    
    # 3. Berechne History (Der "Warm-Up" für das System ab 2015)
    log("🧮 Simuliere Elo-Entwicklung ab 2015...")
    for m in historical_matches:
        w_id = m.get("winner_sackmann_id")
        l_id = m.get("loser_sackmann_id")
        surf = m.get("surface")
        tracker.process_match(w_id, l_id, surf)
        
    # 4. Push in DB
    await push_elo_to_db(tracker)

if __name__ == "__main__":
    asyncio.run(main())
