# -*- coding: utf-8 -*-

import asyncio
import os
import re
import logging
import sys
from typing import List, Dict, Any
from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("NeuralScout_Recalibrator")

def log(msg: str):
    logger.info(msg)

log("🔌 Initialisiere GRAND RECALIBRATION (Form & Surface Update für ALLE Spieler)...")

# Secrets Load (Genau wie in den anderen Scrapern)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Supabase Secrets fehlen! Prüfe GitHub Secrets.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
GLOBAL_SURFACE_MAP: Dict[str, str] = {} 

# =================================================================
# 2. SOTA MOMENTUM V3 ENGINE (xG Model)
# =================================================================
class MomentumV2Engine:  # Behalte den Namen "MomentumV2Engine" bei, damit der Rest des Codes nicht bricht!
    @staticmethod
    def calculate_rating(matches: List[Dict], player_name: str, max_matches: int = 10) -> Dict[str, Any]:
        if not matches: 
            return {"score": 6.5, "text": "Neutral (No Data)", "history_summary": "", "color_hex": "#808080"}

        recent_matches = sorted(matches, key=lambda x: str(x.get('created_at', '')), reverse=True)[:max_matches]
        chrono_matches = recent_matches[::-1]

        base_rating = 6.5 
        cumulative_edge = 0.0
        total_weight = 0.0
        history_log = []
        
        search_name = player_name.split()[-1].lower() if player_name else ""

        for idx, m in enumerate(chrono_matches):
            is_p1 = search_name in str(m.get('player1_name', '')).lower()
            winner = str(m.get('actual_winner_name', '')).lower()
            won = search_name in winner
            
            odds = to_float(m.get('odds1') if is_p1 else m.get('odds2'), 1.85)
            if odds <= 1.01: odds = 1.85
            
            expected_perf = 1 / odds 
            
            actual_perf = 0.5 
            score_str = str(m.get('score', '')).lower()
            
            if "ret" in score_str or "w.o" in score_str:
                actual_perf = 0.6 if won else 0.4
            else:
                sets = re.findall(r'(\d+)-(\d+)', score_str)
                
                if not sets:
                    actual_perf = 0.75 if won else 0.25
                else:
                    player_sets_won = 0
                    opp_sets_won = 0
                    player_games_won = 0
                    opp_games_won = 0
                    
                    for s in sets:
                        try:
                            l, r = int(s[0]), int(s[1])
                            p_games = l if is_p1 else r
                            o_games = r if is_p1 else l
                            
                            player_games_won += p_games
                            opp_games_won += o_games
                            
                            if p_games > o_games: player_sets_won += 1
                            elif o_games > p_games: opp_sets_won += 1
                        except: pass
                    
                    if won:
                        if opp_sets_won == 0: 
                            game_diff = player_games_won - opp_games_won
                            if game_diff >= 8: actual_perf = 1.0      
                            elif game_diff >= 5: actual_perf = 0.9    
                            elif game_diff >= 3: actual_perf = 0.8    
                            else: actual_perf = 0.7                   
                        else: 
                            game_diff = player_games_won - opp_games_won
                            if game_diff >= 4: actual_perf = 0.75     
                            elif game_diff >= 1: actual_perf = 0.65   
                            else: actual_perf = 0.55                  
                    else:
                        if player_sets_won == 1: 
                            game_diff = opp_games_won - player_games_won
                            if game_diff <= 1: actual_perf = 0.45     
                            elif game_diff <= 4: actual_perf = 0.35   
                            else: actual_perf = 0.25                  
                        else: 
                            game_diff = opp_games_won - player_games_won
                            if game_diff <= 3: actual_perf = 0.30     
                            elif game_diff <= 5: actual_perf = 0.20   
                            elif game_diff <= 7: actual_perf = 0.10   
                            else: actual_perf = 0.0                   

                        # --- 3. THE DELTA (Reality vs. Expectation) ---
            # Positiv = Overperformance / Negativ = Underperformance
            match_edge = actual_perf - expected_perf 
            
            # 🚀 DEIN NEUER SIEG-BONUS (Win-Override)
            if won:
                match_edge += 0.40  # <--- HIER: Erhöhe diese Zahl nach Belieben!
            
            # --- 4. TIME DECAY (Gewichtung) ---
            time_weight = 0.3 + (0.7 * (idx / max(1, len(chrono_matches) - 1)))
            
            cumulative_edge += (match_edge * time_weight)
            total_weight += time_weight
            
            history_log.append("W" if won else "L")

        streak_bonus = 0.0
        if len(history_log) >= 3:
            recent_3 = history_log[-3:]
            if recent_3 == ["W", "W", "W"]: streak_bonus = 0.4
            elif recent_3 == ["L", "L", "L"]: streak_bonus = -0.4
            if len(history_log) >= 5:
                recent_5 = history_log[-5:]
                if recent_5.count("W") == 5: streak_bonus = 0.8
                elif recent_5.count("L") == 5: streak_bonus = -0.8

        avg_edge = (cumulative_edge / total_weight) if total_weight > 0 else 0.0
        
        final_rating = base_rating + (avg_edge * 10.0) + streak_bonus
        final_rating = max(1.0, min(10.0, final_rating))
        
        desc = "Average"
        color_hex = "#F0C808" 
        
        if final_rating >= 8.5: 
            desc = "🔥 ELITE"
            color_hex = "#FF00FF" 
        elif final_rating >= 7.2: 
            desc = "📈 Strong"
            color_hex = "#3366FF" 
        elif final_rating >= 6.0: 
            desc = "Solid"
            color_hex = "#00B25B" 
        elif final_rating >= 4.5: 
            desc = "⚠️ Vulnerable"
            color_hex = "#FF9933" 
        else: 
            desc = "❄️ Cold"
            color_hex = "#CC0000" 

        return {
            "score": round(final_rating, 2),
            "text": desc,
            "color_hex": color_hex,
            "history_summary": "".join(history_log[-5:])
        }

def to_float(val: Any, default: float = 0.0) -> float:
    if val is None: 
        return default
    try: 
        return float(val)
    except: 
        return default

class SurfaceIntelligence:
    @staticmethod
    def normalize_surface_key(raw_surface: str) -> str:
        if not raw_surface: return "unknown"
        s = raw_surface.lower()
        if "grass" in s: return "grass"
        if "clay" in s or "sand" in s: return "clay"
        if "hard" in s or "carpet" in s or "acrylic" in s or "indoor" in s: return "hard"
        return "unknown"

    @staticmethod
    def get_matches_by_surface(all_matches: List[Dict], target_surface: str) -> List[Dict]:
        filtered = []
        target = SurfaceIntelligence.normalize_surface_key(target_surface)
        
        for m in all_matches:
            tour_name = str(m.get('tournament', '')).lower()
            ai_text = str(m.get('ai_analysis_text', '')).lower()
            found_surface = "unknown"
            
            match_hist = re.search(r'surface:\s*(hard|clay|grass)', ai_text)
            if match_hist: found_surface = match_hist.group(1)
            elif "hard court" in ai_text or "hard surface" in ai_text: found_surface = "hard"
            elif "red clay" in ai_text or "clay court" in ai_text: found_surface = "clay"
            elif "grass court" in ai_text: found_surface = "grass"
            elif "clay" in tour_name or "roland garros" in tour_name: found_surface = "clay"
            elif "grass" in tour_name or "wimbledon" in tour_name: found_surface = "grass"
            elif "hard" in tour_name or "us open" in tour_name or "australian open" in tour_name: found_surface = "hard"
            
            if SurfaceIntelligence.normalize_surface_key(found_surface) == target:
                filtered.append(m)
        
        return filtered

    @staticmethod
    def compute_player_surface_profile(matches: List[Dict], player_name: str) -> Dict[str, Any]:
        profile = {}
        surfaces_data = {
            "hard": SurfaceIntelligence.get_matches_by_surface(matches, "hard"),
            "clay": SurfaceIntelligence.get_matches_by_surface(matches, "clay"),
            "grass": SurfaceIntelligence.get_matches_by_surface(matches, "grass")
        }
        
        # 🚀 SOTA FIX: Smart Matcher
        search_name = player_name.split()[-1].lower() if player_name else ""

        for surf, surf_matches in surfaces_data.items():
            n_surf = len(surf_matches)
            if n_surf == 0:
                profile[surf] = {"rating": 3.5, "color": "#808080", "matches_tracked": 0, "text": "No Experience"}
                continue
                
            wins = sum(1 for m in surf_matches if search_name in str(m.get('actual_winner_name', '')).lower())
            win_rate = wins / n_surf
            
            vol_score = min(1.0, n_surf / 30.0) * 1.95
            win_score = win_rate * 4.55
            
            final_rating = max(1.0, min(10.0, 3.5 + vol_score + win_score))
            
            desc = "Average"
            if final_rating >= 8.5: desc = "🔥 SPECIALIST"
            elif final_rating >= 7.0: desc = "📈 Strong"
            elif final_rating >= 5.5: desc = "Solid"
            elif final_rating >= 4.5: desc = "⚠️ Vulnerable"
            else: desc = "❄️ Weakness"
            
            color_hex = "#F0C808" 
            if final_rating >= 8.5: color_hex = "#FF00FF" 
            elif final_rating >= 7.5: color_hex = "#3366FF" 
            elif final_rating >= 6.5: color_hex = "#00B25B" 
            elif final_rating >= 5.5: color_hex = "#99CC33" 
            elif final_rating <= 4.5: color_hex = "#CC0000" 
            elif final_rating < 5.5: color_hex = "#FF9933" 

            profile[surf] = {"rating": round(final_rating, 2), "color": color_hex, "matches_tracked": n_surf, "text": desc}
            
        profile['_v95_mastery_applied'] = True
        return profile

# =================================================================
# 3. DER REKALIBRIERUNGS-LOOP
# =================================================================
async def run_recalibration():
    log("📥 Lade alle Spieler aus der Datenbank...")
    res = supabase.table("players").select("id, first_name, last_name").execute()
    players = res.data or []
    
    log(f"✅ {len(players)} Spieler gefunden. Starte Sync...")
    
    updated_count = 0
    for i, p in enumerate(players):
        full_name = p.get('last_name')
        if not full_name:
            continue
            
        try:
            # 🚀 SOTA FIX: Suche in der Match-Historie nach dem letzten Namensteil
            search_name = full_name.split()[-1]
            
            hist_res = supabase.table("market_odds").select("*").or_(
                f"player1_name.ilike.%{search_name}%,player2_name.ilike.%{search_name}%"
            ).order("created_at", desc=True).limit(40).execute()
            
            matches = hist_res.data or []
            
            # Berechne neue Ratings (Die Engine verarbeitet die Namen jetzt richtig!)
            new_form = MomentumV2Engine.calculate_rating(matches, full_name)
            new_surface = SurfaceIntelligence.compute_player_surface_profile(matches, full_name)
            
            # Update Datenbank
            supabase.table("players").update({
                "form_rating": new_form,
                "surface_ratings": new_surface
            }).eq("id", p['id']).execute()
            
            updated_count += 1
            log(f"[{i+1}/{len(players)}] 🔄 {full_name} geupdatet -> Form: {new_form['score']} ({new_form['history_summary']})")
            
            # Kurze Pause, um die Datenbank nicht zu überlasten
            await asyncio.sleep(0.05)
            
        except Exception as e:
            log(f"⚠️ Fehler bei {full_name}: {e}")

    log(f"🏁 GRAND RECALIBRATION ABGESCHLOSSEN! {updated_count} Spieler wurden aktualisiert.")

if __name__ == "__main__":
    asyncio.run(run_recalibration())
