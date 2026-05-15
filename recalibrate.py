# -- coding: utf-8 --

import asyncio
import os
import re
import unicodedata
import math
import logging
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

from supabase import create_client, Client

# =================================================================
# 1. CONFIGURATION & LOGGING
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Schalldämpfer für externe Bibliotheken
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("UI_Sync_Engine")

def log(msg: str):
    logger.info(msg)

log("⚡ Initialisiere Mass UI-Sync Engine (V3.1 DUAL-CORE OMNI SYNC)...")

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
def to_float(val: Any, default: float = 0.0) -> float:
    if val is None: 
        return default
    try: 
        return float(val)
    except: 
        return default

def normalize_db_name(name: str) -> str:
    if not name: 
        return ""
    n = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    n = n.lower().strip()
    n = n.replace('-', ' ').replace("'", "")
    n = re.sub(r'\b(de|van|von|der)\b', '', n).strip()
    return n

def is_same_player(target_name: str, db_name: str) -> bool:
    t_norm = normalize_db_name(target_name)
    d_norm = normalize_db_name(db_name)
    if t_norm == d_norm: return True
    
    t_parts = t_norm.split()
    d_parts = d_norm.split()
    if not t_parts or not d_parts: return False
    
    if t_parts[-1] != d_parts[-1]: return False
    
    t_first = t_parts[0] if len(t_parts) > 1 else ""
    d_first = d_parts[0] if len(d_parts) > 1 else ""
    
    if t_first and d_first:
        return t_first[0] == d_first[0]
        
    return True

# =================================================================
# 3. SOTA MOMENTUM V3 ENGINE (FORM RATING)
# =================================================================
class MomentumV2Engine:  
    @staticmethod
    def calculate_rating(matches: List[Dict], player_name: str, max_matches: int = 15) -> Dict[str, Any]:
        if not matches: 
            return {"score": 6.5, "text": "Neutral (No Data)", "history_summary": "", "color_hex": "#808080"}

        recent_matches = sorted(matches, key=lambda x: str(x.get('created_at', '')), reverse=True)[:max_matches]
        chrono_matches = recent_matches[::-1]

        base_rating = 6.5 
        cumulative_edge = 0.0
        total_weight = 0.0
        history_log = []
        
        for idx, m in enumerate(chrono_matches):
            p1_str = str(m.get('player1_name', ''))
            p2_str = str(m.get('player2_name', ''))
            
            is_p1 = is_same_player(player_name, p1_str)
            is_p2 = is_same_player(player_name, p2_str)
            
            if not is_p1 and not is_p2: continue

            winner = str(m.get('actual_winner_name', ''))
            won = is_same_player(player_name, winner)
            
            odds = to_float(m.get('odds1') if is_p1 else m.get('odds2'), 1.85)
            if odds <= 1.01: odds = 1.85
            
            expected_perf = 1 / odds 
            actual_perf = 0.5 
            
            score_str = str(m.get('score', '')).lower().replace(":", "-").strip()
            score_str = re.sub(r'\.\d+', '', score_str)
            
            if "ret" in score_str or "w.o" in score_str:
                actual_perf = 0.6 if won else 0.4
            else:
                sets = re.findall(r'\b(\d+)\s*-\s*(\d+)\b', score_str)
                
                if not sets:
                    actual_perf = 0.75 if won else 0.25
                elif len(sets) == 1 and (int(sets[0][0]) + int(sets[0][1]) <= 5):
                    l, r = int(sets[0][0]), int(sets[0][1])
                    p_sets = l if is_p1 else r
                    o_sets = r if is_p1 else l
                    
                    if p_sets >= o_sets + 2 or (p_sets == 2 and o_sets == 0):
                        actual_perf = 0.85  
                    elif p_sets > o_sets:
                        actual_perf = 0.65  
                    elif o_sets >= p_sets + 2 or (o_sets == 2 and p_sets == 0):
                        actual_perf = 0.15  
                    elif o_sets > p_sets:
                        actual_perf = 0.35  
                    else:
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

            match_edge = actual_perf - expected_perf 
            
            if won:
                match_edge += 0.40  
            else:
                match_edge -= 0.20
            
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

# =================================================================
# 4. SURFACE INTELLIGENCE ENGINE (TRUE ELO UI SYNC)
# =================================================================
class SurfaceIntelligence:
    @staticmethod
    def compute_player_surface_profile(elo_metrics: Dict, sackmann_metrics: Dict) -> Dict[str, Any]:
        profile = {}
        
        def get_rating_info(elo_val: float):
            if elo_val >= 1850: return 9.5, "🔥 SPECIALIST", "#FF00FF"
            elif elo_val >= 1700: return 8.0, "📈 Strong", "#3366FF"
            elif elo_val >= 1550: return 6.5, "Solid", "#00B25B"
            elif elo_val >= 1400: return 5.0, "Average", "#F0C808"
            else: return 3.5, "❄️ Weakness", "#CC0000"

        if not isinstance(elo_metrics, dict): elo_metrics = {}
        if not isinstance(sackmann_metrics, dict): sackmann_metrics = {}

        for surf in ['hard', 'clay', 'grass']:
            e_val = elo_metrics.get(surf, 1500)
            
            rating = ((e_val - 1400) / 700.0) * 9.0 + 1.0
            rating = max(1.0, min(10.0, rating))
            
            if rating >= 8.5: text, color = "🔥 ELITE", "#FF00FF"
            elif rating >= 7.0: text, color = "📈 STRONG", "#3366FF"
            elif rating >= 5.5: text, color = "✅ SOLID", "#00B25B"
            elif rating >= 4.0: text, color = "⚠️ VULNERABLE", "#F0C808"
            else: text, color = "❄️ WEAKNESS", "#CC0000"

            expected_win_pct = round((1 / (1 + math.pow(10, (1500 - e_val)/400))) * 100, 1)

            profile[surf] = {
                "rating": round(rating, 1),
                "color": color,
                "matches_tracked": elo_metrics.get(f"matches_{surf}", 0),
                "text": text,
                "win_rate": f"{expected_win_pct}% (True Elo)"
            }

        profile['_v95_mastery_applied'] = True
        return profile

# =================================================================
# 5. DATA FETCHING (HYBRID - DUAL CORE)
# =================================================================
async def fetch_player_history_extended(player_last_name: str, limit: int = 20) -> List[Dict]:
    try:
        # Live Scanner
        res_live = supabase.table("market_odds").select("player1_name, player2_name, odds1, odds2, actual_winner_name, score, created_at, tournament").or_(f"player1_name.ilike.%{player_last_name}%,player2_name.ilike.%{player_last_name}%").not_.is_("actual_winner_name", "null").order("created_at", desc=True).limit(limit).execute()
        live = res_live.data or []

        # Data Lake (SOTA V3: ATP & WTA support)
        res_hist = supabase.table("historical_matches").select("winner_name, loser_name, match_date, score, tourney_name, surface").or_(f"winner_name.ilike.%{player_last_name}%,loser_name.ilike.%{player_last_name}%").order("match_date", desc=True).limit(limit).execute()
        hist = res_hist.data or []

        combined = []
        for m in live:
            combined.append({
                "player1_name": m["player1_name"],
                "player2_name": m["player2_name"],
                "actual_winner_name": m["actual_winner_name"],
                "score": m["score"],
                "created_at": m["created_at"],
                "odds1": m.get("odds1", 1.85),
                "odds2": m.get("odds2", 1.85),
            })
            
        for m in hist:
            combined.append({
                "player1_name": m["winner_name"],
                "player2_name": m["loser_name"],
                "actual_winner_name": m["winner_name"],
                "score": m["score"],
                "created_at": m["match_date"] + "T00:00:00Z", 
                "odds1": 1.85, 
                "odds2": 1.85,
            })

        combined.sort(key=lambda x: str(x["created_at"]), reverse=True)

        seen = set()
        deduped = []
        for m in combined:
            is_p1 = player_last_name.lower() in m["player1_name"].lower()
            opp = m["player2_name"] if is_p1 else m["player1_name"]
            date = str(m["created_at"]).split("T")[0]
            k = f"{date}_{opp.split()[-1].lower()}"
            if k not in seen:
                seen.add(k)
                deduped.append(m)

        return deduped[:limit]
    except Exception as e:
        log(f"History Fetch Error: {e}")
        return []

def fetch_all_players() -> List[Dict]:
    data = []
    offset = 0
    limit = 1000
    while True:
        try:
            res = supabase.table("players").select("id, last_name, first_name").range(offset, offset + limit - 1).execute()
            chunk = res.data or []
            data.extend(chunk)
            if len(chunk) < limit: break
            offset += limit
        except Exception as e:
            log(f"⚠️ Pagination error on players: {e}")
            break
    return data

def fetch_all_skills() -> Dict[str, Dict]:
    data = []
    offset = 0
    limit = 1000
    while True:
        try:
            res = supabase.table("player_skills").select("player_id, elo_metrics, sackmann_metrics").range(offset, offset + limit - 1).execute()
            chunk = res.data or []
            data.extend(chunk)
            if len(chunk) < limit: break
            offset += limit
        except Exception as e:
            break
            
    skills_map = {}
    for entry in data:
        pid = entry.get('player_id')
        if pid:
            skills_map[pid] = {
                'elo_metrics': entry.get('elo_metrics', {}),
                'sackmann_metrics': entry.get('sackmann_metrics', {})
            }
    return skills_map

# =================================================================
# MAIN EXECUTION (THE BULLDOZER)
# =================================================================
async def run_sync():
    log("📥 Lade alle Spieler aus der Datenbank...")
    players = fetch_all_players()
    log(f"✅ {len(players)} Spieler gefunden.")
    
    log("📥 Lade alle Skills/Elos aus der Datenbank...")
    skills_map = fetch_all_skills()
    log(f"✅ {len(skills_map)} Skill-Profile gefunden.")
    
    log("🚀 Starte Batch-Update der UI-Metrics...")
    
    updated_count = 0
    
    chunk_size = 50
    for i in range(0, len(players), chunk_size):
        chunk = players[i:i+chunk_size]
        log(f"🔄 Verarbeite Spieler {i+1} bis {i+len(chunk)} von {len(players)}...")
        
        for p in chunk:
            pid = p['id']
            last_name = p.get('last_name', '')
            full_name = f"{p.get('first_name', '')} {last_name}".strip()
            
            if not last_name: continue
            
            p_skills = skills_map.get(pid, {})
            # 🚀 SOTA FIX: Sichere Fallbacks
            elo_metrics = p_skills.get('elo_metrics') or {}
            sackmann_metrics = p_skills.get('sackmann_metrics') or {}
            
            # 1. Berechne neues Surface Rating (aus Elo)
            new_surface_profile = SurfaceIntelligence.compute_player_surface_profile(elo_metrics, sackmann_metrics)
            
            # 2. Berechne neues Form Rating (Historie Dual-Core)
            p_history = await fetch_player_history_extended(full_name, limit=20)
            new_form_rating = MomentumV2Engine.calculate_rating(p_history, full_name)
            
            # 3. Push in Supabase (Spieler Tabelle)
            try:
                supabase.table('players').update({
                    'surface_ratings': new_surface_profile,
                    'form_rating': new_form_rating
                }).eq('id', pid).execute()
                updated_count += 1
            except Exception as e:
                pass
                
    log(f"🏁 UI SYNC FINISHED. {updated_count} Spieler wurden auf das neue Quant-Niveau (V3.1) aktualisiert!")

if __name__ == "__main__":
    asyncio.run(run_sync())
