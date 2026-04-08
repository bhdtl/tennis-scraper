# -- coding: utf-8 --
import asyncio
import os
import logging
from datetime import datetime, timezone
from supabase import create_client, Client

# Logging Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("RatingMigrator")

# Secrets (Nutzt dieselben wie dein Scraper)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =================================================================
# DIE NEUE CORE-LOGIK (Identisch mit deinem neuen Standard)
# =================================================================

def calculate_surface_rating_v2(wins, total_matches):
    if total_matches == 0:
        return 5.0, "#808080", "No Data"
    
    win_rate = wins / total_matches
    rating = max(1.0, min(10.0, win_rate * 10.0))
    
    if rating >= 8.5: return round(rating, 2), "#FF00FF", "🔥 SPECIALIST"
    if rating >= 7.0: return round(rating, 2), "#3366FF", "📈 Strong"
    if rating >= 5.5: return round(rating, 2), "#00B25B", "Solid"
    if rating >= 4.0: return round(rating, 2), "#F0C808", "Average"
    return round(rating, 2), "#CC0000", "❄️ Weakness"

# =================================================================
# MIGRATION RUNNER
# =================================================================

async def run_global_migration():
    logger.info("🚀 Starte globale Rating-Migration für alle Spieler...")

    # 1. Alle Spieler laden
    players_res = supabase.table("players").select("id, first_name, last_name").execute()
    players = players_res.data or []
    logger.info(f"📦 {len(players)} Spieler gefunden.")

    for i, p in enumerate(players):
        p_id = p['id']
        p_last = p.get('last_name', '')
        p_full = f"{p.get('first_name', '')} {p_last}".strip()
        
        # 2. Historie für diesen Spieler aus der market_odds Tabelle ziehen
        # Wir nehmen bis zu 500 Matches, um ein perfektes Bild zu bekommen
        hist_res = supabase.table("market_odds") \
            .select("player1_name, player2_name, actual_winner_name, tournament, ai_analysis_text") \
            .or_(f"player1_name.ilike.%{p_last}%,player2_name.ilike.%{p_last}%") \
            .not_.is_("actual_winner_name", "null") \
            .execute()
        
        matches = hist_res.data or []
        
        # 3. Surface-Stats extrahieren
        stats = {"hard": {"w": 0, "t": 0}, "clay": {"w": 0, "t": 0}, "grass": {"w": 0, "t": 0}}
        
        for m in matches:
            # Einfaches Surface-Mapping basierend auf Tournament/Text
            tour = str(m.get('tournament', '')).lower()
            text = str(m.get('ai_analysis_text', '')).lower()
            surf = "unknown"
            
            if "clay" in tour or "clay" in text: surf = "clay"
            elif "grass" in tour or "grass" in text: surf = "grass"
            elif "hard" in tour or "hard" in text: surf = "hard"
            
            if surf != "unknown":
                stats[surf]["t"] += 1
                if str(m.get('actual_winner_name', '')).lower() in p_full.lower():
                    stats[surf]["w"] += 1

        # 4. Profile bauen
        new_profile = {"_v95_mastery_applied": True}
        for s_type in ["hard", "clay", "grass"]:
            r, color, txt = calculate_surface_rating_v2(stats[s_type]["w"], stats[s_type]["t"])
            new_profile[s_type] = {
                "rating": r,
                "color": color,
                "text": txt,
                "matches_tracked": stats[s_type]["t"]
            }

        # 5. DB Update
        try:
            supabase.table("players").update({"surface_ratings": new_profile}).eq("id", p_id).execute()
            if i % 10 == 0:
                logger.info(f"✅ [{i}/{len(players)}] {p_full} aktualisiert.")
        except Exception as e:
            logger.error(f"❌ Fehler bei {p_full}: {e}")

    logger.info("🏁 Migration abgeschlossen. Alle Surface Ratings sind jetzt auf Win-Rate Basis.")

if __name__ == "__main__":
    asyncio.run(run_global_migration())
