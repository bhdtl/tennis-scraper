# -*- coding: utf-8 -*-

import asyncio
import json
import os
import logging
import sys
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any

import httpx
from supabase import create_client, Client
from groq import AsyncGroq
from duckduckgo_search import AsyncDDGS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# =================================================================
# 1. CONFIGURATION & LOGGING (SILICON VALLEY STANDARD)
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] 🧬 EVO-V2: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_Evolution_V2")

def log(msg: str):
    logger.info(msg)

# Load Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets missing. Check GitHub/Groq Secrets.")
    sys.exit(1)

# Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
ddgs = AsyncDDGS()

# MODEL STRATEGY: 
# We stick with 'llama-3.3-70b-versatile'. 
# Cost: ~$0.70 per 1M tokens via Groq. This is negligible for the quality increase over 8b.
# We need the 70b parameter count to understand the nuance between "Player is in a slump" (Temporary Form) 
# and "Player has changed their swing mechanics" (Permanent Skill Change).
MODEL_NAME = 'llama-3.3-70b-versatile'

# TARGET SOURCES (The "Truth" List)
TRUSTED_ANALYSTS = [
    "Gil Gross Tennis",
    "Andy Roddick Served Podcast",
    "The Tennis Podcast",
    "Tennis Abstract",
    "Gill Gross", 
    "Tactical Tennis",
    "Intuitive Tennis"
]

# =================================================================
# 2. INTELLIGENCE GATHERING (MEDIA MINING)
# =================================================================

async def get_youtube_transcripts(player_name: str, limit: int = 2) -> str:
    """
    Sucht gezielt nach Video-Analysen der Top-Experten, extrahiert die ID 
    und zieht das Transkript. Das ist "Listening to the experts".
    """
    log(f"   📺 Scanning YouTube Analysts for: {player_name}...")
    combined_transcripts = ""
    
    # Wir suchen spezifisch nach Kombinationen aus Spieler + Experte
    search_queries = [
        f"site:youtube.com {player_name} analysis Gil Gross",
        f"site:youtube.com {player_name} Andy Roddick podcast",
        f"site:youtube.com {player_name} technique analysis"
    ]

    video_ids = set()

    try:
        # 1. Find Video Links via DuckDuckGo
        for q in search_queries:
            results = await ddgs.text(q, max_results=2)
            if results:
                for r in results:
                    href = r.get('href', '')
                    # Extrahiere Video ID (v=XXXXX)
                    match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
                    if match:
                        video_ids.add(match.group(1))
            await asyncio.sleep(0.5) # Rate Limit protection

        # 2. Extract Transcripts
        formatter = TextFormatter()
        
        found_count = 0
        for vid in list(video_ids)[:limit]:
            try:
                # Versuche Transcript zu laden
                transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'de'])
                text = formatter.format_transcript(transcript_list)
                
                # Cleanup (Zeitstempel entfernen, wir brauchen nur den Text-Flow)
                clean_text = text.replace("\n", " ")[:3000] # Limit context window per video
                combined_transcripts += f"\n--- VIDEO TRANSCRIPT (ID: {vid}) ---\n{clean_text}\n"
                found_count += 1
                log(f"      ✅ Ingested Transcript from Video {vid}")
            except Exception:
                # Oft haben Videos keine CC, das ist okay. Ignore.
                continue
                
        if found_count == 0:
            return ""

    except Exception as e:
        log(f"   ⚠️ YouTube Mining Warning: {e}")
        return ""

    return combined_transcripts

async def get_expert_articles(player_name: str) -> str:
    """
    Sucht nach geschriebenen Artikeln auf High-Quality Seiten.
    """
    log(f"   📰 Reading Expert Articles for: {player_name}...")
    queries = [
        f"site:tennisabstract.com {player_name} analysis",
        f"site:tennis.com {player_name} gear technique",
        f"site:lossglare.com {player_name}", # Sehr guter Stats Blog
        f"{player_name} tennis form analysis 2026"
    ]
    
    snippets = []
    try:
        for q in queries:
            results = await ddgs.text(q, max_results=2)
            if results:
                for r in results:
                    # Filter: Nur wenn der Spieler auch im Title/Body vorkommt
                    if player_name.lower() in r['title'].lower() or player_name.lower() in r['body'].lower():
                        snippets.append(f"- ARTICLE ({r['title']}): {r['body']}")
            await asyncio.sleep(0.5)
    except Exception as e:
        log(f"   ⚠️ Article Mining Warning: {e}")
    
    return "\n".join(snippets)

# =================================================================
# 3. THE REASONING ENGINE (COMPARATIVE ANALYSIS)
# =================================================================

async def analyze_and_evolve(target: Dict, intelligence_data: str) -> Optional[Dict]:
    """
    Hier passiert die Magie. Wir geben dem LLM die DB-Werte UND die Experten-Meinung.
    Es muss entscheiden: Liegt die DB falsch?
    """
    if not intelligence_data or len(intelligence_data) < 50:
        log("      Start Skipping: Not enough intelligence data found.")
        return None

    p = target['player']
    s = target['skills']
    
    # Current DB Snapshot
    current_ratings = json.dumps({k: s.get(k, 60) for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]})
    current_report = json.dumps(target['report'])

    prompt = f"""
    ROLE: Elite Tennis Analyst (ATP/WTA Specialist).
    TASK: Audit and Update Player Ratings based on new intelligence.
    
    PLAYER: {p['first_name']} {p['last_name']}
    CURRENT DATABASE RATINGS: {current_ratings}
    CURRENT SCOUTING REPORT (Excerpt): {current_report[:500]}...
    
    NEW INTELLIGENCE (Transcripts & Articles):
    {intelligence_data}
    
    ---------------------------------------------------
    CRITICAL INSTRUCTIONS (The "Silicon Valley" Standard):
    1. **COMPARE**: Does the intelligence contradict the current ratings? 
       - Example: Database says Backhand=75, but Gil Gross says "Her backhand is top 5 on tour". -> ACTION: Boost Backhand significantly (e.g., to 88-90).
       - Example: Database says Mental=80, but Andy Roddick says "He chokes in big moments recently". -> ACTION: Lower Mental.
    2. **IGNORE NOISE**: If the intelligence is generic (e.g. "Good match"), DO NOT change ratings. Only change on specific technical/tactical insights.
    3. **SCALE**: 
       - 60-70: Challenger Level
       - 71-79: Top 100 Solid
       - 80-88: Top 20 / Elite Weapon
       - 89-99: Grand Slam Champion Level / Historic Weapon (e.g. Isner Serve, Graf Forehand)
    4. **REPORT UPDATE**: Rewrite the 'strengths' and 'weaknesses' ONLY if you have new info.
    
    OUTPUT JSON ONLY:
    {{
        "analysis_found": true,
        "confidence_score": <0-100, how sure are you based on text?>,
        "changes_detected": true/false,
        "suggested_ratings": {{
            "serve": <int>, "forehand": <int>, "backhand": <int>, 
            "volley": <int>, "speed": <int>, "stamina": <int>, 
            "power": <int>, "mental": <int>
        }},
        "report_updates": {{
            "strengths": "...",
            "weaknesses": "...",
            "tactical_patterns": "..."
        }},
        "reasoning": "Citing specific sources (e.g. 'Roddick mentioned...'), explain why you changed values."
    }}
    """

    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON-only API. You analyze tennis data with extreme precision."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.2, # Low temp = High precision
            response_format={"type": "json_object"}
        )
        
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        log(f"   ❌ LLM Reasoning Error: {e}")
        return None

# =================================================================
# 4. THE SURGEON (DB EXECUTION)
# =================================================================

async def apply_updates(target: Dict, analysis: Dict):
    """
    Führt die Updates nur durch, wenn Confidence hoch genug ist.
    """
    if not analysis.get('changes_detected') or analysis.get('confidence_score', 0) < 70:
        log(f"      ✋ No significant changes or low confidence ({analysis.get('confidence_score')}%). Skipping DB write.")
        return

    player_id = target['player']['id']
    new_stats = analysis.get('suggested_ratings', {})
    
    # 1. Update Skills
    # Wir laden die alten Stats um sicherzugehen, dass wir nichts überschreiben was fehlt
    current_stats = target['skills']
    
    # Merge Logic: Nur Werte, die im JSON sind, werden geupdated.
    final_stats = {}
    
    # Mapping der Keys sicherstellen
    valid_keys = ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]
    
    changes_log = []
    
    for k in valid_keys:
        old_val = current_stats.get(k, 60) # Default DB fallback
        new_val = new_stats.get(k, old_val)
        
        # Safety Clamp: Prevent AI hallucinations (e.g. 150 rating)
        new_val = max(40, min(99, int(new_val)))
        
        # Log significant changes
        if abs(new_val - old_val) >= 2:
            changes_log.append(f"{k.upper()}: {old_val}->{new_val}")
            
        final_stats[k] = new_val

    final_stats['updated_at'] = datetime.now(timezone.utc).isoformat()
    final_stats['overall_rating'] = int(sum(final_stats.values()) / len(final_stats))

    if changes_log:
        log(f"      ⚡ UPDATING RATINGS: {', '.join(changes_log)}")
        try:
            # Check existance
            res = supabase.table("player_skills").select("id").eq("player_id", player_id).execute()
            if res.data:
                supabase.table("player_skills").update(final_stats).eq("player_id", player_id).execute()
            else:
                final_stats['player_id'] = player_id
                supabase.table("player_skills").insert(final_stats).execute()
        except Exception as e:
            log(f"      ❌ DB Error (Skills): {e}")
    else:
        log("      ℹ️ Ratings stable. No update needed.")

    # 2. Update Report if text changed significantly
    if analysis.get('report_updates'):
        upd = analysis['report_updates']
        # Simple heuristic: if string length > 10, assume valid update
        if len(upd.get('strengths', '')) > 10:
            try:
                rep_payload = {
                    "strengths": upd['strengths'],
                    "weaknesses": upd['weaknesses'],
                    "tactical_patterns": upd['tactical_patterns'],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "author_id": "NEURAL_SCOUT_EVO_V2"
                }
                
                # Check existance
                res_rep = supabase.table("scouting_reports").select("id").eq("player_id", player_id).execute()
                if res_rep.data:
                    supabase.table("scouting_reports").update(rep_payload).eq("player_id", player_id).execute()
                else:
                    rep_payload['player_id'] = player_id
                    supabase.table("scouting_reports").insert(rep_payload).execute()
                log("      📝 Scouting Report rewritten based on Analyst Intel.")
            except Exception as e:
                log(f"      ❌ DB Error (Report): {e}")

# =================================================================
# 5. WORKFLOW CONTROLLER
# =================================================================

async def get_active_targets(limit: int = 5):
    """
    Holt Spieler, die:
    1. In der DB existieren (wir erstellen keine neuen hier)
    2. Aktiv sind (Matches in market_odds)
    """
    log("🔍 Selecting high-priority targets...")
    
    # Hole recent matches
    try:
        matches = supabase.table("market_odds")\
            .select("player1_name, player2_name")\
            .order("created_at", desc=True)\
            .limit(50).execute()
            
        active_names = set()
        for m in matches.data:
            active_names.add(m['player1_name'])
            active_names.add(m['player2_name'])
            
        target_list = list(active_names)[:limit]
        
        # Batch fetch player data
        if not target_list: return []
        
        players_res = supabase.table("players").select("*").in_("last_name", target_list).execute()
        
        full_targets = []
        for p in players_res.data:
            pid = p['id']
            # Fetch relational data individually to handle missing rows gracefully
            s_res = supabase.table("player_skills").select("*").eq("player_id", pid).execute()
            r_res = supabase.table("scouting_reports").select("*").eq("player_id", pid).execute()
            
            skills = s_res.data[0] if s_res.data else {} # Empty dict if no skills yet
            report = r_res.data[0] if r_res.data else {} # Empty dict if no report yet
            
            full_targets.append({
                "player": p,
                "skills": skills,
                "report": report
            })
            
        return full_targets
        
    except Exception as e:
        log(f"❌ Error fetching targets: {e}")
        return []

async def run_neural_scout_evolution():
    log("🚀 NEURAL SCOUT EVOLUTION V2 (The Analyst Edition) Started")
    
    targets = await get_active_targets(limit=3) # Small batch for high quality deep dive
    
    for t in targets:
        p_name = f"{t['player']['first_name']} {t['player']['last_name']}"
        log(f"\n🔎 SCOUTING: {p_name}")
        
        # Phase 1: Media Mining
        yt_data = await get_youtube_transcripts(p_name)
        web_data = await get_expert_articles(p_name)
        
        intel = yt_data + "\n" + web_data
        
        if len(intel) < 100:
            log(f"      ⚠️ No analyst content found for {p_name}. Skipping to preserve DB integrity.")
            continue
            
        # Phase 2: Analysis
        analysis_result = await analyze_and_evolve(t, intel)
        
        # Phase 3: Execution
        if analysis_result:
            if analysis_result.get('changes_detected'):
                log(f"      💡 INSIGHT: {analysis_result.get('reasoning')}")
                await apply_updates(t, analysis_result)
            else:
                log(f"      ✅ Verified: Ratings align with current expert consensus.")
        
        await asyncio.sleep(2) # Respect APIs
        
    log("\n🏁 Evolution Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_neural_scout_evolution())
