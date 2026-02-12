# -*- coding: utf-8 -*-

import asyncio
import json
import os
import logging
import sys
import re
import random # NEU: Für zufällige Wartezeiten (Human Behavior)
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

# MODEL STRATEGY: 
# We stick with 'llama-3.3-70b-versatile'. 
# Cost: ~$0.70 per 1M tokens via Groq.
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

# NEU: Helper Funktion für sicheres Suchen mit Retry-Logik
async def safe_search_text(client: AsyncDDGS, query: str, max_retries: int = 2) -> List[Dict]:
    """
    Führt eine Suche durch, aber fängt Rate Limits ab und wartet.
    """
    for attempt in range(max_retries):
        try:
            # Random Sleep vor jeder Anfrage (Human Jitter)
            sleep_time = random.uniform(3.0, 7.0)
            await asyncio.sleep(sleep_time)
            
            # Die eigentliche Suche
            results = await client.text(query, max_results=3)
            return results if results else []
            
        except Exception as e:
            log(f"      ⚠️ Search Error (Attempt {attempt+1}/{max_retries}): {e}")
            if "Ratelimit" in str(e) or "202" in str(e):
                wait_time = 30 + (attempt * 10)
                log(f"      🛑 Rate Limit hit. Cooling down for {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(2)
    return []

async def get_youtube_transcripts(player_name: str, ddgs_client: AsyncDDGS, limit: int = 2) -> str:
    """
    Sucht gezielt nach Video-Analysen der Top-Experten, extrahiert die ID 
    und zieht das Transkript.
    """
    log(f"   📺 Scanning YouTube Analysts for: {player_name}...")
    combined_transcripts = ""
    
    # 1. Primary Queries (High Quality)
    primary_queries = [
        f"site:youtube.com {player_name} analysis Gil Gross",
        f"site:youtube.com {player_name} Andy Roddick podcast",
        f"site:youtube.com {player_name} technique analysis"
    ]
    
    # 2. Fallback Queries (Falls der Spieler unbekannt ist)
    fallback_queries = [
        f"site:youtube.com {player_name} tennis highlights commentary",
        f"site:youtube.com {player_name} interview 2026"
    ]

    video_ids = set()

    # Search Execution
    found_any = False
    
    # Versuche erst die Experten
    for q in primary_queries:
        results = await safe_search_text(ddgs_client, q)
        if results:
            found_any = True
            for r in results:
                href = r.get('href', '')
                match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
                if match: video_ids.add(match.group(1))

    # Wenn keine Experten-Videos gefunden, nutze Fallback
    if not found_any or len(video_ids) == 0:
        log("      ℹ️ No expert videos found. Trying fallback sources...")
        for q in fallback_queries:
            results = await safe_search_text(ddgs_client, q)
            if results:
                for r in results:
                    href = r.get('href', '')
                    match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
                    if match: video_ids.add(match.group(1))

    # Extract Transcripts
    formatter = TextFormatter()
    found_count = 0
    
    for vid in list(video_ids)[:limit]:
        try:
            # Versuche Transcript zu laden (Englisch oder Generiert)
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'de'])
            except:
                # Fallback auf automatisch generierte Untertitel
                try:
                    list_transcripts = YouTubeTranscriptApi.list_transcripts(vid)
                    transcript_list = list_transcripts.find_generated_transcript(['en']).fetch()
                except:
                    continue # Kein Transcript verfügbar

            text = formatter.format_transcript(transcript_list)
            
            clean_text = text.replace("\n", " ")[:3000] 
            combined_transcripts += f"\n--- VIDEO TRANSCRIPT (ID: {vid}) ---\n{clean_text}\n"
            found_count += 1
            log(f"      ✅ Ingested Transcript from Video {vid}")
            
            # Kurze Pause nach erfolgreichem Download
            await asyncio.sleep(1)
            
        except Exception:
            continue
            
    if found_count == 0:
        return ""

    return combined_transcripts

async def get_expert_articles(player_name: str, ddgs_client: AsyncDDGS) -> str:
    """
    Sucht nach geschriebenen Artikeln.
    """
    log(f"   📰 Reading Expert Articles for: {player_name}...")
    
    queries = [
        f"site:tennisabstract.com {player_name} analysis",
        f"site:tennis.com {player_name} form 2026",
        f"{player_name} tennis match analysis 2026"
    ]
    
    snippets = []
    
    for q in queries:
        results = await safe_search_text(ddgs_client, q)
        if results:
            for r in results:
                # Verbesserter Filter: Prüfe ob Name im Titel ODER Body ist
                if player_name.lower() in r['title'].lower() or player_name.lower() in r['body'].lower():
                    snippets.append(f"- ARTICLE ({r['title']}): {r['body']}")
    
    return "\n".join(snippets)

# =================================================================
# 3. THE REASONING ENGINE (COMPARATIVE ANALYSIS)
# =================================================================

async def analyze_and_evolve(target: Dict, intelligence_data: str) -> Optional[Dict]:
    """
    Hier passiert die Magie. Wir geben dem LLM die DB-Werte UND die Experten-Meinung.
    """
    if not intelligence_data or len(intelligence_data) < 50:
        log("      Start Skipping: Not enough intelligence data found.")
        return None

    p = target['player']
    s = target['skills']
    
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
    CRITICAL INSTRUCTIONS:
    1. **COMPARE**: Does the intelligence contradict the current ratings? 
       - If DB says Backhand=75, but text says "Her backhand is elite", Boost to 85+.
       - If DB says Mental=80, but text says "Choking matches", Lower to 70s.
    2. **IGNORE NOISE**: If text is generic, make NO changes.
    3. **SCALE**: 60-70 (Challenger), 71-79 (Pro), 80-88 (Elite), 89-99 (Legendary).
    
    OUTPUT JSON ONLY:
    {{
        "analysis_found": true,
        "confidence_score": <0-100>,
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
        "reasoning": "Explain why you changed values."
    }}
    """

    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON-only API. You analyze tennis data with extreme precision."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.2, 
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
    Führt die Updates durch.
    """
    if not analysis.get('changes_detected') or analysis.get('confidence_score', 0) < 70:
        log(f"      ✋ Low confidence/No changes ({analysis.get('confidence_score')}%). Skipping DB write.")
        return

    player_id = target['player']['id']
    new_stats = analysis.get('suggested_ratings', {})
    current_stats = target['skills']
    
    final_stats = {}
    valid_keys = ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]
    changes_log = []
    
    for k in valid_keys:
        old_val = current_stats.get(k, 60)
        new_val = new_stats.get(k, old_val)
        new_val = max(40, min(99, int(new_val))) # Clamp
        
        if abs(new_val - old_val) >= 2:
            changes_log.append(f"{k.upper()}: {old_val}->{new_val}")
            
        final_stats[k] = new_val

    final_stats['updated_at'] = datetime.now(timezone.utc).isoformat()
    final_stats['overall_rating'] = int(sum(final_stats.values()) / len(final_stats))

    if changes_log:
        log(f"      ⚡ UPDATING RATINGS: {', '.join(changes_log)}")
        try:
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

    if analysis.get('report_updates'):
        upd = analysis['report_updates']
        if len(upd.get('strengths', '')) > 10:
            try:
                rep_payload = {
                    "strengths": upd['strengths'],
                    "weaknesses": upd['weaknesses'],
                    "tactical_patterns": upd['tactical_patterns'],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "author_id": "NEURAL_SCOUT_EVO_V2"
                }
                
                res_rep = supabase.table("scouting_reports").select("id").eq("player_id", player_id).execute()
                if res_rep.data:
                    supabase.table("scouting_reports").update(rep_payload).eq("player_id", player_id).execute()
                else:
                    rep_payload['player_id'] = player_id
                    supabase.table("scouting_reports").insert(rep_payload).execute()
                log("      📝 Scouting Report rewritten.")
            except Exception as e:
                log(f"      ❌ DB Error (Report): {e}")

# =================================================================
# 5. WORKFLOW CONTROLLER
# =================================================================

async def get_active_targets(limit: int = 10): # UPGRADE: Limit erhöht auf 10
    """
    Holt Spieler.
    """
    log("🔍 Selecting high-priority targets...")
    try:
        matches = supabase.table("market_odds")\
            .select("player1_name, player2_name")\
            .order("created_at", desc=True)\
            .limit(60).execute() # Scan more matches to find unique players
            
        active_names = set()
        for m in matches.data:
            active_names.add(m['player1_name'])
            active_names.add(m['player2_name'])
            
        target_list = list(active_names)[:limit]
        
        if not target_list: return []
        
        players_res = supabase.table("players").select("*").in_("last_name", target_list).execute()
        
        full_targets = []
        for p in players_res.data:
            pid = p['id']
            s_res = supabase.table("player_skills").select("*").eq("player_id", pid).execute()
            r_res = supabase.table("scouting_reports").select("*").eq("player_id", pid).execute()
            
            skills = s_res.data[0] if s_res.data else {} 
            report = r_res.data[0] if r_res.data else {} 
            
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
    
    # UPGRADE: Wir bearbeiten jetzt bis zu 10 Spieler pro Run
    targets = await get_active_targets(limit=10) 
    log(f"🎯 Loaded {len(targets)} Targets for Analysis")
    
    # Init Search Client with Timeout config via environment logic handled by library
    async with AsyncDDGS() as ddgs_instance:
    
        for t in targets:
            p_name = f"{t['player']['first_name']} {t['player']['last_name']}"
            log(f"\n🔎 SCOUTING: {p_name}")
            
            # Phase 1: Media Mining
            yt_data = await get_youtube_transcripts(p_name, ddgs_instance)
            web_data = await get_expert_articles(p_name, ddgs_instance)
            
            intel = yt_data + "\n" + web_data
            
            if len(intel) < 50:
                log(f"      ⚠️ No actionable intel found for {p_name}. Skipping.")
                continue
            
            # Phase 2: Analysis
            analysis_result = await analyze_and_evolve(t, intel)
            
            # Phase 3: Execution
            if analysis_result:
                if analysis_result.get('changes_detected'):
                    log(f"      💡 INSIGHT: {analysis_result.get('reasoning')}")
                    await apply_updates(t, analysis_result)
                else:
                    log(f"      ✅ Verified: Database matches Expert Consensus.")
            
            # WICHTIG: Erhöhter Sleep Timer um Bans zu vermeiden
            wait_time = random.uniform(5.0, 10.0)
            log(f"      💤 Sleeping {round(wait_time,1)}s to protect API reputation...")
            await asyncio.sleep(wait_time)
        
    log("\n🏁 Evolution Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_neural_scout_evolution())
