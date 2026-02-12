# -*- coding: utf-8 -*-

import asyncio
import json
import os
import logging
import sys
import re
import random 
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
    format='[%(asctime)s] 🧬 EVO-V3: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_Evolution_V3")

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

MODEL_NAME = 'llama-3.3-70b-versatile'

# TARGET CHANNELS (High Value Intelligence)
# Wir suchen gezielt nach den neuesten Uploads dieser Kanäle
VIP_CHANNELS = [
    "Gil Gross Tennis",
    "Andy Roddick Served Podcast",
    "Courtside Tennis",
    "The Tennis Podcast", 
    "Monday Match Analysis"
]

# =================================================================
# 2. GLOBAL INTELLIGENCE (THE WATCHER)
# =================================================================

async def safe_search_text(client: AsyncDDGS, query: str, max_retries: int = 2) -> List[Dict]:
    """
    Führt eine Suche durch, aber fängt Rate Limits ab und wartet.
    """
    for attempt in range(max_retries):
        try:
            # Random Sleep vor jeder Anfrage (Human Jitter)
            sleep_time = random.uniform(2.0, 5.0)
            await asyncio.sleep(sleep_time)
            
            # Die eigentliche Suche
            results = await client.text(query, max_results=3)
            return results if results else []
            
        except Exception as e:
            log(f"      ⚠️ Search Error (Attempt {attempt+1}): {e}")
            if "Ratelimit" in str(e) or "202" in str(e):
                wait_time = 30 + (attempt * 10)
                log(f"      🛑 Rate Limit hit. Cooling down for {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(2)
    return []

async def scan_vip_channels(ddgs_client: AsyncDDGS) -> str:
    """
    NEU: Scannt VORHER die neuesten Videos der Top-Analysten.
    Das fängt Infos ab wie 'Gil Gross redet über Alcaraz', ohne dass wir speziell nach Alcaraz suchen mussten.
    """
    log("\n📡 PHASE 1: Scanning VIP Analyst Channels (Global Intel)...")
    global_transcript_buffer = ""
    
    formatter = TextFormatter()
    
    for channel in VIP_CHANNELS:
        log(f"   📺 Tuning into: {channel}...")
        # Suche nach dem allerneuesten Video dieses Kanals
        query = f"site:youtube.com {channel} latest tennis analysis 2026"
        
        results = await safe_search_text(ddgs_client, query, max_retries=2)
        
        for r in results:
            href = r.get('href', '')
            match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
            
            if match:
                vid = match.group(1)
                try:
                    # Transcript ziehen
                    transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'de'])
                    text = formatter.format_transcript(transcript_list)
                    
                    # Kontext speichern (Max 4000 Zeichen pro Video um Token zu sparen)
                    snippet = text.replace("\n", " ")[:4000]
                    global_transcript_buffer += f"\n--- VIP SOURCE: {channel} (VideoID: {vid}) ---\n{snippet}\n"
                    log(f"      ✅ Captured Intel from {channel}")
                    break # Nur das neueste Video pro Kanal reicht oft
                except:
                    continue
    
    log(f"   📝 Global Intelligence gathered: {len(global_transcript_buffer)} chars.\n")
    return global_transcript_buffer

# =================================================================
# 3. PLAYER SPECIFIC INTELLIGENCE (THE DEEP DIVE)
# =================================================================

async def get_player_specific_transcripts(player_name: str, ddgs_client: AsyncDDGS, limit: int = 2) -> str:
    """
    Sucht spezifisch nach dem Spieler, falls er in den VIP Kanälen nicht vorkam.
    """
    log(f"   🔍 Deep Scan for: {player_name}...")
    combined_transcripts = ""
    
    # Spezifische Suche
    queries = [
        f"site:youtube.com {player_name} analysis 2026",
        f"site:youtube.com {player_name} tennis highlights commentary"
    ]

    video_ids = set()
    for q in queries:
        results = await safe_search_text(ddgs_client, q)
        if results:
            for r in results:
                href = r.get('href', '')
                match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
                if match: video_ids.add(match.group(1))

    formatter = TextFormatter()
    found_count = 0
    
    for vid in list(video_ids)[:limit]:
        try:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'de'])
            except:
                try:
                    list_transcripts = YouTubeTranscriptApi.list_transcripts(vid)
                    transcript_list = list_transcripts.find_generated_transcript(['en']).fetch()
                except:
                    continue 

            text = formatter.format_transcript(transcript_list)
            clean_text = text.replace("\n", " ")[:2000] 
            combined_transcripts += f"\n--- SPECIFIC VIDEO (ID: {vid}) ---\n{clean_text}\n"
            found_count += 1
            await asyncio.sleep(1)
            
        except Exception:
            continue

    return combined_transcripts

async def get_expert_articles(player_name: str, ddgs_client: AsyncDDGS) -> str:
    log(f"   📰 Reading News for: {player_name}...")
    queries = [
        f"site:tennisabstract.com {player_name} analysis",
        f"site:tennis.com {player_name} form 2026",
        f"{player_name} tennis analysis 2026"
    ]
    snippets = []
    for q in queries:
        results = await safe_search_text(ddgs_client, q)
        if results:
            for r in results:
                if player_name.lower() in r['title'].lower() or player_name.lower() in r['body'].lower():
                    snippets.append(f"- ARTICLE ({r['title']}): {r['body']}")
    return "\n".join(snippets)

# =================================================================
# 4. THE REASONING ENGINE (AI AGENT)
# =================================================================

async def analyze_and_evolve(target: Dict, global_intel: str, specific_intel: str) -> Optional[Dict]:
    """
    Kombiniert Globales Wissen (Gil Gross etc.) mit spezifischem Wissen.
    """
    full_intel = global_intel + "\n" + specific_intel
    
    if len(full_intel) < 50:
        log("      Start Skipping: Not enough data found.")
        return None

    p = target['player']
    s = target['skills']
    
    current_ratings = json.dumps({k: s.get(k, 60) for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]})
    current_report = json.dumps(target['report'])

    # UPGRADE: Prompt instruiert die KI, gezielt im Global Intel nach dem Spielernamen zu suchen.
    prompt = f"""
    ROLE: Elite Tennis Analyst (Silicon Valley Agent).
    TASK: Audit Player Ratings using Global Analyst Intel & Specific Search Data.
    
    TARGET PLAYER: {p['first_name']} {p['last_name']}
    
    CURRENT RATINGS: {current_ratings}
    CURRENT REPORT: {current_report[:500]}...
    
    === INTELLIGENCE STREAM ===
    {full_intel}
    ===========================
    
    INSTRUCTIONS:
    1. **SCAN**: Look for {p['last_name']} in the Intelligence Stream.
       - Did Gil Gross, Roddick, or recent news mention them?
    2. **EVALUATE**:
       - If an analyst praises a specific shot (e.g. "Forehand looks faster"), BOOST that stat (+2 to +5).
       - If an analyst mentions weakness/injury, LOWER relevant stats.
       - If the player is NOT mentioned or intel is generic, DO NOT CHANGE RATINGS.
    3. **SCALE**: 60 (Challenger), 75 (Top 100), 85 (Top 20), 95 (Legend).
    
    OUTPUT JSON ONLY:
    {{
        "analysis_found": true,
        "mentioned_in_intel": true/false,
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
        "reasoning": "Explain source (e.g. 'Found in Gil Gross video: ...')"
    }}
    """

    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON-only API. Precision is key."},
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
# 5. THE SURGEON (DB OPS)
# =================================================================

async def apply_updates(target: Dict, analysis: Dict):
    if not analysis.get('changes_detected') or analysis.get('confidence_score', 0) < 75: # High Confidence needed
        log(f"      ✋ No update needed (Confidence: {analysis.get('confidence_score')}%).")
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
        new_val = max(40, min(99, int(new_val))) 
        
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

    if analysis.get('report_updates'):
        upd = analysis['report_updates']
        if len(upd.get('strengths', '')) > 10:
            try:
                rep_payload = {
                    "strengths": upd['strengths'],
                    "weaknesses": upd['weaknesses'],
                    "tactical_patterns": upd['tactical_patterns'],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "author_id": "NEURAL_SCOUT_EVO_V3"
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
# 6. WORKFLOW CONTROLLER
# =================================================================

async def get_active_targets(limit: int = 15): 
    """
    Holt Spieler mit kürzlichen Matches.
    Bei 600 Spielern: Da wir market_odds sortieren, erwischen wir immer die, die gerade spielen (am wichtigsten).
    """
    log("🔍 Selecting high-priority targets from active matches...")
    try:
        matches = supabase.table("market_odds")\
            .select("player1_name, player2_name")\
            .order("created_at", desc=True)\
            .limit(80).execute() 
            
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
    log("🚀 NEURAL SCOUT EVOLUTION V3 (Silicon Valley Edition) Started")
    
    # 1. Start Search Engine Context
    async with AsyncDDGS() as ddgs_instance:
    
        # 2. GATHER GLOBAL INTEL (Gil Gross, Roddick, etc.) - Nur 1x pro Run!
        # Das spart API Calls und fängt alle Infos auf einmal.
        global_intel = await scan_vip_channels(ddgs_instance)
        
        # 3. Process Active Players
        targets = await get_active_targets(limit=12) # 12 wichtigste Spieler
        log(f"\n🎯 Processing {len(targets)} active players...")
        
        for t in targets:
            p_name = f"{t['player']['first_name']} {t['player']['last_name']}"
            log(f"\n🔎 ANALYZING: {p_name}")
            
            # Phase A: Get Specific Data
            specific_intel = await get_player_specific_transcripts(p_name, ddgs_instance)
            news_intel = await get_expert_articles(p_name, ddgs_instance)
            
            # Phase B: Combine (Global + Specific) -> AI Brain
            # Hier geben wir der AI das Wissen von Gil Gross (Global) + Spezifische News
            combined_specific = specific_intel + "\n" + news_intel
            
            analysis_result = await analyze_and_evolve(t, global_intel, combined_specific)
            
            # Phase C: Execution
            if analysis_result:
                if analysis_result.get('changes_detected'):
                    log(f"      💡 INSIGHT: {analysis_result.get('reasoning')}")
                    await apply_updates(t, analysis_result)
                else:
                    log(f"      ✅ Verified: Stable.")
            
            wait_time = random.uniform(3.0, 8.0)
            await asyncio.sleep(wait_time)
        
    log("\n🏁 Evolution Cycle Finished.")

if __name__ == "__main__":
    asyncio.run(run_neural_scout_evolution())
