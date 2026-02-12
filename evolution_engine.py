# -*- coding: utf-8 -*-

import asyncio
import json
import os
import logging
import sys
import re
import random
import feedparser
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any

import httpx
from supabase import create_client, Client
from groq import AsyncGroq
from duckduckgo_search import AsyncDDGS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# =================================================================
# 1. CONFIGURATION & LOGGING (MILITARY GRADE)
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] 🧬 LEVIATHAN-V4: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_Leviathan")

def log(msg: str):
    logger.info(msg)

# Secrets Loading
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL: Secrets missing. Check GitHub/Groq Secrets.")
    sys.exit(1)

# Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# MODEL: 70b für Deep Reasoning
MODEL_NAME = 'llama-3.3-70b-versatile'

# --- LAYER 1 SOURCES (RSS - Passive & Fast) ---
# Echte Channel IDs verwenden! (Hier Beispiele)
RSS_SOURCES = [
    {"name": "Tennis TV", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC4yxY8f7l_h-w-h-w-h-w"},
    {"name": "ATP Tour", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC..."},
    {"name": "WTA", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC..."}
]

# --- LAYER 2 SOURCES (Analyst Watchlist - Active Scan) ---
# Diese Kanäle werden proaktiv gescannt, auch wenn kein RSS Alarm kommt.
VIP_ANALYST_CHANNELS = [
    "Gil Gross Tennis",
    "Andy Roddick Served Podcast",
    "Courtside Tennis",
    "The Tennis Podcast", 
    "Monday Match Analysis",
    "Intuitive Tennis"
]

# =================================================================
# 2. UTILITY BELT (Robust Networking)
# =================================================================

async def safe_search_text(client: AsyncDDGS, query: str, max_retries: int = 3) -> List[Dict]:
    """
    Führt eine Websuche durch mit exponentiellem Backoff bei Rate Limits.
    """
    for attempt in range(max_retries):
        try:
            # Human Jitter: Nie sofort feuern
            sleep_time = random.uniform(2.0, 6.0) + (attempt * 2)
            await asyncio.sleep(sleep_time)
            
            results = await client.text(query, max_results=4)
            return results if results else []
            
        except Exception as e:
            log(f"      ⚠️ Search Warn (Attempt {attempt+1}): {e}")
            if "Ratelimit" in str(e) or "202" in str(e):
                wait_time = 45 + (attempt * 20)
                log(f"      🛑 Rate Limit Detected. Holding position for {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(5)
    return []

async def fetch_youtube_transcript(video_id: str) -> str:
    """
    Versucht ein Transkript zu laden (Manuell > Auto-Gen > Fail).
    """
    formatter = TextFormatter()
    try:
        # 1. Try Manual English/German
        t_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'de'])
        return formatter.format_transcript(t_list)
    except:
        try:
            # 2. Try Auto-Generated
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
            gen_t = t_list.find_generated_transcript(['en'])
            return formatter.format_transcript(gen_t.fetch())
        except:
            return ""

# =================================================================
# 3. MODULE A: THE WIRETAP (RSS Monitoring)
# =================================================================
async def run_rss_wiretap() -> str:
    """
    Checkt RSS Feeds auf ganz frische Videos (letzte 24h).
    """
    log("📡 [MODULE A] Wiretap: Scanning RSS Feeds...")
    combined_intel = ""
    yesterday = datetime.now() - timedelta(hours=24)
    
    for source in RSS_SOURCES:
        try:
            feed = feedparser.parse(source['rss'])
            for entry in feed.entries[:3]:
                published = datetime(*entry.published_parsed[:6])
                if published > yesterday:
                    vid = entry.yt_videoid
                    title = entry.title
                    log(f"      ✨ RSS Alert: {title}")
                    
                    text = await fetch_youtube_transcript(vid)
                    if text:
                        combined_intel += f"\n--- RSS SOURCE: {source['name']} ({title}) ---\n{text[:10000]}\n"
        except:
            continue
    return combined_intel

# =================================================================
# 4. MODULE B: THE WATCHER (Global Analyst Scan)
# =================================================================
async def run_analyst_watcher(ddgs_client: AsyncDDGS) -> str:
    """
    Sucht aktiv nach den neuesten Videos der VIP Analysten (Gil Gross, Roddick etc.).
    """
    log("🔭 [MODULE B] Watcher: Scanning VIP Analyst Channels...")
    combined_intel = ""
    
    for channel in VIP_ANALYST_CHANNELS:
        log(f"   📺 Tuning into: {channel}...")
        
        # Wir suchen nach den allerneuesten Analysen von 2026
        query = f"site:youtube.com {channel} tennis analysis 2026"
        results = await safe_search_text(ddgs_client, query)
        
        for r in results:
            href = r.get('href', '')
            match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
            if match:
                vid = match.group(1)
                text = await fetch_youtube_transcript(vid)
                if text:
                    # Wir nehmen nur die ersten 5000 Zeichen pro Video um Token zu sparen,
                    # aber genug Kontext für Spielernamen zu haben.
                    combined_intel += f"\n--- VIP ANALYST: {channel} (ID:{vid}) ---\n{text[:8000]}\n"
                    log(f"      ✅ Transcribed video from {channel}")
                    break # Ein Video pro Analyst reicht meist
    
    return combined_intel

# =================================================================
# 5. MODULE C: THE DEEP SCOUT (Targeted Player Search)
# =================================================================
async def run_deep_scout(player_name: str, ddgs_client: AsyncDDGS) -> str:
    """
    Sucht spezifisch nach EINEM Spieler im Web und auf YouTube.
    Wird nur für High-Priority Targets ausgeführt.
    """
    log(f"   🕵️ [MODULE C] Deep Scout: Hunting intel for {player_name}...")
    combined_intel = ""
    
    # 1. YouTube Specifics
    yt_queries = [
        f"site:youtube.com {player_name} interview 2026",
        f"site:youtube.com {player_name} practice court 2026",
        f"site:youtube.com {player_name} press conference"
    ]
    
    for q in yt_queries:
        results = await safe_search_text(ddgs_client, q)
        for r in results:
            href = r.get('href', '')
            match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
            if match:
                vid = match.group(1)
                text = await fetch_youtube_transcript(vid)
                if text:
                    combined_intel += f"\n--- TARGETED VIDEO: {player_name} (ID:{vid}) ---\n{text[:5000]}\n"
                    break # Ein gutes Video reicht
    
    # 2. Article/Web Search
    web_queries = [
        f"site:tennis.com {player_name} form analysis",
        f"site:tennisabstract.com {player_name} stats",
        f"{player_name} injury update 2026"
    ]
    
    for q in web_queries:
        results = await safe_search_text(ddgs_client, q)
        for r in results:
            if player_name.lower() in r['title'].lower() or player_name.lower() in r['body'].lower():
                combined_intel += f"\n--- WEB ARTICLE: {r['title']} ---\n{r['body']}\n"
                
    return combined_intel

# =================================================================
# 6. MODULE D: THE BRAIN (Reasoning & Decision)
# =================================================================
async def analyze_player_evolution(target: Dict, global_intel: str, specific_intel: str) -> Optional[Dict]:
    """
    Führt alle Datenströme zusammen und trifft eine Entscheidung.
    """
    full_context = global_intel + "\n" + specific_intel
    
    if len(full_context) < 100:
        log("      ⚠️ Insufficient data for analysis.")
        return None

    p = target['player']
    s = target['skills']
    
    current_ratings = json.dumps({k: s.get(k, 60) for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]})
    current_report = json.dumps(target['report'])

    prompt = f"""
    ROLE: Chief Tennis Scout (Silicon Valley AI).
    TASK: Analyze gathered intelligence to update player ratings.
    
    PLAYER: {p['first_name']} {p['last_name']}
    
    CURRENT DB RATINGS: {current_ratings}
    CURRENT REPORT EXCERPT: {current_report[:600]}...
    
    === INTELLIGENCE FEED (Global + Specific) ===
    {full_context[:50000]} 
    =============================================
    
    MISSION:
    1. **SEARCH**: Look specifically for mentions of {p['last_name']} in the text.
       - Note: "Global Intel" might mention them even if the video wasn't about them (e.g. Gil Gross comparing players).
    2. **VERIFY**: Use only FACTS from the text. No hallucinations.
       - If Roddick says "Serve is broken", downgrade Serve.
       - If News says "New coach, more aggressive", upgrade Power/Mental.
    3. **DECIDE**:
       - IF significant info found: Suggest numeric updates (+/- 1 to 5).
       - IF info is generic/old: Return "changes_detected": false.
    
    OUTPUT JSON:
    {{
        "mentioned_in_intel": true/false,
        "changes_detected": true/false,
        "confidence": <0-100>,
        "updates": {{
            "serve": <int>, "forehand": <int>, "backhand": <int>, 
            "volley": <int>, "speed": <int>, "stamina": <int>, 
            "power": <int>, "mental": <int>
        }},
        "report_additions": {{
            "tactical_update": "New text to append...",
            "strengths": "Updated list...",
            "weaknesses": "Updated list..."
        }},
        "source_citation": "e.g. 'Andy Roddick Podcast (Video ID...)'"
    }}
    """

    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "JSON only. Be conservative with rating changes."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.2, 
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        log(f"   ❌ Brain Error: {e}")
        return None

# =================================================================
# 7. MODULE E: THE SURGEON (Database Commit)
# =================================================================
async def commit_to_database(target: Dict, analysis: Dict):
    if not analysis.get('changes_detected') or analysis.get('confidence', 0) < 70:
        log(f"      ✋ No action taken (Confidence: {analysis.get('confidence')}%).")
        return

    pid = target['player']['id']
    new_vals = analysis.get('updates', {})
    curr_vals = target['skills']
    
    final = {}
    changes_txt = []
    
    # 1. Skill Update Logic
    for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]:
        old = curr_vals.get(k, 60)
        proposed = new_vals.get(k, old)
        
        # Sanity Check: Max change +/- 5 per run
        diff = proposed - old
        if abs(diff) > 5: proposed = old + (5 if diff > 0 else -5)
        
        final[k] = max(40, min(99, int(proposed)))
        
        if abs(final[k] - old) >= 1:
            changes_txt.append(f"{k}: {old}->{final[k]}")

    final['updated_at'] = datetime.now(timezone.utc).isoformat()
    final['overall_rating'] = int(sum(final.values()) / len(final))

    if changes_txt:
        log(f"      ⚡ COMMITTING UPDATES: {', '.join(changes_txt)}")
        try:
            # Upsert Pattern
            c = supabase.table("player_skills").select("id").eq("player_id", pid).execute()
            if c.data:
                supabase.table("player_skills").update(final).eq("player_id", pid).execute()
            else:
                final['player_id'] = pid
                supabase.table("player_skills").insert(final).execute()
        except Exception as e:
            log(f"      ❌ DB Skill Error: {e}")

    # 2. Report Update Logic
    if analysis.get('report_additions'):
        adds = analysis['report_additions']
        citation = analysis.get('source_citation', 'AI Scout')
        
        try:
            rep = supabase.table("scouting_reports").select("*").eq("player_id", pid).execute()
            
            payload = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "author_id": "LEVIATHAN_V4"
            }
            
            if adds.get('strengths'): payload['strengths'] = adds['strengths']
            if adds.get('weaknesses'): payload['weaknesses'] = adds['weaknesses']
            
            # Append Tactical Update carefully
            if adds.get('tactical_update') and len(adds['tactical_update']) > 10:
                old_tac = rep.data[0].get('tactical_patterns', '') if rep.data else ""
                new_entry = f"\n[UPDATE {datetime.now().strftime('%Y-%m-%d')} via {citation}]: {adds['tactical_update']}"
                payload['tactical_patterns'] = (new_entry + "\n" + old_tac)[:5000]

            if rep.data:
                supabase.table("scouting_reports").update(payload).eq("player_id", pid).execute()
            else:
                payload['player_id'] = pid
                supabase.table("scouting_reports").insert(payload).execute()
            
            log("      📝 Report successfully enhanced.")
            
        except Exception as e:
            log(f"      ❌ DB Report Error: {e}")

# =================================================================
# 8. MAIN CONTROL LOOP
# =================================================================
async def get_high_priority_targets(limit: int = 8):
    """
    Holt Spieler, die in den market_odds (aktive Matches) stehen.
    """
    log("🎯 Identifying High-Priority Targets (Active Matches)...")
    try:
        # Wir schauen uns die letzten 60 Matches an, um die relevantesten Spieler zu finden
        matches = supabase.table("market_odds").select("player1_name, player2_name").order("created_at", desc=True).limit(60).execute()
        
        active = set()
        for m in matches.data:
            active.add(m['player1_name'])
            active.add(m['player2_name'])
            
        names = list(active)[:limit]
        
        if not names: return []
        
        # Batch Fetch Details
        p_res = supabase.table("players").select("*").in_("last_name", names).execute()
        
        full_targets = []
        for p in p_res.data:
            pid = p['id']
            # Fetch relational data individually to be safe
            s = supabase.table("player_skills").select("*").eq("player_id", pid).execute()
            r = supabase.table("scouting_reports").select("*").eq("player_id", pid).execute()
            
            full_targets.append({
                "player": p,
                "skills": s.data[0] if s.data else {},
                "report": r.data[0] if r.data else {}
            })
            
        return full_targets
    except Exception as e:
        log(f"❌ Target Fetch Error: {e}")
        return []

async def run_leviathan_engine():
    log("🚀 SYSTEM ONLINE: Neural Scout Leviathan V4")
    
    # 1. INITIALIZE SEARCH ENGINE (Context Manager)
    async with AsyncDDGS() as ddgs:
        
        # ---------------------------------------------------------
        # PHASE 1: THE GLOBAL SWEEP (One time per run)
        # ---------------------------------------------------------
        # Sammelt alle Infos von RSS und Analysten VORAB.
        # Das ist das "Weltwissen" für diesen Run.
        
        rss_intel = await run_rss_wiretap()
        analyst_intel = await run_analyst_watcher(ddgs)
        
        global_knowledge_base = rss_intel + "\n" + analyst_intel
        log(f"🧠 Global Knowledge Base constructed ({len(global_knowledge_base)} chars).")
        
        # ---------------------------------------------------------
        # PHASE 2: TARGETED OPERATIONS (Per Player)
        # ---------------------------------------------------------
        targets = await get_high_priority_targets(limit=8) # Bearbeite die 8 wichtigsten
        
        for t in targets:
            p_name = f"{t['player']['first_name']} {t['player']['last_name']}"
            log(f"\n🔬 PROCESSING TARGET: {p_name}")
            
            # A. Deep Scout (Spezifische Suche für diesen Spieler)
            # Wir suchen nur tief, wenn das Global Knowledge nichts hergab,
            # oder um Details zu verifizieren.
            specific_intel = await run_deep_scout(p_name, ddgs)
            
            # B. Synthesis & Reasoning
            analysis = await analyze_player_evolution(t, global_knowledge_base, specific_intel)
            
            # C. Execution
            if analysis:
                if analysis.get('mentioned_in_intel'):
                    log(f"      💡 FOUND INTEL: {analysis.get('source_citation')}")
                await commit_to_database(t, analysis)
            
            # D. Evasion Protocol (Wait to look human)
            wait = random.uniform(5.0, 12.0)
            log(f"      ⏳ Cooling down ({round(wait,1)}s)...")
            await asyncio.sleep(wait)
            
    log("🏁 LEVIATHAN CYCLE COMPLETE.")

if __name__ == "__main__":
    asyncio.run(run_leviathan_engine())
