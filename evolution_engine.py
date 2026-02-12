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
# 1. CONFIGURATION & LOGGING (SILICON VALLEY MONOLITH)
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] 🧬 LEVIATHAN-V6: %(message)s',
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

# MODEL: 70b für maximale Analyse-Tiefe
MODEL_NAME = 'llama-3.3-70b-versatile'

# --- SOURCE CONFIGURATION ---

# 1. VIP CHANNELS (Safe RSS Feeds - The Backbone)
# Diese Kanäle werden IMMER gescannt, ohne Rate Limit Gefahr.
RSS_CHANNELS = [
    {"name": "Gil Gross", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCvQ2e8t_e87rYd5f5C8h7wA"},
    {"name": "Tennis TV", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC4yxY8f7l_h-w-h-w-h-w"},
    {"name": "Andy Roddick", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC_c-T1_C-x3-L-Z-w-h-w"},
    {"name": "The Tennis Podcast", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCg1b0r7tZ_5J5kXg9e8_rjg"},
    {"name": "Cult Tennis", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCOTjB42L-_uK9rD3Kx5Tz5g"},
    {"name": "Intuitive Tennis", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCW4Jjo5a0tXGSh57sL16lXQ"}
]

# 2. NEWS FEEDS (Safe Text Sources)
RSS_NEWS = [
    "https://www.tennis.com/rss",
    "https://www.espn.com/espn/rss/tennis/news"
]

# =================================================================
# 2. UTILITY BELT (Networking & Transcripts)
# =================================================================

async def safe_search_text(client: AsyncDDGS, query: str, max_retries: int = 3) -> List[Dict]:
    """
    Führt eine Websuche durch mit extremem Backoff (Stealth Mode).
    """
    for attempt in range(max_retries):
        try:
            # Human Jitter: Lange Pausen um Bot-Detection zu vermeiden
            sleep_time = random.uniform(4.0, 8.0) + (attempt * 3)
            await asyncio.sleep(sleep_time)
            
            results = await client.text(query, max_results=3)
            return results if results else []
            
        except Exception as e:
            log(f"      ⚠️ Deep Scout Search Warn (Attempt {attempt+1}): {e}")
            if "Ratelimit" in str(e) or "202" in str(e):
                wait_time = 60 + (attempt * 30)
                log(f"      🛑 Rate Limit. Going dark for {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(5)
    return []

async def fetch_transcript_robust(video_id: str) -> str:
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
            # 2. Try Auto-Generated English
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
            gen_t = t_list.find_generated_transcript(['en'])
            return formatter.format_transcript(gen_t.fetch())
        except:
            return ""

# =================================================================
# 3. MODULE A: THE GLOBAL WIRETAP (RSS Aggregation)
# =================================================================
async def gather_global_intel() -> str:
    """
    Sammelt Wissen von ALLEN RSS Quellen (Videos & News).
    Das ist das Fundament der Analyse.
    """
    log("📡 [MODULE A] Wiretap: Aggregating Global Intelligence...")
    combined_intel = ""
    cutoff = datetime.now() - timedelta(hours=48) # Letzte 2 Tage
    
    # 1. RSS Video Feeds
    for source in RSS_CHANNELS:
        try:
            feed = feedparser.parse(source['rss'])
            log(f"   📺 Scanning Feed: {source['name']}...")
            
            for entry in feed.entries[:3]: # Top 3 Videos
                try:
                    pub = datetime(*entry.published_parsed[:6])
                except: pub = datetime.now()
                
                if pub > cutoff:
                    vid = entry.yt_videoid
                    title = entry.title
                    
                    transcript = await fetch_transcript_robust(vid)
                    if transcript:
                        combined_intel += f"\n=== SOURCE: {source['name']} | VIDEO: {title} ===\n{transcript[:12000]}\n"
                        log(f"      ✅ Ingested: {title}")
        except Exception as e:
            log(f"      ⚠️ Feed Error {source['name']}: {e}")

    # 2. RSS News Feeds
    for url in RSS_NEWS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                combined_intel += f"\n=== NEWS ARTICLE: {entry.title} ===\n{entry.get('summary', '')}\n"
        except: continue
        
    return combined_intel

# =================================================================
# 4. MODULE B: THE DEEP SCOUT (Active Targeted Search)
# =================================================================
async def run_deep_scout(player_name: str, ddgs_client: AsyncDDGS) -> str:
    """
    Sucht spezifisch nach EINEM Spieler im Web.
    Nutzt DuckDuckGo, aber sehr vorsichtig.
    """
    log(f"   🕵️ [MODULE B] Deep Scout: Hunting specific intel for {player_name}...")
    combined_intel = ""
    
    # 1. Targeted Video Search (Fallback wenn nicht in RSS)
    yt_queries = [
        f"site:youtube.com {player_name} interview 2026",
        f"site:youtube.com {player_name} practice highlights 2026"
    ]
    
    video_found = False
    for q in yt_queries:
        if video_found: break # Ein Video reicht um Limits zu sparen
        results = await safe_search_text(ddgs_client, q)
        
        for r in results:
            href = r.get('href', '')
            match = re.search(r"v=([a-zA-Z0-9_-]{11})", href)
            if match:
                vid = match.group(1)
                text = await fetch_transcript_robust(vid)
                if text:
                    combined_intel += f"\n--- DEEP SCOUT VIDEO: {player_name} (ID:{vid}) ---\n{text[:6000]}\n"
                    video_found = True
                    break
    
    # 2. Targeted Article Search
    web_queries = [
        f"site:tennis.com {player_name} analysis",
        f"site:tennisabstract.com {player_name} stats"
    ]
    
    for q in web_queries:
        results = await safe_search_text(ddgs_client, q)
        for r in results:
            if player_name.lower() in r['title'].lower() or player_name.lower() in r['body'].lower():
                combined_intel += f"\n--- DEEP SCOUT WEB: {r['title']} ---\n{r['body']}\n"
                
    return combined_intel

# =================================================================
# 5. MODULE C: THE BRAIN (Reasoning Engine)
# =================================================================
async def analyze_player_evolution(target: Dict, global_intel: str, specific_intel: str) -> Optional[Dict]:
    """
    Führt Globales Wissen und Spezifisches Wissen zusammen.
    Entscheidet über Updates.
    """
    full_context = global_intel + "\n" + specific_intel
    
    # Wenn wir gar keine Daten haben, brechen wir ab
    if len(full_context) < 100:
        log("      ⚠️ Insufficient data (Global + Specific empty). Skipping.")
        return None

    p = target['player']
    s = target['skills']
    
    current_ratings = json.dumps({k: s.get(k, 60) for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]})
    current_report = json.dumps(target['report'])

    # Der komplexe Prompt (The Silicon Valley Standard)
    prompt = f"""
    ROLE: Chief Tennis Scout & Data Scientist.
    TASK: Analyze gathered intelligence to update player ratings and reports.
    
    TARGET PLAYER: {p['first_name']} {p['last_name']}
    
    CURRENT DB RATINGS: {current_ratings}
    CURRENT REPORT EXCERPT: {current_report[:600]}...
    
    === INTELLIGENCE FEED ===
    {full_context[:55000]} 
    =========================
    
    MISSION:
    1. **IDENTIFY**: Find mentions of {p['last_name']} in the text. 
       - Look for Analyst Opinions (Gil Gross, Roddick) or Match Reports.
    2. **EVALUATE**:
       - Does the text describe a CHANGE in form, technique, or health?
       - Example: "He added 5mph to his serve" -> Upgrade Serve.
       - Example: "Moving sluggishly on the backhand side" -> Downgrade Speed/Backhand.
    3. **DECIDE**:
       - Suggest numeric updates (+/- 1 to 5 points max).
       - Write a short tactical update note.
    
    OUTPUT JSON ONLY:
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
        "source_citation": "e.g. 'Gil Gross Analysis' or 'Web Article'"
    }}
    """

    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "JSON only. No markdown."},
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
# 6. MODULE D: THE SURGEON (Database Commit)
# =================================================================
async def commit_to_database(target: Dict, analysis: Dict):
    # Nur updaten wenn Änderungen erkannt wurden UND Confidence hoch ist
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
        
        # Sanity Check: Max change +/- 5 per run to prevent glitches
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
                "author_id": "LEVIATHAN_V6"
            }
            
            if adds.get('strengths'): payload['strengths'] = adds['strengths']
            if adds.get('weaknesses'): payload['weaknesses'] = adds['weaknesses']
            
            # Append Tactical Update carefully
            if adds.get('tactical_update') and len(adds['tactical_update']) > 10:
                old_tac = rep.data[0].get('tactical_patterns', '') if rep.data else ""
                new_entry = f"\n[UPDATE {datetime.now().strftime('%Y-%m-%d')} via {citation}]: {adds['tactical_update']}"
                # Keep log manageable (max 5000 chars)
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
# 7. MAIN CONTROL LOOP
# =================================================================
async def get_high_priority_targets(limit: int = 10):
    """
    Holt Spieler, die in den market_odds (aktive Matches) stehen.
    """
    log("🎯 Identifying High-Priority Targets (Active Matches)...")
    try:
        # Wir schauen uns die letzten 60 Matches an
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
    log("🚀 SYSTEM ONLINE: Neural Scout Leviathan V6 (Hybrid Monolith)")
    
    # 1. INITIALIZE SEARCH ENGINE (Context Manager)
    async with AsyncDDGS() as ddgs:
        
        # ---------------------------------------------------------
        # PHASE 1: THE GLOBAL WIRETAP (Safe & Fast)
        # ---------------------------------------------------------
        # Holt Daten über RSS, ohne Limit-Gefahr.
        global_intel = await gather_global_intel()
        log(f"🧠 Global Knowledge Base constructed ({len(global_intel)} chars).")
        
        # ---------------------------------------------------------
        # PHASE 2: TARGETED OPERATIONS (Deep Scan)
        # ---------------------------------------------------------
        targets = await get_high_priority_targets(limit=8) # Wir bearbeiten 8 Spieler pro Run
        
        for t in targets:
            p_name = f"{t['player']['first_name']} {t['player']['last_name']}"
            log(f"\n🔬 PROCESSING TARGET: {p_name}")
            
            # A. Deep Scout (Spezifische Suche mit DuckDuckGo)
            # Nutzt jetzt massive Pausen, um Rate Limits zu vermeiden.
            specific_intel = await run_deep_scout(p_name, ddgs)
            
            # B. Synthesis & Reasoning
            analysis = await analyze_player_evolution(t, global_intel, specific_intel)
            
            # C. Execution
            if analysis:
                if analysis.get('mentioned_in_intel'):
                    log(f"      💡 FOUND INTEL: {analysis.get('source_citation')}")
                await commit_to_database(t, analysis)
            
            # D. Evasion Protocol (Wait to look human)
            # WICHTIG: Lange Pause zwischen Spielern für DDG Sicherheit
            wait = random.uniform(10.0, 20.0)
            log(f"      ⏳ Cooling down ({round(wait,1)}s)...")
            await asyncio.sleep(wait)
            
    log("🏁 LEVIATHAN CYCLE COMPLETE.")

if __name__ == "__main__":
    asyncio.run(run_leviathan_engine())
