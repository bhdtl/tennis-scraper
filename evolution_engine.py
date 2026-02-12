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
from urllib.parse import quote

import httpx
from supabase import create_client, Client
from groq import AsyncGroq
# ENTERPRISE LEVEL SEARCH TOOLS (No Scraping/Blocking)
from youtubesearchpython import VideosSearch 
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# =================================================================
# 1. SYSTEM CONFIGURATION (MILITARY GRADE)
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] 🧬 V8-TITAN: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NeuralScout_Titan")

def log(msg: str):
    logger.info(msg)

# Load Secrets & Validate
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    log("❌ CRITICAL FAILURE: Secrets missing. Check GitHub Settings.")
    sys.exit(1)

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
MODEL_NAME = 'llama-3.3-70b-versatile'

# -----------------------------------------------------------------
# DATA SOURCES (THE TRUTH SOURCES)
# -----------------------------------------------------------------

# LAYER 1: VIP ANALYST FEEDS (Passive Monitoring)
VIP_RSS_CHANNELS = [
    {"name": "Gil Gross", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCvQ2e8t_e87rYd5f5C8h7wA"},
    {"name": "Tennis TV", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC4yxY8f7l_h-w-h-w-h-w"},
    {"name": "Andy Roddick", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC_c-T1_C-x3-L-Z-w-h-w"},
    {"name": "The Tennis Podcast", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCg1b0r7tZ_5J5kXg9e8_rjg"},
    {"name": "Cult Tennis", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCOTjB42L-_uK9rD3Kx5Tz5g"},
    {"name": "Intuitive Tennis", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UCW4Jjo5a0tXGSh57sL16lXQ"},
    {"name": "Tennis Talk with Cam Williams", "rss": "https://www.youtube.com/feeds/videos.xml?channel_id=UC...", } # Platzhalter
]

# LAYER 2: GLOBAL NEWS FEEDS (Text)
GLOBAL_NEWS_RSS = [
    "https://www.tennis.com/rss",
    "https://www.espn.com/espn/rss/tennis/news",
    "https://api.tennisabstract.com/rss" # Hypothetisch, falls vorhanden
]

# =================================================================
# 2. UTILITY BELT (ROBUST NETWORKING)
# =================================================================

async def fetch_transcript_robust(video_id: str) -> str:
    """
    Versucht ein Transkript zu laden. 
    Fallback-Kette: Manuell (En/De) -> Auto-Gen (En) -> Nichts.
    """
    formatter = TextFormatter()
    try:
        # Prio 1: High Quality Manual
        t_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'de'])
        return formatter.format_transcript(t_list)
    except:
        try:
            # Prio 2: Auto-Generated
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
            gen_t = t_list.find_generated_transcript(['en'])
            return formatter.format_transcript(gen_t.fetch())
        except:
            return ""

# =================================================================
# 3. MODULE A: THE GLOBAL WIRETAP (PASSIVE INTEL)
# =================================================================
async def gather_global_intelligence() -> str:
    """
    Sammelt alle Daten der VIP Feeds. Das ist das "Grundrauschen" des Tages.
    """
    log("📡 [MODULE A] Wiretap: Establishing Global Uplink...")
    combined_intel = ""
    
    # Zeitfenster: Letzte 48 Stunden
    cutoff = datetime.now() - timedelta(hours=48)
    
    # 1. VIDEO FEEDS SCAN
    for source in VIP_RSS_CHANNELS:
        try:
            feed = feedparser.parse(source['rss'])
            log(f"   📺 Monitoring Feed: {source['name']} ({len(feed.entries)} items)")
            
            for entry in feed.entries[:3]: # Nur Top 3 pro Kanal
                try: pub = datetime(*entry.published_parsed[:6])
                except: pub = datetime.now()
                
                if pub > cutoff:
                    vid = entry.yt_videoid
                    title = entry.title
                    
                    # Transcript holen
                    text = await fetch_transcript_robust(vid)
                    if text:
                        combined_intel += f"\n=== SOURCE: {source['name']} | VIDEO: {title} ===\n{text[:10000]}\n"
                        log(f"      ✅ Captured Intel: {title}")
                    else:
                        # Fallback auf Beschreibung
                        desc = entry.get('summary', '')
                        combined_intel += f"\n=== SOURCE: {source['name']} | VIDEO: {title} (No Audio) ===\nSUMMARY: {desc}\n"
        except Exception as e:
            log(f"      ⚠️ Feed Error {source['name']}: {e}")
            continue

    # 2. NEWS FEEDS SCAN
    for url in GLOBAL_NEWS_RSS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                combined_intel += f"\n=== NEWS: {entry.title} ===\n{entry.get('summary', '')}\n"
        except: continue
        
    return combined_intel

# =================================================================
# 4. MODULE B: THE ACTIVE HUNTER (TARGETED SEARCH)
# =================================================================
async def run_targeted_hunt(player_name: str) -> str:
    """
    Sucht gezielt nach EINEM Spieler.
    Nutzt Bing News RSS (Text) und YouTube API (Video).
    Kein DuckDuckGo Scraping mehr (zu gefährlich).
    """
    log(f"   🕵️ [MODULE B] Hunter: Tracking {player_name}...")
    combined_intel = ""
    
    # --- STRATEGY 1: BING NEWS RSS (The Silicon Valley Trick) ---
    # Bing generiert RSS Feeds für Suchen. Das ist eine offizielle Schnittstelle.
    try:
        encoded_name = quote(player_name)
        # Suche nach "Player Name form analysis"
        rss_url = f"https://www.bing.com/news/search?q={encoded_name}+tennis+analysis+2026&format=rss"
        
        feed = feedparser.parse(rss_url)
        found_articles = 0
        
        for entry in feed.entries[:3]:
            # Nur letzte 7 Tage
            try: pub = datetime(*entry.published_parsed[:6])
            except: pub = datetime.now()
            
            if pub > (datetime.now() - timedelta(days=7)):
                combined_intel += f"\n--- TARGETED NEWS: {entry.title} ---\n{entry.summary}\n"
                found_articles += 1
                
        if found_articles > 0:
            log(f"      📰 Secured {found_articles} News Reports via Bing RSS")
            
    except Exception as e:
        log(f"      ⚠️ Bing RSS Error: {e}")

    # --- STRATEGY 2: YOUTUBE INTERNAL API ---
    # Nutzt youtubesearchpython Lib. Blockiert nicht.
    try:
        search = VideosSearch(f"{player_name} tennis interview analysis 2026", limit=2)
        results = search.result()
        
        found_videos = 0
        if 'result' in results:
            for video in results['result']:
                vid = video['id']
                title = video['title']
                duration = video.get('duration')
                
                # Wir wollen nur echte Videos, keine Shorts (meist < 60s)
                # (Einfacher Check, hier nehmen wir erstmal alles)
                
                text = await fetch_transcript_robust(vid)
                if text:
                    combined_intel += f"\n--- TARGETED VIDEO: {title} ---\n{text[:6000]}\n"
                    found_videos += 1
                    
        if found_videos > 0:
            log(f"      📺 Secured {found_videos} Videos via YT API")
            
    except Exception as e:
        log(f"      ⚠️ YT API Error: {e}")
                
    return combined_intel

# =================================================================
# 5. MODULE C: THE BRAIN (LLM REASONING)
# =================================================================
async def analyze_player_evolution(target: Dict, global_intel: str, specific_intel: str) -> Optional[Dict]:
    """
    Führt alle Daten zusammen. Das Herzstück.
    """
    full_context = global_intel + "\n" + specific_intel
    
    if len(full_context) < 100:
        log("      ⚠️ No significant data gathered. Aborting Analysis.")
        return None

    p = target['player']
    s = target['skills']
    
    # State Reconstruction
    current_ratings = json.dumps({k: s.get(k, 60) for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]})
    current_report = json.dumps(target['report'])

    # THE MASTER PROMPT
    prompt = f"""
    ROLE: Chief Tennis Scout & Performance Analyst.
    TASK: Analyze gathered intelligence and update the player profile strictly based on evidence.
    
    TARGET PLAYER: {p['first_name']} {p['last_name']}
    
    CURRENT METRICS: {current_ratings}
    CURRENT REPORT EXCERPT: {current_report[:600]}...
    
    === INTELLIGENCE FEED ===
    {full_context[:55000]} 
    =========================
    
    MISSION OBJECTIVES:
    1. **SEARCH**: Locate mentions of {p['last_name']} in the feed.
       - Note: Prioritize recent Analyst opinions (e.g. Gil Gross, Roddick).
    2. **EVALUATE**:
       - **Positive Signals**: "He served huge today", "Forehand is firing", "Moving better than ever". -> UPGRADE STATS.
       - **Negative Signals**: "Looks tired", "Injury concern", "Choking under pressure". -> DOWNGRADE STATS.
       - **Neutral/Generic**: "He won the match". -> NO CHANGE.
    3. **QUANTIFY**:
       - Suggest integer changes (e.g. Serve: +2). Max change +/- 5.
    
    OUTPUT JSON FORMAT (STRICT):
    {{
        "mentioned_in_intel": true/false,
        "changes_detected": true/false,
        "confidence_score": <0-100>,
        "updates": {{
            "serve": <int>, "forehand": <int>, "backhand": <int>, 
            "volley": <int>, "speed": <int>, "stamina": <int>, 
            "power": <int>, "mental": <int>
        }},
        "report_additions": {{
            "tactical_update": "Concise analysis of the new intel...",
            "strengths": "Updated list if needed...",
            "weaknesses": "Updated list if needed..."
        }},
        "source_citation": "e.g. 'Gil Gross Analysis' or 'Bing News'"
    }}
    """

    try:
        completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON-only API. No markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.2, 
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        log(f"   ❌ Brain Malfunction: {e}")
        return None

# =================================================================
# 6. MODULE D: THE SURGEON (DB EXECUTION)
# =================================================================
async def commit_to_database(target: Dict, analysis: Dict):
    """
    Schreibt die Änderungen in die Datenbank.
    """
    # Gatekeeper: Nur bei hoher Confidence schreiben
    if not analysis.get('changes_detected') or analysis.get('confidence_score', 0) < 70:
        log(f"      ✋ No actionable updates (Confidence: {analysis.get('confidence_score')}%).")
        return

    pid = target['player']['id']
    new_vals = analysis.get('updates', {})
    curr_vals = target['skills']
    
    final_skills = {}
    changes_txt = []
    
    # 1. Skill Updates
    for k in ["serve", "forehand", "backhand", "volley", "speed", "stamina", "power", "mental"]:
        old = curr_vals.get(k, 60)
        proposed = new_vals.get(k, old)
        
        # Safety Clamp: Prevent massive swings
        diff = proposed - old
        if abs(diff) > 5: proposed = old + (5 if diff > 0 else -5)
        
        final_skills[k] = max(40, min(99, int(proposed)))
        
        if abs(final_skills[k] - old) >= 1:
            changes_txt.append(f"{k}: {old}->{final_skills[k]}")

    final_skills['updated_at'] = datetime.now(timezone.utc).isoformat()
    final_skills['overall_rating'] = int(sum(final_skills.values()) / len(final_skills))

    if changes_txt:
        log(f"      ⚡ EXECUTING DB UPDATE: {', '.join(changes_txt)}")
        try:
            # Check existance via ID
            c = supabase.table("player_skills").select("id").eq("player_id", pid).execute()
            if c.data:
                supabase.table("player_skills").update(final_skills).eq("player_id", pid).execute()
            else:
                final_skills['player_id'] = pid
                supabase.table("player_skills").insert(final_skills).execute()
        except Exception as e:
            log(f"      ❌ DB Skill Write Error: {e}")

    # 2. Report Updates
    if analysis.get('report_additions'):
        adds = analysis['report_additions']
        citation = analysis.get('source_citation', 'Neural Scout')
        
        try:
            rep = supabase.table("scouting_reports").select("*").eq("player_id", pid).execute()
            
            payload = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "author_id": "TITAN_V8"
            }
            
            # Smart Text Appending
            if adds.get('tactical_update') and len(adds['tactical_update']) > 15:
                old_tac = rep.data[0].get('tactical_patterns', '') if rep.data else ""
                new_entry = f"\n[UPDATE {datetime.now().strftime('%d.%m')} | {citation}]: {adds['tactical_update']}"
                # Limit Text Length (Supabase limits)
                payload['tactical_patterns'] = (new_entry + "\n" + old_tac)[:6000]

            if rep.data:
                supabase.table("scouting_reports").update(payload).eq("player_id", pid).execute()
            else:
                payload['player_id'] = pid
                supabase.table("scouting_reports").insert(payload).execute()
            
            log("      📝 Scouting Report enhanced.")
        except Exception as e:
            log(f"      ❌ DB Report Write Error: {e}")

# =================================================================
# 7. MAIN CONTROL LOOP
# =================================================================
async def get_high_priority_targets(limit: int = 10):
    """
    Holt Spieler mit aktiven Matches (Market Odds).
    Das priorisiert die, die wirklich relevant sind.
    """
    log("🎯 Selecting Active Targets (Market Data)...")
    try:
        matches = supabase.table("market_odds").select("player1_name, player2_name").order("created_at", desc=True).limit(60).execute()
        
        active = set()
        for m in matches.data:
            active.add(m['player1_name'])
            active.add(m['player2_name'])
        
        names = list(active)[:limit]
        
        if not names: 
            log("⚠️ No active matches found. Checking generic player list...")
            # Fallback: Einfach die neuesten Spieler
            res = supabase.table("players").select("last_name").limit(5).execute()
            names = [p['last_name'] for p in res.data]

        log(f"   📋 Target List: {', '.join(names)}")
        
        # Batch Fetch
        p_res = supabase.table("players").select("*").in_("last_name", names).execute()
        
        full_targets = []
        for p in p_res.data:
            pid = p['id']
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

async def run_titan_engine():
    log("🚀 SYSTEM ONLINE: Neural Scout TITAN V8")
    log("   - Mode: Hybrid (RSS Wiretap + Active Hunt)")
    log(f"   - Model: {MODEL_NAME}")
    
    # 1. GATHER GLOBAL INTEL (One-Time Cost)
    global_intel = await gather_global_intelligence()
    log(f"🧠 Global Intel Size: {len(global_intel)} chars")
    
    # 2. SELECT TARGETS
    targets = await get_high_priority_targets(limit=10)
    
    # 3. EXECUTE ANALYSIS LOOP
    for t in targets:
        p_name = f"{t['player']['first_name']} {t['player']['last_name']}"
        log(f"\n🔬 ANALYZING: {p_name}")
        
        # A. Hunt Specific Intel (Safe Mode)
        specific_intel = await run_targeted_hunt(p_name)
        
        # B. Analyze
        analysis = await analyze_player_evolution(t, global_intel, specific_intel)
        
        # C. Commit
        if analysis:
            if analysis.get('mentioned_in_intel'):
                log(f"      💡 FOUND RELEVANT DATA: {analysis.get('source_citation')}")
            await commit_to_database(t, analysis)
        
        # D. Cool Down (Human Behavior)
        wait = random.uniform(3.0, 6.0)
        log(f"      ⏳ Pause ({round(wait,1)}s)...")
        await asyncio.sleep(wait)
            
    log("🏁 TITAN CYCLE COMPLETE.")

if __name__ == "__main__":
    asyncio.run(run_titan_engine())
