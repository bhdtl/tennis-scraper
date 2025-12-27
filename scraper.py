# =================================================================
# RESULT VERIFICATION ENGINE (V80.3 - OPTIMIZED GUARDRAILS)
# =================================================================
async def update_past_results():
    log("🏆 Checking for Match Results (Smart Validation V3)...")
    
    pending_matches = supabase.table("market_odds").select("*").is_("actual_winner_name", "null").execute().data
    if not pending_matches:
        log("   ✅ No pending matches to verify.")
        return

    # 1. TIME-LOCK CALIBRATION (Reduced to 65 Minutes)
    # 105 Min war zu strikt für schnelle 2-Satz Matches oder Aufgaben.
    # 65 Min ist der "Sweet Spot" zwischen Sicherheit und Speed.
    safe_matches = []
    now_utc = datetime.now(timezone.utc)
    
    for pm in pending_matches:
        try:
            created_at_str = pm['created_at'].replace('Z', '+00:00')
            created_at = datetime.fromisoformat(created_at_str)
            minutes_since_start = (now_utc - created_at).total_seconds() / 60
            
            # UPDATE: Reduziert auf 65 Minuten für schnellere Updates
            if minutes_since_start > 65: 
                safe_matches.append(pm)
        except: continue

    if not safe_matches:
        log("   ⏳ Matches are currently running (Time-Locked < 65m). Waiting...")
        return

    for day_offset in range(3): 
        target_date = datetime.now() - timedelta(days=day_offset)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                url = f"https://www.tennisexplorer.com/results/?type=all&year={target_date.year}&month={target_date.month}&day={target_date.day}"
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                content = await page.content()
                await browser.close()
                
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', class_='result')
                if not table: continue

                rows = table.find_all('tr')
                
                for i in range(len(rows)):
                    row = rows[i]
                    if 'flags' in str(row) or 'head' in str(row): continue

                    for pm in safe_matches:
                        p1_last = get_last_name(pm['player1_name'])
                        p2_last = get_last_name(pm['player2_name'])
                        
                        row_text = row.get_text(separator=" ", strip=True).lower()
                        next_row_text = ""
                        if i+1 < len(rows):
                            next_row_text = rows[i+1].get_text(separator=" ", strip=True).lower()
                        
                        match_found = (p1_last in row_text and p2_last in next_row_text) or \
                                      (p2_last in row_text and p1_last in next_row_text) or \
                                      (p1_last in row_text and p2_last in row_text)
                        
                        if match_found:
                            try:
                                # SAFETY CHECK: Ignore lines that look like "14:30" (Startzeit) OHNE Resultat
                                if re.search(r'\d{2}:\d{2}', row_text) and not "ret." in row_text and not re.search(r'\d-\d', row_text):
                                    continue 

                                # RETIREMENT DETECTION
                                is_retirement = "ret." in row_text or "w.o." in row_text

                                cols1 = row.find_all('td', class_='score')
                                cols2 = rows[i+1].find_all('td', class_='score') if i+1 < len(rows) else []
                                
                                def find_set_score(columns):
                                    for col in columns:
                                        txt = col.get_text(strip=True)
                                        if txt in ['0', '1', '2', '3']: return int(txt)
                                    return 0

                                s1 = find_set_score(cols1)
                                s2 = find_set_score(cols2)
                                
                                winner_name = None
                                
                                # LOGIC UPDATE:
                                # A) Normaler Sieg: >= 2 Sätze
                                # B) Retirement Sieg: Einfach mehr Sätze als der Gegner (auch bei 1:0)
                                
                                # Check P1 (Top Row)
                                if (s1 >= 2 and s1 > s2) or (is_retirement and s1 > s2): 
                                    if p1_last in row_text: winner_name = pm['player1_name']
                                    elif p2_last in row_text: winner_name = pm['player2_name']
                                
                                # Check P2 (Bottom Row)
                                elif (s2 >= 2 and s2 > s1) or (is_retirement and s2 > s1): 
                                    if p1_last in next_row_text: winner_name = pm['player1_name']
                                    elif p2_last in next_row_text: winner_name = pm['player2_name']
                                
                                if winner_name:
                                    supabase.table("market_odds").update({"actual_winner_name": winner_name}).eq("id", pm['id']).execute()
                                    log(f"   ✅ Result VERIFIED: {winner_name} won ({s1}:{s2}) {'[RET]' if is_retirement else ''}")
                                    safe_matches = [x for x in safe_matches if x['id'] != pm['id']]
                                    
                            except Exception as e:
                                pass

            except Exception as e:
                log(f"   ⚠️ Parsing Error: {e}")
                await browser.close()
