name: 🔮 Tournament Oracle Scraper

on:
  schedule:
    # Läuft jeden Tag um 04:00 Uhr UTC (Morgens, bevor die Matches starten)
    - cron: '0 4 * * *'
  workflow_dispatch: # Erlaubt es dir, den Scraper jederzeit manuell per Knopfdruck zu starten

jobs:
  run-oracle:
    runs-on: ubuntu-latest

    steps:
      - name: 📥 Checkout Repository
        uses: actions/checkout@v4

      - name: 🐍 Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11' # Nutze eine stabile Python-Version

      - name: 📦 Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install supabase playwright beautifulsoup4 httpx

      - name: 🌐 Install Playwright Browsers
        run: |
          playwright install chromium

      - name: 🔮 Run Oracle Scraper
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }} 
          # 🚀 HIER IST DER FIX: Verbindet dein Secret "FUNCTION_URL" mit dem Skript
          SUPABASE_FUNCTION_URL: ${{ secrets.FUNCTION_URL }} 
        run: |
          python oracle_scraper.py
