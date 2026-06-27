@echo off
rem Bellwether launcher — double-click this, or run it from any folder.
rem cd's to this script's folder first so the brand theme (.streamlit/config.toml)
rem and the data/ files all resolve correctly, then starts the app.
cd /d "%~dp0"
python -m streamlit run app.py
pause
