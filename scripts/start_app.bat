@echo off
cd /d "%~dp0"
set SILICONFLOW_API_KEY=sk-dmhuhgiqzcdngjyhmuajbrrrwzintniushyjzuduwaokvrqx
start "" "D:\anaconda\envs\py39\python.exe" app.py
timeout /t 3 /nobreak
start http://localhost:5000
echo Server running at http://localhost:5000
echo Login: admin / admin123
echo Close this window to stop.
pause
