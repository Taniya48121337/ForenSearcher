@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
python -c "import json,pathlib; p=pathlib.Path('.local/openrouter.json'); d=json.loads(p.read_text(encoding='utf-8')); old=d.get('budget_usd'); d['budget_usd']=40.0; p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8'); print('budget_usd:',old,'->',d['budget_usd'])"
pause >nul
