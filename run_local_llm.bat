@echo off
:: Open the Developer Command Prompt (this line assumes you're already using a dev prompt)
:: If you're using a standard cmd, you can call the VS dev tools batch file explicitly

cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python main.py
pause
