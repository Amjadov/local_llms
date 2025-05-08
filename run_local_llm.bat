@echo off
:: Open the Developer Command Prompt (this line assumes you're already using a dev prompt)
:: If you're using a standard cmd, you can call the VS dev tools batch file explicitly
@echo off
echo ========================================================================================
echo Welcome to DeepSeek Coder 7b - Local LLM
echo ========================================================================================
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python main.py
pause
