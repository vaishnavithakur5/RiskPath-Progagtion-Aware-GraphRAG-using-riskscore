@echo off
title RiskPath - Test Suite Verification
echo ========================================================
echo RiskPath: Running Automated Unit & Integration Tests...
echo ========================================================
cd /d "%~dp0"
python -m pytest tests/ -v
echo ========================================================
pause
