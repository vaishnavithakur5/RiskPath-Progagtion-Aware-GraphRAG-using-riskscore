@echo off
title RiskPath - Interactive Viva Defense Console
echo ========================================================
echo RiskPath: Propagation-Aware Risk Retrieval for GraphRAG
echo Launching Interactive Streamlit Demonstration...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run ui/app.py
pause
