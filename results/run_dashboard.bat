@echo off
REM COVID-19 AI Detection Dashboard Launcher for Windows
REM ======================================================

echo.
echo ============================================
echo  🧬 COVID-19 AI Detection System
echo ============================================
echo.
echo 🚀 Starting the dashboard...
echo.

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Streamlit is not installed!
    echo 📦 Install it with: pip install streamlit
    echo.
    pause
    exit /b 1
)

REM Check if artifacts directory exists
if not exist "artifacts\" (
    echo ⚠️  Warning: artifacts\ directory not found!
    echo    Make sure model files are in the artifacts folder.
    echo.
)

REM Launch Streamlit
echo ✨ Opening dashboard in your browser...
echo 🌐 URL: http://localhost:8501
echo.
echo 💡 Tip: Press Ctrl+C to stop the server
echo.

streamlit run app.py ^
    --server.port 8501 ^
    --server.address localhost ^
    --theme.base dark ^
    --theme.primaryColor "#667eea" ^
    --theme.backgroundColor "#1e1e1e" ^
    --theme.secondaryBackgroundColor "#2d2d2d" ^
    --theme.textColor "#ffffff" ^
    --browser.gatherUsageStats false

pause

