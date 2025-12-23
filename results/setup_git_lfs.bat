@echo off
REM Git LFS Setup Script for Windows
REM ===================================

echo.
echo ============================================
echo  🚀 Setting up Git LFS for Large Models
echo ============================================
echo.

REM Check if git-lfs is installed
git lfs version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git LFS is not installed!
    echo.
    echo 📦 Please install Git LFS first:
    echo    Download from: https://git-lfs.github.com/
    echo.
    pause
    exit /b 1
)

echo ✅ Git LFS is installed
echo.

REM Initialize Git LFS
echo 📝 Initializing Git LFS...
git lfs install

REM Track model files
echo 🎯 Tracking large model files...
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git lfs track "artifacts/*.pkl"

echo.
echo ✅ Git LFS configured!
echo.

REM Show tracked files
echo 🔍 Files tracked by Git LFS:
git lfs track

echo.
echo 📋 Next steps:
echo 1. git add .gitattributes
echo 2. git add artifacts/
echo 3. git commit -m "Add model files via Git LFS"
echo 4. git push origin main
echo.
echo 💡 Git LFS will automatically upload large files
echo.
echo ✨ Setup complete!
echo.

pause

