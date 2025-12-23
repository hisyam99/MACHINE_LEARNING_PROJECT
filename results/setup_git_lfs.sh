#!/bin/bash

# Git LFS Setup Script for Large Model Files
# ============================================

echo "🚀 Setting up Git LFS for Large Model Files"
echo "==========================================="
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null
then
    echo "❌ Git LFS is not installed!"
    echo ""
    echo "📦 Please install Git LFS first:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt-get install git-lfs"
    echo ""
    echo "macOS:"
    echo "  brew install git-lfs"
    echo ""
    echo "Windows:"
    echo "  Download from: https://git-lfs.github.com/"
    echo ""
    exit 1
fi

echo "✅ Git LFS is installed"
echo ""

# Initialize Git LFS
echo "📝 Initializing Git LFS..."
git lfs install

# Track model files
echo "🎯 Tracking large model files..."
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git lfs track "artifacts/*.pkl"

echo ""
echo "✅ Git LFS configured!"
echo ""

# Check file sizes
echo "📊 Checking model file sizes..."
echo ""
find artifacts/ -type f \( -name "*.h5" -o -name "*.pt" \) -exec ls -lh {} \; 2>/dev/null || echo "No model files found yet"

echo ""
echo "📋 Next steps:"
echo "1. git add .gitattributes"
echo "2. git add artifacts/"
echo "3. git commit -m 'Add model files via Git LFS'"
echo "4. git push origin main"
echo ""
echo "💡 Git LFS will automatically upload large files to LFS storage"
echo ""

# Show what will be tracked
echo "🔍 Files that will be tracked by Git LFS:"
git lfs track

echo ""
echo "✨ Setup complete! You can now commit and push large files."
echo ""

