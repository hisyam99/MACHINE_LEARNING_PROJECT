#!/bin/bash

# COVID-19 AI Detection Dashboard Launcher
# ==========================================

echo "🧬 COVID-19 AI Detection System"
echo "================================"
echo ""
echo "🚀 Starting the dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Error: Streamlit is not installed!"
    echo "📦 Install it with: pip install streamlit"
    echo ""
    exit 1
fi

# Check if artifacts directory exists
if [ ! -d "./artifacts" ]; then
    echo "⚠️  Warning: artifacts/ directory not found!"
    echo "   Make sure model files are in the artifacts folder."
    echo ""
fi

# Launch Streamlit
echo "✨ Opening dashboard in your browser..."
echo "🌐 URL: http://localhost:8501"
echo ""
echo "💡 Tip: Press Ctrl+C to stop the server"
echo ""

streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --theme.base dark \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#1e1e1e" \
    --theme.secondaryBackgroundColor "#2d2d2d" \
    --theme.textColor "#ffffff" \
    --browser.gatherUsageStats false

