#!/bin/bash

echo "=================================="
echo "Diabetes Prediction App Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate  # For Linux/Mac
# venv\Scripts\activate  # Uncomment for Windows

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Train the model: python devops_miniproject_pycaret.py"
echo "2. Run Streamlit app: streamlit run app.py"
echo ""
echo "For deployment:"
echo "- Push to GitHub"
echo "- Deploy on share.streamlit.io"
echo ""