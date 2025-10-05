#!/bin/bash

echo "Starting NASA Space Biology Knowledge Engine Dashboard..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the dashboard directory."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if data files exist
echo "Checking for data files..."
data_path="../ai_nlp/outputs/summaries/paper_summaries.csv"
if [ ! -f "$data_path" ]; then
    echo "Warning: AI NLP pipeline data not found at $data_path"
    echo "The dashboard will run with sample data for demonstration."
    echo "To use real data, please run the AI NLP pipeline first."
    echo
fi

# Start Streamlit
echo
echo "Starting dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo
streamlit run app.py