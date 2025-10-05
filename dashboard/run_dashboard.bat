@echo off
echo Starting NASA Space Biology Knowledge Engine Dashboard...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist app.py (
    echo Error: app.py not found. Please run this script from the dashboard directory.
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Check if data files exist
echo Checking for data files...
set "data_path=..\ai_nlp\outputs\summaries\paper_summaries.csv"
if not exist "%data_path%" (
    echo Warning: AI NLP pipeline data not found at %data_path%
    echo The dashboard will run with sample data for demonstration.
    echo To use real data, please run the AI NLP pipeline first.
    echo.
)

REM Start Streamlit
echo.
echo Starting dashboard...
echo Dashboard will be available at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause