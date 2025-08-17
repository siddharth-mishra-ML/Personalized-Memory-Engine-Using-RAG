@echo off
REM This script starts the Ollama server and the Streamlit Chatbot app.

REM --- STEP 1: Go to your project directory ---
REM !!! REPLACE THIS PATH with the actual path to your project folder !!!
cd "C:\Users\crazz\OneDrive\Documents\OneDrive_BITS Pilani\Dissertation\Personalized-Memory-Engine"

REM --- STEP 2: Start the Ollama server in a new window ---
REM The 'start' command runs the program without waiting for it to close.
echo Starting Ollama server in a new window...
start "" "C:\Users\crazz\AppData\Local\Programs\Ollama\ollama.exe"

REM --- STEP 3: Wait a few seconds for the server to initialize ---
echo Waiting 5 seconds for Ollama to start...
timeout /t 5 /nobreak

REM --- STEP 4: Start the Streamlit app ---
echo Starting Streamlit chat interface...
streamlit run app.py

pause