#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Start Backend in background
echo "Starting FastAPI Backend..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start Frontend
echo "Starting Streamlit Dashboard..."
streamlit run dashboard.py

# Cleanup on exit
kill $BACKEND_PID
