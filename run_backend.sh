#!/bin/bash
source venv/bin/activate
# Run the backend server with reload enabled
uvicorn main:app --reload --host 127.0.0.1 --port 8000
