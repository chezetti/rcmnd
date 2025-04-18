#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setting environment variables..."
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo ""
echo "Starting the backend server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 