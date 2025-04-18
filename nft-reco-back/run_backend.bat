@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setting environment variables...
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONPATH=%PYTHONPATH%;%CD%

echo.
echo Starting the backend server...
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause 