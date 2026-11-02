"""Start the FastAPI backend server on http://127.0.0.1:8000."""
import sys
from pathlib import Path
import uvicorn

# Ensure backend directory is in sys.path even when executed from project root
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
