import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller extracts the application into its runtime directory.
    app_dir = Path(sys._MEIPASS)
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

import uvicorn

host = os.getenv("BRAHMA_BACKEND_HOST", "127.0.0.1")
port = int(os.getenv("BRAHMA_BACKEND_PORT", "8000"))

uvicorn.run(
    "app.main:app",
    host=host,
    port=port,
    log_level="info",
)
