import os
import uvicorn

host = os.getenv("BRAHMA_BACKEND_HOST", "127.0.0.1")
port = int(os.getenv("BRAHMA_BACKEND_PORT", "8000"))
uvicorn.run("app.main:app", host=host, port=port, log_level="info")
