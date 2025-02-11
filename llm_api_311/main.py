# main.py

import logging
import time
import signal
from threading import Thread, Lock, current_thread, Event
from queue import Empty

from fastapi import FastAPI
import uvicorn

# --- Controllers ---
from controllers import api_router  # Import router from controllers

###############################################################################
# LOGGER
###############################################################################
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###############################################################################
# SHUTDOWN EVENT
###############################################################################
shutdown_event = Event()

###############################################################################
# FASTAPI SETUP
###############################################################################
def create_api_app() -> FastAPI:
    """
    Create and return the FastAPI application, including routers and any
    additional endpoints.
    """
    api_app = FastAPI(
        title="My Scalable API",
        description="A standalone FastAPI service for LLM interactions & Document processing",
        version="1.0.0"
    )

    # Include your controllers under a single router prefix
    api_app.include_router(api_router, prefix="/docs")

    @api_app.get("/health")
    def health_check():
        return {"status": "ok", "message": "FastAPI is running separately!"}

    return api_app

def start_fastapi_server(api_app: FastAPI, host="0.0.0.0", port=8675):
    """
    Blocking call to run Uvicorn.
    """
    uvicorn.run(api_app, host=host, port=port, log_level="info")

###############################################################################
# GRACEFUL SHUTDOWN SIGNAL HANDLER
###############################################################################
def handle_signal(signum, frame):
    logger.info(f"Received shutdown signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

###############################################################################
# MAIN ENTRY POINT
###############################################################################
def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Starting standalone FastAPI LLM application...")

    # Create FastAPI app
    api_app = create_api_app()

    # Start the FastAPI server on the main thread or in a separate thread
    # If you want to run it on the main thread, do:
    start_fastapi_server(api_app=api_app, host="0.0.0.0", port=8675)

    # Keep the main thread alive if you have background threads:
    while not shutdown_event.is_set():
        time.sleep(1)

    logger.info("Application shutdown complete.")

if __name__ == "__main__":
    main()
