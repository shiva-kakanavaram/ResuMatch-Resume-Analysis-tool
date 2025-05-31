import sys
import os
import pathlib
from pathlib import Path
import logging
import time
import threading
from datetime import datetime
import subprocess

# Add the project root to Python path
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("server")

class TimeoutThread(threading.Thread):
    """Thread that monitors startup time and exits if it takes too long."""
    def __init__(self, timeout=60):
        super().__init__()
        self.daemon = True
        self.timeout = timeout
        self.start_time = time.time()
        self.completed = False
        
    def run(self):
        while not self.completed:
            if time.time() - self.start_time > self.timeout:
                logger.error(f"Server startup timed out after {self.timeout} seconds")
                os._exit(1)  # Force exit
            time.sleep(1)
            
    def mark_completed(self):
        self.completed = True

def main():
    """Start the server with optimized settings."""
    start_time = time.time()
    logger.info("Starting optimized ResuMatch server")

    # Import the necessary modules
    try:
        import uvicorn
        from fastapi import FastAPI
        logger.info("FastAPI and Uvicorn imported successfully")
    except ImportError as e:
        logger.error(f"Error importing server dependencies: {e}")
        logger.info("Try running: pip install fastapi uvicorn[standard]")
        return 1

    # Check if main.py exists
    main_path = os.path.join(project_root, "backend", "main.py")
    if not os.path.exists(main_path):
        logger.error(f"Server file not found: {main_path}")
        return 1

    # Add backward compatibility patch for analyze_resume method
    try:
        from backend.ml.resume_analyzer import ResumeAnalyzer
        if hasattr(ResumeAnalyzer, 'analyze') and not hasattr(ResumeAnalyzer, 'analyze_resume'):
            ResumeAnalyzer.analyze_resume = ResumeAnalyzer.analyze
            logger.info("Added backward compatibility for analyze_resume method")
    except ImportError as e:
        logger.warning(f"Could not apply compatibility patch: {e}")

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "127.0.0.1")
    
    # Log environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Server host:port: {host}:{port}")
    logger.info(f"Project root: {project_root}")
    
    # Set server startup options
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    workers = int(os.environ.get("WORKERS", "1"))
    
    # Start the server
    logger.info(f"Starting server with {workers} workers, reload={reload}")
    logger.info(f"Server initialization took {time.time() - start_time:.2f} seconds")
    
    try:
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level="info",
            timeout_keep_alive=60,  # 60 seconds keep-alive timeout
            limit_concurrency=50,   # Limit concurrent connections
            loop="auto"             # Auto-select best event loop
        )
        return 0
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return 1

def get_server_host():
    """Get the host to bind the server to."""
    return os.environ.get("HOST", "127.0.0.1")

def get_server_port():
    """Get the port to bind the server to."""
    return int(os.environ.get("PORT", "8000"))

def is_debug_mode():
    """Check if the server should run in debug mode."""
    return os.environ.get("DEBUG", "false").lower() == "true"

def start_server():
    """Start the FastAPI server with uvicorn."""
    # Enable fallback mode by default to prevent server hangs
    #os.environ["USE_FALLBACK"] = "true"
    
    # Run the server command
    host = get_server_host()
    port = get_server_port()
    debug = is_debug_mode()
    
    logger.info(f"Starting server on {host}:{port} (debug={debug}, fallback={os.environ.get('USE_FALLBACK', 'false')})")
    
    try:
        uvicorn_cmd = ["uvicorn", "backend.main:app", "--host", host, "--port", str(port)]
        if debug:
            uvicorn_cmd.append("--reload")
            
        # Start the server
        subprocess.run(uvicorn_cmd)
        return True
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

if __name__ == "__main__":
    # Set a timeout for the entire startup process
    startup_timeout = 120  # 2 minutes
    
    # Start the server with fallback mode enabled by default
    if start_server():
        logger.info("Server started successfully")
    else:
        logger.error("Failed to start server")
        sys.exit(1) 