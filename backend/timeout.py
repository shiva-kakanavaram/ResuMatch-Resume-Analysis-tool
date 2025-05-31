import threading
import time
import sys
import os

class Timeout:
    """
    Cross-platform context manager for timing out operations.
    Works on both Windows and Unix-based systems.
    
    Usage:
        try:
            with Timeout(seconds):
                # Operation that might take too long
        except TimeoutError:
            # Handle timeout
    """
    def __init__(self, seconds, error_message="Operation timed out"):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = None
        self._timeout_occurred = False
        
    def _timeout_handler(self):
        self._timeout_occurred = True
        
    def __enter__(self):
        # Start the timer
        self.timer = threading.Timer(self.seconds, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        self._start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cancel the timer
        if self.timer:
            self.timer.cancel()
            
        # Check if timeout occurred
        if self._timeout_occurred:
            raise TimeoutError(self.error_message)
            
        # Check if we're about to time out
        if time.time() - self._start_time >= self.seconds:
            raise TimeoutError(self.error_message) 