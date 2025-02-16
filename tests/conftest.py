# tests/conftest.py
import pytest
import subprocess
import time
import requests
import os
import signal
from typing import Generator

@pytest.fixture(scope="session")
def app_url() -> Generator[str, None, None]:
    """
    Starts Gunicorn with the FastAPI application for testing.
    Yields the base URL of the running application.
    Shuts down Gunicorn after tests are complete.
    """
    port = 8000  # Choose a port that's not likely to be in use.  Use a different port if needed
    command = [
        "gunicorn",
        "main:app",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"0.0.0.0:{port}",  # Bind to localhost for testing
        "--workers", "1"  # Use only 1 worker for testing, simplifies things
    ]
    # Start Gunicorn in a separate process.
    process = subprocess.Popen(command)

    # Wait for Gunicorn to start up.  Use requests.get to check.
    base_url = f"http://0.0.0.0:{port}"
    start_time = time.time()
    timeout = 60  # seconds

    while True:
      try:
          response = requests.get(base_url, timeout=1) #Short request timeout for faster startup checks
          response.raise_for_status()  # Raise an exception for bad status codes
          break  # Gunicorn is up and responding
      except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
        if time.time() - start_time > timeout:
            process.terminate()
            process.wait()  #Ensure process is cleaned up
            raise TimeoutError(f"Gunicorn failed to start within {timeout} seconds")
        time.sleep(0.5)  # Wait a bit before retrying

    yield base_url  # Provide the base URL to the tests

    # Teardown:  Shutdown Gunicorn gracefully after tests finish
    # Use SIGTERM for a graceful shutdown. SIGKILL is more forceful.
    os.kill(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5) #Wait for process termination with timeout
    except subprocess.TimeoutExpired: # If the process doesn't terminate
        print("Gunicorn process did not terminate gracefully, killing it.")
        os.kill(process.pid, signal.SIGKILL) #Kill the process
        process.wait() #Wait to avoid zombie processes
    print("Gunicorn server shut down.")