"""End-to-end tests: talk to a REAL running service over HTTP.

The same tests run in two modes:

* Locally (default): the test starts its own uvicorn server on a free port,
  waits until it is ready, and shuts it down afterwards.
* In CI: set the BASE_URL environment variable (for example
  BASE_URL=http://127.0.0.1:8000). The tests then use the service that is
  already running, which in our workflow is the Docker container.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _free_port():
    """Ask the operating system for an unused port, so we never clash with another server."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_ready(base_url, proc=None, timeout=30):
    """Poll /health until the service answers, instead of sleeping for a fixed time."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("The server process exited before it became ready")
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Service at {base_url} was not ready after {timeout} seconds")


@pytest.fixture(scope="module")
def base_url():
    external_url = os.environ.get("BASE_URL", "").strip().rstrip("/")
    if external_url:
        print(f"\nUsing the service that is already running at {external_url}")
        _wait_until_ready(external_url)
        yield external_url
        return

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\nStarting uvicorn on {url} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(port)],
        cwd=PROJECT_ROOT,
    )
    try:
        _wait_until_ready(url, proc)
        yield url
    finally:
        print("\nStopping the server ...")
        proc.terminate()
        proc.wait(timeout=10)


def test_service_is_healthy(base_url):
    response = requests.get(f"{base_url}/health", timeout=5)
    print(f"GET /health -> {response.status_code} {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "features, expected_class",
    [
        ([5.1, 3.5, 1.4, 0.2], 0),  # a typical setosa
        ([6.0, 2.9, 4.5, 1.5], 1),  # a typical versicolor
        ([5.9, 3.0, 5.1, 1.8], 2),  # a typical virginica
    ],
)
def test_prediction_over_http(base_url, features, expected_class):
    response = requests.post(f"{base_url}/predict", json={"x": features}, timeout=5)
    print(f"POST /predict {features} -> {response.status_code} {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"prediction": expected_class}


def test_invalid_request_is_rejected_over_http(base_url):
    response = requests.post(f"{base_url}/predict", json={"x": [1.0, 2.0]}, timeout=5)
    print(f"POST /predict with 2 values -> {response.status_code}")

    assert response.status_code == 422
