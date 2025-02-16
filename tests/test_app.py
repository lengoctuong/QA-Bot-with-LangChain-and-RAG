# tests/test_app.py
import requests
import pytest

def test_root(app_url):
    """Test the root endpoint."""
    response = requests.get(app_url)
    assert response.status_code == 200