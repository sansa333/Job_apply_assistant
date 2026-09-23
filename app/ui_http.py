"""Forward the owner token only to the configured backend, never arbitrary UI URLs."""
from __future__ import annotations

import os
from urllib.parse import urlparse

import requests


def backend_request(method: str, url: str, **kwargs):
    configured = urlparse(os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    target = urlparse(url)
    headers = dict(kwargs.pop("headers", {}))
    if (configured.scheme, configured.netloc) == (target.scheme, target.netloc):
        token = os.getenv("API_TOKEN", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return requests.request(method, url, headers=headers, allow_redirects=False, **kwargs)
