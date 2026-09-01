"""Helpers for releasing Chroma's persistent resources deterministically."""

from __future__ import annotations

from typing import Any


def close_chroma(db: Any) -> None:
    """Stop one Chroma system and evict only its shared-client cache entry.

    Chroma's client does not expose a public close method in the pinned
    release. Stopping its underlying system is required on Windows before a
    persistent directory can be rebuilt or removed.
    """

    client = getattr(db, "_client", None)
    if client is None:
        return
    identifier = getattr(client, "_identifier", None)
    system = getattr(client, "_system", None)
    stop = getattr(system, "stop", None)
    if callable(stop):
        stop()

    # Chroma caches systems by persist-directory identifier. Remove this one
    # stopped system so a later rebuild creates a fresh client without
    # disrupting unrelated collections in the same process.
    try:
        from chromadb.api.client import SharedSystemClient

        systems = getattr(SharedSystemClient, "_identifier_to_system", None)
        if identifier is not None and isinstance(systems, dict):
            systems.pop(identifier, None)
    except (ImportError, AttributeError):
        pass
