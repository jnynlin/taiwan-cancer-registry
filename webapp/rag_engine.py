"""
RAG engine client for the 'Search Manuscripts' tab.

Communicates with the review_rag FastAPI server (port 8600) via HTTP.
The server is auto-started on first use using review_rag's own venv
(which has all heavy deps: llama-index, qdrant-client, sentence-transformers).
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

_RAG_SERVER_URL = "http://127.0.0.1:8600"
_RAG_PYTHON     = "/home/jnynlin/review_rag/.venv/bin/python"
_RAG_SERVER_PY  = "/home/jnynlin/review_rag/api_server.py"
_STARTUP_TIMEOUT = 90   # seconds — embedding model load takes ~30 s on first run


@dataclass
class RAGResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    error: str = ""


class RAGEngine:
    """HTTP client around the review_rag FastAPI server.

    The server is started lazily: the first call to any method that hits the
    server will block until it is healthy (or until _STARTUP_TIMEOUT).
    """

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=_RAG_SERVER_URL, timeout=120.0)
        self._ready = False

    # ── server lifecycle ──────────────────────────────────────────────────────

    def _is_alive(self) -> bool:
        try:
            r = self._client.get("/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def ensure_server(self) -> bool:
        """Start server if not running; return True once healthy."""
        if self._ready and self._is_alive():
            return True

        if self._is_alive():
            self._ready = True
            return True

        # Launch as background daemon
        subprocess.Popen(
            [_RAG_PYTHON, _RAG_SERVER_PY],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.time() + _STARTUP_TIMEOUT
        while time.time() < deadline:
            time.sleep(2)
            if self._is_alive():
                self._ready = True
                return True

        return False

    # ── public API ────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Collection info (vector count, status). Returns error dict on failure."""
        if not self.ensure_server():
            return {"error": "RAG server did not start within timeout"}
        try:
            return self._client.get("/info").json()
        except Exception as exc:
            return {"error": str(exc)}

    def ingested_docs(self) -> list[str]:
        """Names of PDF files already indexed."""
        if not self.ensure_server():
            return []
        try:
            return self._client.get("/ingested").json().get("documents", [])
        except Exception:
            return []

    def query(self, question: str, top_k: int = 6) -> RAGResult:
        if not self.ensure_server():
            return RAGResult(answer="", error="RAG server failed to start.")
        try:
            resp = self._client.post(
                "/query",
                json={"question": question, "top_k": top_k},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return RAGResult(answer=data.get("answer", ""), sources=data.get("sources", []))
        except httpx.HTTPStatusError as exc:
            return RAGResult(answer="", error=f"Server error {exc.response.status_code}")
        except Exception as exc:
            return RAGResult(answer="", error=str(exc))

    def ingest(self, pdf_path: str | Path) -> dict[str, Any]:
        if not self.ensure_server():
            return {"error": "RAG server failed to start."}
        try:
            resp = self._client.post(
                "/ingest",
                json={"path": str(pdf_path)},
                timeout=300.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}
