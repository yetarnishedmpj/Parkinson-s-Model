"""
session_store.py
----------------
SQLite-backed time-series store for Digital Twin telemetry sessions.

Every WebSocket connection creates a new session.  Each tick is persisted
as a JSON blob so that researchers can replay or analyse sessions offline.

Usage:
    store = SessionStore()
    session_id = store.new_session()
    store.log_tick(session_id, timestamp_iso, vitals_dict, analytics_dict)
    ticks = store.get_session(session_id)
    sessions = store.list_sessions()
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sessions.db")
_DB_PATH = os.path.normpath(_DB_PATH)


class SessionStore:
    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at   TEXT,
                    tick_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts         TEXT NOT NULL,
                    vitals     TEXT NOT NULL,
                    analytics  TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_session ON ticks(session_id)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def new_session(self) -> str:
        """Create a new session and return its UUID."""
        sid = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, started_at) VALUES (?, ?)", (sid, now)
            )
        return sid

    def close_session(self, session_id: str):
        """Mark a session as ended."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?", (now, session_id)
            )

    # ── Tick logging ──────────────────────────────────────────────────────────

    def log_tick(
        self,
        session_id: str,
        timestamp: str,
        vitals: dict,
        analytics: dict,
    ):
        """Persist one telemetry frame.  Called at 20 Hz — must be fast."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ticks (session_id, ts, vitals, analytics) VALUES (?,?,?,?)",
                (session_id, timestamp, json.dumps(vitals), json.dumps(analytics)),
            )
            conn.execute(
                "UPDATE sessions SET tick_count = tick_count + 1 WHERE id = ?",
                (session_id,),
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """Return metadata for all sessions, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, started_at, ended_at, tick_count FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> list[dict]:
        """Return all ticks for a session, each with parsed vitals + analytics."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, vitals, analytics FROM ticks WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "ts":        r["ts"],
                "vitals":    json.loads(r["vitals"]),
                "analytics": json.loads(r["analytics"]),
            }
            for r in rows
        ]

    def delete_session(self, session_id: str):
        """Remove a session and all its ticks."""
        with self._connect() as conn:
            conn.execute("DELETE FROM ticks WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# Singleton for use across the app
_store: SessionStore | None = None

def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
