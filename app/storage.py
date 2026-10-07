"""
app/storage.py — SQLite persistence for predictions (plan §7). Zero external
server: the database is one file (data/db.sqlite3), created on first use.

Design decisions (per plan):
  - A short-lived connection is opened PER CALL. FastAPI serves sync endpoints
    from a thread pool, and one shared sqlite3 connection is not safe across
    threads — per-call connections sidestep the issue entirely.
  - probabilities is stored as a JSON string and is_uncertain as 0/1, since
    SQLite has no dict/bool types. Keeping the full probability vector means
    bad stored results can be re-analyzed later WITHOUT re-running the model.
  - Duplicate tiles are allowed: every classification attempt is its own
    auditable row (plan §4).
"""

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "db.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    tile_hash TEXT,
    predicted_label TEXT,
    confidence REAL,
    probabilities TEXT,      -- JSON string of all class probabilities
    is_uncertain BOOLEAN,    -- SQLite stores this as 0/1
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _connect() -> sqlite3.Connection:
    """Fresh connection per call — see module docstring for why."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the table if it doesn't exist. Safe to call repeatedly —
    main.py calls this at startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def save_prediction(
    filename: str,
    tile_hash: str,
    result: dict,
    model_version: str,
) -> int:
    """Insert one prediction row; return the new row's id (so the API
    response can trace a curl call straight to its database row)."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (filename, tile_hash, predicted_label, confidence,
                 probabilities, is_uncertain, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                tile_hash,
                result["predicted_label"],        # raw label, always
                result["confidence"],
                json.dumps(result["probabilities"]),
                int(result["is_uncertain"]),      # bool -> 0/1
                model_version,
            ),
        )
        return int(cur.lastrowid)


def get_prediction(prediction_id: int) -> dict | None:
    """Fetch one row by id -> dict (probabilities parsed back to a dict),
    or None if the id doesn't exist."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
    if row is None:
        return None
    row = dict(row)
    row["probabilities"] = json.loads(row["probabilities"])
    return row


def count_predictions() -> int:
    """Total rows — used by /health to prove the DB is alive and writable."""
    with _connect() as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
    return int(n)