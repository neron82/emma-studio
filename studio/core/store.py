"""SQLite schema + store for Emma Studio.

Three tables, no more:
  personas   — appearance/voice/personality references
  jobs       — the central state machine object (script -> speech -> render)
  facts      — RAG fact cards (FTS5-indexed)

Design rules:
  - Every job step is persisted and individually re-triggerable.
  - Facts are curated cards, not session dumps.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    personality_prompt TEXT NOT NULL DEFAULT '',
    ref_image TEXT,             -- path for video messages (photoreal layer)
    voice_sample TEXT,          -- path to voice reference audio (h3_native)
    voice_profile TEXT,         -- OmniVoice profile id (tts_pre), e.g. 'emma_alicia'
    vrm_model TEXT,             -- VRM preset name for facetime layer
    language TEXT NOT NULL DEFAULT 'German',
    silent_summary INTEGER NOT NULL DEFAULT 1,   -- Kamera-Silent-Turns erlauben
    silent_ratio REAL NOT NULL DEFAULT 0.8,      -- Mindestanteil verworfener Silent-Turns
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL REFERENCES personas(id),
    kind TEXT NOT NULL CHECK (kind IN ('video_message', 'call_turn')),
    state TEXT NOT NULL CHECK (state IN (
        'draft', 'scripted', 'voiced', 'rendering', 'done', 'failed'
    )),
    user_input TEXT NOT NULL,           -- topic / message / user utterance
    script TEXT,                        -- agent-written scene direction + lines
    speech_path TEXT,                   -- audio file (tts_pre) or NULL (h3_native)
    speech_mode TEXT,                   -- 'h3_native' | 'tts_pre' | NULL
    render_path TEXT,                   -- final mp4
    error TEXT,
    search_used INTEGER NOT NULL DEFAULT 0,
    timings_json TEXT,                  -- per-phase latency measurements
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_persona ON jobs(persona_id, created_at DESC);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL REFERENCES personas(id),
    state TEXT NOT NULL CHECK (state IN ('active', 'finalized')) DEFAULT 'active',
    started_at REAL NOT NULL,
    ended_at REAL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT                          -- LLM-Zusammenfassung, nach der Finalisierung
);
CREATE INDEX IF NOT EXISTS idx_sessions_persona ON sessions(persona_id, started_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL REFERENCES personas(id),
    session_id TEXT REFERENCES sessions(id),   -- NULL = keine Call-Session (Batch/Telefonat)
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    job_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_persona ON messages(persona_id, created_at DESC);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    persona_id TEXT REFERENCES personas(id),  -- NULL = global fact
    fact TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact, content='facts', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, fact) VALUES (new.rowid, new.fact);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, fact) VALUES ('delete', old.rowid, old.fact);
END;
"""


def _now() -> float:
    return time.time()


def _as_bool(value) -> bool:
    """Formulare/Frontend liefern 'on', '1', 'true' — alles andere ist False."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "on", "yes", "ja"}


def _clamp_ratio(value) -> float:
    """Verwurfsquote auf 0..1 begrenzen (0 = antwortet immer, 1 = nie)."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return 0.8
    return min(1.0, max(0.0, ratio))


def _uid() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    """Synchronous store; the app is single-user, contention is a non-issue."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: realtime send-loop threads persist turns
        # alongside request-handler threads (sqlite serializes internally).
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()

    def _migrate(self) -> None:
        """Fehlende Spalten in bestehenden Datenbanken nachziehen."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(personas)")}
        for name, ddl in (
            ("silent_summary", "ALTER TABLE personas ADD COLUMN silent_summary INTEGER NOT NULL DEFAULT 1"),
            ("silent_ratio", "ALTER TABLE personas ADD COLUMN silent_ratio REAL NOT NULL DEFAULT 0.8"),
        ):
            if name not in cols:
                self.conn.execute(ddl)
        msg_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(messages)")}
        if "session_id" not in msg_cols:
            self.conn.execute("ALTER TABLE messages ADD COLUMN session_id TEXT REFERENCES sessions(id)")
        # erst nach der Spalte: auf alten DBs fehlt sie sonst noch (Startup-Crash)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- personas -----------------------------------------------------------

    def create_persona(self, name: str, *, personality_prompt: str = "",
                       ref_image: str | None = None, voice_sample: str | None = None,
                       voice_profile: str | None = None,
                       vrm_model: str | None = None, language: str = "German",
                       silent_summary: bool | int = True,
                       silent_ratio: float = 0.8) -> dict:
        now = _now()
        pid = _uid()
        self.conn.execute(
            "INSERT INTO personas (id, name, personality_prompt, ref_image, voice_sample,"
            " voice_profile, vrm_model, language, silent_summary, silent_ratio,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, name, personality_prompt, ref_image, voice_sample, voice_profile,
             vrm_model, language, 1 if silent_summary else 0, _clamp_ratio(silent_ratio),
             now, now),
        )
        self.conn.commit()
        return self.get_persona(pid)

    def get_persona(self, persona_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM personas WHERE id=?", (persona_id,)).fetchone()
        return dict(row) if row else None

    def list_personas(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM personas ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def update_persona(self, persona_id: str, **fields) -> dict | None:
        allowed = {"name", "personality_prompt", "ref_image", "voice_sample", "voice_profile",
                   "vrm_model", "language", "silent_summary", "silent_ratio"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if "silent_summary" in sets:
            sets["silent_summary"] = 1 if _as_bool(sets["silent_summary"]) else 0
        if "silent_ratio" in sets:
            sets["silent_ratio"] = _clamp_ratio(sets["silent_ratio"])
        if not sets:
            return self.get_persona(persona_id)
        sets["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in sets)
        self.conn.execute(f"UPDATE personas SET {cols} WHERE id=?", (*sets.values(), persona_id))
        self.conn.commit()
        return self.get_persona(persona_id)

    # -- jobs ---------------------------------------------------------------

    def create_job(self, persona_id: str, kind: str, user_input: str) -> dict:
        now = _now()
        jid = _uid()
        self.conn.execute(
            "INSERT INTO jobs (id, persona_id, kind, state, user_input, created_at, updated_at)"
            " VALUES (?,?,?,'draft',?,?,?)",
            (jid, persona_id, kind, user_input, now, now),
        )
        self.conn.commit()
        return self.get_job(jid)

    def get_job(self, job_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def set_job_state(self, job_id: str, state: str, **fields) -> dict | None:
        allowed = {"script", "speech_path", "speech_mode", "render_path", "error", "search_used"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        sets["state"] = state
        sets["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in sets)
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*sets.values(), job_id))
        self.conn.commit()
        return self.get_job(job_id)

    def list_jobs(self, persona_id: str | None = None, limit: int = 50) -> list[dict]:
        if persona_id:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE persona_id=? ORDER BY created_at DESC LIMIT ?",
                (persona_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- facts (RAG cards) ----------------------------------------------------

    def add_fact(self, fact: str, *, persona_id: str | None = None, source: str = "user") -> dict:
        fid = _uid()
        self.conn.execute(
            "INSERT INTO facts (id, persona_id, fact, source, created_at) VALUES (?,?,?,?,?)",
            (fid, persona_id, fact.strip(), source, _now()),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM facts WHERE id=?", (fid,)).fetchone()
        return dict(row)

    def delete_fact(self, fact_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def search_facts(self, query: str, *, persona_id: str | None = None, limit: int = 8) -> list[dict]:
        """Hybrid fact search: FTS5 keyword match + embedding similarity.

        Falls back gracefully: no embed server -> FTS only; no FTS hit ->
        embedding-only; both empty -> [].
        """
        results: dict[str, tuple[float, dict]] = {}  # id -> (score, row)

        # 1) keyword path
        sql = """
            SELECT f.* FROM facts f
            JOIN facts_fts fts ON fts.rowid = f.rowid
            WHERE facts_fts MATCH ?
        """
        params: list = [query]
        if persona_id:
            sql += " AND (f.persona_id = ? OR f.persona_id IS NULL)"
            params.append(persona_id)
        sql += " ORDER BY (f.persona_id IS NULL) ASC, f.created_at DESC LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []  # malformed FTS query — never crash the dialog on this
        for rank, r in enumerate(rows):
            # earlier ranks score higher; persona-scoped facts get a boost
            boost = 1.0 if not r["persona_id"] or r["persona_id"] == persona_id else 0.0
            results[r["id"]] = ((len(rows) - rank) / len(rows) + boost, dict(r))

        # 2) semantic path (best effort)
        try:
            emb_hits = self._semantic_search(query, persona_id=persona_id, limit=limit)
        except Exception:  # noqa: BLE001 — embed server down is non-fatal
            emb_hits = []
        for rank, r in enumerate(emb_hits):
            score = (len(emb_hits) - rank) / max(len(emb_hits), 1) * 0.9
            if r["id"] in results:
                results[r["id"]] = (results[r["id"]][0] + score, r)
            else:
                results[r["id"]] = (score, r)

        ranked = sorted(results.values(), key=lambda t: -t[0])
        return [row for _score, row in ranked[:limit]]

    def _all_facts_scoped(self, persona_id: str | None) -> list[dict]:
        if persona_id:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE persona_id=? OR persona_id IS NULL", (persona_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM facts").fetchall()
        return [dict(r) for r in rows]

    def _semantic_search(self, query: str, *, persona_id: str | None, limit: int) -> list[dict]:
        """Embedding similarity via the local bge-m3 server (OpenAI-style API)."""
        import httpx

        candidates = self._all_facts_scoped(persona_id)
        if not candidates:
            return []
        url = f"{config.EMBED_URL.rstrip('/')}/v1/embeddings"
        with httpx.Client(timeout=10) as client:
            q_resp = client.post(
                url, json={"input": query, "model": config.EMBED_MODEL}
            )
            q_resp.raise_for_status()
            q_vec = _first_embedding(q_resp.json())
            texts = [c["fact"] for c in candidates]
            d_resp = client.post(url, json={"input": texts, "model": config.EMBED_MODEL})
            d_resp.raise_for_status()
            d_vecs = _all_embeddings(d_resp.json())
        scored = sorted(
            zip(candidates, d_vecs), key=lambda cr: -_cosine(q_vec, cr[1])
        )
        # keep only reasonably similar hits
        return [c for c, v in scored[:limit] if _cosine(q_vec, v) > 0.4]

    def count_facts(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # -- messages (persistent conversation history) --------------------------

    def add_message(self, persona_id: str, role: str, content: str,
                    *, job_id: str | None = None, session_id: str | None = None) -> dict:
        mid = _uid()
        self.conn.execute(
            "INSERT INTO messages (id, persona_id, session_id, role, content, job_id, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (mid, persona_id, session_id, role, content, job_id, _now()),
        )
        if session_id:
            # turn_count zählt User- und Avatar-Beiträge (nicht Silent-Discard)
            self.conn.execute("UPDATE sessions SET turn_count = turn_count + 1 WHERE id=?", (session_id,))
        self.conn.commit()
        return {"id": mid, "persona_id": persona_id, "session_id": session_id,
                "role": role, "content": content, "job_id": job_id}

    def get_history(self, persona_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE persona_id=? ORDER BY created_at DESC LIMIT ?",
            (persona_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- sessions (Facetime-Calls) -------------------------------------------

    def create_session(self, persona_id: str) -> dict:
        sid = _uid()
        now = _now()
        self.conn.execute(
            "INSERT INTO sessions (id, persona_id, state, started_at) VALUES (?,?, 'active', ?)",
            (sid, persona_id, now),
        )
        self.conn.commit()
        return self.get_session(sid)

    def get_session(self, session_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, persona_id: str | None = None, limit: int = 50) -> list[dict]:
        if persona_id:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE persona_id=? ORDER BY started_at DESC LIMIT ?",
                (persona_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def session_transcript(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Transkript einer Session in Gesprächsreihenfolge (User + Avatar, ohne VLM)."""
        sql = "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC"
        params: list = [session_id]
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def finalize_session(self, session_id: str, *, summary: str | None = None) -> dict | None:
        """active → finalized (idempotent). Zweiter Aufruf liefert die Zeile unverändert."""
        row = self.get_session(session_id)
        if row is None:
            return None
        if row["state"] != "active":
            return row
        sets = ["state='finalized'", "ended_at=?"]
        params: list = [_now()]
        if summary is not None:
            sets.append("summary=?")
            params.append(summary)
        self.conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id=?", (*params, session_id))
        self.conn.commit()
        return self.get_session(session_id)

    def finalize_session_now(self, session_id: str) -> dict | None:
        """Transition an active session to finalized.

        Returns the row only when this call performed the transition (so callers
        can schedule post-call work exactly once); returns None otherwise.
        """
        row = self.get_session(session_id)
        if row is None or row["state"] != "active":
            return None
        cur = self.conn.execute(
            "UPDATE sessions SET state='finalized', ended_at=? WHERE id=? AND state='active'",
            (_now(), session_id),
        )
        if cur.rowcount != 1:
            return None  # der andere Finalisierer war zuerst
        self.conn.commit()
        return self.get_session(session_id)

    def set_session_summary(self, session_id: str, summary: str) -> dict | None:
        self.conn.execute("UPDATE sessions SET summary=? WHERE id=?", (summary, session_id))
        self.conn.commit()
        return self.get_session(session_id)


def _first_embedding(data) -> list[float]:
    """Extract one embedding vector from common server response shapes."""
    if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
        emb = data["data"][0].get("embedding") if isinstance(data["data"][0], dict) else None
        if isinstance(emb, list):
            return emb
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]
    if isinstance(data, dict):
        for key in ("embedding", "embeddings"):
            v = data.get(key)
            if isinstance(v, list) and v:
                return v[0] if isinstance(v[0], list) else v
    raise ValueError("unrecognized embedding response shape")


def _all_embeddings(data) -> list[list[float]]:
    if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
        first = data["data"][0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return [item["embedding"] for item in data["data"]]
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data
    if isinstance(data, dict):
        v = data.get("embeddings")
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v
    raise ValueError("unrecognized embeddings response shape")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
