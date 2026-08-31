import os
import sqlite3
import threading
import time

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fabric.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    source TEXT NOT NULL DEFAULT 'operator',
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts DESC);
CREATE TABLE IF NOT EXISTS miniapp_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    action TEXT NOT NULL,
    choice_kind TEXT,
    choice_id TEXT,
    spec_id TEXT,
    spec_json TEXT,
    gates_json TEXT,
    passed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON miniapp_traces(ts DESC);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    last_result TEXT
);
CREATE INDEX IF NOT EXISTS idx_goals_ts ON goals(ts DESC);
"""


class LogStore:
    def __init__(self, path):
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def append(self, text, source="operator", level="info"):
        ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO logs (ts, level, source, text) VALUES (?, ?, ?, ?)",
                (ts, level, source, text),
            )
            self._conn.commit()
        return {"ts": ts, "level": level, "source": source, "text": text}

    def recent(self, limit=50, level=None, source=None):
        query = "SELECT id, ts, level, source, text FROM logs"
        clauses, params = [], []
        if level:
            clauses.append("level = ?")
            params.append(level)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        keys = ("id", "ts", "level", "source", "text")
        return [dict(zip(keys, r)) for r in rows]

    def trace(self, query: str, action: str, choice_kind: str = None,
              choice_id: str = None, spec_id: str = None,
              spec=None, gates=None, passed: bool = False) -> dict:
        import json

        ts = time.time()
        spec_json = json.dumps(spec, default=str) if spec is not None else None
        gates_json = json.dumps(gates, default=str) if gates is not None else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO miniapp_traces "
                "(ts, query, action, choice_kind, choice_id, spec_id, "
                "spec_json, gates_json, passed) VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, query or "", action, choice_kind, choice_id, spec_id,
                 spec_json, gates_json, 1 if passed else 0),
            )
            self._conn.commit()
            row_id = cur.lastrowid
        return {"id": row_id, "ts": ts, "query": query, "action": action,
                "passed": bool(passed)}

    def traces(self, limit: int = 50, passed: bool | None = None) -> list:
        import json

        q = ("SELECT id, ts, query, action, choice_kind, choice_id, spec_id, "
             "spec_json, gates_json, passed FROM miniapp_traces")
        clauses, params = [], []
        if passed is True:
            clauses.append("passed = 1")
        elif passed is False:
            clauses.append("passed = 0")
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        out = []
        keys = ("id", "ts", "query", "action", "choice_kind", "choice_id",
                "spec_id", "spec_json", "gates_json", "passed")
        for r in rows:
            item = dict(zip(keys, r))
            item["passed"] = bool(item["passed"])
            try:
                item["spec"] = json.loads(item["spec_json"]) if item["spec_json"] else None
            except Exception:
                item["spec"] = None
            try:
                item["gates"] = json.loads(item["gates_json"]) if item["gates_json"] else None
            except Exception:
                item["gates"] = None
            del item["spec_json"]
            del item["gates_json"]
            out.append(item)
        return out

    def sft_examples(self, limit: int = 200) -> list:
        """Accepted (query → spec) rows for a later script-slot LoRA."""
        rows = self.traces(limit=limit, passed=True)
        examples = []
        for r in rows:
            spec = r.get("spec") or {}
            if not r.get("query") or not spec:
                continue
            examples.append({
                "query": r["query"],
                "spec": {
                    "id": spec.get("id"),
                    "title": spec.get("title"),
                    "kernel": spec.get("kernel"),
                    "template": spec.get("template"),
                    "fields": spec.get("fields") or [],
                    "principle": spec.get("principle"),
                    "group": spec.get("group"),
                },
                "choice_id": r.get("choice_id"),
                "action": r.get("action"),
            })
        return examples

    def goal_add(self, text: str, status: str = "open") -> dict:
        ts = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO goals (ts, text, status) VALUES (?, ?, ?)",
                (ts, text, status),
            )
            self._conn.commit()
            row_id = cur.lastrowid
        return {"id": row_id, "ts": ts, "text": text, "status": status}

    def goal_update(self, goal_id: int, status: str = None,
                    last_result: str = None) -> None:
        sets, params = [], []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if last_result is not None:
            sets.append("last_result = ?")
            params.append(last_result)
        if not sets:
            return
        params.append(int(goal_id))
        with self._lock:
            self._conn.execute(
                "UPDATE goals SET " + ", ".join(sets) + " WHERE id = ?",
                params,
            )
            self._conn.commit()

    def goals(self, limit: int = 20) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, text, status, last_result FROM goals "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        keys = ("id", "ts", "text", "status", "last_result")
        return [dict(zip(keys, r)) for r in rows]


_cache = {"path": None, "store": None}


def get_log_store():
    path = os.environ.get("FABRIC_DB_PATH", _DEFAULT_DB)
    if _cache["store"] is None or _cache["path"] != path:
        _cache["store"] = LogStore(path)
        _cache["path"] = path
    return _cache["store"]


def reset_log_store():
    _cache["store"] = None
    _cache["path"] = None
