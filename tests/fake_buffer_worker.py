"""Fake ple-buffer-worker for tests: speaks the M1 JSON-line protocol."""

import json
import sys

print(json.dumps({"ok": True, "ready": True, "layer": 63, "prior_rows": 0}),
      flush=True)
seq = 0
sessions = {}
for line in sys.stdin:
    seq += 1
    try:
        req = json.loads(line)
        op = req.get("op", "store")
        if op == "store":
            sess = req.get("session", "default")
            sessions.setdefault(sess, 0)
            sessions[sess] += req["shape"][0]
            resp = {"ok": True, "seq": seq, "session": sess,
                    "span_tokens": req["shape"][0], "unique_ids": 7,
                    "n_new": 7, "evicted": 0, "pinned": 0,
                    "table_rows": sessions[sess], "gemm_ms": 1.0}
        elif op == "activate":
            sess = req.get("session", "default")
            resp = {"ok": True, "seq": seq, "session": sess,
                    "table_rows": sessions.get(sess, 0)}
        elif op == "stats":
            resp = {"ok": True, "seq": seq,
                    "sessions": {s: {"rows": r, "pinned": 0}
                                 for s, r in sessions.items()}}
        else:
            resp = {"ok": False, "error": f"ValueError: unknown op {op}"}
    except Exception as exc:  # noqa: BLE001
        resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(resp), flush=True)
