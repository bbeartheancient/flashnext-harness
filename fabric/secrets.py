"""KEY=VALUE env-file loader for fabric/data secrets (mode 0600).

Same trust tier as harness.json: values stay on the ship, never logged,
never returned by GET routes. Modules read what they need at call time.
"""

from __future__ import annotations

import os
from pathlib import Path


def load(name: str = "media-keys.env") -> dict[str, str]:
    data_dir = os.environ.get(
        "FABRIC_SECRETS_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    path = Path(data_dir) / name
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            out[key.strip()] = val
    except OSError:
        pass
    return out
