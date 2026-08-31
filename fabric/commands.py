"""Harness slash commands — operator ABI for the command bar.

Typing `/` in the LCARS command line lists these. They cover the whole
fabric (duty LLM, Pi coding lane, pipecat voice stub, ship tools), not
only MiniApp development. Source of truth for GET /api/commands.
"""

from __future__ import annotations

COMMANDS = [
    {"id": "help", "usage": "/help",
     "blurb": "harness command index", "lane": "command"},
    {"id": "status", "usage": "/status",
     "blurb": "ship power caps and slot health", "lane": "duty"},
    {"id": "log", "usage": "/log [n]",
     "blurb": "recent ship log", "lane": "duty"},
    {"id": "chat", "usage": "/chat <prompt>",
     "blurb": "duty agent + tools (default bar route)", "lane": "duty"},
    {"id": "loop", "usage": "/loop <prompt>",
     "blurb": "duty tool loop — goal, act, report", "lane": "duty"},
    {"id": "code", "usage": "/code <task>",
     "blurb": "OMP coding agent (read/bash/edit/grep)", "lane": "omp"},
    {"id": "pi", "usage": "/pi <task>",
     "blurb": "alias for /code", "lane": "omp"},
    {"id": "goal", "usage": "/goal [statement]",
     "blurb": "set or list a coding goal; OMP runs it", "lane": "omp"},
    {"id": "goals", "usage": "/goals",
     "blurb": "list stored goals", "lane": "omp"},
    {"id": "voice", "usage": "/voice",
     "blurb": "pipecat voice-leg status (Track 2 last)", "lane": "pipecat"},
    {"id": "sensors", "usage": "/sensors [prefix]",
     "blurb": "MQTT sensor bus", "lane": "duty"},
    {"id": "wizard", "usage": "/wizard",
     "blurb": "new signal-path route wizard", "lane": "ui"},
    {"id": "library", "usage": "/library",
     "blurb": "open the app library", "lane": "ui"},
    {"id": "theme", "usage": "/theme [name]",
     "blurb": "set LCARS palette", "lane": "ui"},
    {"id": "harness", "usage": "/harness",
     "blurb": "orchestrator + API keys (Grok / Kimi / Zen)", "lane": "ui"},
    {"id": "settings", "usage": "/settings",
     "blurb": "alias for /harness", "lane": "ui"},
    # --- bicameral / OMP lanes -----------------------------------------
    {"id": "omp", "usage": "/omp <task>",
     "blurb": "run OMP coding agent (fabric skill harness)", "lane": "omp"},
    {"id": "bicameral", "usage": "/bicameral [n_frames] [alpha]",
     "blurb": "run the two-card test stand (GPU1→GPU0 pipeline)", "lane": "bicameral"},
    {"id": "swarm", "usage": "/swarm <prompt>",
     "blurb": "multi-agent swarm vs duty LLM (n_per_persona=1)", "lane": "swarm"},
    {"id": "trace", "usage": "/trace [start|stop|snapshot]",
     "blurb": "DMA trace: start/stop recording or dump snapshot", "lane": "bicameral"},
    {"id": "improve", "usage": "/improve <prompt>",
     "blurb": "self-improvement loop: generate→capture→score→critique", "lane": "bicameral"},
    {"id": "atomize", "usage": "/atomize <source> <dest>",
     "blurb": "trace DMA path → atomic program or n-gram teacher/student", "lane": "bicameral"},
    {"id": "domain", "usage": "/domain [intent]",
     "blurb": "route intent → domain shard swap; no arg = state", "lane": "bicameral"},
    {"id": "shard", "usage": "/shard [list|export|import]",
     "blurb": "list / export / import .mv2 domain shards", "lane": "bicameral"},
    {"id": "atomic", "usage": "/atomic <description>",
     "blurb": "compile natural-language → AtomicProgram via duty LLM", "lane": "bicameral"},
    {"id": "teach", "usage": "/teach [intent]",
     "blurb": "build a teaching prompt for a task; no arg = list examples", "lane": "bicameral"},
    {"id": "roundtrip", "usage": "/roundtrip <app_id>",
     "blurb": "decompose app → modify → recompile", "lane": "bicameral"},
    {"id": "buffer", "usage": "/buffer [state|publish|trigger]",
     "blurb": "buffer bus: dump state, fire publish, or named trigger",
     "lane": "bicameral"},
    {"id": "optimize", "usage": "/optimize [focus]",
     "blurb": "self-analyse: shards, DMA trace, buffer, process RSS", "lane": "bicameral"},
]


def catalog() -> dict:
    return {"commands": list(COMMANDS)}


def parse(q: str) -> tuple[str, str]:
    raw = (q or "").strip()
    if not raw.startswith("/"):
        return "", raw
    body = raw[1:].strip()
    if not body:
        return "help", ""
    if " " in body:
        cmd, rest = body.split(" ", 1)
        return cmd.lower(), rest.strip()
    return body.lower(), ""


def resolve(q: str) -> dict:
    """Map a slash line to a console match, a lane POST, or a UI action."""
    cmd, args = parse(q)
    known = {c["id"]: c for c in COMMANDS}
    if cmd in ("pi",):
        cmd = "code"
    meta = known.get(cmd)
    if meta is None:
        return {
            "query": q, "slash": True, "command": cmd, "args": args,
            "match": None, "autorun": False, "fallback": "help",
            "candidates": [{"id": c["id"], "title": c["usage"],
                            "group": c["lane"]} for c in COMMANDS],
            "hint": f"unknown /{cmd} — type /help",
            "department": 100,
        }
    if cmd == "help":
        return {
            "query": q, "slash": True, "command": "help", "args": args,
            "match": None, "autorun": False, "fallback": "help",
            "candidates": [{"id": c["id"], "title": c["usage"] + " — " + c["blurb"],
                            "group": c["lane"]} for c in COMMANDS],
            "hint": "harness commands", "department": 100,
        }
    if cmd == "status":
        return _fn("ship", q, {}, True)
    if cmd == "log":
        fields = {}
        if args.isdigit():
            fields["limit"] = int(args)
        return _fn("log", q, fields, True)
    if cmd == "sensors":
        fields = {"prefix": args} if args else {}
        return _fn("sensors", q, fields, True)
    if cmd == "wizard":
        return {
            "query": q, "slash": True, "command": "wizard", "args": args,
            "match": None, "autorun": False, "wizard": True,
            "department": 100, "hint": "MiniApp wizard",
        }
    if cmd == "library":
        return {
            "query": q, "slash": True, "command": "library", "args": args,
            "ui": "library", "autorun": False, "department": 100,
        }
    if cmd == "theme":
        return {
            "query": q, "slash": True, "command": "theme", "args": args,
            "ui": "theme", "autorun": False, "department": 100,
        }
    if cmd in ("harness", "settings"):
        return {
            "query": q, "slash": True, "command": "harness", "args": args,
            "ui": "harness", "autorun": False, "department": 100,
        }
    if cmd == "voice":
        return {
            "query": q, "slash": True, "command": "voice", "args": args,
            "method": "GET", "url": "/api/voice", "autorun": True,
            "viewport": "term", "department": 100, "lane": "pipecat",
        }
    if cmd == "goals":
        return {
            "query": q, "slash": True, "command": "goals", "args": "",
            "method": "GET", "url": "/api/goals", "autorun": True,
            "viewport": "term", "department": 100, "lane": "omp",
        }
    if cmd == "goal":
        if not args:
            return {
                "query": q, "slash": True, "command": "goals", "args": "",
                "method": "GET", "url": "/api/goals", "autorun": True,
                "viewport": "term", "department": 100, "lane": "omp",
            }
        return {
            "query": q, "slash": True, "command": "goal", "args": args,
            "method": "POST", "url": "/api/pi/goal",
            "body": {"goal": args}, "autorun": True,
            "viewport": "both", "department": 100, "lane": "omp",
        }
    if cmd == "code":
        if not args:
            return {
                "query": q, "slash": True, "command": "code", "args": "",
                "autorun": False, "hint": "/code <task> — OMP coding lane",
                "department": 100, "lane": "omp",
            }
        return {
            "query": q, "slash": True, "command": "code", "args": args,
            "method": "POST", "url": "/api/pi/code",
            "body": {"prompt": args}, "autorun": True,
            "viewport": "both", "department": 100, "lane": "omp",
        }
    if cmd in ("chat", "loop"):
        from . import console

        chat = next(f for f in console.FUNCTIONS if f["id"] == "chat")
        prefix = "[GOAL LOOP] " if cmd == "loop" else ""
        return {
            "query": q, "slash": True, "command": cmd, "args": args,
            "match": {k: chat[k] for k in
                      ("id", "title", "group", "method", "url", "viewport",
                       "fields") if k in chat},
            "fields": {"message": prefix + (args or q)},
            "autorun": True, "department": 100, "lane": "duty",
        }
    if cmd == "omp":
        if not args:
            return {
                "query": q, "slash": True, "command": "omp", "args": "",
                "autorun": False, "hint": "/omp <task> — OMP coding agent",
                "department": 100, "lane": "omp",
            }
        return {
            "query": q, "slash": True, "command": "omp", "args": args,
            "method": "POST", "url": "/api/omp/run",
            "body": {"message": args}, "autorun": True,
            "viewport": "both", "department": 100, "lane": "omp",
        }
    if cmd == "bicameral":
        toks = args.split() if args else []
        n_frames = int(toks[0]) if toks and toks[0].isdigit() else 60
        alpha = float(toks[1]) if len(toks) > 1 else 0.2
        return {
            "query": q, "slash": True, "command": "bicameral", "args": args,
            "method": "POST", "url": "/api/bicameral/run",
            "body": {"n_frames": n_frames, "alpha": alpha},
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "swarm":
        if not args:
            return {
                "query": q, "slash": True, "command": "swarm", "args": "",
                "autorun": False, "hint": "/swarm <prompt>",
                "department": 100, "lane": "swarm",
            }
        return {
            "query": q, "slash": True, "command": "swarm", "args": args,
            "method": "POST", "url": "/api/swarm/run",
            "body": {"prompt": args}, "autorun": True,
            "viewport": "term", "department": 100, "lane": "swarm",
        }
    if cmd == "trace":
        sub = args.strip().lower() or "snapshot"
        if sub == "start":
            return {
                "query": q, "slash": True, "command": "trace", "args": sub,
                "method": "POST", "url": "/api/trace/start",
                "body": {}, "autorun": True,
                "viewport": "term", "department": 100, "lane": "bicameral",
            }
        if sub == "stop":
            return {
                "query": q, "slash": True, "command": "trace", "args": sub,
                "method": "POST", "url": "/api/trace/stop",
                "body": {}, "autorun": True,
                "viewport": "term", "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "trace", "args": sub,
            "method": "GET", "url": "/api/trace/snapshot",
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "improve":
        if not args:
            return {
                "query": q, "slash": True, "command": "improve", "args": "",
                "autorun": False, "hint": "/improve <prompt>",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "improve", "args": args,
            "method": "POST", "url": "/api/improve/run",
            "body": {"prompt": args, "max_iterations": 3, "compact_rate": 0.5},
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "atomize":
        # /atomize <source> [destination]
        # source = live | recent | trace  (defaults: live if DMA trace active)
        # destination = atomic | ngram | both  (default: atomic)
        toks = args.split() if args else []
        if not toks:
            return {
                "query": q, "slash": True, "command": "atomize", "args": "",
                "autorun": False,
                "hint": "/atomize <source> [atomic|ngram|both] "
                        "— trace DMA path → assembly or n-grams",
                "department": 100, "lane": "bicameral",
            }
        source = toks[0].lower() if toks[0] else "live"
        dest = toks[1].lower() if len(toks) > 1 else "atomic"
        if source not in ("live", "recent", "snapshot", "trace"):
            return {
                "query": q, "slash": True, "command": "atomize", "args": args,
                "autorun": False,
                "hint": f"/atomize: source '{source}' must be live|recent|trace",
                "department": 100, "lane": "bicameral",
            }
        if dest not in ("atomic", "ngram", "both"):
            return {
                "query": q, "slash": True, "command": "atomize", "args": args,
                "autorun": False,
                "hint": f"/atomize: dest '{dest}' must be atomic|ngram|both",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "atomize", "args": args,
            "method": "POST", "url": "/api/atomize/run",
            "body": {"source": source, "dest": dest},
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "domain":
        if not args:
            return {
                "query": q, "slash": True, "command": "domain", "args": "",
                "method": "GET", "url": "/api/domain/state",
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "domain", "args": args,
            "method": "POST", "url": "/api/domain/route",
            "body": {"intent": args, "top_k": 1},
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "shard":
        sub = args.strip().lower() or "list"
        if sub == "list":
            return {
                "query": q, "slash": True, "command": "shard", "args": sub,
                "method": "GET", "url": "/api/shard/list",
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        if sub.startswith("export"):
            parts = sub.split(maxsplit=1)
            domain = parts[1] if len(parts) > 1 else ""
            return {
                "query": q, "slash": True, "command": "shard", "args": sub,
                "method": "POST", "url": "/api/shard/export",
                "body": {"domain": domain} if domain else {},
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        if sub.startswith("import"):
            parts = sub.split(maxsplit=1)
            src_path = parts[1] if len(parts) > 1 else ""
            return {
                "query": q, "slash": True, "command": "shard", "args": sub,
                "method": "POST", "url": "/api/shard/import",
                "body": {"src_path": src_path} if src_path else {},
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "shard", "args": sub,
            "method": "GET", "url": "/api/shard/list",
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "atomic":
        if not args:
            return {
                "query": q, "slash": True, "command": "atomic", "args": "",
                "method": "GET", "url": "/api/atomic/examples",
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "atomic", "args": args,
            "method": "POST", "url": "/api/atomic/from_description",
            "body": {"description": args}, "autorun": True,
            "viewport": "term", "department": 100, "lane": "bicameral",
        }
    if cmd == "teach":
        if not args:
            return {
                "query": q, "slash": True, "command": "teach", "args": "",
                "method": "GET", "url": "/api/teach/examples",
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "teach", "args": args,
            "method": "POST", "url": "/api/teach/prompt",
            "body": {"intent": args}, "autorun": True,
            "viewport": "term", "department": 100, "lane": "bicameral",
        }
    if cmd == "roundtrip":
        if not args:
            return {
                "query": q, "slash": True, "command": "roundtrip", "args": "",
                "autorun": False,
                "hint": "/roundtrip <app_id> — decompose, modify, recompile",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "roundtrip", "args": args,
            "method": "POST", "url": "/api/roundtrip/apply",
            "body": {"app_id": args, "modifications": {}},
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "buffer":
        sub = args.strip().lower() or "state"
        if sub == "state" or sub == "":
            return {
                "query": q, "slash": True, "command": "buffer", "args": sub,
                "method": "GET", "url": "/api/buffer/state",
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        if sub.startswith("trigger"):
            parts = sub.split(maxsplit=2)
            name = parts[1] if len(parts) > 1 else "layer_swap"
            layer = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else -1
            return {
                "query": q, "slash": True, "command": "buffer", "args": sub,
                "method": "POST", "url": "/api/buffer/trigger",
                "body": {"name": name, "layer": layer},
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        if sub.startswith("publish"):
            parts = sub.split(maxsplit=2)
            name = parts[1] if len(parts) > 1 else "compaction_90"
            val = float(parts[2]) if len(parts) > 2 else 0.5
            return {
                "query": q, "slash": True, "command": "buffer", "args": sub,
                "method": "POST", "url": "/api/buffer/publish",
                "body": {"name": name, "value": val},
                "autorun": True, "viewport": "term",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "buffer", "args": sub,
            "method": "GET", "url": "/api/buffer/state",
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    if cmd == "optimize":
        focus = args.strip().lower() or ""
        if focus and focus not in ("shard", "trace", "buffer", "domain",
                                     "memory", "process", "teach"):
            return {
                "query": q, "slash": True, "command": "optimize", "args": args,
                "autorun": False,
                "hint": f"/optimize: focus '{focus}' must be one of "
                        f"shard|trace|buffer|domain|memory|process|teach",
                "department": 100, "lane": "bicameral",
            }
        return {
            "query": q, "slash": True, "command": "optimize", "args": focus,
            "method": "POST", "url": "/api/optimize/analyze",
            "body": {"focus": focus} if focus else {},
            "autorun": True, "viewport": "term",
            "department": 100, "lane": "bicameral",
        }
    return {
        "query": q, "slash": True, "command": cmd, "args": args,
        "autorun": False, "department": 100,
    }


def _fn(fid: str, q: str, fields: dict, autorun: bool) -> dict:
    from . import console

    fn = next(f for f in console.FUNCTIONS if f["id"] == fid)
    return {
        "query": q, "slash": True, "command": fid, "args": "",
        "match": {k: fn[k] for k in
                  ("id", "title", "group", "method", "url", "viewport",
                   "fields") if k in fn},
        "fields": fields, "autorun": autorun,
        "department": 100 if fn.get("group") == "command" else 400,
        "lane": "duty",
    }
