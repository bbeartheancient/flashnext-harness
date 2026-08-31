"""Self-analysis harness for /optimize.

The model looks at its own process state — memory shards, response
patterns, DMA trace history, context budgets, and prompt caches —
then makes concrete, actionable recommendations.

The /optimize command (lane=bicameral) calls analyze() with no
arguments; the model receives a structured snapshot and makes
recommendations.  Each finding is typed (BLOAT | STALE | WASTED |
INEFFICIENT | MISSING) so the model can route to different fix tools.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Probe functions — each returns typed findings for the model
# ---------------------------------------------------------------------------

def _shard_probe() -> list[dict]:
    """Examine .mv2 memory shards: size, doc count, access age."""
    findings: list[dict] = []
    try:
        from . import dept_memory
        rows = dept_memory.status()
        for row in rows.get("shards", []):
            sz = row.get("bytes", 0)
            dept = row.get("dept", "?")
            shard = row.get("shard", "?")
            # Size thresholds (BLOAT > 10 MB, OK < 1 MB)
            if sz > 10 * 1024 * 1024:
                findings.append({
                    "type": "BLOAT",
                    "subject": f"shard:{dept}",
                    "detail": f"shard '{shard}' is {sz // (1024*1024)} MB "
                               f"(>{10} MB — consider trimming old docs)",
                    "severity": "high",
                    "fix": f"search for low-relevance docs with "
                           f"/shard export {dept}; manually delete "
                           f"and re-import a trimmed version",
                })
            elif sz == 0:
                findings.append({
                    "type": "MISSING",
                    "subject": f"shard:{dept}",
                    "detail": f"shard '{shard}' is empty or uninitialised",
                    "severity": "low",
                    "fix": "add initial docs with dept_memory_add",
                })
            else:
                findings.append({
                    "type": "OK",
                    "subject": f"shard:{dept}",
                    "detail": f"shard '{shard}' is {sz // 1024} KB, healthy",
                    "severity": "info",
                    "fix": None,
                })
    except Exception as e:
        findings.append({"type": "ERROR", "subject": "shard_probe",
                         "detail": str(e), "severity": "high", "fix": None})
    return findings


def _trace_probe() -> list[dict]:
    """Examine the DMA trace circular buffer: hot/cold modules."""
    findings: list[dict] = []
    try:
        from . import dma_trace
        snap = dma_trace.snapshot()
        frames = snap.get("frames", [])
        n = len(frames)
        max_f = snap.get("max_frames", "?")

        if n == 0:
            findings.append({
                "type": "MISSING",
                "subject": "dma_trace",
                "detail": "No frames recorded — run /trace start first",
                "severity": "low",
                "fix": "/trace start",
            })
            return findings

        # Count per module_id
        counts: dict[str, int] = {}
        for f in frames:
            mid = f.get("module_id", "?")
            counts[mid] = counts.get(mid, 0) + 1

        hot = sorted(counts.items(), key=lambda x: -x[1])[:3]
        cold = [k for k, v in counts.items() if v == 1]

        fill_pct = int(n / max_f * 100) if isinstance(max_f, int) else "?"
        findings.append({
            "type": "OK",
            "subject": "dma_trace",
            "detail": (f"{n} frames, {max_f} max "
                       f"({fill_pct}% full); "
                       f"hot: {dict(hot)}; {len(cold)} cold module(s)"),
            "severity": "info",
            "fix": None,
        })

        if fill_pct != "?" and fill_pct > 80:
            findings.append({
                "type": "INEFFICIENT",
                "subject": "dma_trace:buffer",
                "detail": f"Buffer {fill_pct}% full — oldest frames will "
                           f"drop soon. Export or /trace stop to preserve.",
                "severity": "medium",
                "fix": "/trace stop then /atomize recent atomic",
            })

        if len(cold) > len(counts) * 0.6:
            findings.append({
                "type": "WASTED",
                "subject": "dma_trace:cold_modules",
                "detail": f"{len(cold)}/{len(counts)} modules recorded "
                           f"only once — not enough data to atomise",
                "severity": "low",
                "fix": "Run the same signal chain more times "
                       "before /atomize",
            })
    except Exception as e:
        findings.append({"type": "ERROR", "subject": "trace_probe",
                         "detail": str(e), "severity": "high", "fix": None})
    return findings


def _buffer_probe() -> list[dict]:
    """Examine the MicroFX signal buffer ring."""
    findings: list[dict] = []
    try:
        from . import qwen_buffer
        snap = qwen_buffer.snapshot()
        events = snap.get("events", [])
        capacity = snap.get("capacity", 256)
        fill = len(events)
        fill_pct = int(fill / capacity * 100) if capacity else 0

        findings.append({
            "type": "OK",
            "subject": "qwen_buffer",
            "detail": f"{fill}/{capacity} events "
                       f"({fill_pct}% used)",
            "severity": "info",
            "fix": None,
        })

        if fill_pct > 90:
            findings.append({
                "type": "INEFFICIENT",
                "subject": "qwen_buffer:ring",
                "detail": f"Event ring {fill_pct}% full — "
                           f"old events will be dropped on next push",
                "severity": "medium",
                "fix": "fire_mv2_tap() or export the buffer "
                       "before it wraps",
            })

        # Check for hot module events
        names: dict[str, int] = {}
        for ev in events:
            n = ev.get("name", "?")
            names[n] = names.get(n, 0) + 1
        hot = sorted(names.items(), key=lambda x: -x[1])[:5]
        if hot:
            findings.append({
                "type": "OK",
                "subject": "qwen_buffer:hot_modules",
                "detail": f"Most active modules: {dict(hot)}",
                "severity": "info",
                "fix": None,
            })
    except Exception as e:
        findings.append({"type": "ERROR", "subject": "buffer_probe",
                         "detail": str(e), "severity": "medium", "fix": None})
    return findings


def _domain_probe() -> list[dict]:
    """Check loaded domain shards vs the router catalog."""
    findings: list[dict] = []
    try:
        from . import domain_router
        loaded = domain_router.list_loaded()
        catalog = domain_router.domain_summary()

        if not loaded.get("current"):
            findings.append({
                "type": "MISSING",
                "subject": "domain_router",
                "detail": "No domain shard currently loaded",
                "severity": "low",
                "fix": "/domain <topic> to load a shard",
            })
        else:
            findings.append({
                "type": "OK",
                "subject": "domain_router",
                "detail": (f"Loaded: {loaded.get('current')}; "
                           f"{len(catalog.get('entries', []))} catalog entries"),
                "severity": "info",
                "fix": None,
            })
    except Exception as e:
        findings.append({"type": "ERROR", "subject": "domain_probe",
                         "detail": str(e), "severity": "low", "fix": None})
    return findings


def _dept_memory_probe() -> list[dict]:
    """Analyze dept memory content: shard size vs doc count, recency."""
    findings: list[dict] = []
    try:
        from . import dept_memory
        from .departments import DEPARTMENTS
        for row in DEPARTMENTS:
            dept = row.get("name", "?")
            # Use the dept name as query — guaranteed in the charter seed
            t0 = time.perf_counter()
            result = dept_memory.search(dept, dept, k=1)
            latency_ms = (time.perf_counter() - t0) * 1000
            hits = len(result.get("hits", []))
            err = result.get("error")
            if err:
                findings.append({
                    "type": "ERROR",
                    "subject": f"dept_memory:{dept}",
                    "detail": f"Shard search error: {err}",
                    "severity": "medium",
                    "fix": "verify memvid-sdk install + shard file",
                })
            elif latency_ms > 200:
                findings.append({
                    "type": "INEFFICIENT",
                    "subject": f"dept_memory:{dept}",
                    "detail": f"Shard search took {latency_ms:.0f} ms "
                               f"(>{200} ms threshold)",
                    "severity": "medium",
                    "fix": f"Consider /shard export {dept} and "
                           f"re-import a compact version",
                })
            elif hits == 0:
                findings.append({
                    "type": "STALE",
                    "subject": f"dept_memory:{dept}",
                    "detail": "Shard returns no hits — charter seed may be "
                               "missing (re-seed the shard)",
                    "severity": "low",
                    "fix": f"Add charter docs with dept_memory_add",
                })
            else:
                findings.append({
                    "type": "OK",
                    "subject": f"dept_memory:{dept}",
                    "detail": f"Shard search {latency_ms:.1f} ms, "
                               f"{hits} hit(s)",
                    "severity": "info",
                    "fix": None,
                })
    except Exception as e:
        findings.append({"type": "ERROR", "subject": "dept_memory_probe",
                         "detail": str(e), "severity": "medium", "fix": None})
    return findings


def _process_probe() -> list[dict]:
    """OS-level memory and process stats for the fabric process."""
    findings: list[dict] = []
    try:
        pid = os.getpid()
        # /proc/self/status is Linux-specific
        status = Path("/proc/self/status").read_text()
        lines = {l.split(":", 1)[0]: l.split(":", 1)[1].strip()
                 for l in status.splitlines() if ":" in l}
        rss_kb = int(lines.get("VmRSS", "0").split()[0])
        vms_kb = int(lines.get("VmSize", "0").split()[0])
        rss_mb = rss_kb / 1024
        vms_mb = vms_kb / 1024
        findings.append({
            "type": "OK",
            "subject": "process:memory",
            "detail": f"RSS {rss_mb:.0f} MB / VMS {vms_mb:.0f} MB",
            "severity": "info",
            "fix": None,
        })
        if rss_mb > 2000:
            findings.append({
                "type": "BLOAT",
                "subject": "process:memory",
                "detail": f"RSS {rss_mb:.0f} MB (>{2000} MB — "
                           f"process is using significant memory)",
                "severity": "medium",
                "fix": "Reduce shard load or flush unused caches",
            })
    except Exception:
        pass
    return findings


def _teach_probe() -> list[dict]:
    """Check teaching example registry for gaps vs the catalog."""
    findings: list[dict] = []
    try:
        from . import atomic_program, microfx
        examples = atomic_program.get_teacher_examples(limit=100)
        registered = {e["summary"]["name"] for e in examples}
        catalog = microfx.module_catalog()
        catalog_primitives = set(catalog.keys())
        missing = catalog_primitives - registered
        if missing:
            findings.append({
                "type": "MISSING",
                "subject": "teach:examples",
                "detail": f"{len(missing)} primitives have no teaching "
                           f"example: {sorted(missing)[:5]}",
                "severity": "low",
                "fix": "Record DMA traces for these modules, "
                       "then /atomize recent ngram",
            })
        else:
            findings.append({
                "type": "OK",
                "subject": "teach:examples",
                "detail": (f"{len(examples)} examples registered, "
                           f"all {len(catalog_primitives)} catalog "
                           f"primitives covered"),
                "severity": "info",
                "fix": None,
            })
    except Exception as e:
        findings.append({"type": "ERROR", "subject": "teach_probe",
                         "detail": str(e), "severity": "low", "fix": None})
    return findings


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze(focus: str | None = None) -> dict:
    """Run all probes and return a structured findings list.

    focus: optional probe name to run only that probe
           (shard | trace | buffer | domain | memory | process | teach)
    """
    probes = {
        "shard":    ("Memory shards (.mv2)",         _shard_probe),
        "trace":    ("DMA trace history",            _trace_probe),
        "buffer":   ("MicroFX signal buffer",        _buffer_probe),
        "domain":   ("Domain router state",           _domain_probe),
        "memory":   ("Dept memory shard health",      _dept_memory_probe),
        "process":  ("Process RSS/VMS",               _process_probe),
        "teach":    ("Teaching example coverage",     _teach_probe),
    }

    if focus:
        probes = {k: v for k, v in probes.items() if k == focus}

    all_findings: list[dict] = []
    probe_order: list[str] = []

    for key, (label, fn) in probes.items():
        findings = fn()
        for f in findings:
            f["probe"] = key
            f["probe_label"] = label
        all_findings.extend(findings)
        probe_order.append(key)

    # Tally by type
    tally: dict[str, int] = {}
    for f in all_findings:
        tally[f["type"]] = tally.get(f["type"], 0) + 1

    # High/medium severity findings (non-OK, non-info)
    action_items = [
        f for f in all_findings
        if f["type"] in ("BLOAT", "STALE", "WASTED",
                          "INEFFICIENT", "MISSING", "ERROR")
    ]

    return {
        "probes_run": probe_order,
        "n_findings": len(all_findings),
        "tally": tally,
        "action_items": len(action_items),
        "summary": _summarize(action_items),
        "findings": all_findings,
        "recommendations": _recommendations(action_items),
    }


def _summarize(action_items: list[dict]) -> str:
    if not action_items:
        return "All probes healthy — no action items."
    bloat = [f for f in action_items if f["type"] == "BLOAT"]
    inefficient = [f for f in action_items if f["type"] == "INEFFICIENT"]
    missing = [f for f in action_items if f["type"] == "MISSING"]
    stale = [f for f in action_items if f["type"] == "STALE"]
    parts = []
    if bloat:
        parts.append(f"{len(bloat)} bloat finding(s)")
    if inefficient:
        parts.append(f"{len(inefficient)} inefficient finding(s)")
    if missing:
        parts.append(f"{len(missing)} missing component(s)")
    if stale:
        parts.append(f"{len(stale)} stale shard(s)")
    return "; ".join(parts) + "."


def _recommendations(action_items: list[dict]) -> list[dict]:
    """Extract actionable recommendations, sorted by severity."""
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    recs = [
        {"subject": f["subject"], "type": f["type"],
         "severity": f["severity"], "detail": f["detail"],
         "fix": f.get("fix") or "manual inspection required"}
        for f in action_items
        if f.get("fix")
    ]
    recs.sort(key=lambda r: severity_order.get(r["severity"], 3))
    return recs
