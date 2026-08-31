"""CAS hook: SageMath from ~/hoa64-venv if present, else SymPy.

Operator path is `$HOME/hoa64-venv` (not ~/hoa64 and not the
documented sage-dev conda env). That venv is currently CPython 3.14 +
numpy — `import sage.all` misses. Until sage is installed there (or
SAGE_BIN is set), evaluate with sympy in .venv-fabric.
"""

from __future__ import annotations

import os
import shutil
import subprocess

_SAGE_CANDIDATES = [
    os.environ.get("SAGE_BIN"),
    os.environ.get("SAGE_PYTHON"),
    os.path.expanduser("~/hoa64-venv/bin/sage"),
    os.path.expanduser("~/hoa64-venv/bin/python"),
    os.path.expanduser("~/miniforge3/envs/sage-dev/bin/sage"),
    os.path.expanduser("~/miniforge3/envs/sage-dev/bin/python"),
    shutil.which("sage"),
]

_avail_cache: dict | None = None


def _is_sage(bin_: str) -> bool:
    name = os.path.basename(bin_)
    if name == "sage":
        return True
    # Probe python interpreters that look like the operator's Sage
    # homes (hoa64-venv, sage-dev) or SAGE_PYTHON. Skip unrelated
    # CPythons so we don't pay 2s per miss.
    marked = ("hoa64-venv", "sage-dev", "sage")
    if not any(m in bin_.lower() for m in marked) and not os.environ.get("SAGE_PYTHON"):
        return False
    try:
        proc = subprocess.run(
            [bin_, "-c", "import sage.all"],
            capture_output=True, timeout=2)
        return proc.returncode == 0
    except Exception:
        return False


def sage_available() -> dict:
    global _avail_cache
    if _avail_cache is not None:
        return dict(_avail_cache)
    sage = None
    backend = None
    sage_ok = False
    probed = None
    for p in _SAGE_CANDIDATES:
        if not (p and os.path.isfile(p) and os.access(p, os.X_OK)):
            continue
        probed = p
        if _is_sage(p):
            sage, sage_ok, backend = p, True, "sage"
            break
    if sage is None:
        sage = probed
    sympy_ok = False
    try:
        import sympy  # noqa: F401

        sympy_ok = True
        if backend is None:
            backend = "sympy"
    except ImportError:
        pass
    _avail_cache = {
        "backend": backend,
        "sage_bin": sage if sage_ok else None,
        "sympy": sympy_ok,
        "probed": sage,
        "hint": None if sage_ok else (
            "hoa64-venv has no sage.all (CPython 3.14 + numpy). "
            "Install Sage into that venv or set SAGE_BIN; sympy fallback."),
    }
    return dict(_avail_cache)


def sage_eval(expr: str, timeout_s: float = 8.0) -> dict:
    """Evaluate a CAS expression. Sage preferred; sympy fallback."""
    expr = (expr or "").strip()
    if not expr or len(expr) > 2000:
        return {"error": "empty or too-long expression"}
    banned = ("__", "import ", "open(", "os.", "subprocess", "eval(", "exec(")
    low = expr.lower()
    if any(b in low for b in banned):
        return {"error": "expression rejected"}
    info = sage_available()
    if info["sage_bin"]:
        bin_ = info["sage_bin"]
        if os.path.basename(bin_) == "sage":
            cmd = [bin_, "-c", f"print({expr})"]
        else:
            cmd = [bin_, "-c",
                   "from sage.all import *\nprint(" + expr + ")"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=float(timeout_s),
            )
            out = (proc.stdout or proc.stderr or "").strip()
            return {"backend": "sage", "expr": expr, "result": out,
                    "ok": proc.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"error": "sage timed out"}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
    if info["sympy"]:
        try:
            import sympy as S

            loc = {name: getattr(S, name) for name in (
                "factor", "expand", "simplify", "collect", "together", "apart",
                "diff", "integrate", "limit", "series", "solve", "sqrt",
                "sin", "cos", "tan", "exp", "log", "pi", "I", "E", "N",
                "Matrix", "Integer", "Rational", "symbols", "Eq", "gcd",
                "lcm", "factorial", "binomial")}
            loc.update(dict(zip("xyztn", S.symbols("x y z t n"))))
            val = S.sympify(expr, locals=loc)
            return {
                "backend": "sympy",
                "expr": expr,
                "result": str(val),
                "latex": S.latex(val),
                "ok": True,
            }
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "backend": "sympy"}
    return {"error": "no CAS: install sympy in .venv-fabric or sage on PATH",
            "hint": "sudo pacman -S sagemath"}
