import os
import re
import time

_PLAN_PATH = os.environ.get(
    "FABRIC_PLAN_DOC",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
        "harness-plan.md",
    ),
)
_SECTIONS = ("Device registry", "Department catalog", "Status summary")
_STAGES = (
    "promoted",
    "wired",
    "measured",
    "specced",
    "audited",
    "concept",
    "partial",
    "parked",
    "dead",
)


def _clean(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("`", "").replace("~~", "")
    return re.sub(r"\s+", " ", text).strip()


def _parse_table(lines):
    header = [_clean(c) for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for line in lines[1:]:
        cells = [_clean(c) for c in line.strip().strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        stage = next((s for s in _STAGES if any(s in c.lower() for c in cells)), "")
        rows.append({"cells": cells, "stage": stage})
    return {"columns": header, "rows": rows}


def parse_sections(path=None):
    with open(path or _PLAN_PATH) as f:
        md = f.read()
    out = {}
    for name in _SECTIONS:
        match = re.search(
            rf"^## {re.escape(name)}.*?\n(.*?)(?=^## |\Z)", md, re.M | re.S
        )
        if not match:
            continue
        body = match.group(1)
        table_lines, in_table = [], False
        for line in body.splitlines():
            if line.lstrip().startswith("|"):
                in_table = True
                table_lines.append(line)
            elif in_table and line.strip():
                break
        if len(table_lines) >= 2:
            out[name] = _parse_table(table_lines)
    stat = os.stat(path or _PLAN_PATH)
    return {
        "doc": _PLAN_PATH,
        "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
        "sections": out,
    }
