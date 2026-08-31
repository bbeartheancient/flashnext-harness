"""Skills: procedural memory, Hermes-style (agentskills.io SKILL.md shape).

A skill is a markdown file under NEOH_STATE_DIR/skills/:

    ---
    name: clean-and-place
    description: how to clean an object and place it
    ---
    1. locate the object ...

Every skill is mirrored into the Experience store as a ``kind="task"`` entry
(skill_id ``skill:<name>``), so the existing ``recall[query]`` harness action
retrieves skills and bumps their usage counts — one memory system, not two.

``create_from_trajectory`` closes the Hermes learning loop: after a
successful episode it distills the env-action verb sequence into a candidate
skill (deterministic v1; an LLM distiller can replace it behind the same
interface later).
"""

from __future__ import annotations

import re
from pathlib import Path

from .bpe import BPEState, state_dir

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def skills_dir() -> Path:
    d = state_dir() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


class SkillStore:
    def __init__(self, state: BPEState, directory: str | Path | None = None):
        self.state = state
        self.dir = Path(directory) if directory else skills_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- SKILL.md IO --
    def save(self, name: str, description: str, body: str) -> Path:
        if not _NAME_RE.match(name):
            raise ValueError(f"skill name must match {_NAME_RE.pattern}, got {name!r}")
        path = self.dir / f"{name}.md"
        path.write_text(
            f"---\nname: {name}\ndescription: {description.strip()}\n---\n"
            + body.strip() + "\n"
        )
        # Mirror into the Experience store (Z axis) so recall finds it.
        self.state.experience.add(
            "task", name, f"{description.strip()}\n{body.strip()}",
            skill_id=f"skill:{name}",
        )
        return path

    def load(self, name: str) -> dict | None:
        path = self.dir / f"{name}.md"
        if not path.exists():
            return None
        return self._parse(path.read_text())

    def _parse(self, text: str) -> dict:
        m = re.match(
            r"---\s*\nname:\s*(.+)\ndescription:\s*(.+?)\n---\s*\n(.*)",
            text, re.S,
        )
        if not m:
            raise ValueError("not a SKILL.md file (missing frontmatter)")
        return {
            "name": m.group(1).strip(),
            "description": m.group(2).strip(),
            "body": m.group(3).strip(),
        }

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self.dir.glob("*.md")):
            try:
                out.append(self._parse(path.read_text()))
            except ValueError:
                continue
        return out

    def load_all_into_experience(self) -> int:
        """Re-mirror every on-disk skill into the Experience store."""
        n = 0
        for skill in self.list():
            self.state.experience.add(
                "task", skill["name"],
                f"{skill['description']}\n{skill['body']}",
                skill_id=f"skill:{skill['name']}",
            )
            n += 1
        return n

    # -- learning loop --
    def create_from_trajectory(self, ep, name: str | None = None) -> Path | None:
        """Distill a successful episode into a candidate skill.

        Only runs on solved episodes. The body is the ordered, deduped
        env-action sequence that led to success, plus any insights the policy
        noted during the episode.
        """
        if not ep.solved or not ep.steps:
            return None
        name = name or _slugify(ep.task)
        verbs = []
        for step in ep.steps:
            if step.kind in ("env", "tool"):
                call = step.verb + (f" {step.arg}" if step.arg else "")
                if not verbs or verbs[-1] != call:
                    verbs.append(call)
        notes = [
            s.detail["note"]["text"] for s in ep.steps
            if s.kind == "harness" and s.verb == "note" and "note" in s.detail
        ]
        body = f"Task: {ep.task}\n\nSteps that worked:\n"
        body += "\n".join(f"{i}. {v}" for i, v in enumerate(verbs, 1))
        if notes:
            body += "\n\nLessons:\n" + "\n".join(f"- {n}" for n in notes)
        return self.save(name, f"how to: {ep.task}", body)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:48] or "unnamed-goal")
