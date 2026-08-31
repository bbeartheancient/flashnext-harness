"""Agent-harness action protocol (EvoHarness-RL §2.2).

The duty model emits exactly one action per turn:

    <action>verb[arg]</action>

Harness verbs A_bpe = {track, commit, recall, note}:

- ``track[object]`` / ``track[world]``  -> read Belief      (X axis)
- ``commit[subgoal | status]``          -> write Progress   (Y axis)
       status is optional (default ``done``); e.g. ``commit[find mug | done]``
- ``recall[query]``                     -> read Experience  (Z axis)
- ``note[kind | insight]``              -> write Experience note buffer
       kind is optional (default ``general``); e.g. ``note[prior | knives in drawers]``

Anything else inside the tag is an environment action and is passed through
to the caller's environment. Malformed actions return kind="invalid".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bpe import BPEState

HARNESS_VERBS = ("track", "commit", "recall", "note")

# BPE component -> n-gram buffer coordinate axis (user directive:
# "BPE vectors can be used as the XYZ vectors in the n-gram buffer").
AXIS_FOR_VERB = {"track": "x", "commit": "y", "recall": "z", "note": "z"}

_ACTION_RE = re.compile(r"<action>\s*([A-Za-z_][\w -]*?)\s*(?:\[(.*?)\])?\s*</action>", re.S)


@dataclass
class ActionResult:
    kind: str  # "harness" | "env" | "invalid"
    verb: str
    arg: str = ""
    observation: str = ""
    axis: str | None = None  # XYZ axis touched, harness writes only
    detail: dict = field(default_factory=dict)


def parse_action(text: str) -> tuple[str, str] | None:
    """Extract (verb, arg) from the LAST <action> tag in the model output.

    The last tag wins: the duty model reasons at length and may quote earlier
    candidate actions in its thinking.
    """
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None
    verb, arg = matches[-1]
    return verb.strip().lower(), (arg or "").strip()


def _split_arg(arg: str) -> tuple[str, str | None]:
    """Split 'main | extra' once; extra may be None."""
    if "|" in arg:
        main, extra = arg.split("|", 1)
        return main.strip(), extra.strip()
    return arg.strip(), None


def dispatch(
    text: str,
    state: BPEState,
    on_write=None,
) -> ActionResult:
    """Dispatch one model action against the BPE state.

    ``on_write(axis, verb, rendered_text, detail)`` is called after every
    harness WRITE (commit/note) and after belief-affecting reads, so the XYZ
    vector lane can mirror the state change into the n-gram buffer. It is
    fire-and-forget: exceptions are caught and recorded in the result detail.
    """
    parsed = parse_action(text)
    if parsed is None:
        return ActionResult(
            kind="invalid", verb="", observation="no <action> tag found"
        )
    verb, arg = parsed
    if verb not in HARNESS_VERBS:
        return ActionResult(kind="env", verb=verb, arg=arg)

    axis = AXIS_FOR_VERB[verb]
    try:
        if verb == "track":
            obs = state.belief.track(arg or "world")
            return ActionResult("harness", verb, arg, obs, axis, {"write": False})

        if verb == "commit":
            subgoal, status = _split_arg(arg)
            entry = state.progress.commit(subgoal, status or "done")
            result = ActionResult(
                "harness", verb, arg,
                f"committed: [{entry['status']}] {entry['subgoal']}",
                axis, {"write": True, "entry": dict(entry)},
            )

        elif verb == "recall":
            hits = state.experience.recall(arg)
            if hits:
                obs = "\n".join(
                    f"({h['kind']}) {h['title']}: {h['text']}" for h in hits
                )
            else:
                obs = "(no matching experience)"
            return ActionResult(
                "harness", verb, arg, obs, axis,
                {"write": False, "hits": [h["skill_id"] for h in hits]},
            )

        elif verb == "note":
            first, second = _split_arg(arg)
            # note[kind | insight]; a bare note[insight] defaults to general.
            kind, insight = (first, second) if second is not None else ("general", first)
            note = state.experience.note(insight, kind or "general")
            result = ActionResult(
                "harness", verb, arg, f"noted: {note['text']}", axis,
                {"write": True, "note": dict(note)},
            )

    except (ValueError, KeyError) as exc:
        return ActionResult("invalid", verb, arg, f"{verb} failed: {exc}", axis)

    # write path (commit / note): mirror to the XYZ vector lane
    if on_write is not None:
        rendered = _render_axis(state, result.axis)
        try:
            on_write(result.axis, verb, rendered, result.detail)
        except Exception as exc:  # fire-and-forget, never block the harness
            result.detail["vector_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _render_axis(state: BPEState, axis: str) -> str:
    """The text that represents one XYZ coordinate of the buffer."""
    if axis == "x":
        return state.belief.render()
    if axis == "y":
        return state.progress.render()
    return state.experience.render()
