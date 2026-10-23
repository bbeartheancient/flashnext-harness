"""Context handed between threads - the budgeted selection contract.

Transcribed from Backplane's ``handoff.bend`` / LAWS.bend, which this project
treats as its contract:

    "It is a budgeted selection of the source thread's own messages and tool
     lines, never an agent-written summary: short conversations go whole;
     longer ones keep their newest items that fit the budget, plus the first
     request, and say how many earlier items were left out and where to read
     them."                                              (handoff.bend)

So a handoff is VERBATIM + POINTERED. The dropped items are counted and named,
never paraphrased: an agent-written summary is exactly what this contract
forbids, because it loses facts with no trace. Laws honoured here:
``hand_no_holes``, ``hand_newest_fit``, ``hand_cut_says``,
``handoff_goes_first``, ``handoff_never_runs_alone``, ``handoff_with_message_runs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# the queue item prefix that marks handed-over context rather than a message
PREFIX = "handoff:"

# the most characters of history one handoff carries (Hand.budget)
BUDGET = 16000

# the most characters of one item (Hand.cap); a longer item is cut, and says so
CAP = 4000


def is_handoff(item_id: str) -> bool:
    return str(item_id).startswith(PREFIX)


@dataclass
class Sel:
    """The outcome of a budgeted selection (HD.Sel)."""
    left: int                       # budget remaining after the kept items
    kept: list[str] = field(default_factory=list)   # verbatim, in prompt order
    truncated: bool = False         # whether the budget forced a drop
    omitted: int = 0                # how many older items were left out


def cut(text: str, n: int) -> str:
    """A long item is cut, and says by how much (law hand_cut_says)."""
    if len(text) <= n:
        return text
    return text[:n] + f" ...[{len(text) - n} more characters]"


def cut_item(text: str) -> str:
    return cut(text, CAP)


# Backplane's Sel.left counts the separator too: with budget 17 over 7-character
# items it reports left=1, not 3, so a kept item costs len + 1 (law hand_newest_fit).
def _cost(item: str) -> int:
    return len(cut_item(item)) + 1


def select(items: list[str], budget: int = BUDGET) -> Sel:
    """Keep the NEWEST items that fit the budget; the kept set is a suffix.

    This is what ``hand_no_holes`` demands: once one entry is left out, every
    older one is too. A selection that kept items 1 and 3 but dropped 2 would
    silently splice unrelated history together, so it is never produced.
    """
    left = budget
    keep_from = len(items)
    for i in range(len(items) - 1, -1, -1):
        c = _cost(items[i])
        if c > left:
            break
        left -= c
        keep_from = i
    kept = [cut_item(x) for x in items[keep_from:]]
    omitted = keep_from
    return Sel(left=left, kept=kept, truncated=omitted > 0, omitted=omitted)


def queue_texts(items: list[tuple[str, str]]) -> str:
    """Render a queue in order (law handoff_goes_first: the handoff leads)."""
    return "\n\n".join(text for _id, text in items)


def queue_real(items: list[tuple[str, str]]) -> bool:
    """Does this queue start a turn? Context alone does not; context with a
    real message does (laws handoff_never_runs_alone, handoff_with_message_runs)."""
    return any(not is_handoff(item_id) for item_id, _text in items)


def render(items: list[tuple[str, str]], budget: int = BUDGET) -> str:
    """The handoff as it reaches a target thread: ordered, budgeted, counted."""
    body = queue_texts(items)
    return body if queue_real(items) else ""   # alone -> it never runs
