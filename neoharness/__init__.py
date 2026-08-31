"""NeoHarness — BPE harness for the duty LLM on the bicameral n-gram buffer.

Implements the EvoHarness-RL harness-state model (Belief / Progress /
Experience, arXiv:2608.05446) with the M1Multitronic n-gram buffer as the
latent substrate: the three BPE state components are the XYZ coordinate axes
of the n-gram buffer (Belief→X, Progress→Y, Experience→Z).
"""

from .agent import Agent
from .bpe import BPEState, BeliefStore, ExperienceStore, ProgressStore
from .goals import GoalRunner, GoalResult
from .harness import Harness
from .skills import SkillStore
from .tools import ToolRegistry, make_core_tools

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "BPEState",
    "BeliefStore",
    "ProgressStore",
    "ExperienceStore",
    "GoalResult",
    "GoalRunner",
    "Harness",
    "SkillStore",
    "ToolRegistry",
    "make_core_tools",
    "__version__",
]
