"""Goal autonomy: long-horizon goals on top of the agent loop.

GoalRunner ties the pieces together:

1. Seed Progress (Y axis) with the goal's subgoals.
2. Run the Agent until <done> or the step budget.
3. On success: distill a skill from the trajectory (Hermes learning loop)
   and note the insights.
4. On budget exhaustion: commit the goal blocked and note a mistake.
5. consolidate() at goal end — the EvoHarness epoch boundary, which is also
   the only persist=True write to the n-gram shard.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent
from .harness import EpisodeResult
from .skills import SkillStore


@dataclass
class GoalResult:
    goal: str
    episode: EpisodeResult
    skill_path: str | None
    consolidation: object  # ConsolidationResult

    @property
    def solved(self) -> bool:
        return self.episode.solved


class GoalRunner:
    def __init__(self, agent: Agent, skills: SkillStore | None = None):
        self.agent = agent
        self.state = agent.harness.state
        self.skills = skills or SkillStore(self.state)

    def run_goal(self, goal: str, subgoals: list[str] | None = None) -> GoalResult:
        for sg in subgoals or [goal]:
            self.state.progress.commit(sg, "pending")

        ep = self.agent.run(goal)

        skill_path = None
        if ep.solved:
            for sg in subgoals or [goal]:
                self.state.progress.commit(sg, "done")
            path = self.skills.create_from_trajectory(ep)
            skill_path = str(path) if path else None
        else:
            self.state.progress.commit(goal, "blocked")
            self.state.experience.note(
                f"goal not completed within budget: {goal}", kind="mistake"
            )

        summary = f"{'completed' if ep.solved else 'failed'} goal: {goal}"
        consolidation = self.agent.harness.consolidate(trajectory_summary=summary)
        return GoalResult(goal, ep, skill_path, consolidation)
