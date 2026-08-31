"""Tests for goals.py — success writes a skill + consolidates; failure blocks."""

from __future__ import annotations

from neoharness.agent import Agent
from neoharness.bpe import BPEState
from neoharness.goals import GoalRunner
from neoharness.harness import Harness
from neoharness.skills import SkillStore
from neoharness.tools import make_core_tools


class StubLLM:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, **kw):
        return self.script.pop(0) if self.script else "<done>fallback</done>"


def make_runner(tmp_path, script):
    llm = StubLLM(script)
    h = Harness(llm, state=BPEState(), trajectory_dir=tmp_path / "traj")
    agent = Agent(llm, harness=h, tools=make_core_tools(tmp_path), max_steps=6)
    skills = SkillStore(h.state, directory=tmp_path / "skills")
    return GoalRunner(agent, skills)


def test_success_path_creates_skill_and_consolidates(tmp_path):
    runner = make_runner(tmp_path, [
        '<tool>write_file {"path": "report.txt", "content": "done"}</tool>',
        "<action>note[general | small files write fast]</action>",
        "<done>report written</done>",
    ])
    result = runner.run_goal("write the report",
                             subgoals=["draft report", "save report"])
    assert result.solved
    # subgoals committed done
    statuses = {e["subgoal"]: e["status"] for e in runner.state.progress.entries}
    assert statuses == {"draft report": "done", "save report": "done"}
    # skill distilled and mirrored into experience
    assert result.skill_path is not None
    assert runner.skills.load("write-the-report") is not None
    assert "skill:write-the-report" in runner.state.experience.skills
    # consolidation consumed the note into the bank
    assert result.consolidation.notes_consumed == 1
    assert not runner.state.experience.note_buffer


def test_failure_path_blocks_goal_and_notes_mistake(tmp_path):
    runner = make_runner(tmp_path, [
        '<tool>list_dir {}</tool>',
    ])  # script exhausts to "<done>fallback</done>"... force failure:
    runner.agent.llm.script = ['<tool>list_dir {}</tool>'] * 6  # never <done>
    result = runner.run_goal("impossible goal")
    assert not result.solved
    statuses = {e["subgoal"]: e["status"] for e in runner.state.progress.entries}
    assert statuses["impossible goal"] == "blocked"
    assert result.skill_path is None
    # the mistake note was consolidated into the bank
    kinds = [s["kind"] for s in runner.state.experience.skills.values()]
    assert "mistake" in kinds
