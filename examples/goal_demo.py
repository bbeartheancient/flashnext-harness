"""Offline goal demo: a multi-step workspace task solved autonomously.

Shows the full v0.2 stack: tool loop, BPE interleave, subagent delegation,
skill creation from the successful trajectory, and epoch consolidation.
No services required — a scripted policy stands in for the duty LLM.
With a live model, swap StubLLM for DutyLLM and set NEOH_LATENT=1.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from neoharness import BPEState, Harness
from neoharness.agent import Agent
from neoharness.goals import GoalRunner
from neoharness.skills import SkillStore
from neoharness.tools import make_core_tools


class StubLLM:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, **kw):
        return self.script.pop(0) if self.script else "<done>finished</done>"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "inventory.txt").write_text("mug:dirty\ntowel:clean\nplate:dirty\n")

        script = [
            # check priors, then inspect the workspace
            "<action>recall[cleaning workflow]</action>\n<tool>list_dir {}</tool>",
            '<tool>read_file {"path": "inventory.txt"}</tool>',
            # delegate counting the dirty items to a subagent
            '<tool>delegate {"goal": "count dirty items in inventory.txt",'
            ' "context": "use grep with pattern :dirty"}</tool>',
            '<tool>grep {"pattern": ":dirty", "path": "inventory.txt"}</tool>',
            "<done>subagent counted</done>",  # subagent turn
            # record progress and an insight, then write the report
            "<action>commit[inventory reviewed | done]</action>",
            "<action>note[prior | mugs and plates tend to be dirty]</action>",
            '<tool>write_file {"path": "cleaning-plan.txt",'
            ' "content": "clean mug, clean plate, done"}</tool>',
            "<done>cleaning plan written</done>",
        ]

        llm = StubLLM(script)
        harness = Harness(llm, state=BPEState(), trajectory_dir=ws / "traj")
        harness.state.experience.add(
            "prior", "cleaning workflow", "inspect inventory before cleaning")
        agent = Agent(llm, harness=harness, tools=make_core_tools(ws))
        runner = GoalRunner(agent, SkillStore(harness.state, directory=ws / "skills"))

        result = runner.run_goal(
            "write a cleaning plan for the dirty inventory",
            subgoals=["review inventory", "identify dirty items", "write plan"])

        print("solved:", result.solved)
        print("cost:", result.episode.cost)
        print("skill written:", result.skill_path)
        print("plan file:", (ws / "cleaning-plan.txt").read_text())
        print("\nfinal harness state:\n" + harness.state.render())


if __name__ == "__main__":
    main()
