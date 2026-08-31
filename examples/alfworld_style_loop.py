"""ALFWorld-style toy loop: a scripted long-horizon task against a stub LLM.

Runs fully offline. With a live duty model, swap StubLLM for DutyLLM:

    from neoharness.llm import DutyLLM
    h = Harness(DutyLLM(model="<duty-model-path>"))

and set NEOH_LATENT=1 to mirror BPE writes into the n-gram buffer (requires
the vLLM hidden-states hook on :8000 and the M1 ple-buffer-worker).
"""

from __future__ import annotations

import tempfile

from neoharness import BPEState, Harness


class StubLLM:
    """A deterministic 'policy' that uses the harness the way EvoHarness-RL
    teaches: track to disambiguate, commit at subgoal boundaries, recall for
    priors, note new insights."""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, **kw):
        return self.script.pop(0) if self.script else "<action>look</action>"


class KitchenEnv:
    """Minimal ALFWorld-flavored text world."""

    def __init__(self):
        self.mug_clean = False
        self.placed = False

    def reset(self, task):
        return ("You are in the kitchen. You see a countertop, a sink, "
                "and a cabinet. A dirty mug is on the countertop.")

    def step(self, verb, arg):
        if verb.startswith("take"):
            return "You pick up the dirty mug."
        if verb.startswith("clean"):
            self.mug_clean = True
            return "The mug is now clean."
        if verb.startswith("place"):
            if not self.mug_clean:
                return "Nothing happens. The mug is still dirty."
            self.placed = True
            return "You put the clean mug in the cabinet. Task complete."
        return "Nothing happens."

    def solved(self):
        return self.placed


def main() -> None:
    state = BPEState()
    state.experience.add("prior", "mug locations",
                         "mugs are usually on countertops or in cabinets")

    llm = StubLLM([
        "<action>track[world]</action>",
        "<action>recall[mug locations]</action>",
        "<action>take mug 1 from countertop 1</action>",
        "<action>commit[find mug | done]</action>",
        "<action>clean mug 1 in sink 1</action>",
        "<action>commit[clean mug | done]</action>",
        "<action>note[mistake | placing a dirty mug fails silently]</action>",
        "<action>place mug 1 in cabinet 1</action>",
    ])

    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness(llm, state=state, trajectory_dir=tmp)
        ep = harness.run_episode("clean the mug and put it in the cabinet",
                                 KitchenEnv())
        result = harness.consolidate(
            trajectory_summary="clean-before-place ordering matters")

    print("solved:", ep.solved)
    print("cost:", ep.cost)
    print("consolidation: added=%s updated=%s" % (result.added, result.updated))
    print("\nfinal harness state:\n" + state.render())


if __name__ == "__main__":
    main()
