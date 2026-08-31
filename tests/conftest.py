"""Shared fixtures: keep tests out of the real NEOH_STATE_DIR."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOH_STATE_DIR", str(tmp_path / "neoh-state"))
