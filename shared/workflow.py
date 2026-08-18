"""Load the LangGraph workflow without shadowing the installed langgraph package."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

from shared.config import PROJECT_ROOT


@lru_cache(maxsize=1)
def load_workflow_module():
    path = PROJECT_ROOT / "langgraph" / "graph.py"
    spec = importlib.util.spec_from_file_location("risk_workflow", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_risk_workflow(state: dict) -> dict:
    module = load_workflow_module()
    return module.run_workflow(state)
