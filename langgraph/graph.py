"""Compile the multimodal risk LangGraph.

Loaded via importlib as `risk_workflow`. The installed `langgraph` package is
imported after temporarily removing the project root from sys.path so this
folder does not shadow it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent


def _load_langgraph_library():
    removed: list[str] = []
    for entry in list(sys.path):
        resolved = Path(entry).resolve() if entry not in {"", "."} else Path.cwd()
        if resolved == _ROOT.resolve() or entry in {"", "."}:
            sys.path.remove(entry)
            removed.append(entry)
    sys.modules.pop("langgraph", None)
    sys.modules.pop("langgraph.graph", None)
    from langgraph.graph import END, START, StateGraph  # type: ignore  # installed package

    for entry in reversed(removed):
        sys.path.insert(0, entry)
    return StateGraph, START, END


StateGraph, START, END = _load_langgraph_library()


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_DIR / filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state_mod = _load("risk_state", "state.py")
nodes = _load("risk_nodes", "nodes.py")
RiskState = state_mod.RiskState


def build_graph():
    graph = StateGraph(RiskState)
    graph.add_node("load_data", nodes.load_data)
    graph.add_node("structured_analysis", nodes.structured_analysis)
    graph.add_node("ocr_analysis", nodes.ocr_analysis)
    graph.add_node("vision_analysis", nodes.vision_analysis)
    graph.add_node("text_analysis", nodes.text_analysis)
    graph.add_node("llm_analysis", nodes.llm_analysis)
    graph.add_node("fusion", nodes.fusion)
    graph.add_node("risk_prediction", nodes.risk_prediction)
    graph.add_node("generate_report", nodes.generate_report)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "ocr_analysis")
    graph.add_edge("ocr_analysis", "structured_analysis")
    graph.add_edge("ocr_analysis", "vision_analysis")
    graph.add_edge("ocr_analysis", "text_analysis")
    graph.add_edge("structured_analysis", "llm_analysis")
    graph.add_edge("vision_analysis", "llm_analysis")
    graph.add_edge("text_analysis", "llm_analysis")
    graph.add_edge("llm_analysis", "fusion")
    graph.add_edge("fusion", "risk_prediction")
    graph.add_edge("risk_prediction", "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()


def run_workflow(initial_state: dict) -> dict:
    app = build_graph()
    return app.invoke(initial_state)
