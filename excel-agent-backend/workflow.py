from __future__ import annotations

import random
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent import generate_code
from sandbox import run_sandboxed


class AgentState(TypedDict, total=False):
    prompt: str
    rows: list[dict[str, Any]]
    model: str
    history: list[dict[str, str]]
    columns: list[str]
    sample_rows: list[dict[str, Any]]
    code: str
    assistant_reply: str
    result_rows: list[dict[str, Any]]
    visualization: dict[str, Any] | None
    query_output: str | None
    mutation: bool
    highlight_indices: list[int]


def _prepare_context(state: AgentState) -> AgentState:
    rows = state.get("rows", [])
    columns = list(rows[0].keys()) if rows else []

    sample_size = min(5, len(rows))
    sample_rows = random.sample(rows, sample_size) if sample_size > 0 else []

    return {
        "columns": columns,
        "sample_rows": sample_rows,
    }


def _generate_code(state: AgentState) -> AgentState:
    generated = generate_code(
        prompt=state.get("prompt", ""),
        model_name=state.get("model", "gemini-3-flash-preview"),
        columns=state.get("columns", []),
        sample_rows=state.get("sample_rows", []),
        history=state.get("history", []),
    )
    return {
        "code": generated["code"],
        "assistant_reply": generated["assistant_reply"],
    }


def _execute_code(state: AgentState) -> AgentState:
    result = run_sandboxed(state.get("code", ""), state.get("rows", []))
    return {
        "result_rows": result.rows,
        "visualization": result.visualization,
        "query_output": result.query_output,
        "mutation": result.mutation,
        "highlight_indices": result.highlight_indices,
    }


def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("prepare_context", _prepare_context)
    graph.add_node("generate_code", _generate_code)
    graph.add_node("execute_code", _execute_code)

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", END)

    return graph.compile()
