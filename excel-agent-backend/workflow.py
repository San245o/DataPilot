from __future__ import annotations

import random
import traceback
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from agent import generate_code, generate_fix, draft_final_reply
from sandbox import run_sandboxed

_MUTATION_KW = {'change', 'update', 'replace', 'rename', 'delete', 'add', 'edit', 'merge', 'set', 'modify', 'remove', 'fix', 'correct'}
_VIZ_KW = {'chart', 'plot', 'graph', 'pie', 'bar', 'scatter', 'visual', 'histogram', 'heatmap', 'line', 'bubble', 'trend'}


class AgentState(TypedDict, total=False):
    prompt: str
    rows: list[dict[str, Any]]
    model: str
    history: list[dict[str, str]]
    columns: list[str]
    dtypes: dict[str, str]
    nulls: dict[str, int]
    value_ranges: dict[str, dict]
    top_categories: dict[str, list]
    sample_rows: list[dict[str, Any]]
    action_type: str
    code: str
    assistant_reply: str
    result_rows: list[dict[str, Any]]
    visualization: dict[str, Any] | None
    query_output: str | None
    mutation: bool
    highlight_indices: list[int]
    token_usage: dict[str, int]
    table_request: dict[str, Any]
    query_table_rows: list[dict[str, Any]] | None
    query_tables: list[dict[str, Any]]
    error: str | None
    retry_count: int


def _truncate_cell(value: Any, max_len: int = 80) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    return value


def _is_mutation_only(prompt: str) -> bool:
    """Returns True if this is a pure mutation (no visualization)."""
    p = prompt.lower()
    has_mutation = any(k in p for k in _MUTATION_KW)
    has_viz = any(k in p for k in _VIZ_KW)
    return has_mutation and not has_viz


def _prepare_context(state: AgentState) -> AgentState:
    try:
        rows = state.get("rows") or []
        columns = list(rows[0].keys()) if rows else []
        history = state.get("history") or []
        prompt = state.get("prompt") or ""
        is_mutation_only = _is_mutation_only(prompt)

        # Compute dtypes from sample
        dtypes: dict[str, str] = {}
        sample_for_types = rows[:100]
        for col in columns:
            vals = [r.get(col) for r in sample_for_types if r.get(col) is not None and r.get(col) != ""]
            if not vals:
                dtypes[col] = "unknown"
            elif all(isinstance(v, bool) for v in vals):
                dtypes[col] = "bool"
            elif all(isinstance(v, int) for v in vals):
                dtypes[col] = "int"
            elif all(isinstance(v, (int, float)) for v in vals):
                dtypes[col] = "float"
            else:
                dtypes[col] = "str"

        # Compute null counts (only include non-zero)
        nulls: dict[str, int] = {}
        for col in columns:
            ct = sum(1 for r in rows if r.get(col) is None or r.get(col) == "")
            if ct > 0:
                nulls[col] = ct

        # Compute metadata for non-pure-mutations
        value_ranges: dict[str, dict] = {}
        top_categories: dict[str, list] = {}

        if not is_mutation_only:
            # Value ranges for numeric columns (limit to 12)
            numeric_cols = [c for c in columns if dtypes.get(c) in ("int", "float")][:12]
            for col in numeric_cols:
                nums = []
                for r in rows:
                    v = r.get(col)
                    if v is not None and v != "":
                        try:
                            nums.append(float(v))
                        except (ValueError, TypeError):
                            pass
                if nums:
                    value_ranges[col] = {
                        "min": round(min(nums), 2),
                        "max": round(max(nums), 2),
                        "mean": round(sum(nums) / len(nums), 2),
                    }

            # Top categories for string columns with <25 unique values (limit to 10 cols)
            str_cols = [c for c in columns if dtypes.get(c) == "str"][:10]
            for col in str_cols:
                vals = [r.get(col) for r in rows if r.get(col) is not None and r.get(col) != ""]
                uniq = set(vals)
                if 0 < len(uniq) < 25:
                    from collections import Counter
                    top_5 = [item for item, _ in Counter(vals).most_common(5)]
                    top_categories[col] = top_5

        # Sample rows: only on first turn (no history), 3 rows, truncated cells
        sample_rows: list[dict[str, Any]] = []
        if len(history) == 0 and rows:
            n = min(3, len(rows))
            sampled = random.sample(rows, n)
            sample_rows = [{k: _truncate_cell(v) for k, v in row.items()} for row in sampled]

        return {
            "columns": columns,
            "dtypes": dtypes,
            "nulls": nulls,
            "value_ranges": value_ranges,
            "top_categories": top_categories,
            "sample_rows": sample_rows,
            "retry_count": 0,
            "error": None,
        }
    except Exception as e:
        return {
            "columns": [],
            "dtypes": {},
            "nulls": {},
            "value_ranges": {},
            "top_categories": {},
            "sample_rows": [],
            "retry_count": 0,
            "error": f"Context preparation failed: {e}",
        }


def _generate_code(state: AgentState) -> AgentState:
    # If there's already an error from context preparation, skip code generation
    if state.get("error"):
        return {
            "code": "",
            "assistant_reply": state.get("error", "An error occurred"),
            "action_type": "query",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    try:
        result = generate_code(
            prompt=state.get("prompt") or "",
            model_name=state.get("model") or "gemini-2.0-flash",
            columns=state.get("columns") or [],
            dtypes=state.get("dtypes") or {},
            nulls=state.get("nulls") or {},
            value_ranges=state.get("value_ranges") or {},
            top_categories=state.get("top_categories") or {},
            sample_rows=state.get("sample_rows") or [],
            history=state.get("history") or [],
        )
        return {
            "code": result.get("code", ""),
            "assistant_reply": result.get("assistant_reply", "Done."),
            "action_type": result.get("action_type", "query"),
            "token_usage": result.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "table_request": result.get("table_request", {"enabled": False, "title": "Query Table"}),
            "error": None,
        }
    except Exception as e:
        return {
            "code": "",
            "assistant_reply": f"Failed to generate code: {e}",
            "action_type": "query",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "table_request": {"enabled": False, "title": "Query Table"},
            "error": str(e),
        }


def _execute_code(state: AgentState) -> AgentState:
    # If there's an error or no code, skip execution
    code = state.get("code") or ""
    if state.get("error") or not code.strip():
        rows = state.get("rows") or []
        return {
            "result_rows": rows,
            "visualization": None,
            "query_output": None,
            "query_table_rows": None,
            "query_tables": [],
            "mutation": False,
            "highlight_indices": [],
            "error": state.get("error") or "No code to execute",
        }

    try:
        rows = state.get("rows") or []
        result = run_sandboxed(code, rows)
        table_request = state.get("table_request") or {"enabled": False, "title": "Query Table"}
        table_enabled = bool(table_request.get("enabled", False))
        table_title = str(table_request.get("title") or "Query Table").strip() or "Query Table"

        query_tables: list[dict[str, Any]] = []
        if table_enabled and not result.mutation:
            table_rows = result.query_table_rows or []
            if not table_rows and result.rows:
                table_rows = result.rows[:50]
            if table_rows:
                query_tables = [{
                    "id": uuid4().hex,
                    "title": table_title,
                    "rows": table_rows,
                }]

        return {
            "result_rows": result.rows,
            "visualization": result.visualization,
            "query_output": result.query_output,
            "query_table_rows": result.query_table_rows,
            "query_tables": query_tables,
            "mutation": result.mutation,
            "highlight_indices": result.highlight_indices,
            "error": None,
        }
    except Exception as e:
        rows = state.get("rows") or []
        return {
            "result_rows": rows,
            "visualization": None,
            "query_output": None,
            "query_table_rows": None,
            "query_tables": [],
            "mutation": False,
            "highlight_indices": [],
            "error": str(e),
        }


def _fix_code(state: AgentState) -> AgentState:
    retry = (state.get("retry_count") or 0) + 1

    try:
        result = generate_fix(
            prompt=state.get("prompt") or "",
            model_name=state.get("model") or "gemini-2.0-flash",
            columns=state.get("columns") or [],
            dtypes=state.get("dtypes") or {},
            original_code=state.get("code") or "",
            error_message=state.get("error") or "Unknown error",
        )

        cur = state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        new = result.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        merged = {
            "prompt_tokens": (cur.get("prompt_tokens") or 0) + (new.get("prompt_tokens") or 0),
            "completion_tokens": (cur.get("completion_tokens") or 0) + (new.get("completion_tokens") or 0),
            "total_tokens": (cur.get("total_tokens") or 0) + (new.get("total_tokens") or 0),
        }

        return {
            "code": result.get("code", ""),
            "assistant_reply": result.get("assistant_reply", "Done."),
            "action_type": result.get("action_type") or state.get("action_type") or "query",
            "token_usage": merged,
            "table_request": result.get("table_request", {"enabled": False, "title": "Query Table"}),
            "retry_count": retry,
            "error": None,
        }
    except Exception as e:
        return {
            "retry_count": retry,
            "error": f"Failed to fix code: {e}",
        }


def _needs_final_reply(query_output: str | None, action_type: str | None) -> bool:
    """Determine if we need a second LLM call to interpret results."""
    if (action_type or "query") in ("mutation", "visualization"):
        return False
    if not query_output or len(query_output.strip()) < 10:
        return False
    return True


def _draft_final_reply(state: AgentState) -> AgentState:
    try:
        result = draft_final_reply(
            prompt=state.get("prompt") or "",
            model_name=state.get("model") or "gemini-2.0-flash",
            query_output=state.get("query_output"),
            initial_reply=state.get("assistant_reply") or "",
        )

        cur = state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        new = result.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        merged = {
            "prompt_tokens": (cur.get("prompt_tokens") or 0) + (new.get("prompt_tokens") or 0),
            "completion_tokens": (cur.get("completion_tokens") or 0) + (new.get("completion_tokens") or 0),
            "total_tokens": (cur.get("total_tokens") or 0) + (new.get("total_tokens") or 0),
        }

        return {
            "assistant_reply": result.get("reply") or state.get("assistant_reply") or "Done.",
            "token_usage": merged,
        }
    except Exception as e:
        # If drafting fails, keep the original reply
        return {
            "assistant_reply": state.get("assistant_reply") or "Done.",
        }


def _format_error_reply(state: AgentState) -> AgentState:
    error = state.get("error") or "Unknown error"
    # Clean up error message for display
    if "Execution error:" in error:
        error = error.replace("Execution error: ", "")
    return {
        "assistant_reply": f"I encountered an error: {error}\n\nPlease try rephrasing your request or check if the column names are correct.",
        "result_rows": state.get("rows") or [],
    }


def _should_draft_final(state: AgentState) -> AgentState:
    """Pass-through node for conditional routing."""
    return {}


def _route_after_execute(state: AgentState) -> Literal["fix_code", "format_error", "should_draft_final"]:
    error = state.get("error")
    retry_count = state.get("retry_count") or 0

    if error:
        if retry_count < 1:
            return "fix_code"
        else:
            return "format_error"
    return "should_draft_final"


def _route_after_should_draft(state: AgentState) -> Literal["draft_final_reply", "__end__"]:
    query_output = state.get("query_output")
    action_type = state.get("action_type")

    if _needs_final_reply(query_output, action_type):
        return "draft_final_reply"
    return "__end__"


def build_workflow():
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("prepare_context", _prepare_context)
    g.add_node("generate_code", _generate_code)
    g.add_node("execute_code", _execute_code)
    g.add_node("fix_code", _fix_code)
    g.add_node("format_error", _format_error_reply)
    g.add_node("should_draft_final", _should_draft_final)
    g.add_node("draft_final_reply", _draft_final_reply)

    # Set entry point
    g.set_entry_point("prepare_context")

    # Linear edges
    g.add_edge("prepare_context", "generate_code")
    g.add_edge("generate_code", "execute_code")

    # Conditional after execute
    g.add_conditional_edges(
        "execute_code",
        _route_after_execute,
        {
            "fix_code": "fix_code",
            "format_error": "format_error",
            "should_draft_final": "should_draft_final",
        },
    )

    # Fix code loops back to execute
    g.add_edge("fix_code", "execute_code")

    # Error formatting ends
    g.add_edge("format_error", END)

    # Conditional after should_draft_final
    g.add_conditional_edges(
        "should_draft_final",
        _route_after_should_draft,
        {
            "draft_final_reply": "draft_final_reply",
            "__end__": END,
        },
    )

    # Final reply ends
    g.add_edge("draft_final_reply", END)

    return g.compile()
