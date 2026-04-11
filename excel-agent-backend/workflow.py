from __future__ import annotations

import random
import re
import traceback
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from agent import generate_code, generate_fix
from sandbox import run_sandboxed

_MUTATION_KW = {
    'change', 'update', 'replace', 'rename', 'delete', 'add', 'edit', 'merge', 'set', 'modify', 'remove', 'fix', 'correct',
    'append', 'insert', 'sort', 'reorder', 'order by', 'arrange',
    'add row', 'add rows', 'append row', 'append rows', 'insert row', 'insert rows',
}
_VIZ_KW = {'chart', 'plot', 'graph', 'pie', 'bar', 'scatter', 'visual', 'histogram', 'heatmap', 'line', 'bubble', 'trend'}
_ROW_DELETE_KW = {
    'delete row',
    'delete rows',
    'remove row',
    'remove rows',
    'drop row',
    'drop rows',
    'drop duplicate',
    'drop duplicates',
    'remove duplicate',
    'remove duplicates',
    'deduplicate',
    'dedup',
    'delete entries',
    'remove entries',
    'drop entries',
    'delete items',
    'remove items',
    'drop items',
}


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


def _has_mutation_intent(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in _MUTATION_KW)


def _allows_row_deletion(prompt: str) -> bool:
    p = prompt.lower()
    if any(k in p for k in _ROW_DELETE_KW):
        return True
    return bool(
        re.search(
            r"\b(delete|remove|drop)\b[^\n]*\b(row|rows|record|records|duplicate|duplicates|movie|movies|item|items|entry|entries)\b",
            p,
        )
    )


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
            model_name=state.get("model") or "gemini-3.1-flash-lite-preview",
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
        prompt = state.get("prompt") or ""
        action_type = (state.get("action_type") or "query").lower()
        has_mutation_intent = _has_mutation_intent(prompt)
        row_growth = len(result.rows) > len(rows)
        is_mutation_action = action_type in ("mutation", "combined") or has_mutation_intent or row_growth
        row_deletion_blocked = len(result.rows) < len(rows) and not _allows_row_deletion(prompt)

        mutation_applied = bool(result.mutation and is_mutation_action and not row_deletion_blocked)
        final_rows = result.rows if mutation_applied else rows

        query_output = result.query_output
        if row_deletion_blocked:
            guard_note = (
                "Safety guard: row deletions were ignored because the prompt did not explicitly ask to "
                "delete, drop, or remove rows."
            )
            query_output = f"{query_output}\n\n{guard_note}" if query_output else guard_note

        query_tables: list[dict[str, Any]] = []
        if table_enabled and not mutation_applied:
            table_rows = result.query_table_rows or []
            if not table_rows and final_rows:
                table_rows = final_rows[:50]
            if table_rows:
                query_tables = [{
                    "id": uuid4().hex,
                    "title": table_title,
                    "rows": table_rows,
                }]

        return {
            "result_rows": final_rows,
            "visualization": result.visualization,
            "query_output": query_output,
            "query_table_rows": result.query_table_rows,
            "query_tables": query_tables,
            "mutation": mutation_applied,
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
            model_name=state.get("model") or "gemini-3.1-flash-lite-preview",
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
        # Keep the original execution error so users see the real failure,
        # not a secondary "fix generation" failure.
        return {
            "retry_count": retry,
            "error": state.get("error") or str(e),
        }


def _extract_safety_notes(query_output: str) -> str:
    lines = [line.strip() for line in query_output.splitlines() if line.strip()]
    notes = [line for line in lines if line.lower().startswith("safety guard:")]
    return "\n".join(notes)


def _compact_query_output(query_output: str, max_lines: int = 8, max_chars: int = 700) -> str:
    lines = [line.strip() for line in query_output.splitlines() if line.strip()]
    if not lines:
        return ""

    dataframe_line = next((line for line in lines if line.startswith("[DataFrame:")), "")
    compact_lines: list[str] = []

    if dataframe_line:
        compact_lines.append(dataframe_line)

    for line in lines:
        if line == dataframe_line:
            continue
        if line.lower().startswith("showing first"):
            continue
        if re.match(r"^\d+\s", line):
            continue
        compact_lines.append(line)
        if len(compact_lines) >= max_lines:
            break

    if not compact_lines:
        compact_lines = lines[:max_lines]

    text = "\n".join(compact_lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n... [truncated]"
    return text


def _derive_reply_from_query_output(query_output: str) -> str:
    lines = [line.strip() for line in query_output.splitlines() if line.strip()]
    if not lines:
        return ""

    meaningful_lines = [
        line for line in lines
        if not line.lower().startswith("showing first")
        and not line.startswith("[DataFrame:")
        and not line.startswith("[Series:")
        and not re.match(r"^\d+\s", line)
    ]
    if not meaningful_lines:
        return ""

    if len(meaningful_lines) == 1:
        return meaningful_lines[0]

    scalar_lines = [
        line for line in meaningful_lines
    ]
    if scalar_lines:
        return "\n".join(scalar_lines[:3])

    return "\n".join(meaningful_lines[:3])


def _build_table_acknowledgement(query_tables: list[dict[str, Any]]) -> str:
    if not query_tables:
        return ""

    first_table = query_tables[0]
    title = str(first_table.get("title") or "Extracted Table").strip() or "Extracted Table"
    row_count = len(first_table.get("rows") or [])

    if len(query_tables) == 1:
        if row_count > 0:
            return f"Extracted {row_count} rows into {title}."
        return f"Created {title}."

    return f"Created {len(query_tables)} extracted tables."


def _compose_assistant_reply(state: AgentState) -> AgentState:
    """Compose final assistant reply without calling the model again.

    This keeps the workflow single-hop for normal query execution:
    query -> sandbox -> final response.
    """
    reply = (state.get("assistant_reply") or "Done.").strip()
    query_output = (state.get("query_output") or "").strip()
    action_type = (state.get("action_type") or "query").lower()
    query_tables = state.get("query_tables") or []

    if not query_output:
        table_ack = _build_table_acknowledgement(query_tables)
        if table_ack:
            return {"assistant_reply": table_ack}
        return {"assistant_reply": reply}

    safety_notes = _extract_safety_notes(query_output)
    reply_lower = reply.lower()
    is_generic_reply = reply_lower in {"done", "done.", "completed", "completed."} or len(reply) < 16

    # For chart-focused requests, avoid dumping table previews into chat.
    if action_type in ("visualization", "combined"):
        if safety_notes and safety_notes.lower() not in reply_lower:
            if reply:
                return {"assistant_reply": f"{reply}\n\n{safety_notes}"}
            return {"assistant_reply": safety_notes}
        return {"assistant_reply": reply}

    table_ack = _build_table_acknowledgement(query_tables)
    if table_ack:
        if safety_notes and safety_notes.lower() not in table_ack.lower():
            return {"assistant_reply": f"{table_ack}\n\n{safety_notes}"}
        return {"assistant_reply": table_ack}

    derived_reply = _derive_reply_from_query_output(query_output)
    if derived_reply:
        if safety_notes and safety_notes.lower() not in derived_reply.lower():
            return {"assistant_reply": f"{derived_reply}\n\n{safety_notes}"}
        return {"assistant_reply": derived_reply}

    # Fallback only if we could not derive anything deterministic.
    if not is_generic_reply:
        if safety_notes and safety_notes.lower() not in reply_lower:
            return {"assistant_reply": f"{reply}\n\n{safety_notes}"}
        return {"assistant_reply": reply}

    output_excerpt = _compact_query_output(query_output)
    if not output_excerpt:
        return {"assistant_reply": reply}

    if safety_notes and safety_notes.lower() not in output_excerpt.lower():
        return {"assistant_reply": f"{output_excerpt}\n\n{safety_notes}"}
    return {"assistant_reply": output_excerpt}


def _format_error_reply(state: AgentState) -> AgentState:
    error = state.get("error") or "Unknown error"
    # Clean up error message for display
    if "Execution error:" in error:
        error = error.replace("Execution error: ", "")
    guidance = "Please try rephrasing your request or check if the column names are correct."
    if "invalid model json response" in error.lower():
        guidance = (
            "The selected model returned malformed structured output. "
            "Please retry, or switch to another available model."
        )
    elif "no valid code returned" in error.lower() or "no valid code in fix" in error.lower():
        guidance = (
            "The selected model returned an incomplete structured response (missing code). "
            "Please retry, or switch to another available model."
        )
    return {
        "assistant_reply": f"I encountered an error: {error}\n\n{guidance}",
        "result_rows": state.get("rows") or [],
    }


def _route_after_execute(state: AgentState) -> Literal["fix_code", "format_error", "compose_assistant_reply"]:
    error = state.get("error")
    retry_count = state.get("retry_count") or 0

    if error:
        if retry_count < 1:
            return "fix_code"
        else:
            return "format_error"
    return "compose_assistant_reply"


def build_workflow():
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("prepare_context", _prepare_context)
    g.add_node("generate_code", _generate_code)
    g.add_node("execute_code", _execute_code)
    g.add_node("fix_code", _fix_code)
    g.add_node("format_error", _format_error_reply)
    g.add_node("compose_assistant_reply", _compose_assistant_reply)

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
            "compose_assistant_reply": "compose_assistant_reply",
        },
    )

    # Fix code loops back to execute
    g.add_edge("fix_code", "execute_code")

    # Error formatting ends
    g.add_edge("format_error", END)

    # Compose final response and end.
    g.add_edge("compose_assistant_reply", END)

    return g.compile()
