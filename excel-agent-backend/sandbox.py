from __future__ import annotations

import ast
import builtins
import json
import threading
from dataclasses import dataclass, field
from typing import Any
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types, NaN, Inf, and plotly bdata to JSON-safe values."""
    if isinstance(obj, dict):
        # Convert plotly binary-encoded arrays: {"dtype": ..., "bdata": ...} -> list
        if "bdata" in obj and "dtype" in obj and len(obj) <= 3:
            try:
                import base64
                import struct

                dtype_map = {
                    "f8": ("d", 8), "f4": ("f", 4),
                    "i4": ("i", 4), "i2": ("h", 2), "i1": ("b", 1),
                    "u4": ("I", 4), "u2": ("H", 2), "u1": ("B", 1),
                }
                dtype = obj["dtype"]
                raw = base64.b64decode(obj["bdata"])
                if dtype in dtype_map:
                    fmt, size = dtype_map[dtype]
                    count = len(raw) // size
                    values = list(struct.unpack(f"<{count}{fmt}", raw[:count * size]))
                    return [_sanitize_for_json(v) for v in values]
            except Exception:
                pass
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(item) for item in obj.tolist()]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.api.types.is_scalar(obj) and pd.isna(obj):
        return None
    return obj

FORBIDDEN_NAMES = {
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
}


class SandboxViolation(Exception):
    pass


@dataclass
class SandboxResult:
    rows: list[dict[str, Any]]
    visualization: dict[str, Any] | None
    query_output: str | None
    mutation: bool = False
    highlight_indices: list[int] = field(default_factory=list)


def _validate_code(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxViolation(f"Syntax error in generated code: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation("Imports are not allowed in sandbox")
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
            raise SandboxViolation(f"Dunder access is not allowed: {node.attr}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxViolation(f"Forbidden identifier: {node.id}")


def _execute_in_sandbox(code: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_code(code)

    df = pd.DataFrame(rows)
    df.replace("", np.nan, inplace=True)
    original_df = df.copy()
    query_logs: list[str] = []
    mutation_flag: list[bool] = [False]
    highlight_idx: list[int] = []

    env: dict[str, Any] = {}

    def _get_df() -> pd.DataFrame:
        current_df = env.get("df")
        if not isinstance(current_df, pd.DataFrame):
            raise SandboxViolation("df is not a pandas DataFrame")
        return current_df

    def _set_df(updated_df: pd.DataFrame) -> pd.DataFrame:
        env["df"] = updated_df
        mutation_flag[0] = True
        return updated_df

    def log_output(message: Any) -> None:
        query_logs.append(str(message))

    def safe_print(*args: Any, **kwargs: Any) -> None:
        output = " ".join(str(a) for a in args)
        query_logs.append(output)

    def highlight_rows(indices: list[int]) -> None:
        highlight_idx.clear()
        highlight_idx.extend(indices)

    def print_query(query_expr: str, max_rows: int = 20) -> pd.DataFrame:
        current_df = _get_df()
        queried_df = current_df.query(query_expr, engine="python")
        highlight_idx.clear()
        highlight_idx.extend(queried_df.index.tolist()[:max_rows])
        query_logs.append(queried_df.head(max_rows).to_string(index=False))
        return queried_df

    def print_table(max_rows: int = 20) -> pd.DataFrame:
        current_df = _get_df()
        query_logs.append(current_df.head(max_rows).to_string(index=False))
        return current_df

    def add_column(name: str, default: Any = None) -> pd.DataFrame:
        current_df = _get_df().copy()
        current_df[name] = default
        return _set_df(current_df)

    def delete_column(name: str) -> pd.DataFrame:
        current_df = _get_df().copy()
        if name not in current_df.columns:
            raise SandboxViolation(f"Column '{name}' does not exist")
        current_df = current_df.drop(columns=[name])
        return _set_df(current_df)

    def add_row(row_data: dict[str, Any]) -> pd.DataFrame:
        current_df = _get_df().copy()
        normalized_row = dict(row_data)
        for key in normalized_row:
            if key not in current_df.columns:
                current_df[key] = pd.NA
        ordered_row = {column: normalized_row.get(column, pd.NA) for column in current_df.columns}
        updated_df = pd.concat([current_df, pd.DataFrame([ordered_row])], ignore_index=True)
        return _set_df(updated_df)

    def delete_row(index: int) -> pd.DataFrame:
        current_df = _get_df().copy()
        if index < 0 or index >= len(current_df.index):
            raise SandboxViolation("Row index out of range")
        updated_df = current_df.drop(index=index).reset_index(drop=True)
        return _set_df(updated_df)

    def edit_cell(row_index: int, column: str, value: Any) -> pd.DataFrame:
        current_df = _get_df().copy()
        if column not in current_df.columns:
            raise SandboxViolation(f"Column '{column}' does not exist")
        if row_index < 0 or row_index >= len(current_df.index):
            raise SandboxViolation("Row index out of range")
        current_df.at[row_index, column] = value
        # Auto-highlight the edited row so the UI shows what changed
        if row_index not in highlight_idx:
            highlight_idx.append(row_index)
        return _set_df(current_df)

    def rename_column(old_name: str, new_name: str) -> pd.DataFrame:
        current_df = _get_df().copy()
        if old_name not in current_df.columns:
            raise SandboxViolation(f"Column '{old_name}' does not exist")
        updated_df = current_df.rename(columns={old_name: new_name})
        return _set_df(updated_df)

    def table_to_csv(max_rows: int = 200) -> str:
        current_df = _get_df()
        buffer = StringIO()
        current_df.head(max_rows).to_csv(buffer, index=False)
        value = buffer.getvalue().strip()
        query_logs.append(value)
        return value

    safe_builtins = {
        "abs": builtins.abs,
        "all": builtins.all,
        "any": builtins.any,
        "bool": builtins.bool,
        "dict": builtins.dict,
        "enumerate": builtins.enumerate,
        "filter": builtins.filter,
        "float": builtins.float,
        "format": builtins.format,
        "frozenset": builtins.frozenset,
        "hasattr": builtins.hasattr,
        "hash": builtins.hash,
        "int": builtins.int,
        "isinstance": builtins.isinstance,
        "issubclass": builtins.issubclass,
        "iter": builtins.iter,
        "len": builtins.len,
        "list": builtins.list,
        "map": builtins.map,
        "max": builtins.max,
        "min": builtins.min,
        "next": builtins.next,
        "print": safe_print,
        "range": builtins.range,
        "repr": builtins.repr,
        "reversed": builtins.reversed,
        "round": builtins.round,
        "set": builtins.set,
        "slice": builtins.slice,
        "sorted": builtins.sorted,
        "str": builtins.str,
        "sum": builtins.sum,
        "tuple": builtins.tuple,
        "type": builtins.type,
        "zip": builtins.zip,
        "True": True,
        "False": False,
        "None": None,
        "ValueError": builtins.ValueError,
        "TypeError": builtins.TypeError,
        "KeyError": builtins.KeyError,
        "IndexError": builtins.IndexError,
        "Exception": builtins.Exception,
    }

    env.update({
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "df": df.copy(),
        "log_output": log_output,
        "print": safe_print,
        "print_query": print_query,
        "print_table": print_table,
        "add_column": add_column,
        "delete_column": delete_column,
        "add_row": add_row,
        "delete_row": delete_row,
        "edit_cell": edit_cell,
        "rename_column": rename_column,
        "table_to_csv": table_to_csv,
        "highlight_rows": highlight_rows,
    })

    exec(code, env, env)  # noqa: S102

    # Determine if mutation happened
    result_df = env.get("result_df", env.get("df"))
    if not isinstance(result_df, pd.DataFrame):
        raise SandboxViolation("result_df must be a pandas DataFrame")

    # Auto-detect mutation: compare shape and content with original
    if not mutation_flag[0]:
        if result_df.shape != original_df.shape:
            mutation_flag[0] = True
        elif not result_df.columns.equals(original_df.columns):
            mutation_flag[0] = True
        else:
            try:
                if not result_df.equals(original_df):
                    mutation_flag[0] = True
            except Exception:
                mutation_flag[0] = True

    fig = env.get("fig")
    visualization = None
    if fig is not None and hasattr(fig, "to_plotly_json"):
        try:
            raw_viz = fig.to_plotly_json()
            visualization = _sanitize_for_json(raw_viz)
        except Exception:
            visualization = None

    query_output = "\n\n".join([item for item in query_logs if item]).strip() or None

    return {
        "ok": True,
        "rows": _sanitize_for_json(result_df.to_dict(orient="records")),
        "visualization": visualization,
        "query_output": query_output,
        "mutation": mutation_flag[0],
        "highlight_indices": highlight_idx,
    }


def run_sandboxed(code: str, rows: list[dict[str, Any]], timeout_seconds: int = 30) -> SandboxResult:
    result_container: dict[str, Any] = {}
    error_container: dict[str, str] = {}

    def _target() -> None:
        try:
            result_container.update(_execute_in_sandbox(code, rows))
        except SandboxViolation as exc:
            error_container["error"] = f"Sandbox violation: {exc}"
        except Exception as exc:
            error_container["error"] = f"Execution error: {exc}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Sandbox execution timed out after {timeout_seconds}s")

    if error_container:
        raise RuntimeError(error_container["error"])

    if not result_container:
        raise RuntimeError("Sandbox returned no output")

    if not result_container.get("ok"):
        raise RuntimeError(result_container.get("error", "Sandbox error"))

    return SandboxResult(
        rows=result_container["rows"],
        visualization=result_container.get("visualization"),
        query_output=result_container.get("query_output"),
        mutation=result_container.get("mutation", False),
        highlight_indices=result_container.get("highlight_indices", []),
    )
