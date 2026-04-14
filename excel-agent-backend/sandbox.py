from __future__ import annotations

import ast
import builtins
import json
import math
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any
from io import StringIO

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _repair_mojibake(value: str) -> str:
    try:
        repaired = value.encode("latin-1").decode("utf-8")
        return repaired
    except Exception:
        return value


def _normalize_text_value(value: Any) -> str:
    text = str(value).strip()
    candidates = [text]

    repaired = _repair_mojibake(text)
    if repaired != text:
        candidates.append(repaired)

    normalized: list[str] = []
    for candidate in candidates:
        candidate = unicodedata.normalize("NFKC", candidate)
        candidate = unicodedata.normalize("NFKD", candidate)
        candidate = "".join(ch for ch in candidate if not unicodedata.combining(ch))
        normalized.append(candidate.casefold())

    return normalized[-1]


def _compute_changed_row_indices(original_df: pd.DataFrame, result_df: pd.DataFrame) -> list[int]:
    changed: list[int] = []

    shared_columns = [column for column in original_df.columns if column in result_df.columns]
    shared_len = min(len(original_df.index), len(result_df.index))

    for row_index in range(shared_len):
        original_row = original_df.iloc[row_index]
        result_row = result_df.iloc[row_index]
        row_changed = False

        for column in shared_columns:
            original_value = original_row[column]
            result_value = result_row[column]
            if pd.isna(original_value) and pd.isna(result_value):
                continue
            if original_value != result_value:
                row_changed = True
                break

        if row_changed:
            changed.append(row_index)

    if len(result_df.index) > len(original_df.index):
        changed.extend(range(len(original_df.index), len(result_df.index)))

    return changed


def _reshape_flat_values(values: list[Any], shape: Any) -> Any:
    if shape is None:
        return values

    try:
        if isinstance(shape, str):
            dims = [int(part.strip()) for part in shape.split(",") if part.strip()]
        elif isinstance(shape, (list, tuple)):
            dims = [int(part) for part in shape]
        else:
            return values

        if not dims:
            return values

        total = 1
        for dim in dims:
            total *= dim
        if total != len(values):
            return values

        def _reshape(seq: list[Any], current_dims: list[int]) -> Any:
            if len(current_dims) == 1:
                return seq
            chunk = 1
            for dim in current_dims[1:]:
                chunk *= dim
            return [
                _reshape(seq[i * chunk:(i + 1) * chunk], current_dims[1:])
                for i in range(current_dims[0])
            ]

        return _reshape(values, dims)
    except Exception:
        return values


def _df_to_records(df_obj: Any) -> list[dict[str, Any]]:
    # If the DataFrame has a custom index (like a pivot table), we MUST reset it so it becomes a column
    # Otherwise, to_dict("records") will drop the index entirely and the data looks broken on the frontend
    if not isinstance(df_obj.index, pd.RangeIndex):
        df_obj = df_obj.reset_index()
    return _sanitize_for_json(df_obj.to_dict(orient="records"))

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
                    values = [_sanitize_for_json(v) for v in values]
                    return _reshape_flat_values(values, obj.get("shape"))
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

FORBIDDEN_ATTR_CALLS = {
    "show",
    "to_html",
    "write_html",
    "to_json",
}


class SandboxViolation(Exception):
    pass


@dataclass
class SandboxResult:
    rows: list[dict[str, Any]]
    visualization: dict[str, Any] | None
    query_output: str | None
    query_table_rows: list[dict[str, Any]] | None
    mutation: bool = False
    highlight_indices: list[int] = field(default_factory=list)
    highlighted_columns: list[str] = field(default_factory=list)


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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in FORBIDDEN_ATTR_CALLS:
                raise SandboxViolation(
                    f"Calling '{attr_name}()' is not allowed. Assign chart to 'fig' and let the UI render it."
                )


def _execute_in_sandbox(code: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_code(code)

    df = pd.DataFrame(rows)
    df.replace("", np.nan, inplace=True)

    # Normalize leading/trailing whitespace in string-like cells to reduce fragile exact-match filters.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    original_df = df.copy()
    query_logs: list[str] = []
    query_table_rows: list[dict[str, Any]] | None = None
    mutation_flag: list[bool] = [False]
    highlight_idx: list[int] = []
    highlighted_columns: list[str] = []

    env: dict[str, Any] = {}

    def _contains_plotly_figure(value: Any) -> bool:
        if hasattr(value, "to_plotly_json"):
            return True
        if isinstance(value, dict):
            return any(_contains_plotly_figure(v) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(_contains_plotly_figure(v) for v in value)
        return False

    def _get_df() -> pd.DataFrame:
        current_df = env.get("df")
        if not isinstance(current_df, pd.DataFrame):
            raise SandboxViolation("df is not a pandas DataFrame")
        return current_df

    def _set_df(updated_df: pd.DataFrame) -> pd.DataFrame:
        env["df"] = updated_df
        mutation_flag[0] = True
        return updated_df

    def text_equals(column: str, value: str) -> pd.Series:
        current_df = _get_df()
        if column not in current_df.columns:
            raise SandboxViolation(f"Column '{column}' does not exist")
        needle = _normalize_text_value(value)
        return current_df[column].map(_normalize_text_value).eq(needle)

    def text_contains(column: str, value: str) -> pd.Series:
        current_df = _get_df()
        if column not in current_df.columns:
            raise SandboxViolation(f"Column '{column}' does not exist")
        needle = _normalize_text_value(value)
        return current_df[column].map(_normalize_text_value).str.contains(re.escape(needle), na=False)

    def to_numeric_clean(values: str | pd.Series) -> pd.Series:
        current_df = _get_df()
        if isinstance(values, str):
            if values not in current_df.columns:
                raise SandboxViolation(f"Column '{values}' does not exist")
            series = current_df[values]
        elif isinstance(values, pd.Series):
            series = values
        else:
            raise SandboxViolation("to_numeric_clean expects a column name or pandas Series")

        cleaned = series.astype(str).str.replace(r"[^0-9.+-]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    def log_output(message: Any) -> None:
        nonlocal query_table_rows
        # Don't stringify plotly figures - they produce massive JSON
        if _contains_plotly_figure(message):
            query_logs.append("[Plotly Figure - see visualization]")
        elif isinstance(message, pd.DataFrame):
            df_len = len(message)
            msg = f"[DataFrame: {df_len} rows, {len(message.columns)} columns]"
            if df_len > 10:
                msg += f"\nShowing first 10 rows:\n{message.head(10).to_string()}"
            else:
                msg += f"\n{message.to_string()}"
            query_logs.append(msg)
            query_table_rows = _df_to_records(message)
        elif isinstance(message, pd.Series):
            s_len = len(message)
            msg = f"[Series: {s_len} items]"
            if s_len > 10:
                msg += f"\nShowing first 10 items:\n{message.head(10).to_string()}"
            else:
                msg += f"\n{message.to_string()}"
            query_logs.append(msg)
            series_df = message.to_frame(name=message.name or "value").reset_index()
            query_table_rows = _df_to_records(series_df)
        else:
            s = str(message)
            if len(s) > 5000:
                s = s[:5000] + "... [truncated]"
            query_logs.append(s)

    def safe_print(*args: Any, **kwargs: Any) -> None:
        nonlocal query_table_rows
        parts = []
        for a in args:
            if _contains_plotly_figure(a):
                parts.append("[Plotly Figure - see visualization]")
            elif isinstance(a, pd.DataFrame):
                df_len = len(a)
                msg = f"[DataFrame: {df_len} rows]\n"
                msg += a.head(10).to_string() if df_len > 10 else a.to_string()
                parts.append(msg)
                query_table_rows = _df_to_records(a)
            elif isinstance(a, pd.Series):
                s_len = len(a)
                msg = f"[Series: {s_len} items]\n"
                msg += a.head(10).to_string() if s_len > 10 else a.to_string()
                parts.append(msg)
                series_df = a.to_frame(name=a.name or "value").reset_index()
                query_table_rows = _df_to_records(series_df)
            else:
                s = str(a)
                if len(s) > 5000:
                    s = s[:5000] + "... [truncated]"
                parts.append(s)
        output = " ".join(parts)
        query_logs.append(output)

    def highlight_rows(indices: list[int]) -> None:
        highlight_idx.clear()
        # Always output all indices, never truncate highlighting
        highlight_idx.extend(indices)

    def highlight_column(column: str) -> None:
        current_df = _get_df()
        if column not in current_df.columns:
            raise SandboxViolation(f"Column '{column}' does not exist")
        if column not in highlighted_columns:
            highlighted_columns.append(column)

    def highlight_columns(columns: list[str]) -> None:
        for column in columns:
            highlight_column(column)

    def print_query(query_expr: str, max_rows: int = 10) -> pd.DataFrame:
        nonlocal query_table_rows
        current_df = _get_df()
        queried_df = current_df.query(query_expr, engine="python")
        highlight_idx.clear()
        highlight_idx.extend(queried_df.index.tolist())
        df_len = len(queried_df)
        msg = f"Found {df_len} matching rows."
        if df_len > max_rows:
            msg += f"\nShowing first {max_rows} rows:\n{queried_df.head(max_rows).to_string(index=False)}"
        else:
            msg += f"\n{queried_df.to_string(index=False)}"
        query_logs.append(msg)
        query_table_rows = _df_to_records(queried_df)
        return queried_df

    def print_table(max_rows: int = 10) -> pd.DataFrame:
        nonlocal query_table_rows
        current_df = _get_df()
        df_len = len(current_df)
        msg = f"[DataFrame: {df_len} rows total]"
        if df_len > max_rows:
            msg += f"\nShowing first {max_rows} rows:\n{current_df.head(max_rows).to_string(index=False)}"
        else:
            msg += f"\n{current_df.to_string(index=False)}"
        query_logs.append(msg)
        query_table_rows = _df_to_records(current_df)
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
        if row_index not in highlight_idx:
            highlight_idx.append(row_index)
        highlight_column(column)
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
        "ascii": builtins.ascii,
        "bin": builtins.bin,
        "bool": builtins.bool,
        "callable": builtins.callable,
        "chr": builtins.chr,
        "complex": builtins.complex,
        "dict": builtins.dict,
        "divmod": builtins.divmod,
        "enumerate": builtins.enumerate,
        "filter": builtins.filter,
        "float": builtins.float,
        "format": builtins.format,
        "frozenset": builtins.frozenset,
        "hasattr": builtins.hasattr,
        "hash": builtins.hash,
        "hex": builtins.hex,
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
        "oct": builtins.oct,
        "ord": builtins.ord,
        "pow": builtins.pow,
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
        "math": math,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
        "re": re,
        "unicodedata": unicodedata,
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
        "highlight_column": highlight_column,
        "highlight_columns": highlight_columns,
        "text_equals": text_equals,
        "text_contains": text_contains,
        "to_numeric_clean": to_numeric_clean,
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

    if mutation_flag[0] and not highlight_idx:
        highlight_idx.extend(_compute_changed_row_indices(original_df, result_df))

    if mutation_flag[0] and not highlighted_columns:
        for row_index in highlight_idx:
            if 0 <= row_index < len(original_df.index) and 0 <= row_index < len(result_df.index):
                shared_columns = [column for column in original_df.columns if column in result_df.columns]
                for column in shared_columns:
                    original_value = original_df.iloc[row_index][column]
                    result_value = result_df.iloc[row_index][column]
                    if pd.isna(original_value) and pd.isna(result_value):
                        continue
                    if original_value != result_value:
                        if column not in highlighted_columns:
                            highlighted_columns.append(column)
                        break

    return {
        "ok": True,
        "rows": _df_to_records(result_df),
        "visualization": visualization,
        "query_output": query_output,
        "query_table_rows": query_table_rows,
        "mutation": mutation_flag[0],
        "highlight_indices": highlight_idx,
        "highlighted_columns": highlighted_columns,
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
        query_table_rows=result_container.get("query_table_rows"),
        mutation=result_container.get("mutation", False),
        highlight_indices=result_container.get("highlight_indices", []),
        highlighted_columns=result_container.get("highlighted_columns", []),
    )
