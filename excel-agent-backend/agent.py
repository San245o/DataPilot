from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Any

import google.generativeai as genai

logger = logging.getLogger("excel-agent-backend.model")

SYSTEM_PROMPT_BASE = """You are a data analysis agent using pandas and plotly.

RESPONSE FORMAT - Return ONLY valid JSON (no markdown):
{"code": "python code", "assistant_reply": "markdown response", "action_type": "query|mutation|visualization|combined", "table_request": {"enabled": false, "title": ""}}

ENV: df (DataFrame), pd(pandas), np(numpy), math, px(plotly.express), go(plotly.graph_objects), make_subplots(plotly.subplots.make_subplots), re(regex), unicodedata, pre-loaded. NO imports. End with result_df=df.

HELPERS:
- log_output(msg) - log text or pass a DataFrame/Series directly (e.g. log_output(df)) to populate the UI table. Do NOT use .to_string() for tables!
- print_query('filter_expr') - filter df with query string like 'Price>100'
- print_table(max_rows=10) - show df snapshot
- highlight_rows(indices) - pass a list of ALL matching indices (do not truncate or slice the list)
- highlight_column(column) - highlight one important column header in the UI
- highlight_columns(columns) - highlight multiple relevant column headers in the UI while rows stay highlighted
- text_equals(column, value) - case-insensitive exact match with stripped whitespace
- text_contains(column, value) - case-insensitive contains match
- to_numeric_clean(column_or_series) - strips currency/non-numeric chars then converts with pd.to_numeric(errors='coerce')
- Mutations: To add rows use `df = pd.concat([df, pd.DataFrame([...])], ignore_index=True)`. Do NOT use `df.append()`! edit_cell(row,col,val), add_column(name), delete_column(name), rename_column(old,new), delete_row(idx)

PATTERNS:
A) Aggregation: result=df.groupby('X')['Y'].sum().reset_index(); log_output(result); result_df=df
B) Filter: print_query('Rating>8'); result_df=df
C) Complex filter: mask=(df['A']>5)&(df['B']=='x'); log_output(df[mask]); result_df=df
D) Mutation: df['Col']=df['Col'].str.replace('old','new'); result_df=df

RULES:
- Single quotes in code, pass DataFrames directly to log_output for rendering, handle NaN with dropna/fillna
- String match: .str.contains('x',case=False,na=False)
- Never use direct case-sensitive equality for user-provided text filters (e.g., Item == 'smoothie'). Use text_equals('Item', 'smoothie') or .astype(str).str.strip().str.casefold().eq('smoothie').
- For money/revenue totals, prefer amount columns like 'Total Spent', 'Total_Spent', 'Amount', 'Revenue' and parse with to_numeric_clean(...).fillna(0) before sum.
- For COUNT / HOW MANY / TOTAL NUMBER questions, ALWAYS compute the scalar and emit the real executed answer with `log_output(...)` or `print(...)`. Do not leave the count only in Python variables.
- For scalar query answers (counts, sums, averages, minimums, maximums), keep `assistant_reply` short and generic like 'Done.' so the final reply can be composed from executed output instead of guessing.
- For non-ASCII / mojibake / encoding checks, prefer string methods like `.isascii()` when possible; `ord(...)`, `unicodedata`, and `math` are available if needed for simple compatibility logic.
- Charts: assign to fig, never call fig.show/to_html/to_json
- Prefer `px` for standard single-chart visualizations because it is the default/popular path here and produces simpler, more reliable code.
- Use `go` and `make_subplots(...)` only when you need custom traces, secondary axes, or multi-panel/subplot layouts.
- Subplots ARE allowed. For subplot layouts, build the figure with `make_subplots(...)`, add traces with `fig.add_trace(...)`, and keep the final figure in `fig`.
- Never delete/drop/remove/filter out rows unless the user explicitly asks to delete/drop/remove rows (or duplicates). If not explicitly requested, preserve the original row count.
- If you mutate data (add/delete/update rows, append multiple rows, sort/reorder rows, or modify columns), set action_type='mutation' unless any additional output is also requested, then use action_type='combined'.
- Multi-step tasks are allowed. If the user asks to clean/modify/filter/aggregate and then visualize or summarize, do all requested steps in one code block using the updated `df`, and return the final chart in `fig`.
- For query/visualization requests, never replace `df` or `result_df` with a filtered subset; use print_query/log_output/highlight_rows and keep `result_df=df`.
- For structural MUTATIONS (add/drop columns/rows, clean datatypes): Modify `df` directly, return `result_df=df`, and set `table_request.enabled=False`. Do NOT create separate tables for this.
- Cleaning, standardizing, coercing numeric columns, removing strings from numeric fields, and making values positive are mutation tasks. Persist the cleaned `df` instead of only logging summary statistics.
- For FINDING/SEARCHING/FILTERING: use `highlight_rows(df[mask].index.tolist())` and also highlight the most relevant worked-with columns using `highlight_column(...)` or `highlight_columns([...])`.
- If the user asks to identify, point out, mark, fix, update, rename, or correct values in a specific column, highlight that column header with `highlight_column(column)`.
- If multiple columns are central to the request, prefer `highlight_columns(['col1', 'col2'])` so those headers glow while the matched rows remain highlighted.
- For LISTING/PRINTING (e.g. "list the top 5 students"): If the result has < 10 rows, put the actual answer in `assistant_reply` as a short markdown list. Do NOT create a separate table unless explicitly requested.
- Only use `table_request.enabled=True` when the user explicitly asks for a separate/new/extracted/pivot table. For normal queries, answer in chat and keep `table_request.enabled=False`.
- If the request includes `[FORCE_EXTRACT_TABLE]`, you MUST create a separate subset table with `log_output(subset_df)`, set `table_request.enabled=True`, keep `result_df=df`, and avoid answering with only chat text.
- table_request.title MUST be very short (max 3-5 words), e.g., "Low Involvement".
"""

JSON_OUTPUT_CONTRACT = """STRICT OUTPUT CONTRACT:
- Return exactly one JSON object only.
- No markdown, no code fences, no extra commentary.
- No planning, no reasoning, no self-checks, no bullet points, no prose before or after the JSON.
- Required keys: code, assistant_reply, action_type, table_request.
- table_request must include enabled (bool) and title (string).
"""

THINKING_TRACE_PROMPT = """THINKING MODE - visible ReAct trace:
- Add a concise, user-visible reasoning_steps array that summarizes what you will do.
- reasoning_steps must NOT reveal private chain-of-thought. Keep them short, practical, and execution-focused.
- Include 2 to 4 steps only.
- Each step must be an object with keys: kind, title, content.
- kind must be either "thought" or "action".
- Titles should be short (2-4 words). Content should be one short sentence.
- Do not invent execution results in reasoning_steps. Real observations are added later by the system.
"""

THINKING_JSON_OUTPUT_CONTRACT = """STRICT OUTPUT CONTRACT:
- Return exactly one JSON object only.
- No markdown, no code fences, no extra commentary.
- Required keys: reasoning_steps, code, assistant_reply, action_type, table_request.
- reasoning_steps must be an array with 2 to 4 objects.
- Each reasoning_steps item must include kind, title, and content.
- kind must be "thought" or "action".
- table_request must include enabled (bool) and title (string).
"""

THINKING_SUPPORTED_MODELS = {
    "models/gemma-4-31b-it",
    "minimaxai/minimax-m2.5",
}

VIZ_RULES = """
VIZ RULES - template='plotly_dark', no text on bars/slices, limit 10-15 cats, single series showlegend=False

DEFAULT CHOICE:
- Use px for normal bar/line/scatter/pie/histogram charts.
- Use go + make_subplots only for subplot grids, mixed trace types, or highly custom layouts.

BAR (horizontal for long labels):
data=df.groupby('Cat')['Val'].sum().nlargest(10).reset_index()
fig=px.bar(data,x='Val',y='Cat',orientation='h',template='plotly_dark',title='Top 10')
fig.update_layout(showlegend=False,yaxis={'categoryorder':'total ascending'})

LINE: fig=px.line(trend,x='Year',y='Value',template='plotly_dark',markers=True)

PIE: fig=px.pie(values=counts.values,names=counts.index,template='plotly_dark')
fig.update_traces(textinfo='percent',hoverinfo='label+value+percent')

SCATTER: fig=px.scatter(df,x='X',y='Y',template='plotly_dark',hover_data=['Name'])

SUBPLOTS:
fig = make_subplots(rows=1, cols=2, subplot_titles=['Left', 'Right'])
fig.add_trace(go.Bar(x=left['x'], y=left['y'], name='Left'), row=1, col=1)
fig.add_trace(go.Scatter(x=right['x'], y=right['y'], mode='lines', name='Right'), row=1, col=2)
fig.update_layout(template='plotly_dark', showlegend=False)

HEATMAP MATRIX RULE:
- If data already has a metric/count column (e.g., accident_count, total, amount, revenue), aggregate with sum on that column.
- Only use groupby(...).size() when each row is a raw event and no metric/count column exists.
- Prefer pivot_table(values='metric_col', index='row_cat', columns='col_cat', aggfunc='sum', fill_value=0)
"""

MERGE_RULES = """
MERGE: Only one df available. Self-join: subset=df[df['Type']=='A']; merged=df.merge(subset[['Key','Val']],on='Key')
"""

STATS_RULES = """
STATS (no scipy): .corr(), .describe(), .mean(), .std(), .quantile()
Outliers: Q1,Q3=col.quantile([.25,.75]); IQR=Q3-Q1; outliers=col[(col<Q1-1.5*IQR)|(col>Q3+1.5*IQR)]
Regression: np.polyfit(x,y,1)
"""

CORR_RULES = """
CORRELATION RULES:
- Use numeric only: num_df=df.select_dtypes(include=[np.number])
- Drop cols with <=1 non-null; need >=2 numeric cols
- Matrix: px.imshow(corr,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,template='plotly_dark')
- Pair: scatter with trendline='ols', show r in title
"""

TIME_RULES = """
TIME: pd.to_datetime(df['date'],errors='coerce'); resample('M').mean(); groupby(df['date'].dt.year); rolling(7).mean()
"""

_VIZ_KW = {'chart', 'plot', 'graph', 'pie', 'bar', 'scatter', 'visual', 'histogram', 'heatmap', 'line', 'bubble', 'distribution', 'trend', 'compare'}
_MERGE_KW = {'merge', 'join', 'combine tables', 'match records', 'link'}
_STATS_KW = {'outlier', 'significance', 'p-value', 'statistical', 'std', 'variance', 'regression', 'quartile'}
_CORR_KW = {'correlation', 'correlate', 'corr matrix', 'relationship between'}
_TIME_KW = {'over time', 'by year', 'by month', 'by date', 'forecast', 'time series', 'yearly', 'monthly', 'daily'}
_MUTATION_KW = {
    'change', 'update', 'replace', 'rename', 'delete', 'add', 'edit', 'merge', 'set', 'modify', 'remove', 'fix', 'correct', 'convert',
    'append', 'insert', 'sort', 'reorder', 'order by', 'arrange',
    'add row', 'add rows', 'append row', 'append rows', 'insert row', 'insert rows',
    'clean', 'sanitize', 'normalize', 'standardize', 'coerce', 'cast',
    'make numeric', 'convert to numeric', 'numbers only', 'no strings',
    'only positive', 'make positive', 'absolute values', 'absolute value',
    'fix datatypes', 'fix data types', 'convert type', 'convert column',
}
_TABLE_INTENT_KW = {
    'separate table',
    'separate result table',
    'separate extracted table',
    'another table',
    'new table',
    'result table',
    'extracted table',
    'extract columns',
    'extract these columns',
    'only these columns',
    'table view',
    'split table',
    'pivot table',
    'pivot-style matrix',
}


def _build_system_prompt(prompt: str) -> str:
    p = prompt.lower()
    parts = [SYSTEM_PROMPT_BASE]
    if any(k in p for k in _VIZ_KW):
        parts.append(VIZ_RULES)
    if any(k in p for k in _MERGE_KW):
        parts.append(MERGE_RULES)
    if any(k in p for k in _STATS_KW):
        parts.append(STATS_RULES)
    if any(k in p for k in _CORR_KW):
        parts.append(CORR_RULES)
    if any(k in p for k in _TIME_KW):
        parts.append(TIME_RULES)
    return "\n".join(parts)


def supports_thinking_model(model_name: str) -> bool:
    return model_name in THINKING_SUPPORTED_MODELS


def _build_output_contract(thinking_mode: bool) -> str:
    return THINKING_JSON_OUTPUT_CONTRACT if thinking_mode else JSON_OUTPUT_CONTRACT


def _trim_reasoning_text(value: Any, max_len: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _normalize_reasoning_steps(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue

        kind = str(item.get("kind") or "").strip().lower()
        title = _trim_reasoning_text(item.get("title"), 40)
        content = _trim_reasoning_text(item.get("content"), 220)
        status = str(item.get("status") or "completed").strip().lower()

        if kind not in {"thought", "action", "observation"}:
            continue
        if not title or not content:
            continue
        if status not in {"completed", "error"}:
            status = "completed"

        normalized.append({
            "kind": kind,
            "title": title,
            "content": content,
            "status": status,
        })

    return normalized


def _describe_action_type(action_type: str) -> str:
    mapping = {
        "query": "query the current dataset",
        "mutation": "modify the working dataset",
        "visualization": "build the requested visualization",
        "combined": "update the data and produce the requested result",
    }
    return mapping.get(action_type, "analyze the current dataset")


def _fallback_reasoning_steps(
    *,
    prompt: str,
    action_type: str,
    columns: list[str] | None = None,
    error_message: str | None = None,
) -> list[dict[str, str]]:
    column_count = len(columns or [])
    dataset_note = (
        f"Use the available schema with {column_count} columns to anchor the plan."
        if column_count
        else "Use the available dataset schema to anchor the plan."
    )
    action_summary = _describe_action_type(action_type)

    steps = [
        {
            "kind": "thought",
            "title": "Understand request",
            "content": _trim_reasoning_text(prompt, 180) or "Interpret the user's data task.",
            "status": "completed",
        },
        {
            "kind": "action",
            "title": "Inspect schema",
            "content": dataset_note,
            "status": "completed",
        },
        {
            "kind": "action",
            "title": "Plan execution",
            "content": f"Generate pandas and Plotly code to {action_summary}.",
            "status": "completed",
        },
    ]

    if error_message:
        steps.append({
            "kind": "thought",
            "title": "Repair failure",
            "content": _trim_reasoning_text(error_message, 180),
            "status": "completed",
        })

    return steps[:4]


def _is_mutation_only(prompt: str) -> bool:
    p = prompt.lower()
    has_mutation = any(k in p for k in _MUTATION_KW)
    has_viz = any(k in p for k in _VIZ_KW)
    return has_mutation and not has_viz


def _extract_mentioned_columns(prompt: str, columns: list[str]) -> set[str]:
    p = prompt.lower()
    mentioned = set()
    for col in columns:
        col_lower = col.lower()
        col_spaced = col_lower.replace('_', ' ')
        if col_lower in p or col_spaced in p:
            mentioned.add(col)
    return mentioned


def _wants_separate_table(prompt: str) -> bool:
    p = prompt.lower()
    return "[force_extract_table]" in p or any(k in p for k in _TABLE_INTENT_KW) or "pivot" in p or bool(re.search(r"\bextract\b", p))


def _extract_first_object(text: str) -> str:
    start = text.find('{')
    if start == -1:
        return text

    depth = 0
    in_string = False
    quote_char = ''
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def _extract_last_likely_json_object(text: str) -> str | None:
    candidates: list[str] = []
    for match in re.finditer(r"\{", text):
        candidate = _extract_first_object(text[match.start():]).strip()
        looks_like_standard = '"code"' in candidate and '"assistant_reply"' in candidate
        looks_like_thinking = '"kind"' in candidate and ('"tool"' in candidate or '"final_answer"' in candidate)
        looks_like_pythonish_thinking = (
            re.search(r"\bkind\b", candidate) and
            (re.search(r"\btool\b", candidate) or re.search(r"\bfinal_answer\b", candidate))
        )
        if candidate.startswith("{") and (looks_like_standard or looks_like_thinking or looks_like_pythonish_thinking):
            candidates.append(candidate)
    return candidates[-1] if candidates else None


def _repair_common_json_issues(text: str) -> str:
    repaired = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    repaired = repaired.replace("\\\r\n", "\\n").replace("\\\n", "\\n")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
    return repaired


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    text = _extract_last_likely_json_object(text) or _extract_first_object(text)

    # First attempt: strict JSON.
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload
    except json.JSONDecodeError:
        pass

    # Second attempt: repair common malformed JSON issues.
    try:
        repaired_json = _repair_common_json_issues(text)
        payload = json.loads(repaired_json)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload
    except Exception:
        pass

    # Fallback for models that return Python-like dicts with single quotes.
    try:
        repaired = _repair_common_json_issues(text)
        repaired = re.sub(r"\btrue\b", "True", repaired)
        repaired = re.sub(r"\bfalse\b", "False", repaired)
        repaired = re.sub(r"\bnull\b", "None", repaired)
        payload = ast.literal_eval(repaired)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload
    except Exception as exc:
        raise ValueError(f"Invalid model JSON response: {exc}") from exc


def _parse_model_response(raw: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        candidate = _extract_json(raw)
        if isinstance(candidate, dict):
            payload = candidate
    except Exception:
        payload = {}

    code = payload.get("code")
    if isinstance(code, str) and code.strip():
        return payload

    text = (raw or "").strip()
    block = re.search(r"```(?:python|py)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if block and block.group(1).strip():
        payload["code"] = block.group(1).strip()
        return payload

    if "result_df" in text and ("df" in text or "log_output" in text or "print_query" in text):
        payload["code"] = text

    return payload


def _invoke_model(*, model_name: str, system_prompt: str, user_message: str) -> tuple[str, dict[str, int]]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    normalized_model = str(model_name or "").strip()
    if not normalized_model:
        raise RuntimeError("Model name is required. Silent fallback to Gemini 3.1 Flash Lite is disabled.")

    logger.info("invoke_model model=%s", normalized_model)

    if normalized_model.startswith("openai/") or normalized_model.startswith("gpt-"):
        if normalized_model.startswith("openai/"):
            normalized_model = normalized_model[7:]
            
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise RuntimeError("GITHUB_TOKEN not set")

        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(github_token),
            retry_total=0,
        )
        response = client.complete(
            messages=[SystemMessage(system_prompt), UserMessage(user_message)],
            temperature=0,
            model=normalized_model,
        )
        raw = response.choices[0].message.content or ""
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
        return raw, usage

    if normalized_model.startswith("minimax") or normalized_model.startswith("meta/"):
        from openai import OpenAI

        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        request_payload = {
            "model": normalized_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 8192,
        }
        try:
            # Prefer strict JSON mode when supported by the model.
            response = client.chat.completions.create(
                **request_payload,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(**request_payload)

        try:
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("No content received from NVIDIA NIM model")
        except Exception as e:
            raise RuntimeError(f"NVIDIA NIM error: {str(e)}")

        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
        return raw, usage

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(normalized_model)
    generation_config: dict[str, Any] = {"temperature": 0}
    # Enable JSON mode for Gemini-hosted Gemini and Gemma models.
    if (
        normalized_model.startswith("gemini")
        or normalized_model.startswith("models/gemini")
        or normalized_model.startswith("gemma")
        or normalized_model.startswith("models/gemma")
    ):
        generation_config["response_mime_type"] = "application/json"

    try:
        response = model.generate_content([system_prompt, user_message], generation_config=generation_config)
    except Exception:
        # Fall back to basic generation if a model does not support response MIME controls.
        response = model.generate_content([system_prompt, user_message], generation_config={"temperature": 0})

    raw = response.text or ""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
        usage["completion_tokens"] = response.usage_metadata.candidates_token_count
        usage["total_tokens"] = response.usage_metadata.total_token_count
    return raw, usage


def invoke_model_json(*, model_name: str, system_prompt: str, user_message: str) -> tuple[dict[str, Any], dict[str, int], str]:
    raw, usage = _invoke_model(model_name=model_name, system_prompt=system_prompt, user_message=user_message)
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from model")
    return payload, usage, raw


def clean_final_reply_text(reply: str, *, initial_reply: str, wants_list: bool = False) -> str:
    text = str(reply or "").replace("\r\n", "\n").strip()
    if not text:
        return initial_reply

    meta_prefixes = (
        "user's goal:",
        "user's specific request",
        "execution output:",
        "constraints:",
        "movies listed:",
        "the execution output",
        "the output shows",
        "provide an extremely concise",
        "prefer one short sentence",
        "based only on",
        "no markdown",
        "no ascii tables",
        "extremely concise",
        "no elaboration",
    )
    meta_contains = (
        "? yes.",
        "? no.",
    )

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(meta_prefixes):
            continue
        if any(token in lower for token in meta_contains):
            continue
        cleaned_lines.append(line)

    if wants_list:
        bullet_lines = [line for line in cleaned_lines if line.startswith(("-", "*"))]
        if bullet_lines:
            unique_bullets: list[str] = []
            for line in bullet_lines:
                normalized = line.strip()
                if normalized not in unique_bullets:
                    unique_bullets.append(normalized)
            return "\n".join(unique_bullets)

    for line in reversed(cleaned_lines):
        candidate = line.strip().strip("`").strip("\"'")
        if not candidate:
            continue
        return candidate

    return initial_reply


def _trim_history(history: list[dict[str, str]], max_turns: int = 3) -> list[dict[str, str]]:
    h = history[-max_turns:] if len(history) > max_turns else list(history)
    result = []
    for turn in h:
        c = turn.get("content", "")
        if turn.get("role") == "assistant":
            result.append({"role": "assistant", "content": c[:200] + "..." if len(c) > 200 else c})
        else:
            result.append({"role": turn.get("role", "user"), "content": c[:150] if len(c) > 150 else c})
    return result


def generate_code(
    *,
    prompt: str,
    model_name: str,
    columns: list[str],
    dtypes: dict[str, str],
    nulls: dict[str, int],
    value_ranges: dict[str, dict],
    top_categories: dict[str, list],
    sample_rows: list[dict[str, Any]],
    history: list[dict[str, str]],
    thinking_mode: bool = False,
) -> dict[str, Any]:
    system_prompt = _build_system_prompt(prompt)
    if thinking_mode:
        system_prompt = f"{system_prompt}\n\n{THINKING_TRACE_PROMPT}"
    trimmed_history = _trim_history(history)
    is_mutation = _is_mutation_only(prompt)
    mentioned = _extract_mentioned_columns(prompt, columns)

    # Build context
    ctx = [f"Request: {prompt}"]
    ctx.append(f"Columns: {json.dumps(columns)}")

    # Compact dtypes - only show types that differ from defaults
    ctx.append(f"Types: {json.dumps(dtypes)}")

    # Nulls if any - only columns with nulls
    has_nulls = {k: v for k, v in nulls.items() if v > 0}
    if has_nulls:
        ctx.append(f"Nulls: {json.dumps(has_nulls)}")

    # Value ranges and categories - only for mentioned columns + few extras (max 4)
    if not is_mutation:
        if value_ranges:
            relevant = {k: v for k, v in value_ranges.items() if k in mentioned}
            for k, v in list(value_ranges.items())[:4]:
                if k not in relevant and len(relevant) < 4:
                    relevant[k] = v
            if relevant:
                ctx.append(f"Ranges: {json.dumps(relevant)}")

        if top_categories:
            relevant = {k: v[:5] for k, v in top_categories.items() if k in mentioned}  # Limit to 5 categories each
            for k, v in list(top_categories.items())[:4]:
                if k not in relevant and len(relevant) < 4:
                    relevant[k] = v[:5]
            if relevant:
                ctx.append(f"Categories: {json.dumps(relevant)}")

    # Sample rows - only 2 rows max
    if sample_rows:
        ctx.append(f"Sample: {json.dumps(sample_rows[:2])}")

    # History - already trimmed
    if trimmed_history:
        ctx.append(f"History: {json.dumps(trimmed_history)}")

    ctx.append(_build_output_contract(thinking_mode))

    user_message = "\n".join(ctx)
    raw, usage = _invoke_model(model_name=model_name, system_prompt=system_prompt, user_message=user_message)

    payload = _parse_model_response(raw)
    code = payload.get("code", "")
    reply = payload.get("assistant_reply", "Done.")
    action = payload.get("action_type", "query")
    reasoning_steps = _normalize_reasoning_steps(payload.get("reasoning_steps")) if thinking_mode else []
    table_request = payload.get("table_request") or {}
    table_enabled = bool(table_request.get("enabled", False))
    table_title = str(table_request.get("title") or "Query Table").strip()
    if not table_title:
        table_title = "Query Table"

    explicit_table_intent = _wants_separate_table(prompt)

    if explicit_table_intent:
        table_enabled = True
        if table_title == "Query Table":
            table_title = "Extracted Table"
    elif not explicit_table_intent:
        table_enabled = False

    # Deterministic fallback keeps execution functional.
    if not isinstance(code, str) or not code.strip():
        code = "print_table(max_rows=10)\nresult_df=df"
        if not isinstance(reply, str) or not reply.strip():
            reply = "I could not generate transformation code reliably, so I returned a snapshot of the current data."
        action = "query"
        if not explicit_table_intent:
            table_enabled = False
            table_title = "Query Table"

    if not isinstance(reply, str):
        reply = "Done."
    if action not in ("query", "mutation", "visualization", "combined"):
        action = "query"
    if thinking_mode and not reasoning_steps:
        reasoning_steps = _fallback_reasoning_steps(prompt=prompt, action_type=action, columns=columns)

    return {
        "code": code,
        "assistant_reply": reply,
        "action_type": action,
        "token_usage": usage,
        "table_request": {"enabled": table_enabled, "title": table_title},
        "reasoning_steps": reasoning_steps,
    }


def generate_fix(
    *,
    prompt: str,
    model_name: str,
    columns: list[str],
    dtypes: dict[str, str],
    original_code: str,
    error_message: str,
    thinking_mode: bool = False,
) -> dict[str, Any]:
    system_prompt = _build_system_prompt(prompt)
    if thinking_mode:
        system_prompt = f"{system_prompt}\n\n{THINKING_TRACE_PROMPT}"

    user_message = f"""Original request: {prompt}
Columns: {json.dumps(columns)}
Types: {json.dumps(dtypes)}

FAILED CODE:
```python
{original_code}
```

ERROR: {error_message}

COMMON FIXES:
- print_query() takes a FILTER STRING like 'Price > 100', NOT a variable name
- To log a computed DataFrame/Series, use: log_output(my_var)
- Ensure variables are defined before use
- Use single quotes for strings

{_build_output_contract(thinking_mode)}

Return fixed JSON."""

    raw, usage = _invoke_model(model_name=model_name, system_prompt=system_prompt, user_message=user_message)

    payload = _parse_model_response(raw)
    code = payload.get("code", "")
    reply = payload.get("assistant_reply", "Done.")
    action = payload.get("action_type", "query")
    reasoning_steps = _normalize_reasoning_steps(payload.get("reasoning_steps")) if thinking_mode else []
    table_request = payload.get("table_request") or {}
    table_enabled = bool(table_request.get("enabled", False))
    table_title = str(table_request.get("title") or "Query Table").strip()
    if not table_title:
        table_title = "Query Table"

    explicit_table_intent = _wants_separate_table(prompt)

    if explicit_table_intent:
        table_enabled = True
        if table_title == "Query Table":
            table_title = "Extracted Table"
    elif not explicit_table_intent:
        table_enabled = False

    if not isinstance(code, str) or not code.strip():
        code = original_code if isinstance(original_code, str) and original_code.strip() else "print_table(max_rows=10)\nresult_df=df"
        if not isinstance(reply, str) or not reply.strip():
            reply = "I could not produce a reliable fix; retrying with the previous code."
        action = "query"
        if explicit_table_intent:
            table_enabled = True
            if table_title == "Query Table":
                table_title = "Extracted Table"

    if action not in ("query", "mutation", "visualization", "combined"):
        action = "query"
    if thinking_mode and not reasoning_steps:
        reasoning_steps = _fallback_reasoning_steps(
            prompt=prompt,
            action_type=action,
            columns=columns,
            error_message=error_message,
        )

    return {
        "code": code,
        "assistant_reply": reply,
        "action_type": action,
        "token_usage": usage,
        "table_request": {"enabled": table_enabled, "title": table_title},
        "reasoning_steps": reasoning_steps,
    }


def draft_final_reply(
    prompt: str,
    model_name: str,
    query_output: str | None,
    initial_reply: str,
) -> dict[str, Any]:
    if not query_output or len(query_output.strip()) < 10:
        return {
            "reply": clean_final_reply_text(initial_reply, initial_reply=initial_reply, wants_list=False),
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    api_key = os.getenv("GEMINI_API_KEY")
    output = query_output[:2000] if len(query_output) > 2000 else query_output
    prompt_lower = prompt.lower()
    wants_list = any(keyword in prompt_lower for keyword in ("list", "print", "give me", "show me"))

    user_message = f"""User asked: {prompt}

Execution output:
{output}

Provide an extremely concise, direct answer based ONLY on this output.
{"If the user asked to list/print/show/give items and the result is 10 items or fewer, format the answer as clean markdown bullet points. Each bullet should contain only the most relevant identifier first, such as title/name/item, plus one important value only if the user asked for it. Do NOT dump entire rows or include index columns unless the user asked for them." if wants_list else "Prefer one short sentence unless the user explicitly asked for a list."}
DO NOT elaborate and DO NOT detail unrelated data characteristics. Focus only on the exact core request.
Never output markdown or ASCII tables in this response."""

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if model_name and (model_name.startswith("openai/") or model_name.startswith("gpt-")):
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            return {"reply": initial_reply, "token_usage": usage}

        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(github_token),
            retry_total=0,
        )
        response = client.complete(
            messages=[
                SystemMessage("Summarize data results clearly and concisely. Do not output markdown or ASCII tables."),
                UserMessage(user_message)
            ],
            temperature=0,
            model=model_name,
        )
        reply = clean_final_reply_text(
            response.choices[0].message.content or initial_reply,
            initial_reply=initial_reply,
            wants_list=wants_list,
        )
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
        return {"reply": reply, "token_usage": usage}
    else:
        if not api_key:
            return {"reply": initial_reply, "token_usage": usage}
        normalized_model = str(model_name or "").strip()
        if not normalized_model:
            return {"reply": clean_final_reply_text(initial_reply, initial_reply=initial_reply, wants_list=wants_list), "token_usage": usage}
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(normalized_model)
        try:
            response = model.generate_content([
                "Summarize data results clearly and concisely. Do not output markdown or ASCII tables.",
                user_message
            ])
            reply = clean_final_reply_text(
                response.text or initial_reply,
                initial_reply=initial_reply,
                wants_list=wants_list,
            )
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                usage["total_tokens"] = response.usage_metadata.total_token_count
            return {"reply": reply, "token_usage": usage}
        except Exception:
            return {"reply": initial_reply, "token_usage": usage}
