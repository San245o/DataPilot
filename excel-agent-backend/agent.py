from __future__ import annotations

import ast
import json
import os
import re
import warnings
from typing import Any

import google.generativeai as genai

SYSTEM_PROMPT_BASE = """You are a data analysis agent using pandas and plotly.

RESPONSE FORMAT - Return ONLY valid JSON (no markdown):
{"code": "python code", "assistant_reply": "markdown response", "action_type": "query|mutation|visualization|combined", "table_request": {"enabled": false, "title": ""}}

ENV: df (DataFrame), pd, np, px, go pre-loaded. NO imports. End with result_df=df.

HELPERS:
- log_output(msg) - log text or pass a DataFrame/Series directly (e.g. log_output(df)) to populate the UI table. Do NOT use .to_string() for tables!
- print_query('filter_expr') - filter df with query string like 'Price>100'
- print_table(max_rows=10) - show df snapshot
- highlight_rows(indices) - pass a list of ALL matching indices (do not truncate or slice the list)
- Mutations: To add rows use `df = pd.concat([df, pd.DataFrame([...])], ignore_index=True)`. Do NOT use `df.append()`! edit_cell(row,col,val), add_column(name), delete_column(name), rename_column(old,new), delete_row(idx)

PATTERNS:
A) Aggregation: result=df.groupby('X')['Y'].sum().reset_index(); log_output(result); result_df=df
B) Filter: print_query('Rating>8'); result_df=df
C) Complex filter: mask=(df['A']>5)&(df['B']=='x'); log_output(df[mask]); result_df=df
D) Mutation: df['Col']=df['Col'].str.replace('old','new'); result_df=df

RULES:
- Single quotes in code, pass DataFrames directly to log_output for rendering, handle NaN with dropna/fillna
- String match: .str.contains('x',case=False,na=False)
- Charts: assign to fig, never call fig.show/to_html/to_json
- For structural MUTATIONS (add/drop columns/rows, clean datatypes): Modify `df` directly, return `result_df=df`, and set `table_request.enabled=False`. Do NOT create separate tables for this.
- For FINDING/SEARCHING (e.g. "find all students"): Use `highlight_rows(df[mask].index.tolist())`. Do NOT use `table_request.enabled=True` unless specifically asked to separate it.
- For LISTING/PRINTING (e.g. "list the top 5 students"): If the result has < 10 rows, format the output as a bulleted markdown list in `assistant_reply` (do NOT use tables or df.to_markdown()). If the result has >= 10 rows, do NOT include the text in chat; instead, use `highlight_rows()` to show the result in the main table. Always set `table_request.enabled=False`.
- For EXTRACTIONS/PIVOTS (e.g. "pivot table", "extract to new table"): Create a subset df, pass it via `log_output(subset_df)`, and set `table_request.enabled=True`.
- table_request.title MUST be very short (max 3-5 words), e.g., "Low Involvement".
"""

VIZ_RULES = """
VIZ RULES - template='plotly_dark', no text on bars/slices, limit 10-15 cats, single series showlegend=False

BAR (horizontal for long labels):
data=df.groupby('Cat')['Val'].sum().nlargest(10).reset_index()
fig=px.bar(data,x='Val',y='Cat',orientation='h',template='plotly_dark',title='Top 10')
fig.update_layout(showlegend=False,yaxis={'categoryorder':'total ascending'})

LINE: fig=px.line(trend,x='Year',y='Value',template='plotly_dark',markers=True)

PIE: fig=px.pie(values=counts.values,names=counts.index,template='plotly_dark')
fig.update_traces(textinfo='percent',hoverinfo='label+value+percent')

SCATTER: fig=px.scatter(df,x='X',y='Y',template='plotly_dark',hover_data=['Name'])
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
_MUTATION_KW = {'change', 'update', 'replace', 'rename', 'delete', 'add', 'edit', 'merge', 'set', 'modify', 'remove', 'fix', 'correct', 'convert'}
_TABLE_INTENT_KW = {
    'separate table',
    'another table',
    'new table',
    'extract columns',
    'extract these columns',
    'only these columns',
    'table view',
    'split table',
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
    return any(k in p for k in _TABLE_INTENT_KW)


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]

    # First attempt: strict JSON.
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload
    except json.JSONDecodeError:
        pass

    # Fallback for models that return Python-like dicts with single quotes.
    try:
        repaired = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        payload = ast.literal_eval(repaired)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload
    except Exception as exc:
        raise ValueError(f"Invalid model JSON response: {exc}") from exc


def _invoke_model(*, model_name: str, system_prompt: str, user_message: str) -> tuple[str, dict[str, int]]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if model_name and (model_name.startswith("openai/") or model_name.startswith("gpt-")):
        if model_name.startswith("openai/"):
            model_name = model_name[7:]
            
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
            model=model_name,
        )
        raw = response.choices[0].message.content or ""
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
        return raw, usage

    if model_name and (model_name.startswith("minimax") or model_name.startswith("google/codegemma") or model_name.startswith("meta/")):
        from openai import OpenAI

        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=8192
            )
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
    model = genai.GenerativeModel(model_name or "gemini-2.0-flash")
    response = model.generate_content([system_prompt, user_message])
    raw = response.text or ""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
        usage["completion_tokens"] = response.usage_metadata.candidates_token_count
        usage["total_tokens"] = response.usage_metadata.total_token_count
    return raw, usage


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
) -> dict[str, Any]:
    system_prompt = _build_system_prompt(prompt)
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

    ctx.append("\nReturn valid JSON only.")

    user_message = "\n".join(ctx)
    raw, usage = _invoke_model(model_name=model_name, system_prompt=system_prompt, user_message=user_message)

    payload = _extract_json(raw)
    code = payload.get("code", "")
    reply = payload.get("assistant_reply", "Done.")
    action = payload.get("action_type", "query")
    table_request = payload.get("table_request") or {}
    table_enabled = bool(table_request.get("enabled", False))
    table_title = str(table_request.get("title") or "Query Table").strip()
    if not table_title:
        table_title = "Query Table"

    if _wants_separate_table(prompt) and action in ("query", "combined"):
        table_enabled = True
        if table_title == "Query Table":
            table_title = "Extracted Table"

    if not isinstance(code, str) or not code.strip():
        raise ValueError("No valid code returned")
    if not isinstance(reply, str):
        reply = "Done."
    if action not in ("query", "mutation", "visualization", "combined"):
        action = "query"

    return {
        "code": code,
        "assistant_reply": reply,
        "action_type": action,
        "token_usage": usage,
        "table_request": {"enabled": table_enabled, "title": table_title},
    }


def generate_fix(
    *,
    prompt: str,
    model_name: str,
    columns: list[str],
    dtypes: dict[str, str],
    original_code: str,
    error_message: str,
) -> dict[str, Any]:
    system_prompt = _build_system_prompt(prompt)

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
- To log a computed DataFrame/Series, use: log_output(my_var.to_string())
- Ensure variables are defined before use
- Use single quotes for strings

Return fixed JSON."""

    raw, usage = _invoke_model(model_name=model_name, system_prompt=system_prompt, user_message=user_message)

    payload = _extract_json(raw)
    code = payload.get("code", "")
    reply = payload.get("assistant_reply", "Done.")
    action = payload.get("action_type", "query")
    table_request = payload.get("table_request") or {}
    table_enabled = bool(table_request.get("enabled", False))
    table_title = str(table_request.get("title") or "Query Table").strip()
    if not table_title:
        table_title = "Query Table"

    if _wants_separate_table(prompt) and action in ("query", "combined"):
        table_enabled = True
        if table_title == "Query Table":
            table_title = "Extracted Table"

    if not isinstance(code, str) or not code.strip():
        raise ValueError("No valid code in fix")
    if action not in ("query", "mutation", "visualization", "combined"):
        action = "query"

    return {
        "code": code,
        "assistant_reply": reply,
        "action_type": action,
        "token_usage": usage,
        "table_request": {"enabled": table_enabled, "title": table_title},
    }


def draft_final_reply(
    prompt: str,
    model_name: str,
    query_output: str | None,
    initial_reply: str,
) -> dict[str, Any]:
    if not query_output or len(query_output.strip()) < 10:
        return {"reply": initial_reply, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    api_key = os.getenv("GEMINI_API_KEY")
    output = query_output[:2000] if len(query_output) > 2000 else query_output

    user_message = f"""User asked: {prompt}

Execution output:
{output}

Provide an extremely concise, direct, one-sentence answer based ONLY on this output. 
DO NOT elaborate, DO NOT detail data characteristics, and DO NOT use bullet points unless the user explicitly asked you to explain or elaborate. Focus only on the exact core request.
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
        reply = response.choices[0].message.content or initial_reply
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
        return {"reply": reply, "token_usage": usage}
    else:
        if not api_key:
            return {"reply": initial_reply, "token_usage": usage}
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-2.0-flash")
        try:
            response = model.generate_content([
                "Summarize data results clearly and concisely. Do not output markdown or ASCII tables.",
                user_message
            ])
            reply = response.text or initial_reply
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                usage["total_tokens"] = response.usage_metadata.total_token_count
            return {"reply": reply, "token_usage": usage}
        except Exception:
            return {"reply": initial_reply, "token_usage": usage}
