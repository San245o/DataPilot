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

ENVIRONMENT (pre-loaded, NO imports allowed):
- df: pandas DataFrame with user data
- pd, np, px (plotly.express), go (plotly.graph_objects)
- MUST set result_df = df at the end of every code block

═══════════════════════════════════════════════════════════════
HELPER FUNCTIONS - READ CAREFULLY:
═══════════════════════════════════════════════════════════════

1. log_output(message: str)
   - Logs ANY text/data to output panel
   - USE THIS to display computed results (DataFrames, Series, values)
   - Example: log_output(my_dataframe.to_string())
   - Example: log_output(f'Average: {df["Price"].mean():.2f}')

2. print_query(filter_expr: str, max_rows=20)
   - ONLY for filtering df with a pandas query expression
   - The argument must be a QUERY STRING like "Age > 30", NOT a variable name
   - Runs df.query(filter_expr) internally
   - Example: print_query('Price > 1000')  ✓
   - Example: print_query('Genre == "Horror"')  ✓
   - WRONG: print_query('my_variable')  ✗ - This will ERROR!

3. print_table(max_rows=20) - Prints current df snapshot

4. highlight_rows(indices: list[int]) - Highlight rows in UI

5. Mutation helpers (these modify df):
   - edit_cell(row_idx, column, value)
   - add_column(name, default=None)
   - delete_column(name)
   - rename_column(old_name, new_name)
   - add_row(row_dict)
   - delete_row(index)

═══════════════════════════════════════════════════════════════
CODE PATTERNS:
═══════════════════════════════════════════════════════════════

PATTERN A - Aggregation/Analysis (use log_output):
```python
# Find top 5 directors by total income
result = df.groupby('Director')['Income'].sum().sort_values(ascending=False).head(5)
log_output('Top 5 Directors by Income:')
log_output(result.to_string())
# Highlight rows for top directors
top_names = result.index.tolist()
mask = df['Director'].isin(top_names)
highlight_rows(df[mask].index.tolist())
result_df = df
```

PATTERN B - Filter with condition (can use print_query):
```python
# Find all movies with rating > 8
print_query('Rating > 8')  # This filters df where Rating > 8
result_df = df
```

PATTERN C - Complex filter (use boolean indexing + log_output):
```python
# Find horror movies from 2020s
mask = (df['Genre'].str.contains('Horror', case=False, na=False)) & (df['Year'] >= 2020)
filtered = df[mask]
log_output(f'Found {len(filtered)} horror movies from 2020s:')
log_output(filtered[['Title', 'Year', 'Rating']].head(20).to_string())
highlight_rows(filtered.index.tolist())
result_df = df
```

PATTERN D - Mutation:
```python
# Replace team name
df['Team'] = df['Team'].str.replace('Old Name', 'New Name', case=False)
mask = df['Team'].str.contains('New Name', case=False, na=False)
highlight_rows(df[mask].index.tolist())
log_output(f'Updated {mask.sum()} rows')
result_df = df
```

═══════════════════════════════════════════════════════════════
RULES:
═══════════════════════════════════════════════════════════════
- NO imports - only use pd, np, px, go
- Use single quotes in code strings
- ALWAYS end with: result_df = df
- Use .to_string() when logging DataFrames/Series
- Handle NaN: use .dropna() or .fillna() as needed
- String matching: .str.contains('pattern', case=False, na=False)
- Never call fig.show(), fig.to_html(), fig.write_html(), fig.to_json(), or offline plotting APIs
- For charts: assign the final figure to variable `fig` and do not print/log the figure object
- If the user explicitly needs a tabular result shown in the UI table area, set table_request.enabled=true and give a short table_request.title
- If table_request.enabled=true, DO NOT render markdown/ASCII tables in assistant_reply; provide a short summary only
- If table_request.enabled=true, ensure code emits tabular data through print_query(...), print_table(...), or log_output(DataFrame/Series)
"""

VIZ_RULES = """
═══════════════════════════════════════════════════════════════
VISUALIZATION RULES - Clean, readable charts:
═══════════════════════════════════════════════════════════════

CRITICAL - Follow these to avoid cluttered charts:
1. Always: template='plotly_dark'
2. NO text on bars/slices - hover only
3. Limit to top 10-15 categories max
4. Single series: showlegend=False
5. Long labels: horizontal bars, automargin

BAR CHART (horizontal for long labels):
```python
data = df.groupby('Category')['Value'].sum().nlargest(10).reset_index()
fig = px.bar(data, x='Value', y='Category', orientation='h', template='plotly_dark',
             title='Top 10 Categories by Value')
fig.update_traces(hovertemplate='%{y}: %{x:,.0f}<extra></extra>')
fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'},
                  yaxis_title='', xaxis_title='Value')
result_df = df
```

LINE CHART (trends over time):
```python
trend = df.groupby('Year')['Value'].mean().reset_index()
fig = px.line(trend, x='Year', y='Value', template='plotly_dark',
              title='Average Value by Year', markers=True)
fig.update_traces(hovertemplate='Year %{x}: %{y:,.0f}<extra></extra>')
result_df = df
```

MULTI-LINE (comparing categories):
```python
# Pivot for multiple series
pivot = df.groupby(['Year', 'Category'])['Value'].sum().reset_index()
fig = px.line(pivot, x='Year', y='Value', color='Category', template='plotly_dark',
              title='Value Trends by Category', markers=True)
fig.update_layout(hovermode='x unified')
result_df = df
```

PIE CHART:
```python
counts = df['Category'].value_counts().head(8)
fig = px.pie(values=counts.values, names=counts.index, template='plotly_dark',
             title='Distribution by Category')
fig.update_traces(textinfo='percent', hoverinfo='label+value+percent')
result_df = df
```

SCATTER PLOT:
```python
fig = px.scatter(df, x='Budget', y='Revenue', template='plotly_dark',
                 hover_data=['Title'], title='Budget vs Revenue')
fig.update_traces(hovertemplate='%{customdata[0]}<br>Budget: %{x:$,.0f}<br>Revenue: %{y:$,.0f}<extra></extra>')
result_df = df
```
"""

MERGE_RULES = """
MERGE/JOIN NOTE: Only one DataFrame (df) available. For self-joins:
```python
subset = df[df['Type'] == 'A']
merged = df.merge(subset[['Key', 'Value']], on='Key', suffixes=('', '_A'))
```
"""

STATS_RULES = """
STATISTICS (no scipy, use numpy/pandas):
- Correlation: df['a'].corr(df['b']) or df.corr()
- Stats: df.describe(), .mean(), .std(), .quantile([0.25, 0.5, 0.75])
- Outliers: Q1, Q3 = col.quantile([0.25, 0.75]); IQR = Q3 - Q1; outliers = col[(col < Q1-1.5*IQR) | (col > Q3+1.5*IQR)]
- Regression: coeffs = np.polyfit(x, y, 1); trend_line = np.poly1d(coeffs)

CORRELATION GRAPH RULES (IMPORTANT):
- Always use numeric columns only: num_df = df.select_dtypes(include=[np.number]).copy()
- Drop columns with <=1 non-null value before correlation
- If fewer than 2 numeric columns remain, do NOT create a chart; log_output a clear message instead
- For full correlation matrix, prefer heatmap with zmin=-1, zmax=1 and diverging color scale
- For two-column correlation request, use scatter with OLS trendline and include Pearson r in title

CORRELATION MATRIX TEMPLATE:
```python
num_df = df.select_dtypes(include=[np.number]).copy()
num_df = num_df.dropna(axis=1, thresh=2)
if num_df.shape[1] < 2:
    log_output('Need at least 2 numeric columns for correlation analysis.')
    result_df = df
else:
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        template='plotly_dark',
        title='Correlation Matrix (Numeric Columns)'
    )
    fig.update_layout(xaxis_title='', yaxis_title='')
    result_df = df
```

PAIR CORRELATION TEMPLATE:
```python
pair = df[['col_x', 'col_y']].dropna()
if len(pair) < 3:
    log_output('Not enough valid rows for correlation scatter.')
    result_df = df
else:
    r = pair['col_x'].corr(pair['col_y'])
    fig = px.scatter(
        pair,
        x='col_x',
        y='col_y',
        trendline='ols',
        template='plotly_dark',
        title=f'col_x vs col_y (r={r:.3f})'
    )
    fig.update_traces(mode='markers', marker={'size': 7, 'opacity': 0.8})
    result_df = df
```
"""

TIME_RULES = """
TIME SERIES:
- Parse: df['date'] = pd.to_datetime(df['date'], errors='coerce')
- Resample: df.set_index('date').resample('M').mean()
- Group: df.groupby(df['date'].dt.year).sum()
- Rolling: df['col'].rolling(window=7).mean()
"""

_VIZ_KW = {'chart', 'plot', 'graph', 'pie', 'bar', 'scatter', 'visual', 'histogram', 'heatmap', 'line', 'bubble', 'distribution', 'trend', 'compare'}
_MERGE_KW = {'merge', 'join', 'combine tables', 'match records', 'link'}
_STATS_KW = {'correlation', 'regression', 'distribution', 'outlier', 'significance', 'p-value', 'statistical', 'std', 'variance'}
_TIME_KW = {'over time', 'by year', 'by month', 'by date', 'forecast', 'time series', 'yearly', 'monthly', 'daily', 'trend'}
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
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise RuntimeError("GITHUB_TOKEN not set")

        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(github_token),
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

    if model_name and model_name.startswith("minimax"):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*type is unknown and inference may fail.*")
            warnings.filterwarnings("ignore", message=".*minimaxai/minimax-m2.5.*")
            client = ChatNVIDIA(
                model=model_name,
                api_key=nvidia_api_key,
                temperature=0,
                max_tokens=8192,
            )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        raw = ""
        last_chunk = None
        for chunk in client.stream(messages):
            raw += chunk.content
            last_chunk = chunk

        if last_chunk is not None and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            usage["prompt_tokens"] = last_chunk.usage_metadata.get("input_tokens", 0)
            usage["completion_tokens"] = last_chunk.usage_metadata.get("output_tokens", 0)
            usage["total_tokens"] = last_chunk.usage_metadata.get("total_tokens", 0)
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


def _trim_history(history: list[dict[str, str]], max_turns: int = 4) -> list[dict[str, str]]:
    h = history[-max_turns:] if len(history) > max_turns else list(history)
    result = []
    for turn in h:
        c = turn.get("content", "")
        if turn.get("role") == "assistant":
            result.append({"role": "assistant", "content": c[:400] + "..." if len(c) > 400 else c})
        else:
            result.append({"role": turn.get("role", "user"), "content": c[:200] if len(c) > 200 else c})
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
    ctx = [f"User request: {prompt}"]
    ctx.append(f"Columns: {json.dumps(columns)}")
    ctx.append(f"Types: {json.dumps(dtypes)}")

    # Nulls if any
    has_nulls = {k: v for k, v in nulls.items() if v > 0}
    if has_nulls:
        ctx.append(f"Nulls: {json.dumps(has_nulls)}")

    # Value ranges and categories for non-mutations
    if not is_mutation:
        if value_ranges:
            relevant = {k: v for k, v in value_ranges.items() if k in mentioned}
            if len(relevant) < 6:
                for k, v in list(value_ranges.items())[:8]:
                    if k not in relevant:
                        relevant[k] = v
            if relevant:
                ctx.append(f"Numeric ranges: {json.dumps(relevant)}")

        if top_categories:
            relevant = {k: v for k, v in top_categories.items() if k in mentioned}
            if len(relevant) < 6:
                for k, v in list(top_categories.items())[:8]:
                    if k not in relevant:
                        relevant[k] = v
            if relevant:
                ctx.append(f"Categories: {json.dumps(relevant)}")

    if sample_rows:
        ctx.append(f"Sample rows: {json.dumps(sample_rows)}")

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

Provide a clear, helpful response based ONLY on this output. Format nicely with bullet points for lists.
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
