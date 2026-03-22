from __future__ import annotations

import json
import os
import re
from typing import Any

import google.generativeai as genai

SYSTEM_TEMPLATE = """
You are a data analysis agent. You work with pandas DataFrames and plotly for visualization.

Return ONLY strict JSON with this exact schema (no markdown fences, no extra text):
{
  "code": "python code as a string",
  "assistant_reply": "your reply to the user in markdown format"
}

ENVIRONMENT (pre-loaded, do NOT import anything):
- `df` : pandas DataFrame with the user's data
- `pd`, `np`, `px` (plotly.express), `go` (plotly.graph_objects)
- `result_df` : you MUST assign this at the end. For read-only queries, set `result_df = df` (unchanged).

QUERY vs MUTATION:
- READ-ONLY queries (list, show, filter, count, describe, find): Do NOT modify df.
  Just filter/query and format results. Set `result_df = df` (original unchanged).
  CRITICAL: Whenever you are asked to "find" or "list" rows, you MUST ALWAYS log the output using `print_query` or `print_table` AND you MUST call `highlight_rows()` with the indices of those found rows.
- MUTATIONS (add/delete/edit column/row, rename, transform): Use the helper functions.
  These automatically update df.

ALWAYS HIGHLIGHT RELEVANT ROWS:
- After ANY filtering, querying, or finding matching rows, ALWAYS call `highlight_rows(indices)`.
- For combined prompts (e.g. "list horror movies AND change rating of X to Y"):
  1. First find and highlight the relevant rows
  2. Then perform the mutation
  3. Call highlight_rows() with the indices of the relevant/matched rows
- Example:
  ```
  horror = df[df['genre'].str.contains('Horror', case=False, na=False)]
  highlight_rows(horror.index.tolist())
  # Then do mutation...
  ```

FORMATTING assistant_reply:
- For listing/query results, format them as bullet points in your assistant_reply.
  Example: "Found 3 movies with rating > 9:\\n- The Shawshank Redemption (9.3)\\n- The Godfather (9.2)\\n- The Dark Knight (9.0)"
- Keep it concise. Max ~20 items in bullet lists, add "...and N more" if truncated.
- For charts, briefly describe what was generated.
- For mutations, confirm what changed.

VISUALIZATION (IMPORTANT - CLEAN, UNCLUTTERED CHARTS):
- For chart requests, assign the plotly figure to `fig`.
- Use dark-friendly colors: use template='plotly_dark' for dark theme compatibility.
- Use `px` for simple charts, `go.Figure` for complex ones.

CRITICAL CHART STYLING RULES (follow these to avoid cluttered charts):
1. NEVER show text labels directly on bars/slices/points - use hover instead
2. For bar charts: use `text=None` or omit text parameter, show values on hover only
3. For pie charts: use `textinfo='none'` or `textinfo='percent'` (not labels), use `hoverinfo='label+value+percent'`
4. For scatter plots: don't show point labels, use `hover_name` and `hover_data` for details
5. Long axis labels: use `fig.update_layout(yaxis_tickangle=0)` and consider abbreviating or using `fig.update_yaxes(ticktext=..., tickvals=...)`
6. Always set `fig.update_layout(showlegend=False)` for single-series charts
7. For horizontal bar charts with long labels, use:
   - `fig.update_layout(margin=dict(l=10), yaxis=dict(automargin=True))`
8. Use `hovertemplate` for custom hover formatting when needed
9. Remove unnecessary gridlines: `fig.update_xaxes(showgrid=False)` or `fig.update_yaxes(showgrid=False)`

Example for clean bar chart:
```python
fig = px.bar(data, x='value', y='category', orientation='h', template='plotly_dark')
fig.update_traces(textposition='none', hovertemplate='%{y}: %{x}<extra></extra>')
fig.update_layout(showlegend=False, yaxis=dict(automargin=True))
```

Example for clean pie chart:
```python
fig = px.pie(data, values='count', names='category', template='plotly_dark')
fig.update_traces(textinfo='percent', hoverinfo='label+value+percent')
```

- ALWAYS set `result_df = df` after creating a chart (charts don't mutate data).

Available helper functions:
- `highlight_rows(indices: list[int])` - highlight rows in the UI table
- `print_query(query_expr, max_rows=20)` - run df.query() and log output
- `print_table(max_rows=20)` - log current df snapshot
- `log_output(message)` - log text to output
- `add_column(name, default=None)` -> DataFrame (mutates)
- `delete_column(name)` -> DataFrame (mutates)
- `rename_column(old_name, new_name)` -> DataFrame (mutates)
- `add_row(row_data: dict)` -> DataFrame (mutates)
- `delete_row(index: int)` -> DataFrame (mutates)
- `edit_cell(row_index, column, value)` -> DataFrame (mutates)
- `table_to_csv(max_rows=200)` -> str

CODE RULES:
- Do NOT import anything (no `import ...` or `from ... import ...`). Only use the libraries provided in the pre-loaded ENVIRONMENT list. Many sandbox issues are created if you try to import other libraries.
- Use single quotes for strings in code to avoid JSON issues.
- ALWAYS set `result_df = df` at the end (even for read-only queries).
  If helpers mutated df, still set `result_df = df` since df reference is updated.
- Handle missing values with .dropna() or .fillna() where needed.
- For filtering, use boolean indexing: `df[df['col'] > val]`
- For string matching, use `.str.contains('pattern', case=False, na=False)`
"""


def _extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Gemini output must be a JSON object")
    return payload


def generate_code(
    *,
    prompt: str,
    model_name: str,
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")

    compact_history = history[-8:]
    user_message = (
        f"User prompt: {prompt}\n"
        f"Columns: {json.dumps(columns, ensure_ascii=True)}\n"
        f"Random 5 sample rows: {json.dumps(sample_rows, ensure_ascii=True)}\n"
        f"Recent chat history: {json.dumps(compact_history, ensure_ascii=True)}\n"
        "Return strict JSON only. Format list results as bullet points in assistant_reply."
    )

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if model_name and (model_name.startswith("openai/") or model_name.startswith("gpt-")):
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise RuntimeError("GITHUB_TOKEN is not set")
            
        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(github_token),
        )
        
        response = client.complete(
            messages=[
                SystemMessage(SYSTEM_TEMPLATE),
                UserMessage(user_message),
            ],
            temperature=0, # Using low temp for code gen
            model=model_name
        )
        
        raw = response.choices[0].message.content or ""
        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
    else:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-3-flash-preview")

        response = model.generate_content([SYSTEM_TEMPLATE, user_message])
        raw = response.text or ""
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
            usage["completion_tokens"] = response.usage_metadata.candidates_token_count
            usage["total_tokens"] = response.usage_metadata.total_token_count

    payload = _extract_json(raw)

    code = payload.get("code", "")
    assistant_reply = payload.get("assistant_reply", "Done.")

    if not isinstance(code, str) or not code.strip():
        raise ValueError("Gemini did not return valid code")
    if not isinstance(assistant_reply, str):
        assistant_reply = "Done."

    return {"code": code, "assistant_reply": assistant_reply, "token_usage": usage}

def draft_final_reply(
    prompt: str,
    model_name: str,
    query_output: str | None,
    initial_reply: str,
) -> dict[str, Any]:
    """Drafts a true final response based on the sandbox execution output."""
    if not query_output or not query_output.strip():
        return {"reply": initial_reply, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    api_key = os.getenv("GEMINI_API_KEY")
    user_message = (
        f"User prompt: '{prompt}'\n"
        f"The python sandbox execution output was:\n"
        f"---------\n{query_output}\n---------\n\n"
        f"Original guessed reply: '{initial_reply}'\n\n"
        "Please provide the final helpful answer to the user based STRICTLY on the code execution output shown above. "
        "Do not make up facts or use external knowledge. If the output says something unexpected (like 'Inf' or 'NaN'), explicitly say it. "
        "Do not explain your internal process. Provide only the final response text without JSON wrapping."
    )

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
                SystemMessage("You analyze execution outputs and answer precisely."),
                UserMessage(user_message),
            ],
            temperature=0,
            model=model_name
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

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-3-flash-preview")

        try:
            response = model.generate_content([
                "You analyze execution outputs and answer the user precisely.",
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
