# DataPilot Guide

## 1. What This Project Is

DataPilot is an AI-assisted spreadsheet analytics application with:

- A FastAPI backend that generates and executes pandas and Plotly code.
- A Next.js dashboard frontend for file upload, chat-driven analysis, table exploration, and charting.
- A sandbox runtime that validates and executes generated Python safely.

The app is designed to let users ask natural-language questions like:

- "Show sales by region as a bar chart"
- "Find rows where category is Electronics"
- "Create a pivot table by product and month"

## 2. Repository Layout

- excel-agent-backend
- excel-agent-dashboard
- README.md
- guide.md

### Backend (excel-agent-backend)

- main.py: API server, routes, dataset cache, workflow invocation.
- workflow.py: LangGraph flow for context prep, code generation, sandbox execution, and error handling.
- agent.py: Prompt building, model invocation, structured response parsing.
- sandbox.py: AST validation plus controlled execution environment for pandas and Plotly code.
- schemas.py: Pydantic request and response models.
- requirements.txt: Python dependencies.

### Frontend (excel-agent-dashboard)

- app/chart-viewer/page.tsx: New-tab chart viewer page.
- components/dashboard/agent-dashboard.tsx: Main interactive dashboard UI.
- components/charts/plotly-board.tsx: Plot rendering wrapper.
- app/globals.css: Styling and theme variables.
- tsconfig.json: TypeScript config.

## 3. Runtime Architecture

### Request lifecycle

1. User uploads CSV/XLSX in the dashboard.
2. User asks a prompt in chat.
3. Frontend sends rows + prompt + model + conversation history to backend endpoint.
4. Backend workflow prepares context metadata (columns, dtypes, nulls, sampled rows).
5. agent.py requests generated code from selected model.
6. sandbox.py validates and executes generated code against DataFrame.
7. Backend returns:
- updated rows (for mutations)
- query output text
- optional query tables
- optional Plotly visualization
- generated code and assistant reply
8. Frontend updates table, chat, and chart canvas.

### New-tab chart lifecycle

1. Dashboard serializes chart payload into localStorage with a unique key.
2. Dashboard opens chart-viewer route and passes chart key as query parameter.
3. chart-viewer reads payload from localStorage and renders Plotly chart.
4. Theme state is initialized from payload and can be toggled in-page.

## 4. Backend Details

## 4.1 API Endpoints

- GET /health
- POST /dataset/register
- POST /agent/execute

### POST /dataset/register

Stores uploaded rows in an in-memory LRU dataset cache and returns dataset metadata.

### POST /agent/execute

Executes the LangGraph workflow and returns a structured AgentResponse used by the UI.

## 4.2 Workflow (workflow.py)

Main nodes:

- prepare_context
- generate_code
- execute_code
- fix_code (single retry)
- compose_assistant_reply
- format_error

Behavior highlights:

- Prevents silent row deletion unless prompt explicitly asks to remove/drop rows.
- Distinguishes mutation/query/visualization actions.
- Returns compact but informative assistant reply.

## 4.3 Model Layer (agent.py)

Supports multiple model providers by model name pattern:

- OpenAI/GitHub Models path for gpt-* and openai/*
- NVIDIA path for minimax* and meta/*
- Gemini path for Gemini model names

Response parsing strategy:

- Prefer strict JSON extraction.
- Repair common malformed JSON formatting.
- Fallback to Python-literal parsing when needed.
- If code is still missing, fallback to a safe deterministic snippet:
- print_table(max_rows=10)
- result_df=df

This keeps UX functional instead of failing hard.

## 4.4 Sandbox (sandbox.py)

Key controls:

- AST parse and validation before execution.
- Blocks imports and dangerous builtins/dunder access.
- Restricts helper functions exposed to generated code.
- Runs in a timed thread with timeout safeguards.

Exposed helper APIs include:

- print_query
- print_table
- log_output
- highlight_rows
- add_column / delete_column / rename_column
- add_row / delete_row / edit_cell
- text_equals / text_contains / to_numeric_clean

## 5. Frontend Details

## 5.1 Dashboard UI (agent-dashboard.tsx)

Main panels:

- Sidebar: upload, context, theme toggle, model selector.
- Data table: virtualized scrolling and row highlighting.
- Canvas: floating chart widgets with drag, resize, close, and open-in-new-tab.
- Chat: prompt input, slash commands, generated code viewer.

### Important UX behaviors

- Theme toggle appears above model selector.
- Send button is centered in the input row.
- Popup-blocked scenarios for chart opening fall back to same-tab navigation.

## 5.2 Chart Viewer (app/chart-viewer/page.tsx)

- Theme-aware rendering with inline light/dark toggle.
- Styled header and modern containerized plot area.
- Defensive handling for missing or invalid chart payload.

## 6. Configuration

## 6.1 Backend env vars

- GEMINI_API_KEY: Required for Gemini requests.
- GITHUB_TOKEN: Required for GitHub Models/OpenAI Azure inference path.
- NVIDIA_API_KEY: Required for NVIDIA model path.

## 6.2 Frontend env vars

- NEXT_PUBLIC_API_BASE_URL: Optional direct backend base URL.
- BACKEND_URL: Optional server-side rewrite destination.

## 6.3 TypeScript config note

tsconfig no longer defines baseUrl, which avoids the deprecation warning in newer TypeScript toolchains.

Path aliases such as @/* continue to work through the existing Next.js/TypeScript setup in this project.

## 7. Local Development

### Backend

1. cd excel-agent-backend
2. python -m pip install -r requirements.txt
3. set environment variables for API keys
4. uvicorn main:app --reload --port 8000

### Frontend

1. cd excel-agent-dashboard
2. pnpm install
3. pnpm dev
4. open http://localhost:3000

## 8. Troubleshooting

### Issue: Model returns malformed response

Symptoms:

- errors about invalid JSON
- missing code in generated response

Checks:

- verify API key for selected provider
- switch to another supported model
- inspect generated code in chat "View code"

### Issue: Chart new-tab does not open

Behavior:

- app now attempts new tab first, and if blocked, navigates in same tab.

### Issue: Sandbox syntax/runtime errors

- Copy the generated code from chat and inspect line-by-line.
- Check column names and query logic for mismatches.
- Rephrase prompt with explicit field names and operation intent.

## 9. Extending the Project

### Add a new model

1. Extend provider/model routing in agent.py.
2. Add model option in dashboard select list.
3. Validate structured response format and code generation consistency.

### Add new helper function for generated code

1. Implement helper in sandbox.py.
2. Add helper to env dictionary.
3. Document helper in SYSTEM_PROMPT_BASE for model awareness.

### Add new UI panel/widget

1. Add component under components/dashboard or components/charts.
2. Wire state into AgentDashboard.
3. Preserve dark/light theme compatibility.

## 10. Operational Notes

- Dataset cache is in-memory and capped (LRU). Restart clears cache.
- Large datasets can increase generation and execution time.
- Generated code is always untrusted; sandbox validation is mandatory.

---

If you want, a follow-up can generate a shorter "runbook" version of this guide for teammates (quick start + common fixes only).
