## Excel Agent Dashboard

This workspace contains:

1. A Next.js dashboard frontend with a visualization board.
2. A FastAPI backend that uses Gemini to generate pandas + Plotly code.
3. A sandbox runner for pandas code execution with timeout and AST safety checks.

## Run Frontend

From the project root:

```bash
npm install
npm run dev
```

Frontend URL: http://localhost:3000

## Run Backend (FastAPI)

Backend is now a separate project at [../excel-agent-backend](../excel-agent-backend).

Install backend dependencies with system Python:

```bash
cd ../excel-agent-backend
python -m pip install -r requirements.txt
```

Set Gemini key (PowerShell):

```powershell
$env:GEMINI_API_KEY="<your-gemini-api-key>"
```

Run API server:

```bash
uvicorn main:app --app-dir d:/Data_Pilot/excel-agent-backend --reload --port 8000
```

Backend URL: http://localhost:8000

## How It Works

1. Upload CSV/XLSX in the left panel.
2. Enter a prompt in the chat box and click Ask Agent.
3. LangGraph workflow prepares context (columns + random 5 sample rows).
4. Backend asks Gemini for pandas/Plotly code.
5. Code executes in a sandbox process.
5. If Plotly figure is produced, the visualization board renders it (2D/3D).
6. If no figure is returned, the board shows the full spreadsheet table.
