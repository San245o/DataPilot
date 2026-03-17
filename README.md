# Data Pilot

An AI-powered Excel/data analytics tool with a Next.js dashboard frontend and FastAPI + LangGraph + Google Gemini AI backend.

## Features

- **Data Table**: Upload and view CSV/XLSX files with sticky headers and smooth scrolling
- **AI Chat**: Natural language queries to analyze, transform, and visualize data
- **Visualization Canvas**: Interactive Plotly charts with drag-and-drop positioning
- **Dark/Light Mode**: Full theme support
- **Thinking Mode**: Step-by-step streaming progress for AI operations
- **Code Viewer**: See the generated Python code for each AI response

## Project Structure

```
Data_Pilot/
├── excel-agent-dashboard/    # Next.js frontend
│   ├── app/                  # Next.js app router
│   ├── components/           # React components
│   │   ├── dashboard/        # Main dashboard component
│   │   ├── charts/           # Plotly chart components
│   │   └── ui/               # shadcn/ui components
│   └── ...
│
└── excel-agent-backend/      # FastAPI backend
    ├── main.py               # API endpoints
    ├── agent.py              # Gemini AI code generation
    ├── workflow.py           # LangGraph workflow
    ├── sandbox.py            # Safe code execution
    └── schemas.py            # Pydantic models
```

## Setup

### Backend

1. Navigate to the backend directory:
   ```bash
   cd excel-agent-backend
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd excel-agent-dashboard
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. **Upload Data**: Click "Upload CSV / XLSX" in the sidebar to load your data
2. **Ask Questions**: Type natural language queries in the chat panel
   - "Show me a bar chart of sales by region"
   - "List all products with price > 100"
   - "Add a new column called 'profit' that is revenue minus cost"
3. **Interact with Charts**: Drag charts around the canvas, click to bring to front
4. **Toggle Thinking Mode**: Click the sparkles button to see step-by-step AI progress

## Models

The application uses Google Gemini models:
- `gemini-3-flash-preview` (default) - Fast and capable
- `gemini-3.1-flash-lite-preview` - Lighter, faster responses

## Tech Stack

### Frontend
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui
- Plotly.js

### Backend
- FastAPI
- LangGraph
- Google Generative AI (Gemini)
- Pandas
- Plotly

## License

MIT
