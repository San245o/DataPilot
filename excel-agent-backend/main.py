from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from schemas import AgentRequest, AgentResponse
from workflow import build_workflow

load_dotenv()

app = FastAPI(title="Excel Agent Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build workflow once at startup
_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def _safe_get(state: dict | None, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    val = state.get(key)
    return val if val is not None else default


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/execute", response_model=AgentResponse)
def execute_agent(payload: AgentRequest) -> AgentResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    try:
        workflow = get_workflow()
        final_state = workflow.invoke({
            "prompt": payload.prompt,
            "rows": payload.rows,
            "model": payload.model,
            "history": [m.model_dump() for m in payload.history],
        })

        if not final_state:
            raise HTTPException(status_code=500, detail="No result from agent")

        return AgentResponse(
            rows=_safe_get(final_state, "result_rows", payload.rows),
            visualization=_safe_get(final_state, "visualization"),
            query_output=_safe_get(final_state, "query_output"),
            code=_safe_get(final_state, "code", ""),
            assistant_reply=_safe_get(final_state, "assistant_reply", "Done."),
            context_preview={
                "columns": _safe_get(final_state, "columns", []),
                "sample_rows": _safe_get(final_state, "sample_rows", []),
            },
            mutation=_safe_get(final_state, "mutation", False),
            highlight_indices=_safe_get(final_state, "highlight_indices", []),
            token_usage=_safe_get(final_state, "token_usage"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Agent error") from exc


async def _stream_agent_execution(payload: AgentRequest) -> AsyncGenerator[str, None]:
    """Stream agent execution with step-by-step progress."""

    def emit(event: str, data: dict[str, Any]) -> str:
        return f"data: {json.dumps({'event': event, 'data': data})}\n\n"

    try:
        yield emit("step", {"id": "schema", "status": "active"})
        workflow = get_workflow()
        yield emit("step", {"id": "schema", "status": "done"})

        yield emit("step", {"id": "generate", "status": "active"})
        final_state = workflow.invoke({
            "prompt": payload.prompt,
            "rows": payload.rows,
            "model": payload.model,
            "history": [m.model_dump() for m in payload.history],
        })
        yield emit("step", {"id": "generate", "status": "done"})

        yield emit("step", {"id": "execute", "status": "active"})
        yield emit("step", {"id": "execute", "status": "done"})

        yield emit("step", {"id": "prepare", "status": "active"})

        if not final_state:
            yield emit("error", {"message": "No result from agent"})
            return

        response = AgentResponse(
            rows=_safe_get(final_state, "result_rows", payload.rows),
            visualization=_safe_get(final_state, "visualization"),
            query_output=_safe_get(final_state, "query_output"),
            code=_safe_get(final_state, "code", ""),
            assistant_reply=_safe_get(final_state, "assistant_reply", "Done."),
            context_preview={
                "columns": _safe_get(final_state, "columns", []),
                "sample_rows": _safe_get(final_state, "sample_rows", []),
            },
            mutation=_safe_get(final_state, "mutation", False),
            highlight_indices=_safe_get(final_state, "highlight_indices", []),
            token_usage=_safe_get(final_state, "token_usage"),
        )

        yield emit("step", {"id": "prepare", "status": "done"})
        yield emit("result", response.model_dump())

    except Exception as exc:
        yield emit("error", {"message": str(exc) or "Agent error"})


@app.post("/agent/stream")
async def stream_agent(payload: AgentRequest) -> StreamingResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    return StreamingResponse(
        _stream_agent_execution(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
