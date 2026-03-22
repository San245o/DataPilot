from __future__ import annotations

import json
import random
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from schemas import AgentRequest, AgentResponse
from workflow import build_workflow
from agent import generate_code, draft_final_reply
from sandbox import run_sandboxed

load_dotenv()

app = FastAPI(title="Excel Agent Backend", version="3.0.0")
workflow = build_workflow()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/execute", response_model=AgentResponse)
def execute_agent(payload: AgentRequest) -> AgentResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    try:
        final_state = workflow.invoke(
            {
                "prompt": payload.prompt,
                "rows": payload.rows,
                "model": payload.model,
                "history": [m.model_dump() for m in payload.history],
            }
        )

        return AgentResponse(
            rows=final_state.get("result_rows", payload.rows),
            visualization=final_state.get("visualization"),
            query_output=final_state.get("query_output"),
            code=final_state.get("code", ""),
            assistant_reply=final_state.get("assistant_reply", "Done."),
            context_preview={
                "columns": final_state.get("columns", []),
                "sample_rows": final_state.get("sample_rows", []),
            },
            mutation=final_state.get("mutation", False),
            highlight_indices=final_state.get("highlight_indices", []),
            token_usage=final_state.get("token_usage"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _stream_agent_execution(payload: AgentRequest) -> AsyncGenerator[str, None]:
    """Stream the agent execution with step-by-step progress events."""
    
    def emit(event: str, data: dict[str, Any]) -> str:
        return f"data: {json.dumps({'event': event, 'data': data})}\n\n"
    
    try:
        rows = payload.rows
        
        # Step 1: Prepare context
        yield emit("step", {"id": "schema", "status": "active"})
        columns = list(rows[0].keys()) if rows else []
        sample_size = min(5, len(rows))
        sample_rows = random.sample(rows, sample_size) if sample_size > 0 else []
        yield emit("step", {"id": "schema", "status": "done"})
        
        # Step 2: Generate code
        yield emit("step", {"id": "generate", "status": "active"})
        generated = generate_code(
            prompt=payload.prompt,
            model_name=payload.model,
            columns=columns,
            sample_rows=sample_rows,
            history=[m.model_dump() for m in payload.history],
        )
        yield emit("step", {"id": "generate", "status": "done"})
        
        # Step 3: Execute code
        yield emit("step", {"id": "execute", "status": "active"})
        result = run_sandboxed(generated["code"], rows)
        yield emit("step", {"id": "execute", "status": "done"})
        
        # Step 4: Prepare response
        yield emit("step", {"id": "prepare", "status": "active"})
        
        final_output = draft_final_reply(
            prompt=payload.prompt,
            model_name=payload.model,
            query_output=result.query_output,
            initial_reply=generated["assistant_reply"],
        )
        
        final_reply = final_output.get("reply", generated["assistant_reply"])
        gen_usage = generated.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        draft_usage = final_output.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        merged_usage = {
            "prompt_tokens": gen_usage.get("prompt_tokens", 0) + draft_usage.get("prompt_tokens", 0),
            "completion_tokens": gen_usage.get("completion_tokens", 0) + draft_usage.get("completion_tokens", 0),
            "total_tokens": gen_usage.get("total_tokens", 0) + draft_usage.get("total_tokens", 0),
        }
        
        response = AgentResponse(
            rows=result.rows,
            visualization=result.visualization,
            query_output=result.query_output,
            code=generated["code"],
            assistant_reply=final_reply,
            context_preview={
                "columns": columns,
                "sample_rows": sample_rows,
            },
            mutation=result.mutation,
            highlight_indices=result.highlight_indices,
            token_usage=merged_usage,
        )
        yield emit("step", {"id": "prepare", "status": "done"})
        
        # Final result
        yield emit("result", response.model_dump())
        
    except Exception as exc:
        yield emit("error", {"message": str(exc)})


@app.post("/agent/stream")
async def stream_agent(payload: AgentRequest) -> StreamingResponse:
    """Streaming endpoint for agent execution with step-by-step progress."""
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
