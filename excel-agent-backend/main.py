from __future__ import annotations

from collections import OrderedDict
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas import AgentRequest, AgentResponse, DatasetRegisterRequest, DatasetRegisterResponse
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
_dataset_store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
MAX_CACHED_DATASETS = 20


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def _cache_dataset(rows: list[dict[str, Any]]) -> str:
    dataset_id = uuid4().hex
    _dataset_store[dataset_id] = rows
    _dataset_store.move_to_end(dataset_id)

    while len(_dataset_store) > MAX_CACHED_DATASETS:
        _dataset_store.popitem(last=False)

    return dataset_id


def _resolve_rows(payload: AgentRequest) -> list[dict[str, Any]]:
    if payload.dataset_id:
        cached_rows = _dataset_store.get(payload.dataset_id)
        if cached_rows is None:
            raise HTTPException(status_code=404, detail="Dataset not found. Upload again.")

        # Mark as recently used in the LRU cache.
        _dataset_store.move_to_end(payload.dataset_id)
        return cached_rows

    return payload.rows


def _safe_get(state: dict | None, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    val = state.get(key)
    return val if val is not None else default


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/dataset/register", response_model=DatasetRegisterResponse)
def register_dataset(payload: DatasetRegisterRequest) -> DatasetRegisterResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    dataset_id = _cache_dataset(payload.rows)
    column_count = len(payload.rows[0]) if payload.rows else 0
    return DatasetRegisterResponse(
        dataset_id=dataset_id,
        row_count=len(payload.rows),
        column_count=column_count,
    )


@app.post("/agent/execute", response_model=AgentResponse)
def execute_agent(payload: AgentRequest) -> AgentResponse:
    rows = _resolve_rows(payload)
    if not rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    try:
        workflow = get_workflow()
        final_state = workflow.invoke({
            "prompt": payload.prompt,
            "rows": rows,
            "model": payload.model,
            "history": [m.model_dump() for m in payload.history],
        })

        if not final_state:
            raise HTTPException(status_code=500, detail="No result from agent")

        return AgentResponse(
            rows=_safe_get(final_state, "result_rows", rows),
            visualization=_safe_get(final_state, "visualization"),
            query_output=_safe_get(final_state, "query_output"),
            query_tables=_safe_get(final_state, "query_tables", []),
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
