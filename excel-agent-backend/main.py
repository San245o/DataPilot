from __future__ import annotations

from collections import OrderedDict
import json
import logging
from queue import Queue
import threading
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from agent import supports_thinking_model
from schemas import AgentRequest, AgentResponse, DatasetRegisterRequest, DatasetRegisterResponse
from thinking import run_thinking_agent
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

logger = logging.getLogger("excel-agent-backend")

# Build workflow once at startup
_workflow = None
_dataset_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
MAX_CACHED_DATASETS = 100


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def _cache_dataset(rows: list[dict[str, Any]], name: str | None = None) -> str:
    dataset_id = uuid4().hex
    _dataset_store[dataset_id] = {
        "rows": rows,
        "name": name or "Dataset",
        "kind": "uploaded",
        "source_dataset_ids": [],
        "modified": False,
    }
    _dataset_store.move_to_end(dataset_id)

    while len(_dataset_store) > MAX_CACHED_DATASETS:
        _dataset_store.popitem(last=False)

    return dataset_id


def _upsert_cached_dataset(
    dataset_id: str,
    *,
    rows: list[dict[str, Any]],
    name: str,
    kind: str = "uploaded",
    source_dataset_ids: list[str] | None = None,
    modified: bool = False,
) -> None:
    _dataset_store[dataset_id] = {
        "rows": rows,
        "name": name,
        "kind": kind,
        "source_dataset_ids": source_dataset_ids or [],
        "modified": modified,
    }
    _dataset_store.move_to_end(dataset_id)

    while len(_dataset_store) > MAX_CACHED_DATASETS:
        _dataset_store.popitem(last=False)


def _get_cached_dataset(dataset_id: str) -> dict[str, Any]:
    cached = _dataset_store.get(dataset_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}. Upload again.")
    _dataset_store.move_to_end(dataset_id)
    return cached


def _resolve_rows(payload: AgentRequest) -> list[dict[str, Any]]:
    dataset_id = payload.active_dataset_id or payload.dataset_id
    if dataset_id:
        cached = _get_cached_dataset(dataset_id)
        return list(cached.get("rows") or [])

    return payload.rows


def _resolve_selected_datasets(payload: AgentRequest) -> dict[str, dict[str, Any]]:
    selected_ids: list[str] = []
    for dataset_id in [payload.active_dataset_id or payload.dataset_id, *payload.selected_dataset_ids]:
        if dataset_id and dataset_id not in selected_ids:
            selected_ids.append(dataset_id)

    resolved: dict[str, dict[str, Any]] = {}
    for dataset_id in selected_ids:
        cached = _get_cached_dataset(dataset_id)
        resolved[dataset_id] = {
            **cached,
            "name": payload.dataset_names.get(dataset_id) or cached.get("name") or dataset_id,
        }
    return resolved


def _safe_get(state: dict | None, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    val = state.get(key)
    return val if val is not None else default


def _normalize_highlighted_columns(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            column = item.get("column")
            if isinstance(column, str) and column:
                normalized.append({"column": column})
        elif isinstance(item, str) and item:
            normalized.append({"column": item})
    return normalized


def _build_agent_response(payload: dict[str, Any], default_rows: list[dict[str, Any]]) -> AgentResponse:
    return AgentResponse(
        rows=_safe_get(payload, "result_rows", default_rows),
        active_dataset_id=_safe_get(payload, "active_dataset_id"),
        updated_datasets=_safe_get(payload, "updated_datasets", []),
        created_datasets=_safe_get(payload, "created_datasets", []),
        visualization=_safe_get(payload, "visualization"),
        query_output=_safe_get(payload, "query_output"),
        query_tables=_safe_get(payload, "query_tables", []),
        code=_safe_get(payload, "code", ""),
        assistant_reply=_safe_get(payload, "assistant_reply", "Done."),
        context_preview=_safe_get(payload, "context_preview", {"columns": [], "sample_rows": []}),
        mutation=_safe_get(payload, "mutation", False),
        highlight_indices=_safe_get(payload, "highlight_indices", []),
        highlighted_columns=_normalize_highlighted_columns(_safe_get(payload, "highlighted_columns", [])),
        thinking_trace=_safe_get(payload, "thinking_trace", []),
        token_usage=_safe_get(payload, "token_usage"),
    )


def _persist_response_datasets(response_payload: dict[str, Any], dataset_names: dict[str, str] | None = None) -> None:
    dataset_names = dataset_names or {}
    for dataset in response_payload.get("updated_datasets") or []:
        dataset_id = dataset.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id:
            _upsert_cached_dataset(
                dataset_id,
                rows=dataset.get("rows") or [],
                name=dataset.get("name") or dataset_names.get(dataset_id) or dataset_id,
                kind=dataset.get("kind") or "uploaded",
                source_dataset_ids=dataset.get("source_dataset_ids") or [],
                modified=bool(dataset.get("modified", True)),
            )

    for dataset in response_payload.get("created_datasets") or []:
        dataset_id = dataset.get("dataset_id") or uuid4().hex
        dataset["dataset_id"] = dataset_id
        _upsert_cached_dataset(
            dataset_id,
            rows=dataset.get("rows") or [],
            name=dataset.get("name") or "Derived Dataset",
            kind="derived",
            source_dataset_ids=dataset.get("source_dataset_ids") or [],
            modified=True,
        )


def _require_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Model is required. The backend will not silently fall back to Gemini 3.1 Flash Lite.")
    return normalized


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/dataset/register", response_model=DatasetRegisterResponse)
def register_dataset(payload: DatasetRegisterRequest) -> DatasetRegisterResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    dataset_id = _cache_dataset(payload.rows, payload.name)
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
    if payload.thinking_mode:
        raise HTTPException(status_code=400, detail="Use /agent/think for thinking mode requests.")
    model_name = _require_model_name(payload.model)
    logger.info("agent.execute model=%s thinking=%s rows=%s", model_name, payload.thinking_mode, len(rows))

    try:
        workflow = get_workflow()
        selected_datasets = _resolve_selected_datasets(payload)
        final_state = workflow.invoke({
            "prompt": payload.prompt,
            "rows": rows,
            "active_dataset_id": payload.active_dataset_id or payload.dataset_id,
            "selected_dataset_ids": payload.selected_dataset_ids,
            "dataset_names": payload.dataset_names,
            "datasets": selected_datasets,
            "model": model_name,
            "history": [m.model_dump() for m in payload.history],
            "selection_context": payload.selection_context,
        })

        if not final_state:
            raise HTTPException(status_code=500, detail="No result from agent")

        response_payload = {
            **final_state,
            "context_preview": {
                "columns": _safe_get(final_state, "columns", []),
                "sample_rows": _safe_get(final_state, "sample_rows", []),
            },
            "thinking_trace": [],
        }

        _persist_response_datasets(response_payload, payload.dataset_names)

        return _build_agent_response(response_payload, rows)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Agent error") from exc


@app.post("/agent/think", response_model=AgentResponse)
def execute_thinking_agent(payload: AgentRequest) -> AgentResponse:
    rows = _resolve_rows(payload)
    if not rows:
        raise HTTPException(status_code=400, detail="Upload data first")
    model_name = _require_model_name(payload.model)
    if not supports_thinking_model(model_name):
        raise HTTPException(
            status_code=400,
            detail="Thinking mode only supports Gemma 4 31B IT and Minimax m2.5.",
        )
    logger.info("agent.think model=%s thinking=%s rows=%s", model_name, payload.thinking_mode, len(rows))

    try:
        result = run_thinking_agent(
            prompt=payload.prompt,
            rows=rows,
            model_name=model_name,
            history=[m.model_dump() for m in payload.history],
            datasets=_resolve_selected_datasets(payload),
            active_dataset_id=payload.active_dataset_id or payload.dataset_id,
            selected_dataset_ids=payload.selected_dataset_ids,
            dataset_names=payload.dataset_names,
            selection_context=payload.selection_context,
        )
        _persist_response_datasets(result, payload.dataset_names)
        return _build_agent_response(result, rows)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Thinking agent error") from exc


@app.post("/agent/think/stream")
def execute_thinking_agent_stream(payload: AgentRequest) -> StreamingResponse:
    rows = _resolve_rows(payload)
    if not rows:
        raise HTTPException(status_code=400, detail="Upload data first")
    model_name = _require_model_name(payload.model)
    if not supports_thinking_model(model_name):
        raise HTTPException(
            status_code=400,
            detail="Thinking mode only supports Gemma 4 31B IT and Minimax m2.5.",
        )
    logger.info("agent.think.stream model=%s thinking=%s rows=%s", model_name, payload.thinking_mode, len(rows))

    event_queue: Queue[dict[str, Any] | None] = Queue()

    def emit(event: dict[str, Any]) -> None:
        event_queue.put(event)

    def runner() -> None:
        try:
            result = run_thinking_agent(
                prompt=payload.prompt,
                rows=rows,
                model_name=model_name,
                history=[m.model_dump() for m in payload.history],
                datasets=_resolve_selected_datasets(payload),
                active_dataset_id=payload.active_dataset_id or payload.dataset_id,
                selected_dataset_ids=payload.selected_dataset_ids,
                dataset_names=payload.dataset_names,
                selection_context=payload.selection_context,
                event_callback=lambda entry: emit({"type": "trace", "entry": entry}),
            )
            _persist_response_datasets(result, payload.dataset_names)
            response = _build_agent_response(result, rows)
            emit({"type": "final", "payload": response.model_dump(mode="json")})
        except Exception as exc:
            logger.exception("agent.think.stream failed model=%s", model_name)
            emit({"type": "error", "error": str(exc) or "Thinking agent error"})
        finally:
            event_queue.put(None)

    threading.Thread(target=runner, daemon=True).start()

    def stream():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
