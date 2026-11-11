from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from itertools import count
import json
import logging
from queue import Empty, Queue
import threading
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from agent import THINKING_MODEL
from auth import FRONTEND_ORIGINS, get_current_user, router as auth_router
from image_import import router as image_import_router
from schemas import AgentRequest, AgentResponse, DatasetRegisterRequest, DatasetRegisterResponse, ReportRequest
from report import REPORT_ANALYST_MODEL, ReportUnavailable, generate_auto_report
from thinking import run_thinking_agent
from workflow import build_workflow

load_dotenv()

app = FastAPI(title="Excel Agent Backend", version="3.0.0")
app.include_router(auth_router)
app.include_router(image_import_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("excel-agent-backend")

# Build workflow once at startup
_workflow = None
_dataset_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
MAX_CACHED_DATASETS = 100
THINKING_STREAM_HEARTBEAT_SECONDS = 10


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def _cache_dataset(rows: list[dict[str, Any]], name: str | None = None, user_id: str = "anonymous") -> str:
    dataset_id = uuid4().hex
    _dataset_store[dataset_id] = {
        "rows": rows,
        "name": name or "Dataset",
        "kind": "uploaded",
        "source_dataset_ids": [],
        "modified": False,
        "user_id": user_id,
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
    user_id: str = "anonymous",
) -> None:
    _dataset_store[dataset_id] = {
        "rows": rows,
        "name": name,
        "kind": kind,
        "source_dataset_ids": source_dataset_ids or [],
        "modified": modified,
        "user_id": user_id,
    }
    _dataset_store.move_to_end(dataset_id)

    while len(_dataset_store) > MAX_CACHED_DATASETS:
        _dataset_store.popitem(last=False)


def _get_cached_dataset(dataset_id: str, requesting_user_id: str = "anonymous") -> dict[str, Any]:
    cached = _dataset_store.get(dataset_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}. Upload again.")

    owner_id = cached.get("user_id", "anonymous")
    # Strict ownership check: if dataset has an owner, requesting user must match
    if owner_id != "anonymous" and owner_id != requesting_user_id:
        logger.warning(f"Unauthorized dataset access attempt: dataset={dataset_id}, owner={owner_id}, requested_by={requesting_user_id}")
        raise HTTPException(status_code=403, detail="Forbidden: You do not have permission to access this dataset.")

    _dataset_store.move_to_end(dataset_id)
    return cached


def _resolve_rows(payload: AgentRequest, user_id: str = "anonymous") -> list[dict[str, Any]]:
    dataset_id = payload.active_dataset_id or payload.dataset_id
    if dataset_id:
        cached = _get_cached_dataset(dataset_id, requesting_user_id=user_id)
        return list(cached.get("rows") or [])
    return payload.rows


def _resolve_selected_datasets(payload: AgentRequest, user_id: str = "anonymous") -> dict[str, dict[str, Any]]:
    selected_ids: list[str] = []
    for dataset_id in [payload.active_dataset_id or payload.dataset_id, *payload.selected_dataset_ids]:
        if dataset_id and dataset_id not in selected_ids:
            selected_ids.append(dataset_id)

    resolved: dict[str, dict[str, Any]] = {}
    for dataset_id in selected_ids:
        cached = _get_cached_dataset(dataset_id, requesting_user_id=user_id)
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
        visualizations=_safe_get(payload, "visualizations", []),
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
        sources=_safe_get(payload, "sources", []),
    )


def _persist_response_datasets(
    response_payload: dict[str, Any],
    dataset_names: dict[str, str] | None = None,
    user_id: str = "anonymous",
) -> None:
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
                user_id=user_id,
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
            user_id=user_id,
        )


def _require_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="Model is required. The backend will not silently fall back to Gemini 3.1 Flash Lite.",
        )
    return THINKING_MODEL if normalized.casefold().removeprefix("models/").startswith("gemma-") else normalized


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/dataset/register", response_model=DatasetRegisterResponse)
def register_dataset(
    payload: DatasetRegisterRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> DatasetRegisterResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    user_id = current_user.get("user_id", "anonymous")
    dataset_id = _cache_dataset(payload.rows, payload.name, user_id=user_id)
    column_count = len(payload.rows[0]) if payload.rows else 0
    return DatasetRegisterResponse(
        dataset_id=dataset_id,
        row_count=len(payload.rows),
        column_count=column_count,
    )


@app.post("/agent/execute", response_model=AgentResponse)
def execute_agent(
    payload: AgentRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AgentResponse:
    user_id = current_user.get("user_id", "anonymous")
    rows = _resolve_rows(payload, user_id=user_id)
    if not rows:
        raise HTTPException(status_code=400, detail="Upload data first")

    if payload.thinking_mode:
        raise HTTPException(status_code=400, detail="Use /agent/think for thinking mode requests.")
    model_name = _require_model_name(payload.model)
    logger.info("agent.execute user=%s model=%s thinking=%s rows=%s", user_id, model_name, payload.thinking_mode, len(rows))

    try:
        workflow = get_workflow()
        selected_datasets = _resolve_selected_datasets(payload, user_id=user_id)
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

        _persist_response_datasets(response_payload, payload.dataset_names, user_id=user_id)

        return _build_agent_response(response_payload, rows)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Agent error") from exc


@app.post("/agent/think", response_model=AgentResponse)
def execute_thinking_agent(
    payload: AgentRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AgentResponse:
    user_id = current_user.get("user_id", "anonymous")
    rows = _resolve_rows(payload, user_id=user_id)
    if not rows:
        raise HTTPException(status_code=400, detail="Upload data first")
    logger.info("agent.think user=%s model=%s rows=%s", user_id, THINKING_MODEL, len(rows))

    try:
        result = run_thinking_agent(
            prompt=payload.prompt,
            rows=rows,
            history=[m.model_dump() for m in payload.history],
            datasets=_resolve_selected_datasets(payload, user_id=user_id),
            active_dataset_id=payload.active_dataset_id or payload.dataset_id,
            selected_dataset_ids=payload.selected_dataset_ids,
            dataset_names=payload.dataset_names,
            selection_context=payload.selection_context,
        )
        response = _build_agent_response(result, rows)
        _persist_response_datasets(result, payload.dataset_names, user_id=user_id)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Thinking agent error") from exc


@app.post("/agent/think/stream")
async def execute_thinking_agent_stream(
    payload: AgentRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    user_id = current_user.get("user_id", "anonymous")
    rows = _resolve_rows(payload, user_id=user_id)
    if not rows:
        raise HTTPException(status_code=400, detail="Upload data first")
    logger.info("agent.think.stream user=%s model=%s rows=%s", user_id, THINKING_MODEL, len(rows))

    datasets = _resolve_selected_datasets(payload, user_id=user_id)
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    disconnected = threading.Event()

    def emit(event: dict[str, Any]) -> None:
        if disconnected.is_set():
            raise RuntimeError("Thinking stream closed")
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    def runner() -> None:
        try:
            result = run_thinking_agent(
                prompt=payload.prompt,
                rows=rows,
                history=[m.model_dump() for m in payload.history],
                datasets=datasets,
                active_dataset_id=payload.active_dataset_id or payload.dataset_id,
                selected_dataset_ids=payload.selected_dataset_ids,
                dataset_names=payload.dataset_names,
                selection_context=payload.selection_context,
                event_callback=lambda entry: emit({"type": "trace", "entry": entry}),
            )
            response = _build_agent_response(result, rows)
            event = {"type": "final", "payload": json.loads(response.model_dump_json())}
            if not disconnected.is_set():
                _persist_response_datasets(result, payload.dataset_names, user_id=user_id)
                emit(event)
        except Exception as exc:
            if not disconnected.is_set():
                logger.exception("agent.think.stream failed model=%s", THINKING_MODEL)
                loop.call_soon_threadsafe(
                    event_queue.put_nowait, {"type": "error", "error": str(exc) or "Thinking agent error"}
                )

    async def stream():
        threading.Thread(target=runner, daemon=True).start()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(event_queue.get(), timeout=THINKING_STREAM_HEARTBEAT_SECONDS)
                except TimeoutError:
                    item = {"type": "heartbeat"}
                yield json.dumps(item) + "\n"
                if item["type"] in {"final", "error"}:
                    return
        finally:
            disconnected.set()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/report/stream")
def execute_auto_report_stream(
    payload: ReportRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Generate a structured report against one immutable, user-owned dataset snapshot."""
    user_id = current_user.get("user_id", "anonymous")
    cached = _get_cached_dataset(payload.dataset_id, requesting_user_id=user_id)
    rows = list(cached.get("rows") or [])
    if not rows:
        raise HTTPException(status_code=400, detail="The active dataset is empty.")
    model_name = REPORT_ANALYST_MODEL

    # Capture identity, name, and rows before the background thread starts. Later UI
    # dataset changes cannot retarget this report run.
    dataset_id = payload.dataset_id
    dataset_name = str(cached.get("name") or "Dataset")
    frozen_rows = [dict(row) for row in rows]
    event_queue: Queue[dict[str, Any] | None] = Queue()
    stop = threading.Event()
    sequence = count(1)

    def queue_trace(entry: dict[str, Any]) -> None:
        event_queue.put({
            "type": "trace",
            "entry": {
                **entry,
                "sequence": next(sequence),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            },
        })

    queue_trace({"kind": "thought", "content": "Starting report analysis for the active dataset.", "status": "completed"})
    logger.info("report stream started dataset=%s user=%s rows=%s", dataset_id, user_id, len(frozen_rows))

    def runner() -> None:
        try:
            report = generate_auto_report(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                rows=frozen_rows,
                model_name=model_name,
                event_callback=queue_trace,
                progress_callback=event_queue.put,
                stop=stop,
            )
            if report is not None and not stop.is_set():
                event_queue.put({"type": "final", "payload": report})
        except ReportUnavailable as exc:
            event_queue.put({"type": "error", "error": str(exc)})
        except Exception as exc:
            logger.exception("agent.report.stream failed dataset=%s", dataset_id)
            queue_trace({"kind": "observation", "content": "Report generation failed before the final report could be completed.", "status": "error"})
            event_queue.put({"type": "error", "error": "Auto Report could not finish. Try running it again."})
        finally:
            event_queue.put(None)
            logger.info("report stream sentinel queued dataset=%s", dataset_id)

    async def stream():
        threading.Thread(target=runner, daemon=True).start()
        try:
            while True:
                try:
                    item = await asyncio.to_thread(event_queue.get, True, 0.5)
                except Empty:
                    continue
                if item is None:
                    break
                yield json.dumps(item, default=str) + "\n"
        finally:
            stop.set()
            logger.info("report stream closed dataset=%s", dataset_id)

    return StreamingResponse(
        stream(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
