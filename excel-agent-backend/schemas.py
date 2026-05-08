from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    prompt: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    active_dataset_id: str | None = None
    selected_dataset_ids: list[str] = Field(default_factory=list)
    dataset_names: dict[str, str] = Field(default_factory=dict)
    model: str = ""
    thinking_mode: bool = False
    selection_context: dict[str, Any] | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class DatasetRegisterRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    name: str | None = None


class DatasetRegisterResponse(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class QueryTable(BaseModel):
    id: str
    title: str
    rows: list[dict[str, Any]] = Field(default_factory=list)


class HighlightedColumn(BaseModel):
    column: str


class DatasetResult(BaseModel):
    dataset_id: str
    name: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    kind: Literal["uploaded", "derived"] = "uploaded"
    source_dataset_ids: list[str] = Field(default_factory=list)
    modified: bool = False


class ThinkingTraceEntry(BaseModel):
    kind: Literal["thought", "action", "observation"]
    content: str
    tool_name: str | None = None
    tool_input: str | None = None
    details: str | None = None
    status: Literal["completed", "error"] = "completed"


class AgentResponse(BaseModel):
    rows: list[dict[str, Any]]
    active_dataset_id: str | None = None
    updated_datasets: list[DatasetResult] = Field(default_factory=list)
    created_datasets: list[DatasetResult] = Field(default_factory=list)
    visualization: dict[str, Any] | None = None
    query_output: str | None = None
    query_tables: list[QueryTable] = Field(default_factory=list)
    code: str
    assistant_reply: str
    context_preview: dict[str, Any]
    mutation: bool = False
    highlight_indices: list[int] = Field(default_factory=list)
    highlighted_columns: list[HighlightedColumn] = Field(default_factory=list)
    thinking_trace: list[ThinkingTraceEntry] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
