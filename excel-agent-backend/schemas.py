from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    prompt: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    model: str = "gemini-2.0-flash"
    history: list[ChatMessage] = Field(default_factory=list)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentResponse(BaseModel):
    rows: list[dict[str, Any]]
    visualization: dict[str, Any] | None = None
    query_output: str | None = None
    code: str
    assistant_reply: str
    context_preview: dict[str, Any]
    mutation: bool = False
    highlight_indices: list[int] = Field(default_factory=list)
    token_usage: TokenUsage | None = None


class StreamEvent(BaseModel):
    """Server-sent event for streaming progress"""
    event: str  # "step" | "result" | "error"
    data: dict[str, Any]
