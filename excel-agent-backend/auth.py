from __future__ import annotations

import os
from secrets import compare_digest, token_urlsafe
from threading import Lock
from time import monotonic
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/auth")
SESSION_COOKIE = "datapilot_test_session"
SESSION_SECONDS = 8 * 60 * 60
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
frontend_port = os.environ.get("FRONTEND_PORT", "3000")
FRONTEND_ORIGINS = [
    f"http://{host}:{frontend_port}" for host in ("127.0.0.1", "localhost", "[::1]")
]
TEST_PASSWORD = b"datapilot-test"
TEST_ACCOUNTS = {
    "analyst@example.test": "Test Analyst",
    "reviewer@example.test": "Test Reviewer",
}


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


_sessions: dict[str, tuple[str, float]] = {}
_session_lock = Lock()


def _require_local_request(request: Request) -> None:
    origin = request.headers.get("origin")
    if (
        request.client is None
        or request.client.host not in LOOPBACK_HOSTS
        or request.url.hostname not in LOOPBACK_HOSTS
        or (origin is not None and origin not in FRONTEND_ORIGINS)
    ):
        raise HTTPException(status_code=403, detail="Test accounts are available on localhost only.")


def get_current_user(request: Request) -> dict[str, str]:
    _require_local_request(request)
    token = request.cookies.get(SESSION_COOKIE, "")
    with _session_lock:
        session = _sessions.get(token)
        if session is not None:
            email, expires = session
            if expires > monotonic():
                return {"user_id": email, "email": email, "name": TEST_ACCOUNTS[email]}
            _sessions.pop(token, None)
    raise HTTPException(status_code=401, detail="Sign in with a DataPilot test account.")


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, bool]:
    _require_local_request(request)
    email = payload.email.strip().lower()
    if email not in TEST_ACCOUNTS or not compare_digest(payload.password.encode(), TEST_PASSWORD):
        raise HTTPException(status_code=401, detail="That test email or password is incorrect.")

    token = token_urlsafe(32)
    now = monotonic()
    with _session_lock:
        for old_token, (_, expires) in list(_sessions.items()):
            if expires <= now:
                del _sessions[old_token]
        _sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
        _sessions[token] = (email, now + SESSION_SECONDS)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=SESSION_SECONDS)
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}


@router.get("/session")
def read_session(user: Annotated[dict[str, str], Depends(get_current_user)], response: Response) -> dict[str, dict[str, str]]:
    response.headers["Cache-Control"] = "no-store"
    return {"user": user}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    _require_local_request(request)
    with _session_lock:
        _sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
    response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="lax")
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}
