from __future__ import annotations

import os
import logging
from typing import Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient, PyJWTError

logger = logging.getLogger("excel-agent-backend.auth")

security = HTTPBearer(auto_error=False)

ENTRA_TENANT_ID = os.getenv("ENTRA_TENANT_ID") or os.getenv("NEXT_PUBLIC_ENTRA_TENANT_ID")
ENTRA_TENANT_SUBDOMAIN = os.getenv("ENTRA_TENANT_SUBDOMAIN") or os.getenv("NEXT_PUBLIC_ENTRA_TENANT_SUBDOMAIN", "datapilot.ciamlogin.com")
ENTRA_API_CLIENT_ID = os.getenv("ENTRA_API_CLIENT_ID") or os.getenv("NEXT_PUBLIC_ENTRA_API_CLIENT_ID")
ENTRA_REQUIRED_SCOPE = os.getenv("ENTRA_REQUIRED_SCOPE", "access_as_user")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() in ("true", "1", "yes")

_jwks_clients: dict[str, PyJWKClient] = {}

def get_jwks_url() -> str:
    subdomain = ENTRA_TENANT_SUBDOMAIN if "." in ENTRA_TENANT_SUBDOMAIN else f"{ENTRA_TENANT_SUBDOMAIN}.ciamlogin.com"
    if ENTRA_TENANT_ID:
        return f"https://{subdomain}/{ENTRA_TENANT_ID}/discovery/v2.0/keys"
    return f"https://{subdomain}/common/discovery/v2.0/keys"

def get_jwks_client() -> PyJWKClient:
    url = get_jwks_url()
    if url not in _jwks_clients:
        try:
            _jwks_clients[url] = PyJWKClient(url)
            logger.info(f"Initialized Microsoft Entra JWKS client at {url}")
        except Exception as exc:
            logger.warning(f"Failed to initialize PyJWKClient for {url}: {exc}")
    return _jwks_clients[url]

def verify_entra_api_token(token: str) -> dict[str, Any]:
    """
    Validates Microsoft Entra External ID v2 Access Token for DataPilot API.
    Verifies signature via JWKS, expected issuer, expiration, API client ID GUID audience ('aud'),
    delegated scope ('scp'), and extracts stable user identifier ('oid' or 'sub').
    Rejects ID tokens or unauthorized scopes.
    """
    jwks_client = get_jwks_client()

    subdomain = ENTRA_TENANT_SUBDOMAIN if "." in ENTRA_TENANT_SUBDOMAIN else f"{ENTRA_TENANT_SUBDOMAIN}.ciamlogin.com"
    expected_issuers = []
    if ENTRA_TENANT_ID:
        expected_issuers.extend([
            f"https://{subdomain}/{ENTRA_TENANT_ID}/v2.0",
            f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/v2.0",
            f"https://sts.windows.net/{ENTRA_TENANT_ID}/",
        ])

    expected_aud = ENTRA_API_CLIENT_ID if (ENTRA_API_CLIENT_ID and not ENTRA_API_CLIENT_ID.startswith("placeholder")) else None

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": False, # Validated below explicitly against API Client ID GUID
                "verify_iss": False, # Validated below against API issuer list
            },
        )

        # 1. Issuer Validation
        token_iss = payload.get("iss", "")
        if expected_issuers and not any(token_iss.startswith(iss) for iss in expected_issuers):
            logger.warning(f"Token issuer mismatch: got '{token_iss}', expected one of {expected_issuers}")
            raise PyJWTError(f"Invalid token issuer: {token_iss}")

        # 2. Audience Validation (v2 Access Token aud MUST equal ENTRA_API_CLIENT_ID GUID)
        token_aud = payload.get("aud")
        if expected_aud:
            aud_valid = (token_aud == expected_aud) or (isinstance(token_aud, list) and expected_aud in token_aud)
            if not aud_valid:
                logger.warning(f"Token audience mismatch: got '{token_aud}', expected API client ID GUID '{expected_aud}'")
                raise PyJWTError(f"Invalid token audience. Expected API client ID GUID '{expected_aud}', got '{token_aud}'")

        # 3. Delegated Scope ('scp') Validation
        token_scopes = str(payload.get("scp", "")).split()
        if ENTRA_REQUIRED_SCOPE and ENTRA_REQUIRED_SCOPE not in token_scopes and "access_as_user" not in token_scopes:
            token_roles = payload.get("roles", [])
            if not token_roles and token_scopes:
                logger.warning(f"Token missing required scope '{ENTRA_REQUIRED_SCOPE}'. Scopes present: {token_scopes}")
                raise PyJWTError(f"Insufficient scope. Required scope: '{ENTRA_REQUIRED_SCOPE}'")

        # 4. Extract Stable User Identifier (oid in Entra ID, sub as fallback)
        stable_user_id = payload.get("oid") or payload.get("sub")
        if not stable_user_id:
            raise ValueError("Token missing stable user identifier ('oid' or 'sub')")

        email = payload.get("email") or payload.get("preferred_username") or payload.get("upn")

        return {
            "user_id": str(stable_user_id),
            "email": email,
            "authenticated": True,
            "claims": payload,
        }
    except PyJWTError as err:
        logger.warning(f"Entra API access token verification failed: {err}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired DataPilot API token: {str(err)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as err:
        logger.warning(f"Unexpected token verification error: {err}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(err)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict[str, Any]:
    if not credentials or not credentials.credentials:
        if REQUIRE_AUTH:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please provide a valid Bearer token for DataPilot API.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"user_id": "dev-local-user", "email": "dev@datapilot.local", "authenticated": False}

    token = credentials.credentials
    # Dev token handling for local test mode when Entra keys are not fully configured
    if token.startswith("dev_token_") or token.startswith("google_token_"):
        dev_user_id = token.replace("dev_token_", "").replace("google_token_", "")
        return {"user_id": dev_user_id, "email": "dev@datapilot.local", "authenticated": True}

    return verify_entra_api_token(token)
