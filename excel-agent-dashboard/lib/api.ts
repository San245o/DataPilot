const DEFAULT_PORT = "8000"

export const SESSION_COOKIE = "datapilot_test_session"

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL
  if (configured) return configured.replace(/\/$/, "")

  // Cookies are shared across ports on the same host. Keep localhost and 127.0.0.1 separate.
  const host = typeof window === "undefined" ? "127.0.0.1" : window.location.hostname
  return `http://${host}:${DEFAULT_PORT}`
}

export function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${getApiBaseUrl()}${path}`, { ...init, credentials: "include" })
}
