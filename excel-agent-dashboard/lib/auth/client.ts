"use client";

import type { UserSession } from "./session";

export async function getClientSession(): Promise<UserSession | null> {
  try {
    const res = await fetch("/api/auth/session", { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    return data.user || null;
  } catch {
    return null;
  }
}

export async function loginWithEmailPassword(email: string, password: string): Promise<{ ok: boolean; error?: string }> {
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();
    if (!res.ok) {
      return { ok: false, error: data.error || "Authentication failed." };
    }

    if (data.redirectUrl) {
      window.location.href = data.redirectUrl;
      return { ok: true };
    }

    return { ok: true };
  } catch (err: unknown) {
    return { ok: false, error: err instanceof Error ? err.message : "Network error." };
  }
}

export async function signUpWithEmailPassword(fullName: string, email: string, password: string): Promise<{ ok: boolean; error?: string; message?: string }> {
  try {
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fullName, email, password }),
    });

    const data = await res.json();
    if (!res.ok) {
      return { ok: false, error: data.error || "Registration failed." };
    }

    if (data.redirectUrl) {
      window.location.href = data.redirectUrl;
      return { ok: true };
    }

    return { ok: true, message: data.message };
  } catch (err: unknown) {
    return { ok: false, error: err instanceof Error ? err.message : "Network error." };
  }
}

export async function loginWithGoogle(): Promise<void> {
  window.location.href = "/api/auth/google";
}

export async function logoutUser(): Promise<void> {
  await fetch("/api/auth/logout", { method: "POST" });
  window.location.href = "/login";
}
