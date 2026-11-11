"use client"

import { apiFetch } from "@/lib/api"

type UserSession = {
  user_id: string
  email: string
  name: string
}

export async function getClientSession(): Promise<UserSession | null> {
  try {
    const response = await apiFetch("/auth/session", { cache: "no-store" })
    if (!response.ok) return null
    const data = await response.json()
    return data.user ?? null
  } catch {
    return null
  }
}

export async function loginWithEmailPassword(email: string, password: string): Promise<void> {
  const response = await apiFetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  })

  if (!response.ok) {
    const data = await response.json().catch(() => null)
    throw new Error(typeof data?.detail === "string" ? data.detail : "Sign-in failed.")
  }
}

export async function logoutUser(): Promise<void> {
  const response = await apiFetch("/auth/logout", { method: "POST" })
  if (!response.ok) throw new Error("Sign-out failed. Please try again.")
  window.location.assign("/login")
}
