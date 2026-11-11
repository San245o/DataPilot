"use client"

import { useState, type FormEvent } from "react"
import { Lock, Mail } from "lucide-react"
import { loginWithEmailPassword } from "@/lib/auth/client"

import AuthField from "./AuthField"
import { AuthHeading, PrimaryButton } from "./AuthExtras"

const TEST_EMAILS = ["analyst@example.test", "reviewer@example.test"]
const TEST_PASSWORD = "datapilot-test"
const WORKSPACE_PATH = /^\/(home|dashboard|report|chart-viewer|history|tutorials)([/?#]|$)/

export function LoginForm() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (loading) return
    setError("")
    setLoading(true)

    try {
      await loginWithEmailPassword(email, password)
      const requested = new URLSearchParams(window.location.search).get("redirectTo")
      window.location.assign(requested && WORKSPACE_PATH.test(requested) ? requested : "/home")
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Sign-in failed.")
      setLoading(false)
    }
  }

  return (
    <div>
      <AuthHeading title="Sign in to DataPilot" subtitle="Choose a test account to open the local workspace." />

      <div className="mb-5 flex flex-wrap gap-2">
        {TEST_EMAILS.map((address) => (
          <button
            key={address}
            type="button"
            disabled={loading}
            onClick={() => {
              setEmail(address)
              setPassword(TEST_PASSWORD)
              setError("")
            }}
            className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:border-emerald-400 hover:text-emerald-300 disabled:opacity-50"
          >
            {address}
          </button>
        ))}
      </div>

      {error && (
        <p role="alert" className="mb-4 rounded-xl border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">
          {error}
        </p>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <AuthField
          label="Test email"
          icon={Mail}
          type="email"
          autoComplete="username"
          placeholder="analyst@example.test"
          value={email}
          onChange={(event) => {
            setEmail(event.target.value)
            setError("")
          }}
          required
        />
        <AuthField
          label="Password"
          icon={Lock}
          isPassword
          autoComplete="current-password"
          placeholder="Enter the test password"
          value={password}
          onChange={(event) => {
            setPassword(event.target.value)
            setError("")
          }}
          required
        />
        <PrimaryButton type="submit" loading={loading}>Sign in</PrimaryButton>
      </form>
    </div>
  )
}

export default LoginForm
