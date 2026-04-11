"use client"

import { useCallback, useRef, useState } from "react"

import {
  expandSlashPrompt,
  normalizeRows,
  type AgentExecuteResponse,
  type ChatMessage,
  type PivotTableTab,
  type QueryTablePayload,
  type SheetRow,
  type TokenUsageSummary,
  type VisualizationPayload,
} from "@/components/dashboard/dashboard-shared"

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend").replace(/\/$/, "")

type RunAgentInput = {
  prompt: string
  rows: SheetRow[]
  modelName: string
  history: ChatMessage[]
}

type AgentRunSuccessData = {
  nextRows: SheetRow[]
  responseTables: PivotTableTab[]
  highlightIndices: number[]
  mutation: boolean
  visualizationPayload: VisualizationPayload | null
  assistantMessage: ChatMessage
  tokenUsage?: TokenUsageSummary
}

export type AgentRunResponse =
  | { ok: true; data: AgentRunSuccessData }
  | { ok: false; error: string }

function normalizeQueryTables(tables: QueryTablePayload[] | undefined): PivotTableTab[] {
  if (!Array.isArray(tables)) return []

  return tables.map((table) => ({
    id: table.id,
    title: table.title,
    rows: normalizeRows(table.rows),
    open: true,
    isNew: true,
    insertions: table.rows?.length ?? 0,
  }))
}

export function useAgentRunner() {
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isRunningRef = useRef(false)

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const runAgent = useCallback(async ({
    prompt,
    rows,
    modelName,
    history,
  }: RunAgentInput): Promise<AgentRunResponse> => {
    if (isRunningRef.current) {
      return { ok: false, error: "A request is already running." }
    }

    if (!prompt.trim()) {
      return { ok: false, error: "Please enter a prompt." }
    }

    const endpoint = `${API_BASE_URL}/agent/execute`
    const expandedPrompt = expandSlashPrompt(prompt.trim())

    isRunningRef.current = true
    setIsRunning(true)
    setError(null)

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: expandedPrompt,
          rows,
          model: modelName,
          history: history.map((message) => ({
            role: message.role,
            content: message.role === "user" ? expandSlashPrompt(message.content) : message.content,
          })),
        }),
      })

      if (!response.ok) {
        const contentType = response.headers.get("content-type") || ""
        if (contentType.includes("application/json")) {
          const errorData = await response.json().catch(() => ({ detail: "Agent execution failed" }))
          throw new Error(errorData.detail ?? "Agent execution failed")
        }

        throw new Error(`Server error: ${response.status} ${response.statusText}`)
      }

      const finalPayload = (await response.json()) as AgentExecuteResponse
      if (!finalPayload) {
        throw new Error("No result received from agent")
      }

      const nextRows = Array.isArray(finalPayload.rows) ? normalizeRows(finalPayload.rows) : []
      const responseTables = normalizeQueryTables(finalPayload.query_tables)
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: finalPayload.assistant_reply ?? "Done.",
        code: finalPayload.code,
        query_output: finalPayload.query_output,
        table_links: responseTables.map((table) => ({ id: table.id, title: table.title })),
      }

      return {
        ok: true,
        data: {
          nextRows,
          responseTables,
          highlightIndices: Array.isArray(finalPayload.highlight_indices) ? finalPayload.highlight_indices : [],
          mutation: Boolean(finalPayload.mutation),
          visualizationPayload: finalPayload.visualization ?? null,
          assistantMessage,
          tokenUsage: finalPayload.token_usage,
        },
      }
    } catch (agentError) {
      const message =
        agentError instanceof TypeError && agentError.message.includes("fetch")
          ? `Failed to reach backend at ${endpoint}. If you are running the dashboard in a remote/devcontainer setup, use the default /api/backend proxy or set NEXT_PUBLIC_API_BASE_URL to a reachable URL.`
          : agentError instanceof Error
            ? agentError.message
            : "Unknown agent error"

      setError(message)
      return { ok: false, error: message }
    } finally {
      isRunningRef.current = false
      setIsRunning(false)
    }
  }, [])

  return {
    clearError,
    error,
    isRunning,
    runAgent,
    setError,
  }
}
