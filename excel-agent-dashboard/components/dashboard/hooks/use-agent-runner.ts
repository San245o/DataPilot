"use client"

import { useCallback, useRef, useState } from "react"

import {
  expandSlashPrompt,
  normalizeRows,
  type AgentExecuteResponse,
  type ChatMessage,
  type HighlightedColumn,
  type PivotTableTab,
  type QueryTablePayload,
  type SheetRow,
  type ThinkingTraceEntry,
  type TokenUsageSummary,
  type VisualizationPayload,
} from "@/components/dashboard/dashboard-shared"

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend").replace(/\/$/, "")

type RunAgentInput = {
  prompt: string
  rows: SheetRow[]
  modelName: string
  history: ChatMessage[]
  thinkingMode: boolean
  onThinkingTrace?: (entry: ThinkingTraceEntry) => void
}

type AgentRunSuccessData = {
  nextRows: SheetRow[]
  responseTables: PivotTableTab[]
  highlightIndices: number[]
  highlightedColumns: HighlightedColumn[]
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

function normalizeFinalPayload(finalPayload: AgentExecuteResponse): AgentRunSuccessData {
  const nextRows = Array.isArray(finalPayload.rows) ? normalizeRows(finalPayload.rows) : []
  const responseTables = normalizeQueryTables(finalPayload.query_tables)
  const assistantMessage: ChatMessage = {
    role: "assistant",
    content: finalPayload.assistant_reply ?? "Done.",
    code: finalPayload.code,
    query_output: finalPayload.query_output,
    table_links: responseTables.map((table) => ({ id: table.id, title: table.title })),
    thinkingTrace: finalPayload.thinking_trace,
  }

  return {
    nextRows,
    responseTables,
    highlightIndices: Array.isArray(finalPayload.highlight_indices) ? finalPayload.highlight_indices : [],
    highlightedColumns: Array.isArray(finalPayload.highlighted_columns) ? finalPayload.highlighted_columns : [],
    mutation: Boolean(finalPayload.mutation),
    visualizationPayload: finalPayload.visualization ?? null,
    assistantMessage,
    tokenUsage: finalPayload.token_usage,
  }
}

type ThinkingStreamEvent =
  | { type: "trace"; entry: ThinkingTraceEntry }
  | { type: "final"; payload: AgentExecuteResponse }
  | { type: "error"; error: string }

class ThinkingStreamBackendError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "ThinkingStreamBackendError"
  }
}

async function fetchThinkingFinalFallback({
  expandedPrompt,
  rows,
  modelName,
  history,
}: {
  expandedPrompt: string
  rows: SheetRow[]
  modelName: string
  history: ChatMessage[]
}): Promise<AgentExecuteResponse> {
  const response = await fetch(`${API_BASE_URL}/agent/think`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: expandedPrompt,
      rows,
      model: modelName,
      thinking_mode: true,
      history: history.map((message) => ({
        role: message.role,
        content: message.role === "user" ? expandSlashPrompt(message.content) : message.content,
      })),
    }),
  })

  if (!response.ok) {
    const contentType = response.headers.get("content-type") || ""
    if (contentType.includes("application/json")) {
      const errorData = await response.json().catch(() => ({ detail: "Thinking fallback failed" }))
      throw new Error(errorData.detail ?? "Thinking fallback failed")
    }
    throw new Error(`Thinking fallback failed: ${response.status} ${response.statusText}`)
  }

  const payload = (await response.json()) as AgentExecuteResponse
  if (!payload) {
    throw new Error("Thinking fallback returned no result")
  }
  return payload
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
    thinkingMode,
    onThinkingTrace,
  }: RunAgentInput): Promise<AgentRunResponse> => {
    if (isRunningRef.current) {
      return { ok: false, error: "A request is already running." }
    }

    if (!prompt.trim()) {
      return { ok: false, error: "Please enter a prompt." }
    }

    const endpoint = thinkingMode
      ? `${API_BASE_URL}/agent/think/stream`
      : `${API_BASE_URL}/agent/execute`
    const expandedPrompt = expandSlashPrompt(prompt.trim())

    isRunningRef.current = true
    setIsRunning(true)
    setError(null)

    let lastAttemptedUrl = endpoint

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: expandedPrompt,
          rows,
          model: modelName,
          thinking_mode: thinkingMode,
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

      if (thinkingMode) {
        const reader = response.body?.getReader()
        if (!reader) {
          throw new Error("Streaming response body was not available.")
        }

        const decoder = new TextDecoder()
        let buffer = ""
        let finalPayload: AgentExecuteResponse | null = null

        const processLine = (line: string) => {
          if (!line.trim()) return
          const event = JSON.parse(line) as ThinkingStreamEvent
          if (event.type === "trace") {
            onThinkingTrace?.(event.entry)
            return
          }
          if (event.type === "error") {
            throw new ThinkingStreamBackendError(event.error || "Thinking stream failed")
          }
          if (event.type === "final") {
            finalPayload = event.payload
          }
        }

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split("\n")
            buffer = lines.pop() ?? ""

            for (const line of lines) {
              processLine(line)
            }
          }
        } catch (streamError) {
          if (streamError instanceof ThinkingStreamBackendError) {
            throw streamError
          }

          onThinkingTrace?.({
            kind: "observation",
            content: "The live stream was interrupted, so I'm finishing the thinking run in the background.",
            status: "error",
          })

          lastAttemptedUrl = `${API_BASE_URL}/agent/think`
          const fallbackPayload = await fetchThinkingFinalFallback({
            expandedPrompt,
            rows,
            modelName,
            history,
          })
          return {
            ok: true,
            data: normalizeFinalPayload(fallbackPayload),
          }
        }

        buffer += decoder.decode()
        if (buffer.trim()) {
          processLine(buffer)
        }

        if (!finalPayload) {
          onThinkingTrace?.({
            kind: "observation",
            content: "The live stream ended before the final answer arrived, so I'm finishing it in the background.",
            status: "error",
          })

          lastAttemptedUrl = `${API_BASE_URL}/agent/think`
          const fallbackPayload = await fetchThinkingFinalFallback({
            expandedPrompt,
            rows,
            modelName,
            history,
          })
          return {
            ok: true,
            data: normalizeFinalPayload(fallbackPayload),
          }
        }

        return {
          ok: true,
          data: normalizeFinalPayload(finalPayload),
        }
      }

      const finalPayload = (await response.json()) as AgentExecuteResponse
      if (!finalPayload) {
        throw new Error("No result received from agent")
      }

      return {
        ok: true,
        data: normalizeFinalPayload(finalPayload),
      }
    } catch (agentError) {
      const message =
        agentError instanceof TypeError && agentError.message.includes("fetch")
          ? `Failed to reach backend at ${lastAttemptedUrl}. If you are running the dashboard in a remote/devcontainer setup, use the default /api/backend proxy or set NEXT_PUBLIC_API_BASE_URL to a reachable URL.`
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
