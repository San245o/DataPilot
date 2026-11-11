"use client"

import { useCallback, useRef, useState } from "react"
import { apiFetch, getApiBaseUrl } from "@/lib/api"
import {
  expandSlashPrompt,
  normalizeRows,
  type AgentExecuteResponse,
  type ChatMessage,
  type DataSelectionContext,
  type HighlightedColumn,
  type PivotTableTab,
  type QueryTablePayload,
  type SheetRow,
  type ThinkingTraceEntry,
  type TokenUsageSummary,
  type VisualizationPayload,
  type WorkspaceDataset,
} from "@/components/dashboard/dashboard-shared"

type RunAgentInput = {
  prompt: string
  rows: SheetRow[]
  activeDatasetId?: string | null
  selectedDatasetIds?: string[]
  datasetNames?: Record<string, string>
  modelName: string
  history: ChatMessage[]
  thinkingMode: boolean
  selectionContext?: DataSelectionContext | null
  onThinkingTrace?: (entry: ThinkingTraceEntry) => void
}

type AgentRunSuccessData = {
  nextRows: SheetRow[]
  responseTables: PivotTableTab[]
  highlightIndices: number[]
  highlightedColumns: HighlightedColumn[]
  mutation: boolean
  visualizations: VisualizationPayload[]
  updatedDatasets: WorkspaceDataset[]
  createdDatasets: WorkspaceDataset[]
  assistantMessage: ChatMessage
  tokenUsage?: TokenUsageSummary
}

export type AgentRunResponse =
  | { ok: true; data: AgentRunSuccessData }
  | { ok: false; error: string; cancelled?: boolean }

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
  const normalizeDatasetResults = (items: NonNullable<AgentExecuteResponse["updated_datasets"]>): WorkspaceDataset[] =>
    items.map((item) => {
      const rows = normalizeRows(item.rows ?? [])
      const [fileName, sheetName] = String(item.name || item.dataset_id).split(" / ", 2)
      return {
        id: item.dataset_id,
        fileName: fileName || item.name || item.dataset_id,
        sheetName,
        displayName: item.name || item.dataset_id,
        rows,
        rowCount: rows.length,
        columnCount: Object.keys(rows[0] ?? {}).length,
        kind: item.kind ?? "uploaded",
        modified: Boolean(item.modified),
        sourceDatasetIds: item.source_dataset_ids ?? [],
      }
    })
  const assistantMessage: ChatMessage = {
    role: "assistant",
    content: finalPayload.assistant_reply ?? "Done.",
    code: finalPayload.code,
    query_output: finalPayload.query_output,
    table_links: responseTables.map((table) => ({ id: table.id, title: table.title })),
    thinkingTrace: finalPayload.thinking_trace,
    sources: finalPayload.sources,
  }

  return {
    nextRows,
    responseTables,
    highlightIndices: Array.isArray(finalPayload.highlight_indices) ? finalPayload.highlight_indices : [],
    highlightedColumns: Array.isArray(finalPayload.highlighted_columns) ? finalPayload.highlighted_columns : [],
    mutation: Boolean(finalPayload.mutation),
    visualizations: finalPayload.visualizations?.length
      ? finalPayload.visualizations
      : finalPayload.visualization ? [finalPayload.visualization] : [],
    updatedDatasets: normalizeDatasetResults(finalPayload.updated_datasets ?? []),
    createdDatasets: normalizeDatasetResults(finalPayload.created_datasets ?? []),
    assistantMessage,
    tokenUsage: finalPayload.token_usage,
  }
}

type ThinkingStreamEvent =
  | { type: "heartbeat" }
  | { type: "trace"; entry: ThinkingTraceEntry }
  | { type: "final"; payload: AgentExecuteResponse }
  | { type: "error"; error: string }

async function readThinkingResult(
  stream: ReadableStream<Uint8Array>,
  onTrace?: (entry: ThinkingTraceEntry) => void,
): Promise<AgentExecuteResponse> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let pending = ""

  try {
    while (true) {
      const { done, value } = await reader.read()
      pending += decoder.decode(value, { stream: !done })
      const lines = pending.split("\n")
      pending = done ? "" : lines.pop() ?? ""

      for (const line of lines) {
        if (!line.trim()) continue
        const event = JSON.parse(line) as ThinkingStreamEvent
        if (event.type === "final" && event.payload) return event.payload
        if (event.type === "error") throw new Error(event.error || "Thinking run failed.")
        if (event.type === "trace") onTrace?.(event.entry)
      }

      if (done) throw new Error("The connection ended before the analysis result arrived.")
    }
  } finally {
    // A complete final event is enough even if the connection closes unexpectedly afterward.
    void reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

export function useAgentRunner() {
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isRunningRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const runAgent = useCallback(async ({
    prompt,
    rows,
    activeDatasetId,
    selectedDatasetIds,
    datasetNames,
    modelName,
    history,
    thinkingMode,
    selectionContext,
    onThinkingTrace,
  }: RunAgentInput): Promise<AgentRunResponse> => {
    if (isRunningRef.current) {
      return { ok: false, error: "A request is already running." }
    }

    if (!prompt.trim()) {
      return { ok: false, error: "Please enter a prompt." }
    }

    const endpoint = thinkingMode
      ? "/agent/think/stream"
      : "/agent/execute"
    const expandedPrompt = expandSlashPrompt(prompt.trim())
    const abortController = new AbortController()

    abortControllerRef.current = abortController
    isRunningRef.current = true
    setIsRunning(true)
    setError(null)

    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" }

      const response = await apiFetch(endpoint, {
        method: "POST",
        headers,
        signal: abortController.signal,
        body: JSON.stringify({
          prompt: expandedPrompt,
          rows,
          active_dataset_id: activeDatasetId,
          selected_dataset_ids: selectedDatasetIds ?? [],
          dataset_names: datasetNames ?? {},
          model: modelName,
          thinking_mode: thinkingMode,
          selection_context: selectionContext ?? null,
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
        if (!response.body) throw new Error("The analysis stream was unavailable.")
        const finalPayload = await readThinkingResult(response.body, onThinkingTrace)
        return { ok: true, data: normalizeFinalPayload(finalPayload) }
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
      if (
        abortController.signal.aborted ||
        (agentError instanceof Error && agentError.name === "AbortError")
      ) {
        return { ok: false, error: "Request stopped.", cancelled: true }
      }

      const message =
        agentError instanceof TypeError && agentError.message.includes("fetch")
          ? `Could not reach DataPilot at ${getApiBaseUrl()}. Check that the local backend is running.`
          : agentError instanceof Error
            ? agentError.message
            : "Unknown agent error"

      setError(message)
      return { ok: false, error: message }
    } finally {
      isRunningRef.current = false
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
      setIsRunning(false)
    }
  }, [])

  const cancelRun = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  return {
    cancelRun,
    clearError,
    error,
    isRunning,
    runAgent,
    setError,
  }
}
