"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode, type SetStateAction } from "react"
import { AlertCircle, BrainCircuit, Check, ChevronDown, Code, Loader2, Paperclip, Plus, Send, Square, Table2, X } from "lucide-react"

import {
  DATASET_DRAG_TYPE,
  SLASH_COMMANDS,
  type ChatMessage,
  type DataSelectionContext,
  type PivotTableTab,
  type SlashCommandOption,
  type ThinkingTraceEntry,
  type WorkspaceDataset,
} from "@/components/dashboard/dashboard-shared"

type ChatPanelProps = {
  width: number
  chatMessages: ChatMessage[]
  resultTables: PivotTableTab[]
  highlightedMessageIndex: number | null
  pendingSelectionContext: DataSelectionContext | null
  attachedSelectionContext: DataSelectionContext | null
  prompt: string
  isRunning: boolean
  isUploading: boolean
  error: string | null
  thinkingMode: boolean
  modelLabel: string
  datasets: WorkspaceDataset[]
  scopeDatasets: WorkspaceDataset[]
  detachableDatasetIds: string[]
  onDropFiles: (files: File[]) => void
  onAttachDataset: (id: string) => void
  onDetachDataset: (id: string) => void
  onPromptChange: (value: string) => void
  onSubmit: () => void
  onCancel: () => void
  onToggleThinkingMode: () => void
  onClearError: () => void
  onOpenTableLink: (tableId: string) => void
  onAttachSelection: () => void
  onClearSelection: () => void
}

function isDatasetDrop(transfer: DataTransfer) {
  return transfer.types.includes("Files") || transfer.types.includes(DATASET_DRAG_TYPE)
}

function ResultTablePreview({ table, onOpen }: { table: PivotTableTab; onOpen: () => void }) {
  const columns = Object.keys(table.rows[0] ?? {})
  return (
    <section aria-label={`Result table: ${table.title}`} className="min-w-0 overflow-hidden rounded-lg border border-border bg-card/70">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 px-2.5 py-2">
        <div className="min-w-0">
          <h4 className="break-words text-xs font-semibold">{table.title}</h4>
          <p className="text-[10px] text-muted-foreground">{table.rows.length.toLocaleString()} rows</p>
        </div>
        <button type="button" onClick={onOpen} className="inline-flex shrink-0 items-center gap-1 rounded border border-primary/30 px-2 py-1 text-[11px] font-semibold text-primary hover:bg-primary/10">
          <Table2 aria-hidden="true" className="size-3" />
          Open table
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[11px]">
          <thead className="bg-muted/60 text-muted-foreground">
            <tr>{columns.map((column) => <th key={column} scope="col" className="whitespace-nowrap px-2 py-1.5">{column}</th>)}</tr>
          </thead>
          <tbody>
            {table.rows.slice(0, 3).map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-border/50">
                {columns.map((column) => (
                  <td key={column} className="max-w-36 truncate px-2 py-1.5" title={String(row[column] ?? "")}>
                    {row[column] == null || row[column] === "" ? "—" : String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.rows.length > 3 && <p className="border-t border-border/60 px-2.5 py-1.5 text-[10px] text-muted-foreground">Showing first 3 rows</p>}
    </section>
  )
}

function renderSimpleMarkdown(text: string) {
  const lines = text.split("\n")
  const elements: ReactNode[] = []
  let listBuffer: string[] = []

  const formatInline = (value: string): ReactNode[] => {
    return value.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={index}>{part.slice(2, -2)}</strong>
      }

      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <code key={index} className="rounded bg-muted px-1 py-0.5 text-[11px] font-mono">
            {part.slice(1, -1)}
          </code>
        )
      }

      return part
    })
  }

  const flushList = () => {
    if (listBuffer.length === 0) return

    elements.push(
      <ul key={`ul-${elements.length}`} className="list-disc pl-4 space-y-0.5">
        {listBuffer.map((item, index) => (
          <li key={index}>{formatInline(item)}</li>
        ))}
      </ul>
    )

    listBuffer = []
  }

  for (const line of lines) {
    const trimmed = line.trimStart()
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      listBuffer.push(trimmed.slice(2))
      continue
    }

    flushList()
    if (trimmed === "") {
      elements.push(<br key={`br-${elements.length}`} />)
      continue
    }

    elements.push(
      <span key={`p-${elements.length}`}>
        {elements.length > 0 && <br />}
        {formatInline(trimmed)}
      </span>
    )
  }

  flushList()
  return <>{elements}</>
}

function getActivitySteps(trace: ThinkingTraceEntry[]) {
  const steps: { action: ThinkingTraceEntry; result?: ThinkingTraceEntry }[] = []
  for (const entry of trace) {
    if (entry.kind === "action") {
      steps.push({ action: entry })
    } else if (entry.kind === "observation") {
      const step = entry.call_id
        ? steps.findLast((item) => item.action.call_id === entry.call_id)
        : steps.at(-1)
      if (step) step.result = entry
      else steps.push({ action: entry, result: entry })
    }
  }
  return steps
}

function AnalysisActivity({
  trace,
  live,
  model,
  open,
  onToggle,
}: {
  trace: ThinkingTraceEntry[]
  live: boolean
  model: string
  open: boolean
  onToggle: () => void
}) {
  const steps = getActivitySteps(trace)

  return (
    <div className="mb-2 w-full max-w-[90%] overflow-hidden rounded-xl border border-border/60">
      <button
        type="button"
        title={model}
        aria-label={open ? "Hide analysis steps" : "Show analysis steps"}
        aria-expanded={open}
        onClick={onToggle}
        className="flex w-full min-w-0 items-center gap-2 px-3 py-2.5 text-left hover:bg-accent/30"
      >
        {live
          ? <Loader2 aria-hidden="true" className="size-4 shrink-0 text-primary motion-safe:animate-spin" />
          : <BrainCircuit aria-hidden="true" className="size-4 shrink-0 text-muted-foreground" />}
        <span role="status" className="flex-1 truncate text-xs font-medium">
          {live ? "Thinking" : "Analysis"}
        </span>
        {steps.length > 0 && <span className="text-[11px] tabular-nums text-muted-foreground">{steps.length}</span>}
        <ChevronDown aria-hidden="true" className={`size-3 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="px-3 pb-3">
          {steps.length === 0 ? (
            <p className="text-xs text-muted-foreground">{live ? "Reviewing request…" : "Answered from context"}</p>
          ) : (
            <ol className="max-h-64 space-y-2 overflow-y-auto">
              {steps.map(({ action, result }, index) => {
                const failed = result?.status === "error"
                const pending = live && !result
                const details = [action.tool_input, result?.details].filter(Boolean)
                const label = action.content.replace(/\s+/g, " ").replace(/[.]$/, "")
                const outcome = result && result !== action
                  ? result.content.split("\n")[0].slice(0, 140)
                  : ""
                const row = (
                  <>
                    {failed ? <AlertCircle aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                      : pending ? <Loader2 aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-primary motion-safe:animate-spin" />
                        : <Check aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />}
                    <span className="min-w-0 flex-1">
                      <span className={`block break-words ${failed ? "text-destructive" : "text-foreground/90"}`}>{label}</span>
                      {outcome && <span className="mt-0.5 block break-words text-muted-foreground">{outcome}</span>}
                    </span>
                  </>
                )
                return (
                  <li key={action.call_id ?? index} className="min-w-0 text-xs leading-relaxed">
                    {details.length > 0 ? (
                      <details>
                        <summary className="flex cursor-pointer list-none items-start gap-2 [&::-webkit-details-marker]:hidden">
                          {row}
                          <Code aria-hidden="true" className="mt-0.5 size-3 shrink-0 text-muted-foreground/60" />
                        </summary>
                        {details.map((preview, previewIndex) => (
                          <pre key={previewIndex} className="mt-2 max-h-40 overflow-auto rounded-md bg-muted/50 p-2 text-[11px]">{preview}</pre>
                        ))}
                      </details>
                    ) : <div className="flex items-start gap-2">{row}</div>}
                  </li>
                )
              })}
            </ol>
          )}
        </div>
      )}
    </div>
  )
}

function selectionTypeLabel(selection: DataSelectionContext) {
  if (selection.selection_type === "rows") return "Rows"
  if (selection.selection_type === "columns") return "Columns"
  if (selection.selection_type === "cells") return "Cells"
  return "Mixed"
}

type ActivePromptTrigger = {
  query: string
  start: number
  end: number
}

type MentionToken = {
  token: string
  label: string
  kind: "file" | "command"
}

function getActivePromptTrigger(text: string, cursor: number, trigger: "/" | "@"): ActivePromptTrigger | null {
  const safeCursor = Math.max(0, Math.min(cursor, text.length))
  let start = safeCursor
  let end = safeCursor

  while (start > 0 && !/\s/.test(text[start - 1])) {
    start -= 1
  }

  while (end < text.length && !/\s/.test(text[end])) {
    end += 1
  }

  if (start >= end || text[start] !== trigger || safeCursor <= start) return null

  const query = text.slice(start + 1, safeCursor).toLowerCase()
  if (query.includes(trigger)) return null

  return { query, start, end }
}

function getDatasetMentionTokens(datasets: WorkspaceDataset[]) {
  const seen = new Set<string>()
  const tokens: MentionToken[] = []

  datasets.forEach((dataset) => {
    ;[dataset.displayName, dataset.fileName].forEach((name) => {
      if (!name) return
      const token = `@${name}`
      const key = token.toLowerCase()
      if (seen.has(key)) return
      seen.add(key)
      tokens.push({ token, label: token, kind: "file" })
    })
  })

  return tokens.sort((a, b) => b.token.length - a.token.length)
}

function startsWithIgnoreCase(text: string, index: number, token: string) {
  return text.slice(index, index + token.length).toLowerCase() === token.toLowerCase()
}

function isTokenStart(text: string, index: number) {
  return index === 0 || /\s/.test(text[index - 1])
}

function isTokenEnd(text: string, index: number) {
  return index >= text.length || /[\s.,;:!?]/.test(text[index])
}

function renderPromptTokens(text: string, datasets: WorkspaceDataset[], surface: "editor" | "message") {
  const datasetTokens = getDatasetMentionTokens(datasets)
  const nodes: ReactNode[] = []
  let index = 0
  let plainStart = 0
  let keyIndex = 0

  const flushPlain = (end: number) => {
    if (end > plainStart) {
      nodes.push(text.slice(plainStart, end))
    }
  }

  const pushToken = (token: MentionToken, rawToken = token.token) => {
    flushPlain(index)
    const label = surface === "message" ? token.label : rawToken
    nodes.push(
      <span key={`${token.kind}-${keyIndex}`} className={`prompt-token prompt-token-${token.kind}`}>
        {label}
      </span>
    )
    keyIndex += 1
    index += rawToken.length
    plainStart = index
  }

  while (index < text.length) {
    const markdownFileMatch = text.slice(index).match(/^\[([^\]\n]+)\]\(([^\)\n]+)\)/)
    if (markdownFileMatch) {
      pushToken({ token: markdownFileMatch[0], label: markdownFileMatch[1], kind: "file" }, markdownFileMatch[0])
      continue
    }

    if (isTokenStart(text, index) && text[index] === "@") {
      const datasetToken = datasetTokens.find(
        (candidate) =>
          startsWithIgnoreCase(text, index, candidate.token) &&
          isTokenEnd(text, index + candidate.token.length)
      )
      if (datasetToken) {
        pushToken(datasetToken, text.slice(index, index + datasetToken.token.length))
        continue
      }
    }

    if (isTokenStart(text, index) && text[index] === "/") {
      const commandMatch = text.slice(index).match(/^\/[^\s]*/)
      const commandToken = commandMatch?.[0] ?? ""
      const comparableCommand = commandToken.toLowerCase().replace(/[.,;:!?]+$/, "")
      if (
        comparableCommand.length > 1 &&
        SLASH_COMMANDS.some((item) => item.command.startsWith(comparableCommand) || comparableCommand.startsWith(item.command))
      ) {
        pushToken({ token: commandToken, label: commandToken, kind: "command" }, commandToken)
        continue
      }
    }

    index += 1
  }

  flushPlain(text.length)
  if (text.endsWith("\n")) {
    nodes.push(<br key="final-line-break" />)
  }

  return nodes
}

export function ChatPanel({
  width,
  chatMessages,
  resultTables,
  highlightedMessageIndex,
  pendingSelectionContext,
  attachedSelectionContext,
  prompt,
  isRunning,
  isUploading,
  error,
  thinkingMode,
  modelLabel,
  datasets,
  scopeDatasets,
  detachableDatasetIds,
  onDropFiles,
  onAttachDataset,
  onDetachDataset,
  onPromptChange,
  onSubmit,
  onCancel,
  onToggleThinkingMode,
  onClearError,
  onOpenTableLink,
  onAttachSelection,
  onClearSelection,
}: ChatPanelProps) {
  const [isDropTarget, setIsDropTarget] = useState(false)
  const dropDepth = useRef(0)
  const attachInput = useRef<HTMLInputElement>(null)
  const [expandedCodeIndex, setExpandedCodeIndex] = useState<number | null>(null)
  const [transcriptVisibilityOverrides, setTranscriptVisibilityOverrides] = useState<Record<number, boolean>>({})
  const [slashSelection, setSlashSelection] = useState({ query: null as string | null, index: 0 })
  const [datasetSelection, setDatasetSelection] = useState({ query: null as string | null, index: 0 })
  const [slashDismissed, setSlashDismissed] = useState(false)
  const [datasetMenuDismissed, setDatasetMenuDismissed] = useState(false)
  const [promptCursor, setPromptCursor] = useState(prompt.length)

  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const followLatestRef = useRef(true)
  const wasStreamingRef = useRef(false)
  const promptInputRef = useRef<HTMLTextAreaElement | null>(null)
  const slashListRef = useRef<HTMLDivElement | null>(null)
  const messageRefs = useRef<Array<HTMLDivElement | null>>([])

  const activeSlashTrigger = useMemo(() => getActivePromptTrigger(prompt, promptCursor, "/"), [prompt, promptCursor])
  const activeDatasetTrigger = useMemo(() => getActivePromptTrigger(prompt, promptCursor, "@"), [prompt, promptCursor])
  const slashQuery = activeSlashTrigger?.query ?? null
  const datasetQuery = activeDatasetTrigger?.query ?? null
  if (slashSelection.query !== slashQuery) setSlashSelection({ query: slashQuery, index: 0 })
  if (datasetSelection.query !== datasetQuery) setDatasetSelection({ query: datasetQuery, index: 0 })
  const selectedSlashIndex = slashSelection.query === slashQuery ? slashSelection.index : 0
  const selectedDatasetIndex = datasetSelection.query === datasetQuery ? datasetSelection.index : 0
  const setSelectedSlashIndex = useCallback((update: SetStateAction<number>) => {
    setSlashSelection((previous) => ({
      query: slashQuery,
      index: typeof update === "number" ? update : update(previous.query === slashQuery ? previous.index : 0),
    }))
  }, [slashQuery])
  const setSelectedDatasetIndex = useCallback((update: SetStateAction<number>) => {
    setDatasetSelection((previous) => ({
      query: datasetQuery,
      index: typeof update === "number" ? update : update(previous.query === datasetQuery ? previous.index : 0),
    }))
  }, [datasetQuery])
  const filteredSlashCommands = useMemo(() => {
    if (slashQuery === null) return []
    if (!slashQuery) return SLASH_COMMANDS
    return SLASH_COMMANDS.filter((item) => item.command.slice(1).startsWith(slashQuery))
  }, [slashQuery])
  const showSlashMenu = slashQuery !== null && !slashDismissed && filteredSlashCommands.length > 0
  const filteredDatasets = useMemo(() => {
    if (datasetQuery === null) return []
    return datasets.filter((dataset) =>
      [dataset.displayName, dataset.fileName].some((name) => name.toLowerCase().includes(datasetQuery))
    )
  }, [datasetQuery, datasets])
  const showDatasetMenu = datasetQuery !== null && !datasetMenuDismissed && filteredDatasets.length > 0
  const activeSlashIndex = showSlashMenu
    ? Math.min(selectedSlashIndex, filteredSlashCommands.length - 1)
    : 0
  const activeDatasetIndex = showDatasetMenu
    ? Math.min(selectedDatasetIndex, filteredDatasets.length - 1)
    : 0

  useEffect(() => {
    const timer = setTimeout(() => {
      setPromptCursor((prev) => Math.min(prev, prompt.length))
    }, 0)
    return () => clearTimeout(timer)
  }, [prompt.length])

  useEffect(() => {
    const latest = chatMessages.at(-1)
    const finishedAnalysis = wasStreamingRef.current && latest?.isStreaming === false
    wasStreamingRef.current = Boolean(latest?.isStreaming)
    if (!followLatestRef.current) return

    if (finishedAnalysis || latest?.table_links?.length) {
      messageRefs.current[chatMessages.length - 1]?.scrollIntoView({ block: "start" })
    } else {
      chatEndRef.current?.scrollIntoView({ block: "end" })
    }
  }, [chatMessages])

  useEffect(() => {
    if (highlightedMessageIndex === null) return
    messageRefs.current[highlightedMessageIndex]?.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [highlightedMessageIndex])

  const updatePromptCursor = useCallback((input = promptInputRef.current) => {
    if (!input) return
    setPromptCursor(input.selectionStart ?? input.value.length)
  }, [])

  const insertSlashCommand = useCallback((item: SlashCommandOption) => {
    if (!activeSlashTrigger) return

    const nextValue = `${prompt.slice(0, activeSlashTrigger.start)}${item.command} ${prompt.slice(activeSlashTrigger.end)}`
    const nextCursor = activeSlashTrigger.start + item.command.length + 1
    onPromptChange(nextValue)
    setPromptCursor(nextCursor)
    setSlashDismissed(false)
    setSelectedSlashIndex(0)

    requestAnimationFrame(() => {
      const input = promptInputRef.current
      if (!input) return
      input.focus()
      input.setSelectionRange(nextCursor, nextCursor)
    })
  }, [activeSlashTrigger, onPromptChange, prompt, setSelectedSlashIndex])

  const insertDatasetMention = useCallback((dataset: WorkspaceDataset) => {
    if (!activeDatasetTrigger) return

    const nextValue = `${prompt.slice(0, activeDatasetTrigger.start)}@${dataset.displayName} ${prompt.slice(activeDatasetTrigger.end)}`
    const nextCursor = activeDatasetTrigger.start + dataset.displayName.length + 2
    onPromptChange(nextValue)
    setPromptCursor(nextCursor)
    setDatasetMenuDismissed(false)
    setSelectedDatasetIndex(0)

    requestAnimationFrame(() => {
      const input = promptInputRef.current
      if (!input) return
      input.focus()
      input.setSelectionRange(nextCursor, nextCursor)
    })
  }, [activeDatasetTrigger, onPromptChange, prompt, setSelectedDatasetIndex])

  const toggleTranscript = useCallback((index: number, nextOpen: boolean) => {
    setTranscriptVisibilityOverrides((prev) => ({
      ...prev,
      [index]: nextOpen,
    }))
  }, [])

  const submitPrompt = () => {
    if (isUploading) return
    followLatestRef.current = true
    onSubmit()
  }

  return (
    <div className="flex shrink-0 flex-col bg-card/30" style={{ width }}>
      <div
        className="flex-1 min-h-0 overflow-auto p-3 space-y-3 scrollbar-thin"
        onScroll={({ currentTarget }) => {
          const { scrollTop, scrollHeight, clientHeight } = currentTarget
          followLatestRef.current = scrollHeight - scrollTop - clientHeight < 64
        }}
      >
        {chatMessages.map((message, index) => (
          (() => {
            const messageThinkingTrace = Array.isArray(message.thinkingTrace) ? message.thinkingTrace : []
            const isTranscriptOpen = transcriptVisibilityOverrides[index] ?? Boolean(message.isStreaming)
            const isRollbackHighlight = highlightedMessageIndex === index

            return (
              <div
                key={`${message.role}-${index}`}
                ref={(node) => {
                  messageRefs.current[index] = node
                }}
                className={`group flex flex-col ${message.role === "user" ? "items-end" : "items-start"}`}
              >
                {message.role === "assistant" && (messageThinkingTrace.length > 0 || message.isStreaming) && (
                  <AnalysisActivity
                    trace={messageThinkingTrace}
                    live={Boolean(message.isStreaming)}
                    model={message.modelLabel ?? modelLabel}
                    open={isTranscriptOpen}
                    onToggle={() => toggleTranscript(index, !isTranscriptOpen)}
                  />
                )}

                {message.role === "user" ? (
                  <div className={`prompt-message-bubble max-w-[90%] rounded-xl bg-primary px-3 py-2 text-[13px] leading-relaxed text-primary-foreground ${isRollbackHighlight ? "chat-rollback-bubble" : ""}`}>
                    {renderPromptTokens(message.content, datasets, "message")}
                    {!!message.datasetScope?.length && (
                      <div aria-label="Datasets used for this prompt" className="mt-2 flex flex-wrap gap-1.5">
                        {message.datasetScope.map((name, scopeIndex) => (
                          <span key={scopeIndex} className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-md bg-black/15 px-1.5 py-0.5 text-[10px]">
                            <Paperclip aria-hidden="true" className="size-3 shrink-0" />
                            <span className="truncate" title={name}>{name}</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  message.content && (
                    <div className={`max-w-[90%] break-words px-1 py-1 text-[13px] leading-relaxed text-foreground/92 ${isRollbackHighlight ? "chat-rollback-bubble rounded-xl" : ""}`}>
                      {renderSimpleMarkdown(message.content)}
                    </div>
                  )
                )}

                {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                  <div className="mt-2 flex max-w-[90%] flex-wrap gap-1.5 px-1">
                    {message.sources.map((source) => (
                      <a key={source.url} href={source.url} target="_blank" rel="noreferrer" className="rounded-md border border-border bg-card px-2 py-1 text-[10px] font-medium text-primary hover:bg-accent">
                        {source.title}
                      </a>
                    ))}
                  </div>
                )}

                {message.role === "assistant" && Boolean(message.table_links?.length) && (
                  <div className="mt-2 w-full max-w-[94%] min-w-0 space-y-2">
                    {message.table_links?.map((link) => {
                      const table = resultTables.find((item) => item.id === link.id)
                      return table && (
                        <ResultTablePreview key={link.id} table={table} onOpen={() => onOpenTableLink(link.id)} />
                      )
                    })}
                  </div>
                )}

                {message.role === "assistant" && !message.table_links?.length && message.query_output?.trim()
                  && message.query_output.trim() !== message.content.trim() && (
                  <details className="mt-1 w-full max-w-[90%] text-xs text-muted-foreground">
                    <summary className="cursor-pointer px-1 py-1">Execution output</summary>
                    <pre className="max-h-48 overflow-auto rounded-lg border border-border bg-card p-2 text-[11px] whitespace-pre-wrap break-words">
                      {message.query_output}
                    </pre>
                  </details>
                )}

                {message.role === "assistant" && message.code && (
                  <div className="mt-1 w-full max-w-[90%]">
                    <button
                      type="button"
                      onClick={() => setExpandedCodeIndex(expandedCodeIndex === index ? null : index)}
                      className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                        expandedCodeIndex === index
                          ? "text-primary bg-primary/10"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                      }`}
                    >
                      <Code className="size-3" />
                      {expandedCodeIndex === index ? "Hide code" : "View code"}
                    </button>

                    {expandedCodeIndex === index && (
                      <div className="mt-1.5 w-full space-y-2">
                        <pre className="w-full max-h-64 overflow-auto rounded-lg bg-[oklch(0.1_0_0)] p-2.5 text-[11px] font-mono text-[oklch(0.7_0_0)] border border-border whitespace-pre">
                          {message.code}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })()
        ))}

        <div ref={chatEndRef} />
      </div>

      {error && (
        <div className="mx-3 mb-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive flex justify-between items-center">
          <span>{error}</span>
          <button
            type="button"
            onClick={onClearError}
            className="text-destructive hover:text-destructive/80 transition-colors"
          >
            <X className="size-3.5" />
          </button>
        </div>
      )}

      <div className="border-t border-border p-3">
        {(pendingSelectionContext || attachedSelectionContext) && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {attachedSelectionContext ? (
              <div className="group/selection-chip inline-flex max-w-full items-center gap-2 rounded-md border border-primary/35 bg-primary/10 px-2.5 py-1.5 text-left text-[11px] font-medium text-foreground shadow-sm">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-primary/15 text-primary">
                  <Table2 className="size-3.5" />
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-primary">{selectionTypeLabel(attachedSelectionContext)} selected</span>
                  <span className="block truncate text-[10px] text-muted-foreground">
                    {attachedSelectionContext.summary} · {attachedSelectionContext.dataset_name}
                    {attachedSelectionContext.compact ? " · compact" : ""}
                  </span>
                </span>
                <button
                  type="button"
                  onClick={onClearSelection}
                  className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                  title="Clear selection"
                  aria-label="Clear selection"
                >
                  <X className="size-3.5" />
                </button>
              </div>
            ) : pendingSelectionContext ? (
              <div className="inline-flex max-w-full items-center rounded-md border border-border bg-secondary text-[11px] font-medium text-foreground shadow-sm transition-colors hover:border-primary/35">
                <button
                  type="button"
                  onClick={onAttachSelection}
                  className="inline-flex min-w-0 items-center gap-2 px-2.5 py-1.5 text-left transition-colors hover:bg-accent"
                  title="Attach selected range to this prompt"
                >
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-card text-muted-foreground">
                    <Plus className="size-3.5" />
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate">Add selected range</span>
                    <span className="block truncate text-[10px] text-muted-foreground">
                      {pendingSelectionContext.summary} · {pendingSelectionContext.dataset_name}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  onClick={onClearSelection}
                  className="mr-1 flex h-6 w-6 shrink-0 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                  title="Clear selection"
                  aria-label="Clear selection"
                >
                  <X className="size-3.5" />
                </button>
              </div>
            ) : null}
          </div>
        )}

        <div className="relative">
          <div
            data-state={showDatasetMenu ? "open" : "closed"}
            data-side="top"
            className="menu-surface menu-popover absolute bottom-full left-0 right-0 mb-2 rounded-lg p-2 shadow-md"
          >
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Datasets
            </div>
            <div className="max-h-56 overflow-y-auto scrollbar-thin pr-1">
              <div className="space-y-1">
                {filteredDatasets.map((dataset, index) => {
                  const isActive = index === activeDatasetIndex
                  return (
                    <button
                      key={dataset.id}
                      type="button"
                      onMouseDown={(event) => {
                        event.preventDefault()
                        insertDatasetMention(dataset)
                      }}
                      onMouseEnter={() => setSelectedDatasetIndex(index)}
                      className={`flex h-8 w-full items-center justify-between gap-2 rounded-md px-2.5 text-left text-[12px] font-medium leading-none transition-colors ${
                        isActive ? "bg-primary/12 text-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent"
                      }`}
                    >
                      <span className="truncate">{dataset.displayName}</span>
                      <span className="shrink-0 text-[10px] text-muted-foreground">
                        {dataset.rowCount}x{dataset.columnCount}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>

          <div
            data-state={showSlashMenu ? "open" : "closed"}
            data-side="top"
            className="menu-surface menu-popover absolute bottom-full left-0 right-0 mb-2 rounded-lg p-2 shadow-md"
          >
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Commands
            </div>
            <div
              ref={slashListRef}
              className="max-h-56 overflow-y-auto scrollbar-thin pr-1"
            >
              <div className="space-y-1">
                {filteredSlashCommands.map((item, index) => {
                  const isActive = index === activeSlashIndex

                  return (
                    <button
                      key={item.command}
                      type="button"
                      data-slash-option="true"
                      onMouseDown={(event) => {
                        event.preventDefault()
                        insertSlashCommand(item)
                      }}
                      onMouseEnter={() => {
                        setSelectedSlashIndex(index)
                      }}
                      className={`flex h-8 w-full items-center justify-between gap-2 rounded-md px-2.5 text-left text-[12px] font-medium leading-none transition-colors ${
                        isActive
                          ? "border border-primary/35 bg-primary/12 text-foreground"
                          : "border border-transparent text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                      }`}
                    >
                      <span className="shrink-0 font-medium text-foreground">{item.command}</span>
                      <span className="truncate text-[11px] text-muted-foreground">
                        {item.description}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>

          <div
            className={`relative rounded-xl border bg-secondary/70 shadow-sm transition focus-within:border-ring/70 focus-within:ring-2 focus-within:ring-ring/15 ${isDropTarget ? "border-primary ring-2 ring-primary/25" : "border-border"}`}
            onDragEnter={(event) => {
              if (!isDatasetDrop(event.dataTransfer)) return
              event.preventDefault()
              if (isRunning || isUploading) return
              dropDepth.current += 1
              setIsDropTarget(true)
            }}
            onDragOver={(event) => {
              if (!isDatasetDrop(event.dataTransfer)) return
              event.preventDefault()
              event.dataTransfer.dropEffect = isRunning || isUploading ? "none" : "copy"
            }}
            onDragLeave={() => {
              dropDepth.current = Math.max(0, dropDepth.current - 1)
              if (!dropDepth.current) setIsDropTarget(false)
            }}
            onDrop={(event) => {
              if (!isDatasetDrop(event.dataTransfer)) return
              event.preventDefault()
              dropDepth.current = 0
              setIsDropTarget(false)
              if (isRunning || isUploading) return
              const files = Array.from(event.dataTransfer.files)
              const datasetId = event.dataTransfer.getData(DATASET_DRAG_TYPE)
              if (files.length) onDropFiles(files)
              else if (datasetId) onAttachDataset(datasetId)
              promptInputRef.current?.focus()
            }}
          >
            {isDropTarget && (
              <div className="pointer-events-none absolute inset-0 z-10 grid place-content-center rounded-xl bg-background/90 text-sm font-medium text-primary">
                Drop to attach to this prompt
              </div>
            )}
            {!!scopeDatasets.length && (
              <div aria-label="Datasets in scope" className="flex max-h-24 flex-wrap items-center gap-1.5 overflow-y-auto border-b border-border/60 px-3 py-2">
                <span className="mr-0.5 text-[10px] font-medium text-muted-foreground">In scope</span>
                {scopeDatasets.map((dataset) => (
                  <span key={dataset.id} className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-md border border-primary/25 bg-primary/10 px-1.5 py-0.5 text-[11px] font-medium text-primary">
                    <span className="truncate" title={dataset.displayName}>@{dataset.displayName}</span>
                    {detachableDatasetIds.includes(dataset.id) && (
                      <button
                        type="button"
                        aria-label={`Detach ${dataset.displayName}`}
                        disabled={isRunning}
                        onClick={() => onDetachDataset(dataset.id)}
                        className="rounded p-0.5 hover:bg-primary/15 disabled:opacity-40"
                      >
                        <X aria-hidden="true" className="size-3" />
                      </button>
                    )}
                  </span>
                ))}
              </div>
            )}
            <div className="relative min-h-20">
              <textarea
                ref={promptInputRef}
                className="relative block max-h-56 min-h-20 w-full resize-y rounded-t-xl bg-transparent px-3 py-2.5 text-sm leading-relaxed text-foreground outline-none placeholder:text-muted-foreground"
                aria-label="Ask DataPilot"
                placeholder="Ask DataPilot… Drop files or use @ to select datasets."
                rows={3}
                spellCheck={false}
                value={prompt}
                onChange={(event) => {
                  const nextValue = event.currentTarget.value
                  const nextCursor = event.currentTarget.selectionStart ?? nextValue.length

                  onPromptChange(nextValue)
                  setPromptCursor(nextCursor)
                  setSlashDismissed(false)
                  setDatasetMenuDismissed(false)

                  if (!getActivePromptTrigger(nextValue, nextCursor, "/")) {
                  }
                }}
                onClick={(event) => updatePromptCursor(event.currentTarget)}
                onKeyUp={(event) => updatePromptCursor(event.currentTarget)}
                onSelect={(event) => updatePromptCursor(event.currentTarget)}
                onKeyDown={(event) => {
                  if (showDatasetMenu) {
                    if (event.key === "ArrowDown") {
                      event.preventDefault()
                      setSelectedDatasetIndex((prev) => (prev + 1) % filteredDatasets.length)
                      return
                    }

                    if (event.key === "ArrowUp") {
                      event.preventDefault()
                      setSelectedDatasetIndex((prev) => prev === 0 ? filteredDatasets.length - 1 : prev - 1)
                      return
                    }

                    if (event.key === "Enter" || event.key === "Tab") {
                      event.preventDefault()
                      const selected = filteredDatasets[activeDatasetIndex]
                      if (selected) {
                        insertDatasetMention(selected)
                      }
                      return
                    }

                    if (event.key === "Escape") {
                      event.preventDefault()
                      setDatasetMenuDismissed(true)
                      return
                    }
                  }

                  if (showSlashMenu) {
                    const scrollSlashOptionIntoView = (index: number) => {
                      const container = slashListRef.current
                      if (!container) return
                      const options = container.querySelectorAll<HTMLElement>("button[data-slash-option]")
                      options[index]?.scrollIntoView({ block: "nearest" })
                    }

                    if (event.key === "ArrowDown") {
                      event.preventDefault()
                      setSelectedSlashIndex((prev) => {
                        const next = (prev + 1) % filteredSlashCommands.length
                        scrollSlashOptionIntoView(next)
                        return next
                      })
                      return
                    }

                    if (event.key === "ArrowUp") {
                      event.preventDefault()
                      setSelectedSlashIndex((prev) => {
                        const next = prev === 0 ? filteredSlashCommands.length - 1 : prev - 1
                        scrollSlashOptionIntoView(next)
                        return next
                      })
                      return
                    }

                    if (event.key === "Enter" || event.key === "Tab") {
                      event.preventDefault()
                      const selected = filteredSlashCommands[activeSlashIndex]
                      if (selected) {
                        insertSlashCommand(selected)
                      }
                      return
                    }

                    if (event.key === "Escape") {
                      event.preventDefault()
                      setSlashDismissed(true)
                      return
                    }
                  }

                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault()
                    if (!isRunning) {
                      submitPrompt()
                    }
                  }
                }}
              />
            </div>

            <div className="flex items-center justify-between gap-2 px-2 py-2">
              <div className="flex min-w-0 items-center gap-2">
                <input
                  ref={attachInput}
                  type="file"
                  multiple
                  accept=".csv,.xlsx,.xls"
                  hidden
                  onChange={(event) => {
                    const files = Array.from(event.currentTarget.files ?? [])
                    event.currentTarget.value = ""
                    if (files.length) onDropFiles(files)
                  }}
                />
                <button
                  type="button"
                  onClick={() => attachInput.current?.click()}
                  disabled={isRunning || isUploading}
                  aria-label="Attach files"
                  title="Attach CSV or Excel files"
                  className="inline-flex size-8 shrink-0 items-center justify-center rounded-md border border-border bg-card/70 text-muted-foreground hover:text-foreground disabled:opacity-50"
                >
                  <Paperclip aria-hidden="true" className="size-3.5" />
                </button>
                <button
                  type="button"
                  onClick={onToggleThinkingMode}
                  disabled={isRunning}
                  title={thinkingMode ? "Disable thinking mode" : "Enable thinking mode"}
                  aria-label={thinkingMode ? "Disable thinking mode" : "Enable thinking mode"}
                  aria-pressed={thinkingMode}
                  className={`inline-flex h-8 items-center gap-1.5 rounded-md border px-2 text-[11px] font-medium transition-colors ${
                    thinkingMode
                      ? "border-primary/35 bg-primary/10 text-primary"
                      : "border-border bg-card/70 text-muted-foreground hover:text-foreground"
                  } disabled:pointer-events-none disabled:opacity-50`}
                >
                  <BrainCircuit className="size-3.5" />
                  Think
                </button>

                {(isRunning || isUploading) && (
                  <div role="status" className="flex min-w-0 items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <Loader2 className="size-3.5 shrink-0 animate-spin" />
                    <span className="truncate">{isUploading ? "Uploading…" : thinkingMode ? "Thinking..." : "Running..."}</span>
                  </div>
                )}
              </div>

              {isRunning ? (
                <button
                  type="button"
                  onClick={onCancel}
                  className="inline-flex h-8 items-center gap-1.5 rounded-md border border-destructive/35 bg-destructive/10 px-2.5 text-xs font-semibold text-destructive transition-colors hover:bg-destructive/15"
                >
                  <Square className="size-3.5 fill-current" />
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  onClick={submitPrompt}
                  disabled={isUploading || !prompt.trim()}
                  className="inline-flex h-8 items-center gap-1.5 rounded-md bg-primary px-3 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:pointer-events-none disabled:opacity-40"
                >
                  <Send className="size-3.5" />
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
