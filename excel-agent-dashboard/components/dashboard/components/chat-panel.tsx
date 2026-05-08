"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { BrainCircuit, ChevronDown, Code, Loader2, Plus, Send, Square, Table2, X } from "lucide-react"

import {
  SLASH_COMMANDS,
  SLASH_DESCRIPTION_DELAY_MS,
  SLASH_ROW_GAP,
  SLASH_ROW_HEIGHT,
  type ChatMessage,
  type DataSelectionContext,
  type SlashCommandOption,
  type ThinkingTraceEntry,
  type WorkspaceDataset,
} from "@/components/dashboard/dashboard-shared"

type ChatPanelProps = {
  width: number
  chatMessages: ChatMessage[]
  highlightedMessageIndex: number | null
  pendingSelectionContext: DataSelectionContext | null
  attachedSelectionContext: DataSelectionContext | null
  prompt: string
  isRunning: boolean
  error: string | null
  thinkingMode: boolean
  modelLabel: string
  datasets: WorkspaceDataset[]
  onPromptChange: (value: string) => void
  onSubmit: () => void
  onCancel: () => void
  onToggleThinkingMode: () => void
  onClearError: () => void
  onOpenTableLink: (tableId: string) => void
  onAttachSelection: () => void
  onClearSelection: () => void
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

type AssistantTextProps = {
  text: string
}

function AssistantText({ text }: AssistantTextProps) {
  const [visibleLength, setVisibleLength] = useState(0)

  useEffect(() => {
    const interval = window.setInterval(() => {
      setVisibleLength((prev) => {
        if (prev >= text.length) {
          window.clearInterval(interval)
          return text.length
        }
        return prev + Math.max(1, Math.ceil((text.length - prev) / 12))
      })
    }, 24)

    return () => window.clearInterval(interval)
  }, [text])

  const visibleText = text.slice(0, visibleLength)
  return (
    <div className="max-w-[90%] px-1 py-1 text-[13px] leading-relaxed text-foreground/92">
      {renderSimpleMarkdown(visibleText)}
      {visibleLength < text.length && (
        <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-[2px] animate-pulse bg-primary/80 align-middle" />
      )}
    </div>
  )
}

function getTraceLabel(entry: ThinkingTraceEntry) {
  if (entry.kind === "thought") return "Thinking"
  if (entry.kind === "action") return "Action"
  return entry.status === "error" ? "Error" : "Observation"
}

function getTraceDotClass(entry: ThinkingTraceEntry) {
  if (entry.kind === "thought") return "bg-primary"
  if (entry.kind === "action") return "bg-amber-500"
  return entry.status === "error" ? "bg-destructive" : "bg-emerald-500"
}

function shouldHideObservationLead(entry: ThinkingTraceEntry) {
  if (entry.kind !== "observation" || !entry.details || entry.status === "error") return false
  return /^(Returned \d+ (rows|items)|Found \d+ matching rows\.|The tool returned )/i.test(entry.content.trim())
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
  highlightedMessageIndex,
  pendingSelectionContext,
  attachedSelectionContext,
  prompt,
  isRunning,
  error,
  thinkingMode,
  modelLabel,
  datasets,
  onPromptChange,
  onSubmit,
  onCancel,
  onToggleThinkingMode,
  onClearError,
  onOpenTableLink,
  onAttachSelection,
  onClearSelection,
}: ChatPanelProps) {
  const [expandedCodeIndex, setExpandedCodeIndex] = useState<number | null>(null)
  const [transcriptVisibilityOverrides, setTranscriptVisibilityOverrides] = useState<Record<number, boolean>>({})
  const [selectedSlashIndex, setSelectedSlashIndex] = useState(0)
  const [selectedDatasetIndex, setSelectedDatasetIndex] = useState(0)
  const [slashDismissed, setSlashDismissed] = useState(false)
  const [datasetMenuDismissed, setDatasetMenuDismissed] = useState(false)
  const [visibleSlashDescriptionFor, setVisibleSlashDescriptionFor] = useState<string | null>(null)
  const [promptCursor, setPromptCursor] = useState(prompt.length)

  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const promptInputRef = useRef<HTMLTextAreaElement | null>(null)
  const slashListRef = useRef<HTMLDivElement | null>(null)
  const messageRefs = useRef<Array<HTMLDivElement | null>>([])
  const slashDescriptionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const activeSlashTrigger = useMemo(() => getActivePromptTrigger(prompt, promptCursor, "/"), [prompt, promptCursor])
  const activeDatasetTrigger = useMemo(() => getActivePromptTrigger(prompt, promptCursor, "@"), [prompt, promptCursor])
  const slashQuery = activeSlashTrigger?.query ?? null
  const datasetQuery = activeDatasetTrigger?.query ?? null
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
  const activeDescription = showSlashMenu ? visibleSlashDescriptionFor : null

  useEffect(() => {
    setPromptCursor((prev) => Math.min(prev, prompt.length))
  }, [prompt.length])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatMessages, isRunning])

  useEffect(() => {
    if (highlightedMessageIndex === null) return
    messageRefs.current[highlightedMessageIndex]?.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [highlightedMessageIndex])

  const clearSlashDescriptionTimer = useCallback(() => {
    if (!slashDescriptionTimerRef.current) return
    clearTimeout(slashDescriptionTimerRef.current)
    slashDescriptionTimerRef.current = null
  }, [])

  const hideSlashDescription = useCallback(() => {
    clearSlashDescriptionTimer()
    setVisibleSlashDescriptionFor(null)
  }, [clearSlashDescriptionTimer])

  const scheduleSlashDescription = useCallback((command: string) => {
    clearSlashDescriptionTimer()
    setVisibleSlashDescriptionFor(null)
    slashDescriptionTimerRef.current = setTimeout(() => {
      setVisibleSlashDescriptionFor(command)
      slashDescriptionTimerRef.current = null
    }, SLASH_DESCRIPTION_DELAY_MS)
  }, [clearSlashDescriptionTimer])

  useEffect(() => {
    return () => clearSlashDescriptionTimer()
  }, [clearSlashDescriptionTimer])

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
    setVisibleSlashDescriptionFor(null)
    clearSlashDescriptionTimer()

    requestAnimationFrame(() => {
      const input = promptInputRef.current
      if (!input) return
      input.focus()
      input.setSelectionRange(nextCursor, nextCursor)
    })
  }, [activeSlashTrigger, clearSlashDescriptionTimer, onPromptChange, prompt])

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
  }, [activeDatasetTrigger, onPromptChange, prompt])

  const toggleTranscript = useCallback((index: number, nextOpen: boolean) => {
    setTranscriptVisibilityOverrides((prev) => ({
      ...prev,
      [index]: nextOpen,
    }))
  }, [])

  return (
    <div className="flex shrink-0 flex-col bg-card/30" style={{ width }}>
      <div className="flex-1 min-h-0 overflow-auto p-3 space-y-3 scrollbar-thin">
        {chatMessages.map((message, index) => (
          (() => {
            const messageThinkingTrace = Array.isArray(message.thinkingTrace) ? message.thinkingTrace : []
            const defaultTranscriptOpen = false
            const isTranscriptOpen = transcriptVisibilityOverrides[index] ?? defaultTranscriptOpen
            const isRollbackHighlight = highlightedMessageIndex === index

            return (
              <div
                key={`${message.role}-${index}`}
                ref={(node) => {
                  messageRefs.current[index] = node
                }}
                className={`group flex flex-col ${message.role === "user" ? "items-end" : "items-start"}`}
              >
                {message.role === "assistant" &&
                  (messageThinkingTrace.length > 0 || Boolean(message.isStreaming)) && (
                <div className="mb-1 w-full max-w-[90%]">
                  <button
                    type="button"
                    onClick={() => toggleTranscript(index, !isTranscriptOpen)}
                    aria-expanded={isTranscriptOpen}
                    className={`flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[10px] font-medium transition-colors ${
                      isTranscriptOpen
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-accent hover:text-foreground"
                    }`}
                  >
                    <BrainCircuit className="size-3" />
                    {message.isStreaming ? "Working" : "Details"}
                    <ChevronDown className={`size-3 transition-transform ${isTranscriptOpen ? "rotate-180" : ""}`} />
                  </button>

                  <div
                    data-state={isTranscriptOpen ? "open" : "closed"}
                    className="thinking-panel"
                    aria-hidden={!isTranscriptOpen}
                  >
                    <div className="thinking-panel-surface mt-2 rounded-xl border border-border/70 bg-card/80 p-3 shadow-sm">
                      <div className="mb-3 flex items-center justify-between gap-2">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                          Thinking Transcript
                        </div>
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                          {message.isStreaming && <span className="thinking-flow inline-block font-medium">Live</span>}
                          <span>{message.modelLabel ?? modelLabel}</span>
                        </div>
                      </div>

                      {messageThinkingTrace.length === 0 && message.isStreaming && (
                        <p className="thinking-step-enter text-[12px] leading-relaxed text-muted-foreground">
                          Preparing the first thinking step...
                        </p>
                      )}

                      <div className="space-y-3">
                        {messageThinkingTrace.map((entry, traceIndex) => (
                          <div
                            key={`${index}-trace-${traceIndex}`}
                            className="thinking-step-enter relative pl-5"
                            style={{ animationDelay: `${Math.min(traceIndex * 40, 180)}ms` }}
                          >
                            {traceIndex < messageThinkingTrace.length - 1 && (
                              <div className="absolute left-[4px] top-3 bottom-[-14px] w-px bg-border/60" />
                            )}
                            <div className={`absolute left-0 top-1.5 size-2 rounded-full ${getTraceDotClass(entry)}`} />

                            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                              {getTraceLabel(entry)}
                            </div>
                            {!shouldHideObservationLead(entry) && (
                              <p className="mt-1 whitespace-pre-wrap text-[12px] leading-relaxed text-foreground/85">
                                {entry.content}
                              </p>
                            )}

                            {entry.tool_name && (
                              <div className="mt-2 text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                                {entry.tool_name}
                              </div>
                            )}

                            {entry.tool_input && (
                              <pre className="mt-1 overflow-x-auto rounded-lg border border-border/70 bg-background/70 p-2.5 text-[11px] font-mono text-foreground/80 whitespace-pre-wrap">
                                {entry.tool_input}
                              </pre>
                            )}

                            {entry.details && (
                              <pre className="thinking-observation-preview mt-2 max-h-44 overflow-auto rounded-lg border border-border/70 bg-background/65 p-2.5 text-[11px] font-mono text-foreground/78 whitespace-pre-wrap">
                                {entry.details}
                              </pre>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                )}

                {message.role === "user" ? (
                  <div className={`prompt-message-bubble max-w-[90%] rounded-xl bg-primary px-3 py-2 text-[13px] leading-relaxed text-primary-foreground ${isRollbackHighlight ? "chat-rollback-bubble" : ""}`}>
                    {renderPromptTokens(message.content, datasets, "message")}
                  </div>
                ) : (
                  message.content ? (
                    index === chatMessages.length - 1 ? (
                      <div className={isRollbackHighlight ? "chat-rollback-bubble rounded-xl" : ""}>
                        <AssistantText key={message.content} text={message.content} />
                      </div>
                    ) : (
                      <div className={`max-w-[90%] px-1 py-1 text-[13px] leading-relaxed text-foreground/92 ${isRollbackHighlight ? "chat-rollback-bubble rounded-xl" : ""}`}>
                        {renderSimpleMarkdown(message.content)}
                      </div>
                    )
                  ) : null
                )}

                {message.role === "assistant" &&
                  Array.isArray(message.table_links) &&
                  message.table_links.length > 0 && (
                    <div className="mt-2 flex max-w-[90%] flex-wrap gap-1.5 px-1">
                      {message.table_links.map((tableLink) => (
                        <button
                          key={`${index}-${tableLink.id}`}
                          type="button"
                          onClick={() => onOpenTableLink(tableLink.id)}
                          className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-[10px] font-medium text-foreground hover:bg-accent transition-colors"
                          title="Open table in Data region"
                        >
                          <span className="text-primary truncate max-w-[200px]" title={tableLink.title}>
                            {tableLink.title}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}

                {message.role === "assistant" && message.code && (
                  <div className="mt-1 w-full max-w-[90%]">
                    <button
                      type="button"
                      onClick={() => setExpandedCodeIndex(expandedCodeIndex === index ? null : index)}
                      className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                        expandedCodeIndex === index
                          ? "text-primary bg-primary/10"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent opacity-0 group-hover:opacity-100"
                      }`}
                    >
                      <Code className="size-3" />
                      {expandedCodeIndex === index ? "Hide code" : "View code"}
                    </button>

                    {expandedCodeIndex === index && (
                      <div className="mt-1.5 w-full space-y-2">
                        <pre className="w-full overflow-x-auto rounded-lg bg-[oklch(0.1_0_0)] p-2.5 text-[11px] font-mono text-[oklch(0.7_0_0)] border border-border whitespace-pre">
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
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground space-y-1">
                Commands
              </div>
              <div
                ref={slashListRef}
                className="max-h-56 overflow-y-auto scrollbar-thin pr-1"
                onMouseLeave={hideSlashDescription}
              >
                <div className="relative space-y-1">
                  <div
                    className="pointer-events-none absolute left-0 right-0 rounded-md border border-primary/35 bg-primary/12 transform-gpu will-change-transform transition-transform duration-280 ease-[cubic-bezier(0.22,1,0.36,1)]"
                    style={{
                      height: `${SLASH_ROW_HEIGHT}px`,
                      transform: `translateY(${activeSlashIndex * (SLASH_ROW_HEIGHT + SLASH_ROW_GAP)}px)`,
                    }}
                  />

                  {filteredSlashCommands.map((item, index) => {
                    const isActive = index === activeSlashIndex
                    const showDelayedDescription = activeDescription === item.command

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
                          setSelectedSlashIndex((prev) => (prev === index ? prev : index))
                          scheduleSlashDescription(item.command)
                        }}
                        className={`relative z-10 h-8 w-full rounded-md px-2.5 text-left text-[12px] font-medium leading-none transition-colors duration-150 ${
                          isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {item.command}
                        {showDelayedDescription && (
                          <span className="pointer-events-none absolute left-2 right-2 bottom-[calc(100%+6px)] z-30 rounded-md border border-border bg-popover/95 px-2.5 py-1.5 text-[11px] font-medium leading-snug text-popover-foreground shadow-sm">
                            {item.description}
                          </span>
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>
          </div>

          <div className="rounded-xl border border-border bg-secondary/70 shadow-sm transition focus-within:border-ring/70 focus-within:ring-2 focus-within:ring-ring/15">
            <div className="relative min-h-20">
              <textarea
                ref={promptInputRef}
                className="relative block max-h-56 min-h-20 w-full resize-y rounded-t-xl bg-transparent px-3 py-2.5 text-sm leading-relaxed text-foreground outline-none placeholder:text-muted-foreground"
                aria-label="Ask DataPilot"
                placeholder="Ask DataPilot... Use @ to attach datasets or / for commands."
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
                    hideSlashDescription()
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
                      hideSlashDescription()
                      setSelectedSlashIndex((prev) => {
                        const next = (prev + 1) % filteredSlashCommands.length
                        scrollSlashOptionIntoView(next)
                        return next
                      })
                      return
                    }

                    if (event.key === "ArrowUp") {
                      event.preventDefault()
                      hideSlashDescription()
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
                      hideSlashDescription()
                      return
                    }
                  }

                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault()
                    if (!isRunning) {
                      onSubmit()
                    }
                  }
                }}
              />
            </div>

            <div className="flex items-center justify-between gap-2 px-2 py-2">
              <div className="flex min-w-0 items-center gap-2">
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

                {isRunning && (
                  <div className="flex min-w-0 items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <Loader2 className="size-3.5 shrink-0 animate-spin" />
                    <span className="truncate">{thinkingMode ? "Thinking..." : "Running..."}</span>
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
                  onClick={onSubmit}
                  disabled={!prompt.trim()}
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
