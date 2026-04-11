"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { Code, Loader2, Send, X } from "lucide-react"

import {
  getActiveSlashQuery,
  SLASH_COMMANDS,
  SLASH_DESCRIPTION_DELAY_MS,
  SLASH_ROW_GAP,
  SLASH_ROW_HEIGHT,
  type ChatMessage,
  type SlashCommandOption,
} from "@/components/dashboard/dashboard-shared"

type ChatPanelProps = {
  width: number
  chatMessages: ChatMessage[]
  prompt: string
  isRunning: boolean
  error: string | null
  onPromptChange: (value: string) => void
  onSubmit: () => void
  onClearError: () => void
  onOpenTableLink: (tableId: string) => void
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

export function ChatPanel({
  width,
  chatMessages,
  prompt,
  isRunning,
  error,
  onPromptChange,
  onSubmit,
  onClearError,
  onOpenTableLink,
}: ChatPanelProps) {
  const [expandedCodeIndex, setExpandedCodeIndex] = useState<number | null>(null)
  const [selectedSlashIndex, setSelectedSlashIndex] = useState(0)
  const [slashDismissed, setSlashDismissed] = useState(false)
  const [visibleSlashDescriptionFor, setVisibleSlashDescriptionFor] = useState<string | null>(null)

  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const promptInputRef = useRef<HTMLTextAreaElement | null>(null)
  const slashListRef = useRef<HTMLDivElement | null>(null)
  const slashDescriptionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const slashQuery = useMemo(() => getActiveSlashQuery(prompt), [prompt])
  const filteredSlashCommands = useMemo(() => {
    if (slashQuery === null) return []
    if (!slashQuery) return SLASH_COMMANDS
    return SLASH_COMMANDS.filter((item) => item.command.slice(1).startsWith(slashQuery))
  }, [slashQuery])
  const showSlashMenu = slashQuery !== null && !slashDismissed && filteredSlashCommands.length > 0
  const activeSlashIndex = showSlashMenu
    ? Math.min(selectedSlashIndex, filteredSlashCommands.length - 1)
    : 0
  const activeDescription = showSlashMenu ? visibleSlashDescriptionFor : null

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatMessages, isRunning])

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

  const insertSlashCommand = useCallback((item: SlashCommandOption) => {
    const nextValue = `${item.command} `
    onPromptChange(nextValue)
    setSlashDismissed(false)
    setSelectedSlashIndex(0)
    setVisibleSlashDescriptionFor(null)
    clearSlashDescriptionTimer()

    requestAnimationFrame(() => {
      const input = promptInputRef.current
      if (!input) return
      input.focus()
      input.setSelectionRange(nextValue.length, nextValue.length)
    })
  }, [clearSlashDescriptionTimer, onPromptChange])

  return (
    <div className="flex shrink-0 flex-col bg-card/30" style={{ width }}>
      <div className="flex-1 min-h-0 overflow-auto p-3 space-y-3 scrollbar-thin">
        {chatMessages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`group flex flex-col ${message.role === "user" ? "items-end" : "items-start"}`}
          >
            {message.role === "user" ? (
              <div className="max-w-[90%] rounded-xl bg-primary px-3 py-2 text-[13px] leading-relaxed text-primary-foreground">
                {message.content}
              </div>
            ) : (
              index === chatMessages.length - 1 ? (
                <AssistantText key={message.content} text={message.content} />
              ) : (
                <div className="max-w-[90%] px-1 py-1 text-[13px] leading-relaxed text-foreground/92">
                  {renderSimpleMarkdown(message.content)}
                </div>
              )
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
        ))}

        {isRunning && (
          <div className="flex items-start px-1 py-1 text-[13px] leading-relaxed text-muted-foreground">
            <span className="thinking-flow inline-block font-medium">Thinking</span>
          </div>
        )}

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
        <div className="relative">
          <div
            data-state={showSlashMenu ? "open" : "closed"}
            data-side="top"
            className="menu-surface menu-popover absolute bottom-full left-0 right-11 mb-2 rounded-lg p-2 shadow-md"
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

          <div className="flex items-center gap-2">
            <textarea
              ref={promptInputRef}
              className="flex-1 resize-none rounded-lg border border-border bg-secondary px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:ring-1 focus:ring-ring"
              placeholder="Ask something... (type / for commands)"
              rows={2}
              value={prompt}
              onChange={(event) => {
                onPromptChange(event.target.value)
                setSlashDismissed(false)
                if (!event.target.value.trimStart().startsWith("/")) {
                  hideSlashDescription()
                }
              }}
              onKeyDown={(event) => {
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
                    const selected = filteredSlashCommands[selectedSlashIndex]
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

            <button
              type="button"
              onClick={() => {
                if (!isRunning) {
                  onSubmit()
                }
              }}
              disabled={isRunning || !prompt.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center self-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:pointer-events-none disabled:opacity-40"
            >
              {isRunning ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
