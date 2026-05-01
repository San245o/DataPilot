"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react"
import Papa from "papaparse"
import * as XLSX from "xlsx"
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Download,
  FileSpreadsheet,
  Moon,
  Sun,
  Upload,
} from "lucide-react"

import {
  clamp,
  DEFAULT_THINKING_MODEL,
  getNextWidgetZIndex,
  INITIAL_CHAT_MESSAGES,
  MODEL_OPTIONS,
  normalizeRows,
  seedRows,
  THINKING_MODEL_OPTIONS,
  transformationSteps,
  type ChatMessage,
  type FullscreenPanel,
  type HighlightedColumn,
  type PivotTableTab,
  type SheetRow,
  type VisualizationPayload,
  type VizWidget,
} from "@/components/dashboard/dashboard-shared"
import { CanvasBoard } from "@/components/dashboard/components/canvas-board"
import { ChatPanel } from "@/components/dashboard/components/chat-panel"
import { DataGrid } from "@/components/dashboard/components/data-grid"
import { useAgentRunner } from "@/components/dashboard/hooks/use-agent-runner"

export function AgentDashboard() {
  const [rows, setRows] = useState<SheetRow[]>(seedRows)
  const [datasetName, setDatasetName] = useState("gapminder-lite.xlsx")
  const [modelName, setModelName] = useState("gemini-3.1-flash-lite-preview")
  const [prompt, setPrompt] = useState("")
  const [widgets, setWidgets] = useState<VizWidget[]>([])
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isDark, setIsDark] = useState(true)
  const [highlightedRows, setHighlightedRows] = useState<Set<number>>(new Set())
  const [highlightedColumns, setHighlightedColumns] = useState<HighlightedColumn[]>([])
  const [fullscreenPanel, setFullscreenPanel] = useState<FullscreenPanel>(null)
  const [splitRatio, setSplitRatio] = useState(0.38)
  const [isSplitterDragging, setIsSplitterDragging] = useState(false)
  const [chatWidth, setChatWidth] = useState(360)
  const [isChatSplitterDragging, setIsChatSplitterDragging] = useState(false)
  const [pivotTables, setPivotTables] = useState<PivotTableTab[]>([])
  const [activeDataTab, setActiveDataTab] = useState("base")
  const [isContextOpen, setIsContextOpen] = useState(false)
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0, total: 0 })
  const [queryCount, setQueryCount] = useState(0)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_CHAT_MESSAGES)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [thinkingMode, setThinkingMode] = useState(false)

  const splitContainerRef = useRef<HTMLDivElement | null>(null)
  const mainContentRef = useRef<HTMLDivElement | null>(null)
  const newTableTimeoutsRef = useRef<number[]>([])
  const modelMenuRef = useRef<HTMLDivElement | null>(null)

  const { clearError, error, isRunning, runAgent, setError } = useAgentRunner()

  const availableModels = useMemo(
    () => (thinkingMode ? THINKING_MODEL_OPTIONS : MODEL_OPTIONS),
    [thinkingMode]
  )
  const selectedModelLabel = useMemo(
    () => availableModels.find((option) => option.value === modelName)?.label ?? modelName,
    [availableModels, modelName]
  )

  const baseShapeLabel = useMemo(() => {
    const baseColumnCount = Object.keys(rows[0] ?? {}).length
    return `${rows.length}X${baseColumnCount}`
  }, [rows])

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark)
  }, [isDark])

  useEffect(() => {
    const timeoutIds = newTableTimeoutsRef.current
    return () => {
      timeoutIds.forEach((timeoutId) => window.clearTimeout(timeoutId))
    }
  }, [])

  useEffect(() => {
    if (!modelMenuOpen) return

    const handleClickOutside = (event: MouseEvent) => {
      if (!modelMenuRef.current?.contains(event.target as Node)) {
        setModelMenuOpen(false)
      }
    }

    document.addEventListener("click", handleClickOutside)
    return () => document.removeEventListener("click", handleClickOutside)
  }, [modelMenuOpen])

  useEffect(() => {
    if (!isSplitterDragging) return

    const handleMouseMove = (event: MouseEvent) => {
      const container = splitContainerRef.current
      if (!container) return

      const rect = container.getBoundingClientRect()
      const nextRatio = clamp((event.clientY - rect.top) / rect.height, 0.15, 0.85)
      setSplitRatio(nextRatio)
    }

    const handleMouseUp = () => setIsSplitterDragging(false)
    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isSplitterDragging])

  useEffect(() => {
    if (!isChatSplitterDragging) return

    const handleMouseMove = (event: MouseEvent) => {
      const container = mainContentRef.current
      if (!container) return

      const rect = container.getBoundingClientRect()
      const nextWidth = clamp(rect.right - event.clientX, 280, 600)
      setChatWidth(nextWidth)
    }

    const handleMouseUp = () => setIsChatSplitterDragging(false)
    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isChatSplitterDragging])

  const toggleTheme = useCallback(() => {
    setIsDark((prev) => !prev)
  }, [])

  const toggleThinkingMode = useCallback(() => {
    setThinkingMode((prev) => {
      const next = !prev
      if (next) {
        setModelName(DEFAULT_THINKING_MODEL)
      }
      return next
    })
    setModelMenuOpen(false)
  }, [])

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const extension = file.name.split(".").pop()?.toLowerCase()
    let parsedRows: Record<string, unknown>[] = []

    if (extension === "xlsx" || extension === "xls") {
      const buffer = await file.arrayBuffer()
      const workbook = XLSX.read(buffer, { type: "array" })
      const firstSheetName = workbook.SheetNames[0]
      const firstSheet = workbook.Sheets[firstSheetName]
      parsedRows = XLSX.utils.sheet_to_json<Record<string, unknown>>(firstSheet, { defval: "" })
    } else {
      const text = await file.text()
      const parsed = Papa.parse<Record<string, unknown>>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
      })
      parsedRows = (parsed.data ?? []).filter((row) => Object.keys(row).length > 0)
    }

    if (parsedRows.length === 0) return

    setRows(normalizeRows(parsedRows))
    setDatasetName(file.name)
    setWidgets([])
    setPivotTables([])
    setActiveDataTab("base")
    setHighlightedRows(new Set())
    setHighlightedColumns([])
    clearError()
  }

  const handleDownloadExcel = () => {
    if (rows.length === 0) return

    const worksheet = XLSX.utils.json_to_sheet(rows)
    const workbook = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(workbook, worksheet, "Data")
    XLSX.writeFile(workbook, `modified_${datasetName.replace(/\.[^/.]+$/, "")}.xlsx`)
  }

  const activateBaseData = useCallback(() => {
    setActiveDataTab("base")
  }, [])

  const openPivotTable = useCallback((pivotId: string) => {
    setPivotTables((prev) => prev.map((tab) => (tab.id === pivotId ? { ...tab, open: true } : tab)))
    setActiveDataTab(pivotId)
  }, [])

  const closePivotTable = useCallback((pivotId: string) => {
    setPivotTables((prev) => prev.map((tab) => (tab.id === pivotId ? { ...tab, open: false } : tab)))
    setActiveDataTab((prev) => (prev === pivotId ? "base" : prev))
  }, [])

  const addVisualizationWidget = useCallback((visualization: VisualizationPayload | null) => {
    if (!visualization || !Array.isArray(visualization.data) || visualization.data.length === 0) return

    setWidgets((prev) => {
      const nextIndex = prev.length
      const nextZIndex = getNextWidgetZIndex(prev)

      return [
        ...prev,
        {
          id: `${Date.now()}-${nextIndex}`,
          title: `Chart ${nextIndex + 1}`,
          data: visualization.data ?? [],
          layout: visualization.layout,
          frames: visualization.frames,
          x: 20 + (nextIndex % 3) * 40,
          y: 20 + (nextIndex % 3) * 30,
          width: 460,
          height: 320,
          zIndex: nextZIndex,
        },
      ]
    })
  }, [])

  const handleRunAgent = useCallback(async () => {
    if (isRunning || !prompt.trim()) return

    clearError()

    const outgoingPrompt = prompt.trim()
    const isHelpCommand = outgoingPrompt.toLowerCase() === "/help"
    const historyWithUser: ChatMessage[] = [
      ...chatMessages,
      { role: "user", content: outgoingPrompt },
    ]
    const streamingPlaceholder: ChatMessage | null = thinkingMode
      ? {
          role: "assistant",
          content: "",
          thinkingTrace: [],
          modelLabel: selectedModelLabel,
          isStreaming: true,
        }
      : null

    setChatMessages(streamingPlaceholder ? [...historyWithUser, streamingPlaceholder] : historyWithUser)
    setPrompt("")

    if (isHelpCommand) {
      setChatMessages((prev) => [
        ...(thinkingMode ? prev.slice(0, -1) : prev),
        {
          role: "assistant",
          content:
            "Try prompts like: `/filter movies after 2010`, `/modify rename Electronic to Electronics`, `/extract titles directed by Christopher Nolan`, `/visualize total revenue by category`, `/summarize this dataset`, or ask in plain English for edits, filters, summaries, and charts.",
        },
      ])
      return
    }

    const result = await runAgent({
      prompt: outgoingPrompt,
      rows,
      modelName,
      history: historyWithUser,
      thinkingMode,
      onThinkingTrace: (entry) => {
        setChatMessages((prev) => {
          if (!thinkingMode || prev.length === 0) return prev
          const next = [...prev]
          const lastMessage = next[next.length - 1]
          if (lastMessage?.role !== "assistant" || !lastMessage.isStreaming) return prev
          next[next.length - 1] = {
            ...lastMessage,
            thinkingTrace: [...(lastMessage.thinkingTrace ?? []), entry],
          }
          return next
        })
      },
    })

    if (!result.ok) {
      if (thinkingMode) {
        setChatMessages((prev) => {
          if (prev.length === 0) return prev
          const next = [...prev]
          const lastMessage = next[next.length - 1]
          if (lastMessage?.role === "assistant" && lastMessage.isStreaming) {
            next[next.length - 1] = {
              ...lastMessage,
              content: `Error: ${result.error}`,
              isStreaming: false,
            }
            return next
          }
          return [...prev, { role: "assistant", content: `Error: ${result.error}` }]
        })
      } else {
        setChatMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${result.error}` },
        ])
      }
      return
    }

    const { data } = result

    setQueryCount((prev) => prev + 1)
    if (data.tokenUsage) {
      setTotalTokens((prev) => ({
        prompt: prev.prompt + (data.tokenUsage?.prompt_tokens || 0),
        completion: prev.completion + (data.tokenUsage?.completion_tokens || 0),
        total: prev.total + (data.tokenUsage?.total_tokens || 0),
      }))
    }

    if (data.responseTables.length > 0) {
      setPivotTables((prev) => {
        const byId = new Map(prev.map((item) => [item.id, item] as const))
        data.responseTables.forEach((table) => {
          byId.set(table.id, table)
        })
        return Array.from(byId.values())
      })

      setActiveDataTab(data.responseTables[0].id)

      const timeoutId = window.setTimeout(() => {
        setPivotTables((prev) =>
          prev.map((tab) =>
            data.responseTables.some((table) => table.id === tab.id)
              ? { ...tab, isNew: false }
              : tab
          )
        )
      }, 2000)
      newTableTimeoutsRef.current.push(timeoutId)
    }

    if (data.mutation && data.nextRows.length > 0) {
      setRows(data.nextRows)
      setActiveDataTab("base")
    }

    setHighlightedRows(new Set(data.highlightIndices))
    setHighlightedColumns(data.highlightedColumns)
    addVisualizationWidget(data.visualizationPayload)
    if (thinkingMode) {
      setChatMessages((prev) => {
        if (prev.length === 0) return prev
        const next = [...prev]
        const lastMessage = next[next.length - 1]
        const finalAssistant = {
          ...data.assistantMessage,
          modelLabel: selectedModelLabel,
          isStreaming: false,
        }
        if (lastMessage?.role === "assistant" && lastMessage.isStreaming) {
          next[next.length - 1] = finalAssistant
          return next
        }
        return [...prev, finalAssistant]
      })
    } else {
      setChatMessages((prev) => [
        ...prev,
        {
          ...data.assistantMessage,
          modelLabel: selectedModelLabel,
        },
      ])
    }
  }, [addVisualizationWidget, chatMessages, clearError, isRunning, modelName, prompt, rows, runAgent, selectedModelLabel, thinkingMode])

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground font-sans antialiased">
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <aside
          className={`shrink-0 flex flex-col border-r border-border bg-card/50 backdrop-blur-sm transition-all duration-200 ${
            sidebarCollapsed ? "w-14" : "w-60"
          }`}
        >
          <div className="flex h-10 items-center justify-end border-b border-border px-2">
            <button
              type="button"
              onClick={() => setSidebarCollapsed((prev) => !prev)}
              className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? <ChevronRight className="size-3.5" /> : <ChevronLeft className="size-3.5" />}
            </button>
          </div>

          <div className="border-b border-border px-2 py-2">
            <div className={`flex items-center gap-2.5 rounded-lg px-2 py-1.5 ${sidebarCollapsed ? "justify-center" : ""}`}>
              <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground shrink-0">
                <FileSpreadsheet className="size-4" />
              </div>
              {!sidebarCollapsed && (
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-bold tracking-tight text-foreground">DataPilot</div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="font-medium">{baseShapeLabel}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-2 space-y-2">
            <label
              className={`group flex cursor-pointer items-center gap-2.5 rounded-lg border border-dashed border-border p-2.5 text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground hover:bg-accent/50 ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
            >
              <Upload className="size-4 shrink-0" />
              {!sidebarCollapsed && <span className="text-xs font-medium truncate">Upload CSV / XLSX</span>}
              <input
                className="hidden"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileUpload}
              />
            </label>

            <button
              type="button"
              onClick={handleDownloadExcel}
              disabled={rows.length === 0}
              className={`w-full group flex items-center gap-2.5 rounded-lg border border-dashed border-border p-2.5 text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground hover:bg-accent/50 disabled:opacity-50 disabled:cursor-not-allowed ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
              title="Download modified data"
            >
              <Download className="size-4 shrink-0" />
              {!sidebarCollapsed && <span className="text-xs font-medium truncate">Download Excel</span>}
            </button>
          </div>

          {!sidebarCollapsed ? (
            <div className="px-3 py-2">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">File</div>
              <div className="text-xs font-medium text-foreground truncate" title={datasetName}>
                {datasetName}
              </div>
            </div>
          ) : (
            <div className="flex justify-center py-2" title={datasetName}>
              <FileSpreadsheet className="size-4 text-muted-foreground" />
            </div>
          )}

          <div className="px-2 py-2 space-y-1">
            {!sidebarCollapsed && (
              <div className="px-1 mb-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                Workflow
              </div>
            )}

            {transformationSteps.map((step, index) => {
              const Icon = step.icon

              if (step.isContextButton) {
                return (
                  <div key={step.label} className="flex flex-col">
                    <button
                      type="button"
                      onClick={() => setIsContextOpen((prev) => !prev)}
                      className={`flex items-center w-full gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors ${
                        sidebarCollapsed ? "justify-center" : ""
                      } text-muted-foreground hover:bg-accent hover:text-foreground`}
                      title={step.label}
                    >
                      <Icon className="size-3.5 shrink-0" />
                      {!sidebarCollapsed && (
                        <div className="flex flex-1 items-center justify-between">
                          <span className="font-medium">{step.label}</span>
                          {isContextOpen ? <ChevronRight className="size-3.5 rotate-90" /> : <ChevronRight className="size-3.5" />}
                        </div>
                      )}
                    </button>

                    {!sidebarCollapsed && isContextOpen && (
                      <div className="space-y-1.5 flex flex-col text-xs text-muted-foreground ml-2 mt-1">
                        <div className="flex justify-between items-center bg-secondary/50 px-2 py-1.5 rounded">
                          <span>Queries:</span>
                          <span className="font-medium text-foreground">{queryCount}</span>
                        </div>
                        <div className="flex justify-between items-center bg-secondary/50 px-2 py-1.5 rounded">
                          <span>Prompt:</span>
                          <span className="font-medium text-foreground">{totalTokens.prompt.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center bg-secondary/50 px-2 py-1.5 rounded">
                          <span>Completion:</span>
                          <span className="font-medium text-foreground">{totalTokens.completion.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center bg-primary/10 text-primary px-2 py-1.5 rounded">
                          <span className="font-semibold">Total Tokens:</span>
                          <span className="font-bold">{totalTokens.total.toLocaleString()}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )
              }

              return (
                <div
                  key={step.label}
                  className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors ${
                    sidebarCollapsed ? "justify-center" : ""
                  } ${index === 0 ? "bg-primary/10 text-primary" : "text-muted-foreground"}`}
                  title={step.label}
                >
                  <Icon className="size-3.5 shrink-0" />
                  {!sidebarCollapsed && <span className="font-medium">{step.label}</span>}
                </div>
              )
            })}

            <button
              type="button"
              onClick={toggleTheme}
              className={`w-full flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
              title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDark ? <Sun className="size-3.5 shrink-0" /> : <Moon className="size-3.5 shrink-0" />}
              {!sidebarCollapsed && <span className="font-medium">{isDark ? "Light mode" : "Dark mode"}</span>}
            </button>
          </div>

          {!sidebarCollapsed && (
            <div className="mt-auto border-t border-border p-3">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">
                Model
              </div>
              <div ref={modelMenuRef} className="relative">
                <button
                  type="button"
                  onClick={() => setModelMenuOpen((prev) => !prev)}
                  className="flex h-9 w-full items-center justify-between rounded-lg border border-border/80 bg-secondary/85 px-3 text-xs font-medium text-foreground outline-none transition-[border-color,background-color,box-shadow,transform] duration-200 ease-[var(--menu-ease)] hover:border-primary/35 hover:bg-accent/70 focus-visible:ring-2 focus-visible:ring-ring/60"
                >
                  <span className="truncate">{selectedModelLabel}</span>
                  <ChevronDown className={`size-3.5 shrink-0 text-muted-foreground transition-transform duration-200 ease-[var(--menu-ease)] ${modelMenuOpen ? "rotate-180 text-foreground" : ""}`} />
                </button>

                <div
                  data-state={modelMenuOpen ? "open" : "closed"}
                  className="menu-surface menu-popover absolute bottom-full left-0 right-0 mb-2 overflow-hidden rounded-xl p-1.5 z-50"
                  data-side="top"
                >
                  {availableModels.map((option) => {
                    const isActive = option.value === modelName
                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => {
                          setModelName(option.value)
                          setModelMenuOpen(false)
                        }}
                        className={`flex w-full items-center rounded-lg px-3 py-2 text-left text-[12px] transition-all duration-200 ease-[var(--menu-ease)] ${
                          isActive
                            ? "bg-primary/14 text-foreground shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--primary)_24%,transparent)]"
                            : "text-muted-foreground hover:bg-accent/80 hover:text-foreground"
                        }`}
                      >
                        <span className="truncate">{option.label}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          )}
        </aside>

        <div ref={mainContentRef} className="flex flex-1 min-h-0 min-w-0 overflow-hidden">
          <div ref={splitContainerRef} className="flex flex-1 min-w-0 flex-col min-h-0 overflow-hidden">
            <div
              className="flex flex-col min-h-[120px] shrink-0 overflow-hidden"
              style={fullscreenPanel === "data" ? {} : { height: `calc(${splitRatio * 100}% - 2px)` }}
            >
              <DataGrid
                rows={rows}
                pivotTables={pivotTables}
                activeDataTab={activeDataTab}
                highlightedRows={highlightedRows}
                highlightedColumns={highlightedColumns}
                fullscreen={fullscreenPanel === "data"}
                onActivateBaseData={activateBaseData}
                onOpenPivotTable={openPivotTable}
                onClosePivotTable={closePivotTable}
                onToggleFullscreen={() => setFullscreenPanel(fullscreenPanel === "data" ? null : "data")}
              />
            </div>

            {fullscreenPanel === null && (
              <div
                className={`shrink-0 splitter-handle ${isSplitterDragging ? "dragging" : ""}`}
                onMouseDown={() => setIsSplitterDragging(true)}
              />
            )}

            <CanvasBoard
              widgets={widgets}
              setWidgets={setWidgets}
              isDark={isDark}
              fullscreen={fullscreenPanel === "canvas"}
              onToggleFullscreen={() => setFullscreenPanel(fullscreenPanel === "canvas" ? null : "canvas")}
              onError={setError}
            />
          </div>

          <div
            className={`w-1 shrink-0 cursor-col-resize splitter-handle-horizontal ${isChatSplitterDragging ? "dragging" : ""}`}
            onMouseDown={() => setIsChatSplitterDragging(true)}
          />

          <ChatPanel
            width={chatWidth}
            chatMessages={chatMessages}
            prompt={prompt}
            isRunning={isRunning}
            error={error}
            thinkingMode={thinkingMode}
            modelLabel={selectedModelLabel}
            onPromptChange={(value) => setPrompt(value)}
            onSubmit={handleRunAgent}
            onToggleThinkingMode={toggleThinkingMode}
            onClearError={clearError}
            onOpenTableLink={openPivotTable}
          />
        </div>
      </div>
    </div>
  )
}
