"use client"

import { useEffect, useMemo, useRef, useState, useCallback } from "react"
import Papa from "papaparse"
import * as XLSX from "xlsx"
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Code,
  Database,
  FileSpreadsheet,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  Moon,
  Send,
  Sparkles,
  Sun,
  Upload,
  X,
} from "lucide-react"
import type { Data, Layout } from "plotly.js"

import { PlotlyBoard } from "@/components/charts/plotly-board"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

type CellValue = string | number | boolean | null
type SheetRow = Record<string, CellValue>
type VisualizationPayload = {
  data?: Data[]
  layout?: Partial<Layout>
}
type ContextPreview = {
  columns?: string[]
  sample_rows?: SheetRow[]
}
type AgentExecuteResponse = {
  rows?: SheetRow[]
  visualization?: VisualizationPayload | null
  query_output?: string | null
  code?: string
  assistant_reply?: string
  context_preview?: ContextPreview
  detail?: string
  mutation?: boolean
  highlight_indices?: number[]
}
type ChatMessage = {
  role: "user" | "assistant"
  content: string
  code?: string
}
type VizWidget = {
  id: string
  title: string
  data: Data[]
  layout?: Partial<Layout>
  x: number
  y: number
  width: number
  height: number
  zIndex: number
}
type DragState = {
  id: string
  offsetX: number
  offsetY: number
} | null

type ResizeState = {
  id: string
  startX: number
  startY: number
  startWidth: number
  startHeight: number
  corner: "se" | "sw" | "ne" | "nw"
} | null

type ThinkingStep = {
  id: string
  label: string
  status: "pending" | "active" | "done"
}

type FullscreenPanel = "data" | "canvas" | null

const seedRows: SheetRow[] = [
  { year: 2000, country: "Kenya", fertility: 4.9, life_expectancy: 52.1, pop_size: 3.7 },
  { year: 2005, country: "Kenya", fertility: 4.6, life_expectancy: 56.7, pop_size: 4.1 },
  { year: 2010, country: "Kenya", fertility: 4.3, life_expectancy: 60.8, pop_size: 4.9 },
  { year: 2000, country: "Japan", fertility: 1.4, life_expectancy: 81.2, pop_size: 3.1 },
  { year: 2005, country: "Japan", fertility: 1.3, life_expectancy: 82.1, pop_size: 2.9 },
  { year: 2010, country: "Japan", fertility: 1.4, life_expectancy: 83.1, pop_size: 3.0 },
  { year: 2000, country: "Brazil", fertility: 2.4, life_expectancy: 70.1, pop_size: 5.9 },
  { year: 2005, country: "Brazil", fertility: 2.1, life_expectancy: 72.4, pop_size: 6.3 },
  { year: 2010, country: "Brazil", fertility: 1.9, life_expectancy: 74.2, pop_size: 6.8 },
]

const transformationSteps = [
  { label: "Upload", icon: Upload },
  { label: "Infer", icon: Database },
  { label: "Context", icon: Layers },
]

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend"

// Virtualization constants
const ROW_HEIGHT = 28 // Height of each row in pixels
const OVERSCAN = 5 // Extra rows to render above/below viewport

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(value, max))
}

/** Lightweight markdown renderer for chat bubbles:
 *  - Lines starting with `- ` or `* ` become <li> bullet items
 *  - **bold** text
 *  - `inline code`
 *  - Newlines become <br /> between non-list lines
 */
function renderSimpleMarkdown(text: string) {
  const lines = text.split("\n")
  const elements: React.ReactNode[] = []
  let listBuffer: string[] = []

  const flushList = () => {
    if (listBuffer.length === 0) return
    elements.push(
      <ul key={`ul-${elements.length}`} className="list-disc pl-4 space-y-0.5">
        {listBuffer.map((item, i) => (
          <li key={i}>{formatInline(item)}</li>
        ))}
      </ul>
    )
    listBuffer = []
  }

  const formatInline = (s: string): React.ReactNode => {
    // Handle **bold** and `code`
    const parts = s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <code key={i} className="rounded bg-muted px-1 py-0.5 text-[11px] font-mono">
            {part.slice(1, -1)}
          </code>
        )
      }
      return part
    })
  }

  for (const line of lines) {
    const trimmed = line.trimStart()
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      listBuffer.push(trimmed.slice(2))
    } else {
      flushList()
      if (trimmed === "") {
        elements.push(<br key={`br-${elements.length}`} />)
      } else {
        elements.push(
          <span key={`p-${elements.length}`}>
            {elements.length > 0 && <br />}
            {formatInline(trimmed)}
          </span>
        )
      }
    }
  }
  flushList()
  return <>{elements}</>
}

function normalizeRows(rows: Record<string, unknown>[]): SheetRow[] {
  return rows.map((row) => {
    const normalized: SheetRow = {}
    Object.entries(row).forEach(([key, value]) => {
      if (
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean" ||
        value === null
      ) {
        normalized[key] = value
      } else {
        normalized[key] = value == null ? "" : String(value)
      }
    })
    return normalized
  })
}

export function AgentDashboard() {
  const [rows, setRows] = useState<SheetRow[]>(seedRows)
  const [datasetName, setDatasetName] = useState("gapminder-lite.xlsx")
  const [modelName, setModelName] = useState("gemini-3-flash-preview")
  const [prompt, setPrompt] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [widgets, setWidgets] = useState<VizWidget[]>([])
  const [maxZIndex, setMaxZIndex] = useState(1)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [dragState, setDragState] = useState<DragState>(null)
  const [resizeState, setResizeState] = useState<ResizeState>(null)
  const [isDark, setIsDark] = useState(true)
  const [highlightedRows, setHighlightedRows] = useState<Set<number>>(new Set())
  const [fullscreenPanel, setFullscreenPanel] = useState<FullscreenPanel>(null)
  const [splitRatio, setSplitRatio] = useState(0.38)
  const [isSplitterDragging, setIsSplitterDragging] = useState(false)
  const [chatWidth, setChatWidth] = useState(360)
  const [isChatSplitterDragging, setIsChatSplitterDragging] = useState(false)
  const [thinkingMode, setThinkingMode] = useState(false)
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([])
  const [expandedCodeIndex, setExpandedCodeIndex] = useState<number | null>(null)
  const [chartMenuOpen, setChartMenuOpen] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Upload a sheet and ask for transformations or charts. I can generate Plotly 2D and 3D visualizations.",
    },
  ])

  const boardRef = useRef<HTMLDivElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const splitContainerRef = useRef<HTMLDivElement | null>(null)
  const tableContainerRef = useRef<HTMLDivElement | null>(null)
  const mainContentRef = useRef<HTMLDivElement | null>(null)
  const headers = useMemo(() => Object.keys(rows[0] ?? {}), [rows])

  // Virtualization state
  const [scrollTop, setScrollTop] = useState(0)
  const [tableHeight, setTableHeight] = useState(400)

  // Convert highlighted rows set to sorted array for navigation
  const highlightedRowsArray = useMemo(() => {
    return Array.from(highlightedRows).sort((a, b) => a - b)
  }, [highlightedRows])
  
  const [currentHighlightIndex, setCurrentHighlightIndex] = useState(0)

  // Calculate virtualized rows
  const virtualizedData = useMemo(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
    const visibleCount = Math.ceil(tableHeight / ROW_HEIGHT) + OVERSCAN * 2
    const endIndex = Math.min(rows.length, startIndex + visibleCount)
    
    return {
      startIndex,
      endIndex,
      visibleRows: rows.slice(startIndex, endIndex),
      totalHeight: rows.length * ROW_HEIGHT,
      offsetTop: startIndex * ROW_HEIGHT
    }
  }, [rows, scrollTop, tableHeight])

  // Reset highlight index when highlighted rows change
  useEffect(() => {
    setCurrentHighlightIndex(0)
  }, [highlightedRows])

  // Update table height on resize
  useEffect(() => {
    const updateTableHeight = () => {
      if (tableContainerRef.current) {
        setTableHeight(tableContainerRef.current.clientHeight)
      }
    }
    updateTableHeight()
    window.addEventListener("resize", updateTableHeight)
    return () => window.removeEventListener("resize", updateTableHeight)
  }, [splitRatio, fullscreenPanel])

  // Navigate to highlighted row
  const navigateToHighlightedRow = useCallback((direction: "prev" | "next") => {
    if (highlightedRowsArray.length === 0) return
    
    let newIndex: number
    if (direction === "prev") {
      newIndex = currentHighlightIndex > 0 ? currentHighlightIndex - 1 : highlightedRowsArray.length - 1
    } else {
      newIndex = currentHighlightIndex < highlightedRowsArray.length - 1 ? currentHighlightIndex + 1 : 0
    }
    setCurrentHighlightIndex(newIndex)
    
    // Scroll to the row using virtualization offset
    const rowIndex = highlightedRowsArray[newIndex]
    const tableContainer = tableContainerRef.current
    if (tableContainer) {
      const targetScrollTop = rowIndex * ROW_HEIGHT - tableHeight / 2 + ROW_HEIGHT / 2
      tableContainer.scrollTop = Math.max(0, targetScrollTop)
    }
  }, [highlightedRowsArray, currentHighlightIndex, tableHeight])

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatMessages])

  // Close chart menu when clicking outside
  useEffect(() => {
    if (!chartMenuOpen) return
    const handleClickOutside = () => setChartMenuOpen(false)
    document.addEventListener("click", handleClickOutside)
    return () => document.removeEventListener("click", handleClickOutside)
  }, [chartMenuOpen])

  // Dark mode toggle
  const toggleTheme = useCallback(() => {
    setIsDark((prev) => {
      const next = !prev
      document.documentElement.classList.toggle("dark", next)
      return next
    })
  }, [])

  // Splitter drag handling (vertical - data/canvas split)
  useEffect(() => {
    if (!isSplitterDragging) return

    const handleMouseMove = (event: MouseEvent) => {
      const container = splitContainerRef.current
      if (!container) return
      const rect = container.getBoundingClientRect()
      const newRatio = clamp((event.clientY - rect.top) / rect.height, 0.15, 0.85)
      setSplitRatio(newRatio)
    }

    const handleMouseUp = () => setIsSplitterDragging(false)

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isSplitterDragging])

  // Chat splitter drag handling (horizontal - main content/chat split)
  useEffect(() => {
    if (!isChatSplitterDragging) return

    const handleMouseMove = (event: MouseEvent) => {
      const container = mainContentRef.current
      if (!container) return
      const rect = container.getBoundingClientRect()
      const newWidth = clamp(rect.right - event.clientX, 280, 600)
      setChatWidth(newWidth)
    }

    const handleMouseUp = () => setIsChatSplitterDragging(false)

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isChatSplitterDragging])

  // Bring widget to front
  const bringToFront = useCallback((widgetId: string) => {
    setMaxZIndex((prev) => {
      const newZ = prev + 1
      setWidgets((widgets) =>
        widgets.map((w) => (w.id === widgetId ? { ...w, zIndex: newZ } : w))
      )
      return newZ
    })
  }, [])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const ext = file.name.split(".").pop()?.toLowerCase()

    if (ext === "xlsx" || ext === "xls") {
      const buffer = await file.arrayBuffer()
      const workbook = XLSX.read(buffer, { type: "array" })
      const firstSheetName = workbook.SheetNames[0]
      const firstSheet = workbook.Sheets[firstSheetName]
      const parsedRows = XLSX.utils.sheet_to_json<Record<string, unknown>>(firstSheet, { defval: "" })
      if (parsedRows.length > 0) {
        setRows(normalizeRows(parsedRows))
        setDatasetName(file.name)
        setWidgets([])
        setHighlightedRows(new Set())
      }
      return
    }

    const text = await file.text()
    const parsed = Papa.parse<Record<string, unknown>>(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    })
    const parsedRows = (parsed.data ?? []).filter((row) => Object.keys(row).length > 0)
    if (parsedRows.length > 0) {
      setRows(normalizeRows(parsedRows))
      setDatasetName(file.name)
      setWidgets([])
      setHighlightedRows(new Set())
    }
  }

  // Widget drag handling
  useEffect(() => {
    if (!dragState) return

    const handleMouseMove = (event: MouseEvent) => {
      const board = boardRef.current
      if (!board) return
      const rect = board.getBoundingClientRect()
      setWidgets((prev) =>
        prev.map((widget) => {
          if (widget.id !== dragState.id) return widget
          const newX = event.clientX - rect.left + board.scrollLeft - dragState.offsetX
          const newY = event.clientY - rect.top + board.scrollTop - dragState.offsetY
          return {
            ...widget,
            x: Math.max(0, newX),
            y: Math.max(0, newY),
          }
        })
      )
    }

    const handleMouseUp = () => setDragState(null)
    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [dragState])

  const startWidgetDrag = (event: React.MouseEvent<HTMLDivElement>, widget: VizWidget) => {
    const board = boardRef.current
    if (!board) return
    const boardRect = board.getBoundingClientRect()
    bringToFront(widget.id)
    setDragState({
      id: widget.id,
      offsetX: event.clientX - boardRect.left + board.scrollLeft - widget.x,
      offsetY: event.clientY - boardRect.top + board.scrollTop - widget.y,
    })
  }

  // Widget resize handling
  const startWidgetResize = (
    event: React.MouseEvent,
    widget: VizWidget,
    corner: "se" | "sw" | "ne" | "nw"
  ) => {
    event.stopPropagation()
    event.preventDefault()
    bringToFront(widget.id)
    setResizeState({
      id: widget.id,
      startX: event.clientX,
      startY: event.clientY,
      startWidth: widget.width,
      startHeight: widget.height,
      corner,
    })
  }

  useEffect(() => {
    if (!resizeState) return

    const MIN_WIDTH = 200
    const MIN_HEIGHT = 150

    const handleMouseMove = (event: MouseEvent) => {
      const deltaX = event.clientX - resizeState.startX
      const deltaY = event.clientY - resizeState.startY

      setWidgets((prev) =>
        prev.map((widget) => {
          if (widget.id !== resizeState.id) return widget

          let newWidth = resizeState.startWidth
          let newHeight = resizeState.startHeight
          let newX = widget.x
          let newY = widget.y

          // Handle horizontal resize based on corner
          if (resizeState.corner === "se" || resizeState.corner === "ne") {
            newWidth = Math.max(MIN_WIDTH, resizeState.startWidth + deltaX)
          } else {
            // sw or nw - resize from left edge
            newWidth = Math.max(MIN_WIDTH, resizeState.startWidth - deltaX)
            if (newWidth > MIN_WIDTH) {
              newX = widget.x + (resizeState.startWidth - newWidth)
            }
          }

          // Handle vertical resize based on corner
          if (resizeState.corner === "se" || resizeState.corner === "sw") {
            newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight + deltaY)
          } else {
            // ne or nw - resize from top edge
            newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight - deltaY)
            if (newHeight > MIN_HEIGHT) {
              newY = widget.y + (resizeState.startHeight - newHeight)
            }
          }

          return { ...widget, width: newWidth, height: newHeight, x: newX, y: newY }
        })
      )
    }

    const handleMouseUp = () => setResizeState(null)

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [resizeState])

  const runAgent = async () => {
    if (!prompt.trim()) return

    const outgoingPrompt = prompt.trim()
    const nextMessages: ChatMessage[] = [
      ...chatMessages,
      { role: "user", content: outgoingPrompt },
    ]
    setChatMessages(nextMessages)
    setPrompt("")
    setIsRunning(true)
    setError(null)

    // Initialize thinking steps only if thinking mode is enabled
    if (thinkingMode) {
      const steps: ThinkingStep[] = [
        { id: "schema", label: "Reading data schema...", status: "pending" },
        { id: "generate", label: "Generating code...", status: "pending" },
        { id: "execute", label: "Executing in sandbox...", status: "pending" },
        { id: "prepare", label: "Preparing response...", status: "pending" },
      ]
      setThinkingSteps(steps)
    }

    try {
      // Use streaming endpoint only if thinking mode is enabled
      const endpoint = thinkingMode ? `${API_BASE_URL}/agent/stream` : `${API_BASE_URL}/agent/execute`
      
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: outgoingPrompt,
          rows,
          model: modelName,
          history: nextMessages.map(m => ({ role: m.role, content: m.content })),
        }),
      })

      if (!response.ok) {
        // For streaming, response might be text/html on error
        const contentType = response.headers.get("content-type") || ""
        if (contentType.includes("application/json")) {
          const errorData = await response.json().catch(() => ({ detail: "Agent execution failed" }))
          throw new Error(errorData.detail ?? "Agent execution failed")
        } else {
          throw new Error(`Server error: ${response.status} ${response.statusText}`)
        }
      }

      let finalPayload: AgentExecuteResponse | null = null

      if (thinkingMode) {
        // Streaming mode with step-by-step updates
        const reader = response.body?.getReader()
        if (!reader) throw new Error("No response body")

        const decoder = new TextDecoder()
        let buffer = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n\n")
          buffer = lines.pop() ?? ""

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue
            try {
              const eventData = JSON.parse(line.slice(6))
              const { event, data } = eventData

              if (event === "step") {
                // Update thinking step status
                setThinkingSteps((prev) =>
                  prev.map((s) => {
                    if (s.id === data.id) {
                      return { ...s, status: data.status }
                    }
                    // If this step is now active, mark previous steps as done
                    if (data.status === "active") {
                      const stepOrder = ["schema", "generate", "execute", "prepare"]
                      const activeIdx = stepOrder.indexOf(data.id)
                      const thisIdx = stepOrder.indexOf(s.id)
                      if (thisIdx < activeIdx && s.status !== "done") {
                        return { ...s, status: "done" }
                      }
                    }
                    return s
                  })
                )
              } else if (event === "result") {
                finalPayload = data as AgentExecuteResponse
              } else if (event === "error") {
                throw new Error(data.message)
              }
            } catch (parseError) {
              // Ignore parse errors for incomplete chunks
            }
          }
        }
      } else {
        // Standard non-streaming mode
        finalPayload = await response.json()
      }

      if (!finalPayload) {
        throw new Error("No result received from agent")
      }

      // Only update table data when the backend signals a real mutation
      if (finalPayload.mutation && finalPayload.rows) {
        setRows(finalPayload.rows)
      }

      // Highlight matching rows (works for both queries and mutations)
      if (
        Array.isArray(finalPayload.highlight_indices) &&
        finalPayload.highlight_indices.length > 0
      ) {
        setHighlightedRows(new Set(finalPayload.highlight_indices))
      } else {
        setHighlightedRows(new Set())
      }

      const visualizationPayload = finalPayload.visualization ?? null
      if (
        visualizationPayload &&
        Array.isArray(visualizationPayload.data) &&
        visualizationPayload.data.length > 0
      ) {
        setWidgets((prev) => {
          const nextIndex = prev.length
          const nextZIndex = maxZIndex + 1
          setMaxZIndex(nextZIndex)
          return [
            ...prev,
            {
              id: `${Date.now()}-${nextIndex}`,
              title: `Chart ${nextIndex + 1}`,
              data: visualizationPayload.data ?? [],
              layout: visualizationPayload.layout,
              x: 20 + (nextIndex % 3) * 40,
              y: 20 + (nextIndex % 3) * 30,
              width: 460,
              height: 320,
              zIndex: nextZIndex,
            },
          ]
        })
      }

      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: finalPayload.assistant_reply ?? "Done.",
          code: finalPayload.code,
        },
      ])
    } catch (agentError) {
      let message: string
      if (agentError instanceof TypeError && agentError.message.includes("fetch")) {
        message = "Cannot reach the backend server. Make sure the backend is running on " + API_BASE_URL
      } else if (agentError instanceof Error) {
        message = agentError.message
      } else {
        message = "Unknown agent error"
      }
      setError(message)
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${message}` },
      ])
    } finally {
      setIsRunning(false)
      setThinkingSteps([])
    }
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground font-sans antialiased">
      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-card/80 px-4 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <FileSpreadsheet className="size-4" />
          </div>
          <h1 className="text-sm font-bold tracking-tight text-foreground">
            Data Pilot
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium">{rows.length} rows</span>
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            className="flex h-7 w-7 items-center justify-center rounded-md border border-border bg-secondary text-muted-foreground transition-colors hover:text-foreground hover:bg-accent"
          >
            {isDark ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          </button>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`shrink-0 flex flex-col border-r border-border bg-card/50 backdrop-blur-sm transition-all duration-200 ${
            sidebarCollapsed ? "w-14" : "w-60"
          }`}
        >
          {/* Sidebar toggle */}
          <div className="flex h-10 items-center justify-end px-2 border-b border-border">
            <button
              type="button"
              onClick={() => setSidebarCollapsed((prev) => !prev)}
              className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            >
              {sidebarCollapsed ? <ChevronRight className="size-3.5" /> : <ChevronLeft className="size-3.5" />}
            </button>
          </div>

          {/* File upload */}
          <div className="p-2">
            <label
              className={`group flex cursor-pointer items-center gap-2.5 rounded-lg border border-dashed border-border p-2.5 text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground hover:bg-accent/50 ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
            >
              <Upload className="size-4 shrink-0" />
              {!sidebarCollapsed && (
                <span className="text-xs font-medium truncate">Upload CSV / XLSX</span>
              )}
              <input
                className="hidden"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileUpload}
              />
            </label>
          </div>

          {/* Active file */}
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

          {/* Workflow steps */}
          <div className="px-2 py-2 space-y-1">
            {!sidebarCollapsed && (
              <div className="px-1 mb-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                Workflow
              </div>
            )}
            {transformationSteps.map((step, idx) => {
              const Icon = step.icon
              return (
                <div
                  key={step.label}
                  className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors ${
                    sidebarCollapsed ? "justify-center" : ""
                  } ${idx === 0 ? "bg-primary/10 text-primary" : "text-muted-foreground"}`}
                  title={step.label}
                >
                  <Icon className="size-3.5 shrink-0" />
                  {!sidebarCollapsed && <span className="font-medium">{step.label}</span>}
                </div>
              )
            })}
          </div>

          {/* Model selector at bottom */}
          {!sidebarCollapsed && (
            <div className="mt-auto border-t border-border p-3">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">
                Model
              </div>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full h-7 rounded-md border border-border bg-secondary px-2 text-xs font-medium text-foreground outline-none transition focus:ring-1 focus:ring-ring"
              >
                <option value="gemini-3-flash-preview">Gemini 3 Flash</option>
                <option value="gemini-3.1-flash-lite-preview">Gemini 3.1 Flash Lite</option>
                <option value="openai/gpt-4.1">GPT 4.1 (GitHub Models)</option>
              </select>
            </div>
          )}
        </aside>

        {/* Main content area */}
        <div ref={mainContentRef} className="flex flex-1 min-h-0 min-w-0 overflow-hidden">
          {/* Center: Data + Canvas */}
          <div ref={splitContainerRef} className="flex flex-1 min-w-0 flex-col min-h-0 overflow-hidden">
            {/* Data Table */}
            <div
              className={`flex flex-col min-h-[120px] shrink-0 overflow-hidden ${
                fullscreenPanel === "data" ? "fullscreen-overlay" : ""
              }`}
              style={fullscreenPanel === "data" ? {} : { height: `calc(${splitRatio * 100}% - 2px)` }}
            >
              <div className="flex items-center justify-between h-10 px-4 border-b border-border bg-card/30 shrink-0">
                <span className="text-xs font-semibold text-foreground">Data</span>
                <div className="flex items-center gap-2">
                  {/* Highlighted row navigation */}
                  {highlightedRowsArray.length > 0 && (
                    <div className="flex items-center gap-1 mr-2">
                      <button
                        type="button"
                        onClick={() => navigateToHighlightedRow("prev")}
                        className="flex h-6 w-6 items-center justify-center rounded text-primary hover:bg-primary/10 transition-colors"
                        title="Previous highlighted row"
                      >
                        <ChevronLeft className="size-4" />
                      </button>
                      <span className="text-[10px] font-medium text-primary min-w-[3rem] text-center">
                        {currentHighlightIndex + 1} / {highlightedRowsArray.length}
                      </span>
                      <button
                        type="button"
                        onClick={() => navigateToHighlightedRow("next")}
                        className="flex h-6 w-6 items-center justify-center rounded text-primary hover:bg-primary/10 transition-colors"
                        title="Next highlighted row"
                      >
                        <ChevronRight className="size-4" />
                      </button>
                    </div>
                  )}
                  <span className="text-[10px] text-muted-foreground">{headers.length} columns</span>
                  <button
                    type="button"
                    onClick={() => setFullscreenPanel(fullscreenPanel === "data" ? null : "data")}
                    className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                    title={fullscreenPanel === "data" ? "Exit fullscreen" : "Fullscreen"}
                  >
                    {fullscreenPanel === "data" ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
                  </button>
                </div>
              </div>
              {/* Table with synced scrolling */}
              <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                <div 
                  ref={tableContainerRef}
                  className="flex-1 min-h-0 overflow-auto scrollbar-thin"
                  onScroll={(e) => {
                    const target = e.currentTarget
                    setScrollTop(target.scrollTop)
                  }}
                >
                  <Table className="w-max min-w-full">
                    <TableHeader>
                      <TableRow className="border-b-border hover:bg-transparent">
                        {headers.map((header) => (
                          <TableHead
                            key={header}
                            className="sticky top-0 z-10 h-8 bg-card/95 backdrop-blur px-3 text-[11px] font-semibold text-muted-foreground whitespace-nowrap"
                          >
                            {header}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {virtualizedData.startIndex > 0 && (
                        <tr style={{ height: virtualizedData.offsetTop }} />
                      )}
                      {virtualizedData.visibleRows.map((row, localIdx) => {
                        const idx = virtualizedData.startIndex + localIdx
                        const isHighlighted = highlightedRows.has(idx)
                        const isCurrentHighlight = highlightedRowsArray.length > 0 && highlightedRowsArray[currentHighlightIndex] === idx
                        return (
                          <TableRow
                            key={idx}
                            style={{ height: ROW_HEIGHT }}
                            className={`border-b-border/50 ${
                              isHighlighted
                                ? isCurrentHighlight
                                  ? "bg-primary/25 border-l-[3px] border-l-primary"
                                  : "bg-primary/15 border-l-[3px] border-l-primary"
                                : "hover:bg-accent/30"
                            }`}
                          >
                            {headers.map((header) => (
                              <TableCell
                                key={`${idx}-${header}`}
                                className={`px-3 py-1.5 text-xs whitespace-nowrap ${
                                  isHighlighted ? "text-foreground font-medium" : "text-foreground/80"
                                }`}
                              >
                                {String(row[header] ?? "")}
                              </TableCell>
                            ))}
                          </TableRow>
                        )
                      })}
                      {virtualizedData.endIndex < rows.length && (
                        <tr style={{ height: (rows.length - virtualizedData.endIndex) * ROW_HEIGHT }} />
                      )}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </div>

            {/* Draggable Splitter */}
            {fullscreenPanel === null && (
              <div
                className={`shrink-0 splitter-handle ${isSplitterDragging ? "dragging" : ""}`}
                onMouseDown={() => setIsSplitterDragging(true)}
              />
            )}

            {/* Visualization Canvas - clean board */}
            <div
              className={`flex flex-col flex-1 min-h-0 overflow-hidden ${
                fullscreenPanel === "canvas" ? "fullscreen-overlay" : ""
              }`}
            >
              <div className="flex items-center justify-between h-10 px-4 border-b border-border bg-card/30">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold text-foreground">Canvas</span>
                  {widgets.length > 0 && (
                    <div className="relative">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setChartMenuOpen((prev) => !prev)
                        }}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                      >
                        <span>{widgets.length} chart{widgets.length !== 1 ? "s" : ""}</span>
                        <ChevronDown className={`size-3 transition-transform ${chartMenuOpen ? "rotate-180" : ""}`} />
                      </button>
                      {chartMenuOpen && (
                        <div 
                          className="absolute top-full left-0 mt-1 min-w-[160px] rounded-lg border border-border bg-card shadow-lg py-1 z-50"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {widgets.map((widget) => (
                            <button
                              key={widget.id}
                              type="button"
                              onClick={() => {
                                bringToFront(widget.id)
                                setChartMenuOpen(false)
                              }}
                              className="w-full text-left px-3 py-1.5 text-xs text-foreground hover:bg-accent transition-colors"
                            >
                              {widget.title}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setFullscreenPanel(fullscreenPanel === "canvas" ? null : "canvas")}
                  className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                  title={fullscreenPanel === "canvas" ? "Exit fullscreen" : "Fullscreen"}
                >
                  {fullscreenPanel === "canvas" ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
                </button>
              </div>
              <div className="flex-1 min-h-0 relative overflow-hidden">
                <div
                  ref={boardRef}
                  className="absolute inset-0 overflow-auto"
                  style={{
                    backgroundImage: isDark
                      ? "radial-gradient(circle, rgba(148,163,184,0.08) 1px, transparent 1px)"
                      : "radial-gradient(circle, rgba(0,0,0,0.06) 1px, transparent 1px)",
                    backgroundSize: "20px 20px",
                  }}
                >
                  {widgets.length > 0 ? (
                    widgets.map((widget) => (
                      <div
                        key={widget.id}
                        className="absolute flex flex-col rounded-lg border border-border bg-card shadow-lg"
                        style={{
                          left: `${widget.x}px`,
                          top: `${widget.y}px`,
                          width: `${widget.width}px`,
                          height: `${widget.height}px`,
                          zIndex: widget.zIndex,
                        }}
                        onClick={() => bringToFront(widget.id)}
                      >
                        <div
                          className="flex h-8 shrink-0 cursor-move items-center justify-between border-b border-border px-3"
                          onMouseDown={(event) => startWidgetDrag(event, widget)}
                        >
                          <span className="text-[11px] font-semibold text-foreground">{widget.title}</span>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              setWidgets((prev) => prev.filter((item) => item.id !== widget.id))
                            }}
                            className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground hover:bg-accent"
                          >
                            <X className="size-3" />
                          </button>
                        </div>
                        <div className="min-h-0 flex-1 p-2">
                          <PlotlyBoard data={widget.data} layout={widget.layout} isDark={isDark} />
                        </div>
                        {/* Resize handles at corners */}
                        <div
                          className="absolute top-0 left-0 w-4 h-4 cursor-nw-resize group/resize"
                          onMouseDown={(e) => startWidgetResize(e, widget, "nw")}
                        >
                          <div className="absolute top-0.5 left-0.5 w-2.5 h-2.5 border-t-2 border-l-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-tl" />
                        </div>
                        <div
                          className="absolute top-0 right-0 w-4 h-4 cursor-ne-resize group/resize"
                          onMouseDown={(e) => startWidgetResize(e, widget, "ne")}
                        >
                          <div className="absolute top-0.5 right-0.5 w-2.5 h-2.5 border-t-2 border-r-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-tr" />
                        </div>
                        <div
                          className="absolute bottom-0 left-0 w-4 h-4 cursor-sw-resize group/resize"
                          onMouseDown={(e) => startWidgetResize(e, widget, "sw")}
                        >
                          <div className="absolute bottom-0.5 left-0.5 w-2.5 h-2.5 border-b-2 border-l-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-bl" />
                        </div>
                        <div
                          className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize group/resize"
                          onMouseDown={(e) => startWidgetResize(e, widget, "se")}
                        >
                          <div className="absolute bottom-0.5 right-0.5 w-2.5 h-2.5 border-b-2 border-r-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-br" />
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="flex h-full items-center justify-center">
                      <p className="text-xs text-muted-foreground">
                        Charts will appear here
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Chat Splitter */}
          <div
            className={`w-1 shrink-0 cursor-col-resize splitter-handle-horizontal ${isChatSplitterDragging ? "dragging" : ""}`}
            onMouseDown={() => setIsChatSplitterDragging(true)}
          />

          {/* Right Panel: Chat */}
          <div className="flex shrink-0 flex-col bg-card/30" style={{ width: chatWidth }}>
            {/* Chat messages */}
            <div className="flex-1 min-h-0 overflow-auto p-3 space-y-3 scrollbar-thin">
              {chatMessages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`group flex flex-col ${message.role === "user" ? "items-end" : "items-start"}`}
                >
                  <div
                    className={`max-w-[90%] rounded-xl px-3 py-2 text-[13px] leading-relaxed ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-secondary-foreground"
                    }`}
                  >
                    {message.role === "assistant"
                      ? renderSimpleMarkdown(message.content)
                      : message.content}
                  </div>
                  {/* Code toggle button - only for assistant messages with code */}
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
                        <pre className="mt-1.5 w-full overflow-x-auto rounded-lg bg-[oklch(0.1_0_0)] p-2.5 text-[11px] font-mono text-[oklch(0.7_0_0)] border border-border whitespace-pre">
                          {message.code}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              ))}

              {isRunning && (
                <div className="flex justify-start">
                  {thinkingMode && thinkingSteps.length > 0 ? (
                    <div className="rounded-xl bg-secondary px-3 py-2.5 text-xs">
                      <div className="flex items-center gap-2 text-muted-foreground mb-2">
                        <Loader2 className="size-3 animate-spin" />
                        <span className="font-medium">Thinking...</span>
                      </div>
                      <div className="space-y-1.5">
                        {thinkingSteps.map((step) => (
                          <div
                            key={step.id}
                            className={`flex items-center gap-2 text-[11px] transition-all duration-200 ${
                              step.status === "done"
                                ? "text-primary"
                                : step.status === "active"
                                ? "text-foreground"
                                : "text-muted-foreground/50"
                            }`}
                          >
                            {step.status === "done" ? (
                              <svg className="size-3" viewBox="0 0 16 16" fill="currentColor">
                                <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" />
                              </svg>
                            ) : step.status === "active" ? (
                              <Loader2 className="size-3 animate-spin" />
                            ) : (
                              <div className="size-3 rounded-full border border-current opacity-40" />
                            )}
                            <span>{step.label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 rounded-xl bg-secondary px-3 py-2 text-xs text-muted-foreground">
                      <Loader2 className="size-3 animate-spin" />
                      Processing...
                    </div>
                  )}
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Error */}
            {error && (
              <div className="mx-3 mb-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {error}
              </div>
            )}

            {/* Input area */}
            <div className="border-t border-border p-3">
              <div className="flex items-end gap-2">
                <textarea
                  className="flex-1 resize-none rounded-lg border border-border bg-secondary px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:ring-1 focus:ring-ring"
                  placeholder="Ask something..."
                  rows={2}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      runAgent()
                    }
                  }}
                />
                <div className="flex flex-col gap-1.5">
                  <button
                    type="button"
                    onClick={() => setThinkingMode((prev) => !prev)}
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border transition-colors ${
                      thinkingMode
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent"
                    }`}
                    title={thinkingMode ? "Thinking mode ON" : "Thinking mode OFF"}
                  >
                    <Sparkles className="size-4" />
                  </button>
                  <button
                    type="button"
                    onClick={runAgent}
                    disabled={isRunning || !prompt.trim()}
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-40 disabled:pointer-events-none"
                  >
                    {isRunning ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Send className="size-4" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}