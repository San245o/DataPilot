"use client"

import { useEffect, useMemo, useRef, useState, useCallback } from "react"
import Papa from "papaparse"
import * as XLSX from "xlsx"
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Code,
  ExternalLink,
  FileSpreadsheet,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  Moon,
  Send,
  Sun,
  Upload,
  Download,
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
type QueryTablePayload = {
  id: string
  title: string
  rows: SheetRow[]
}
type AgentExecuteResponse = {
  rows?: SheetRow[]
  visualization?: VisualizationPayload | null
  query_output?: string | null
  query_tables?: QueryTablePayload[]
  code?: string
  assistant_reply?: string
  context_preview?: ContextPreview
  detail?: string
  mutation?: boolean
  highlight_indices?: number[]
  token_usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}
type ChatMessage = {
  role: "user" | "assistant"
  content: string
  code?: string
  query_output?: string | null
  table_links?: Array<{ id: string; title: string }>
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
type PivotTableTab = {
  id: string
  title: string
  rows: SheetRow[]
  open: boolean
  isNew?: boolean
  insertions?: number
  deletions?: number
}
type DragState = {
  id: string
  offsetX: number
  offsetY: number
} | null

type SlashCommandOption = {
  command: string
  description: string
  rewritePrefix: string
}

type ResizeState = {
  id: string
  startX: number
  startY: number
  startWidth: number
  startHeight: number
  startWidgetX: number
  startWidgetY: number
  corner: "se" | "sw" | "ne" | "nw"
} | null

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
  { label: "Context", icon: Layers, isContextButton: true },
]

const SLASH_COMMANDS: SlashCommandOption[] = [
  {
    command: "/visualize",
    description: "Build a chart (bar, line, scatter, pie, heatmap).",
    rewritePrefix: "Create a visualization.",
  },
  {
    command: "/modify",
    description: "Update rows/columns in the main dataset.",
    rewritePrefix: "Modify the dataset.",
  },
  {
    command: "/extract",
    description: "Create a separate result table from selected rows/columns.",
    rewritePrefix: "Extract a separate result table.",
  },
  {
    command: "/filter",
    description: "Find matching rows without mutating source data.",
    rewritePrefix: "Filter the dataset to find matching rows.",
  },
  {
    command: "/summarize",
    description: "Get concise KPIs, totals, and summary stats.",
    rewritePrefix: "Summarize the key metrics concisely.",
  },
  {
    command: "/pivot",
    description: "Create a pivot/matrix view from categorical dimensions.",
    rewritePrefix: "Create a pivot-style matrix.",
  },
  {
    command: "/compare",
    description: "Compare categories, segments, or time periods.",
    rewritePrefix: "Compare groups or categories.",
  },
  {
    command: "/correlate",
    description: "Analyze numeric relationships/correlation.",
    rewritePrefix: "Analyze correlation between relevant numeric columns.",
  },
  {
    command: "/trend",
    description: "Analyze changes over time.",
    rewritePrefix: "Analyze the trend over time.",
  },
  {
    command: "/clean",
    description: "Standardize values and clean missing/dirty data.",
    rewritePrefix: "Clean and standardize the data.",
  },
  {
    command: "/help",
    description: "Show what this dataset assistant can do.",
    rewritePrefix: "Show a concise list of things you can do with this dataset.",
  },
]

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend").replace(/\/$/, "")
const SLASH_ROW_HEIGHT = 32
const SLASH_ROW_GAP = 4
const SLASH_DESCRIPTION_DELAY_MS = 1000

// Virtualization constants
const ROW_HEIGHT = 28 // Height of each row in pixels
const OVERSCAN = 5 // Extra rows to render above/below viewport

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(value, max))
}

function getActiveSlashQuery(text: string): string | null {
  const trimmed = text.trimStart()
  if (!trimmed.startsWith("/")) return null

  const firstToken = trimmed.split(/\s+/, 1)[0]
  if (firstToken.length < 1) return null
  if (trimmed.includes(" ")) return null

  return firstToken.slice(1).toLowerCase()
}

function expandSlashPrompt(rawPrompt: string): string {
  const trimmed = rawPrompt.trim()
  if (!trimmed.startsWith("/")) {
    return rawPrompt
  }

  const [commandToken, ...restParts] = trimmed.split(/\s+/)
  const rest = restParts.join(" ").trim()
  const matched = SLASH_COMMANDS.find((item) => item.command === commandToken.toLowerCase())
  if (!matched) {
    return rawPrompt
  }
  if (!rest) {
    return matched.rewritePrefix
  }
  return `${matched.rewritePrefix} ${rest}`
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
  const [modelName, setModelName] = useState("gemini-3.1-flash-lite-preview")
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
  const [expandedCodeIndex, setExpandedCodeIndex] = useState<number | null>(null)
  const [chartMenuOpen, setChartMenuOpen] = useState(false)
  const [pivotMenuOpen, setPivotMenuOpen] = useState(false)
  const [pivotTables, setPivotTables] = useState<PivotTableTab[]>([])
  const [activeDataTab, setActiveDataTab] = useState<string>("base")
  const [isContextOpen, setIsContextOpen] = useState(false)
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0, total: 0 })
  const [queryCount, setQueryCount] = useState(0)
  const [selectedSlashIndex, setSelectedSlashIndex] = useState(0)
  const [slashDismissed, setSlashDismissed] = useState(false)
  const [visibleSlashDescriptionFor, setVisibleSlashDescriptionFor] = useState<string | null>(null)
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
  const promptInputRef = useRef<HTMLTextAreaElement | null>(null)
  const slashListRef = useRef<HTMLDivElement | null>(null)
  const slashDescriptionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isRunningRef = useRef(false)
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
  const activePivotTable = useMemo(
    () => pivotTables.find((table) => table.id === activeDataTab && table.open),
    [pivotTables, activeDataTab]
  )
  const isBaseDataTab = activeDataTab === "base" || !activePivotTable

  useEffect(() => {
    setSelectedSlashIndex(0)
  }, [slashQuery])

  const clearSlashDescriptionTimer = useCallback(() => {
    if (slashDescriptionTimerRef.current) {
      clearTimeout(slashDescriptionTimerRef.current)
      slashDescriptionTimerRef.current = null
    }
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

  const handleSlashOptionHover = useCallback((index: number, command: string) => {
    setSelectedSlashIndex((prev) => (prev === index ? prev : index))
    scheduleSlashDescription(command)
  }, [scheduleSlashDescription])

  useEffect(() => {
    if (showSlashMenu) return
    hideSlashDescription()
  }, [showSlashMenu, hideSlashDescription])

  useEffect(() => {
    return () => {
      clearSlashDescriptionTimer()
    }
  }, [clearSlashDescriptionTimer])

  const insertSlashCommand = useCallback((item: SlashCommandOption) => {
    const nextValue = `${item.command} `
    setPrompt(nextValue)
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
  }, [clearSlashDescriptionTimer])
  const tableRows = isBaseDataTab ? rows : (activePivotTable?.rows ?? rows)
  const headers = useMemo(() => Object.keys(tableRows[0] ?? {}), [tableRows])
  const baseShapeLabel = useMemo(() => {
    const baseColumnCount = Object.keys(rows[0] ?? {}).length
    return `${rows.length}X${baseColumnCount}`
  }, [rows])
  const tableShapeLabel = useMemo(
    () => `${tableRows.length}X${headers.length}`,
    [tableRows.length, headers.length]
  )

  // Virtualization state
  const [scrollTop, setScrollTop] = useState(0)
  const [tableHeight, setTableHeight] = useState(400)

  // Column width state
  const [colWidths, setColWidths] = useState<Record<string, number>>({})
  const resizingCol = useRef<{ key: string, startX: number, startWidth: number } | null>(null)

  const totalTableWidth = useMemo(() => {
    if (!headers || headers.length === 0) return "100%"
    const width = headers.reduce((sum, h) => sum + (colWidths[h] || 150), 0)
    return Math.max(width, 100) // fallback minimum
  }, [headers, colWidths])

  // Convert highlighted rows set to sorted array for navigation
  const highlightedRowsArray = useMemo(() => {
    if (!isBaseDataTab) return []
    return Array.from(highlightedRows).sort((a, b) => a - b)
  }, [highlightedRows, isBaseDataTab])
  
  const [currentHighlightIndex, setCurrentHighlightIndex] = useState(0)

  // Calculate virtualized rows
  const virtualizedData = useMemo(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
    const visibleCount = Math.ceil(tableHeight / ROW_HEIGHT) + OVERSCAN * 2
    const endIndex = Math.min(tableRows.length, startIndex + visibleCount)
    
    return {
      startIndex,
      endIndex,
      visibleRows: tableRows.slice(startIndex, endIndex),
      totalHeight: tableRows.length * ROW_HEIGHT,
      offsetTop: startIndex * ROW_HEIGHT
    }
  }, [tableRows, scrollTop, tableHeight])

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

  // Column resizing effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizingCol.current) return
      const { key, startX, startWidth } = resizingCol.current
      const newWidth = Math.max(50, startWidth + (e.clientX - startX)) // 50px min width
      setColWidths((prev) => ({ ...prev, [key]: newWidth }))
    }

    const handleMouseUp = () => {
      if (resizingCol.current) {
        resizingCol.current = null
        document.body.style.cursor = ""
      }
    }

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [])

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

  const closePivotTab = useCallback((pivotId: string) => {
    setPivotTables((prev) => prev.map((tab) => (tab.id === pivotId ? { ...tab, open: false } : tab)))
    setActiveDataTab((prev) => (prev === pivotId ? "base" : prev))
  }, [])

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatMessages])

  // Close dropdown menus when clicking outside
  useEffect(() => {
    if (!chartMenuOpen && !pivotMenuOpen) return
    const handleClickOutside = () => {
      setChartMenuOpen(false)
      setPivotMenuOpen(false)
    }
    document.addEventListener("click", handleClickOutside)
    return () => document.removeEventListener("click", handleClickOutside)
  }, [chartMenuOpen, pivotMenuOpen])

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

  const openWidgetInNewTab = useCallback((widget: VizWidget) => {
    if (typeof window === "undefined") return

    try {
      const storageKey = `datapilot:chart:${widget.id}:${Date.now()}`
      const targetUrl = `/chart-viewer?chartKey=${encodeURIComponent(storageKey)}`

      const popup = window.open("about:blank", "_blank")
      localStorage.setItem(
        storageKey,
        JSON.stringify({
          title: widget.title,
          data: widget.data,
          layout: widget.layout,
          isDark,
        })
      )

      if (popup) {
        popup.location.href = targetUrl
      } else {
        // Fallback when popup is blocked: continue in current tab.
        window.location.href = targetUrl
      }
    } catch {
      setError("Unable to open chart in a new tab.")
    }
  }, [isDark])

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
        setPivotTables([])
        setActiveDataTab("base")
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
      setPivotTables([])
      setActiveDataTab("base")
      setHighlightedRows(new Set())
    }
  }

  const handleDownloadExcel = () => {
    if (!rows || rows.length === 0) return
    const worksheet = XLSX.utils.json_to_sheet(rows)
    const workbook = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(workbook, worksheet, "Data")
    XLSX.writeFile(workbook, `modified_${datasetName.replace(/\.[^/.]+$/, "")}.xlsx`)
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
      startWidgetX: widget.x,
      startWidgetY: widget.y,
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
          let newX = resizeState.startWidgetX
          let newY = resizeState.startWidgetY

          // Handle horizontal resize based on corner
          if (resizeState.corner === "se" || resizeState.corner === "ne") {
            newWidth = Math.max(MIN_WIDTH, resizeState.startWidth + deltaX)
          } else {
            // sw or nw - resize from left edge
            newWidth = Math.max(MIN_WIDTH, resizeState.startWidth - deltaX)
            if (newWidth > MIN_WIDTH) {
              newX = Math.max(0, resizeState.startWidgetX + (resizeState.startWidth - newWidth))
            } else {
              // Reached min width, cap the X position movement
              newX = Math.max(0, resizeState.startWidgetX + (resizeState.startWidth - MIN_WIDTH))
            }
          }

          // Handle vertical resize based on corner
          if (resizeState.corner === "se" || resizeState.corner === "sw") {
            newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight + deltaY)
          } else {
            // ne or nw - resize from top edge
            newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight - deltaY)
            if (newHeight > MIN_HEIGHT) {
              newY = Math.max(0, resizeState.startWidgetY + (resizeState.startHeight - newHeight))
            } else {
              // Reached min height, cap the Y position movement
              newY = Math.max(0, resizeState.startWidgetY + (resizeState.startHeight - MIN_HEIGHT))
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
    if (isRunningRef.current) return
    if (!prompt.trim()) return

    const outgoingPrompt = prompt.trim()
    const expandedPrompt = expandSlashPrompt(outgoingPrompt)
    const nextMessages: ChatMessage[] = [
      ...chatMessages,
      { role: "user", content: outgoingPrompt },
    ]
    setChatMessages(nextMessages)
    setPrompt("")
    isRunningRef.current = true
    setIsRunning(true)
    setError(null)

    const endpoint = `${API_BASE_URL}/agent/execute`

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: expandedPrompt,
          rows,
          model: modelName,
          history: nextMessages.map((m) => ({
            role: m.role,
            content: m.role === "user" ? expandSlashPrompt(m.content) : m.content,
          })),
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

      const finalPayload: AgentExecuteResponse = await response.json()

      if (!finalPayload) {
        throw new Error("No result received from agent")
      }

      setQueryCount((prev) => prev + 1)
      if (finalPayload.token_usage) {
        setTotalTokens((prev) => ({
          prompt: prev.prompt + (finalPayload.token_usage?.prompt_tokens || 0),
          completion: prev.completion + (finalPayload.token_usage?.completion_tokens || 0),
          total: prev.total + (finalPayload.token_usage?.total_tokens || 0),
        }))
      }

      const nextRows = Array.isArray(finalPayload.rows)
        ? normalizeRows(finalPayload.rows as Record<string, unknown>[])
        : []
      const responseTables = Array.isArray(finalPayload.query_tables) ? finalPayload.query_tables : []
      const createdTableLink: { id: string; title: string } | null = null

        if (responseTables.length > 0) {
          const normalizedTables = responseTables.map((table) => ({
            id: table.id,
            title: table.title,
            rows: normalizeRows(table.rows as Record<string, unknown>[]),
            open: true,
            isNew: true,
            insertions: table.rows ? table.rows.length : 0,
          }))

          setPivotTables((prev) => {
            const byId = new Map(prev.map((item) => [item.id, item]))
            normalizedTables.forEach((table) => {
              byId.set(table.id, table)
            })
            return Array.from(byId.values())
          })
          setActiveDataTab(normalizedTables[0].id)	
          
          setTimeout(() => {
            setPivotTables((prev) =>
              prev.map((tab) => (normalizedTables.find(t => t.id === tab.id) ? { ...tab, isNew: false } : tab))
            )
          }, 2000)
        }

        if (finalPayload.mutation && nextRows.length > 0) {
          setRows(nextRows)
          setActiveDataTab("base")
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
          query_output: finalPayload.query_output,
          table_links: [
            ...responseTables.map((table) => ({ id: table.id, title: table.title })),
            ...(createdTableLink ? [createdTableLink] : []),
          ],
        },
      ])
    } catch (agentError) {
      let message: string
      if (agentError instanceof TypeError && agentError.message.includes("fetch")) {
        message = `Failed to reach backend at ${endpoint}. If you are running the dashboard in a remote/devcontainer setup, use the default /api/backend proxy or set NEXT_PUBLIC_API_BASE_URL to a reachable URL.`
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
      isRunningRef.current = false
      setIsRunning(false)
    }
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background text-foreground font-sans antialiased">
      {/* Main Layout */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`shrink-0 flex flex-col border-r border-border bg-card/50 backdrop-blur-sm transition-all duration-200 ${
            sidebarCollapsed ? "w-14" : "w-60"
          }`}
        >
          {/* Sidebar toggle */}
          <div className="flex h-10 items-center justify-end px-2 border-b border-border px-2">
            <button
              type="button"
              onClick={() => setSidebarCollapsed((prev) => !prev)}
              className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? <ChevronRight className="size-3.5" /> : <ChevronLeft className="size-3.5" />}
            </button>
          </div>

          {/* Sidebar brand */}
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

          {/* File upload & download */}
          <div className="p-2 space-y-2">
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

            <button
              onClick={handleDownloadExcel}
              disabled={rows.length === 0}
              className={`w-full group flex items-center gap-2.5 rounded-lg border border-dashed border-border p-2.5 text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground hover:bg-accent/50 disabled:opacity-50 disabled:cursor-not-allowed ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
              title="Download modified data"
            >
              <Download className="size-4 shrink-0" />
              {!sidebarCollapsed && (
                <span className="text-xs font-medium truncate">Download Excel</span>
              )}
            </button>
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
              if (step.isContextButton) {
                return (
                  <div key={step.label} className="flex flex-col">
                    <button
                      onClick={() => setIsContextOpen(!isContextOpen)}
                      className={`flex items-center w-full gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors ${
                        sidebarCollapsed ? "justify-center" : ""
                      } text-muted-foreground hover:bg-accent hover:text-foreground`}
                      title={step.label}
                    >
                      <Icon className="size-3.5 shrink-0" />
                      {!sidebarCollapsed && (
                        <div className="flex flex-1 items-center justify-between">
                          <span className="font-medium">{step.label}</span>
                          {isContextOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
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
                  } ${idx === 0 ? "bg-primary/10 text-primary" : "text-muted-foreground"}`}
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
                <option value="gpt-4o-mini">GPT-4o Mini (Fast)</option>
                <option value="gpt-4o">GPT-4o (GitHub Models)</option>
                <option value="minimaxai/minimax-m2.5">Minimax m2.5</option>
                <option value="meta/llama-3.1-405b-instruct">Llama 3.1 405B (NVIDIA)</option>
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
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 group relative overflow-hidden px-1.5 py-0.5 rounded-sm">
                    {/* Add cool swipe background for new tables */}
                    {!isBaseDataTab && activePivotTable?.isNew && (
                      <div className="absolute inset-0 z-0 bg-gradient-to-r from-transparent via-cyan-500/20 to-transparent animate-swipe" />
                    )}

                    <span className="relative z-10 text-xs font-semibold text-foreground max-w-[250px] truncate">
                      {isBaseDataTab ? "Data" : activePivotTable?.title || "Data"}
                    </span>

                    {!isBaseDataTab && activePivotTable && (activePivotTable.insertions || activePivotTable.deletions) && (
                      <span className="relative z-10 opacity-0 group-hover:opacity-100 flex gap-1 text-[10px] items-center px-1 transition-opacity ml-1 font-bold">
                        {activePivotTable.insertions != null && <span className="text-green-500">+{activePivotTable.insertions}</span>}
                        {activePivotTable.deletions != null && <span className="text-destructive">-{activePivotTable.deletions}</span>}
                      </span>
                    )}

                    {!isBaseDataTab && activePivotTable && (
                      <button
                        type="button"
                        onClick={() => closePivotTab(activePivotTable.id)}
                        className="relative z-10 flex h-5 w-5 items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                        title="Close pivot table"
                      >
                        <X className="size-3" />
                      </button>
                    )}
                  </div>
                  {pivotTables.length > 0 && (
                    <div className="relative">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setPivotMenuOpen((prev) => !prev)
                        }}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                      >
                        <span>{pivotTables.length} pivot{pivotTables.length !== 1 ? "s" : ""}</span>
                        <ChevronDown className={`size-3 transition-transform ${pivotMenuOpen ? "rotate-180" : ""}`} />
                      </button>
                      {pivotMenuOpen && (
                        <div
                          className="absolute top-full left-0 mt-1 min-w-[220px] rounded-lg border border-border bg-card shadow-lg py-1 z-50"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            onClick={() => {
                              setActiveDataTab("base")
                              setPivotMenuOpen(false)
                            }}
                            className={`w-full text-left px-3 py-1.5 text-xs transition-colors ${
                              isBaseDataTab ? "text-primary bg-primary/10" : "text-foreground hover:bg-accent"
                            }`}
                          >
                            Base Data
                          </button>
                          {pivotTables.map((pivot) => (
                            <div key={pivot.id} className="flex items-center group relative overflow-hidden">
                              <button
                                type="button"
                                onClick={() => {
                                  if (!pivot.open) {
                                    setPivotTables((prev) =>
                                      prev.map((tab) => (tab.id === pivot.id ? { ...tab, open: true } : tab))
                                    )
                                  }
                                  setActiveDataTab(pivot.id)
                                  setPivotMenuOpen(false)
                                }}
                                className={`flex-1 flex items-center overflow-hidden text-left px-3 py-1 text-xs transition-colors ${
                                  activeDataTab === pivot.id
                                    ? "text-primary bg-primary/10"
                                    : "text-foreground hover:bg-accent"
                                }`}
                                title={pivot.title}
                              >
                                <span className="truncate inline-block align-bottom max-w-[150px]">{pivot.title}</span>

                                {(pivot.insertions || pivot.deletions) && (
                                  <span className="opacity-0 group-hover:opacity-100 flex gap-1 text-[10px] items-center px-1 transition-opacity ml-auto font-bold pl-2 bg-gradient-to-r from-transparent to-background/80">
                                    {pivot.insertions != null && <span className="text-green-500">+{pivot.insertions}</span>}
                                    {pivot.deletions != null && <span className="text-destructive">-{pivot.deletions}</span>}
                                  </span>
                                )}
                              </button>
                              
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {/* Highlighted row navigation */}
                  {isBaseDataTab && highlightedRowsArray.length > 0 && (
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
                  <span className="text-[10px] text-muted-foreground">{tableShapeLabel}</span>
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
                  <Table className="table-fixed" style={{ width: typeof totalTableWidth === "number" ? `max(100%, ${totalTableWidth}px)` : totalTableWidth }}>
                    <TableHeader>
                      <TableRow className="border-b-border hover:bg-transparent">
                        {headers.map((header) => (
                          <TableHead
                            key={header}
                            style={{ width: colWidths[header] || 150 }}
                            className="sticky top-0 z-10 h-8 bg-card/95 backdrop-blur px-3 text-[11px] font-semibold text-muted-foreground truncate group"
                          >
                            {header}
                            {/* Resizer handle */}
                            <div
                              className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize opacity-0 group-hover:opacity-100 hover:opacity-100 hover:bg-primary/50 transition-opacity"
                              onMouseDown={(e) => {
                                e.stopPropagation()
                                e.preventDefault()
                                resizingCol.current = {
                                  key: header,
                                  startX: e.clientX,
                                  startWidth: colWidths[header] || 150,
                                }
                                document.body.style.cursor = "col-resize"
                              }}
                            />
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
                        const isHighlighted = isBaseDataTab && highlightedRows.has(idx)
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
                                className={`px-3 py-1.5 text-xs truncate ${
                                  isHighlighted ? "text-foreground font-medium" : "text-foreground/80"
                                }`}
                              >
                                {String(row[header] ?? "")}
                              </TableCell>
                            ))}
                          </TableRow>
                        )
                      })}
                      {virtualizedData.endIndex < tableRows.length && (
                        <tr style={{ height: (tableRows.length - virtualizedData.endIndex) * ROW_HEIGHT }} />
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
                    backgroundColor: isDark ? "transparent" : "rgba(239, 229, 207, 0.42)",
                    backgroundImage: isDark
                      ? "radial-gradient(circle, rgba(148,163,184,0.08) 1px, transparent 1px)"
                      : "radial-gradient(circle, rgba(106, 86, 52, 0.2) 1.2px, transparent 1.2px)",
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
                          <div className="flex items-center gap-1">
                            <div className="group/open-tab relative">
                              <button
                                type="button"
                                onMouseDown={(e) => {
                                  e.preventDefault()
                                  e.stopPropagation()
                                }}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  openWidgetInNewTab(widget)
                                }}
                                className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground hover:bg-accent"
                                aria-label="Open in new tab"
                              >
                                <ExternalLink className="size-3" />
                              </button>
                              <span className="pointer-events-none absolute right-full top-1/2 mr-1.5 -translate-y-1/2 whitespace-nowrap rounded bg-popover px-1.5 py-0.5 text-[10px] text-popover-foreground opacity-0 shadow transition-opacity group-hover/open-tab:opacity-100">
                                Open in new tab
                              </span>
                            </div>
                            <button
                              type="button"
                              onMouseDown={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                              }}
                              onClick={(e) => {
                                e.stopPropagation()
                                setWidgets((prev) => prev.filter((item) => item.id !== widget.id))
                              }}
                              className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground hover:bg-accent"
                            >
                              <X className="size-3" />
                            </button>
                          </div>
                        </div>
                        <div className="min-h-0 flex-1 p-1">
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
                    {message.role === "assistant" && Array.isArray(message.table_links) && message.table_links.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {message.table_links.map((tableLink) => (
                          <button
                            key={`${index}-${tableLink.id}`}
                            type="button"
                            onClick={() => {
                              setPivotTables((prev) =>
                                prev.map((tab) => (tab.id === tableLink.id ? { ...tab, open: true } : tab))
                              )
                              setActiveDataTab(tableLink.id)
                            }}
                            className="inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-[10px] font-medium text-foreground hover:bg-accent transition-colors"
                            title="Open table in Data region"
                          >
                            
                            <span className="text-primary truncate max-w-[200px]" title={tableLink.title}>{tableLink.title}</span>
                          </button>
                        ))}
                      </div>
                    )}
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
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-xl bg-secondary px-3 py-2 text-xs text-muted-foreground">
                    <Loader2 className="size-3 animate-spin" />
                    Processing...
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Error */}
            {error && (
              <div className="mx-3 mb-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive flex justify-between items-center">
                <span>{error}</span>
                <button
                   onClick={() => setError(null)}
                   className="text-destructive hover:text-destructive/80 transition-colors"
                >
                  <X className="size-3.5" />
                </button>
              </div>
            )}

            {/* Input area */}
            <div className="border-t border-border p-3">
              <div className="relative">
                {showSlashMenu && (
                  <div className="absolute bottom-full left-0 right-11 mb-2 rounded-lg border border-border bg-card/95 p-2 shadow-md">
                    <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground space-y-1">
                      Commands
                    </div>
                    <div ref={slashListRef} className="max-h-56 overflow-y-auto scrollbar-thin pr-1" onMouseLeave={hideSlashDescription}>
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
                          const showDelayedDescription = visibleSlashDescriptionFor === item.command
                          return (
                            <button
                              key={item.command}
                              type="button"
                              onMouseDown={(e) => {
                                e.preventDefault()
                                insertSlashCommand(item)
                              }}
                              onMouseEnter={() => handleSlashOptionHover(index, item.command)}
                              className={`relative z-10 h-8 w-full rounded-md px-2.5 text-left text-[12px] font-medium leading-none transition-colors duration-150 ${
                                isActive
                                  ? "text-foreground"
                                  : "text-muted-foreground hover:text-foreground"
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
                )}
                <div className="flex items-center gap-2">
                  <textarea
                    ref={promptInputRef}
                    className="flex-1 resize-none rounded-lg border border-border bg-secondary px-3 py-2 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:ring-1 focus:ring-ring"
                    placeholder="Ask something... (type / for commands)"
                    rows={2}
                    value={prompt}
                    onChange={(e) => {
                      setPrompt(e.target.value)
                      setSlashDismissed(false)
                      if (!e.target.value.trimStart().startsWith("/")) {
                        hideSlashDescription()
                      }
                    }}
                    onKeyDown={(e) => {
                      if (showSlashMenu) {
                        if (e.key === "ArrowDown") {
                          e.preventDefault()
                          hideSlashDescription()
                          setSelectedSlashIndex((prev) => {
                            const next = (prev + 1) % filteredSlashCommands.length;
                            const container = slashListRef.current;
                            if (container) {
                              const el = container.querySelector(`button:nth-child(${next + 2})`) as HTMLElement;
                              if (el) el.scrollIntoView({ block: "nearest" });
                            }
                            return next;
                          })
                          return
                        }
                        if (e.key === "ArrowUp") {
                          e.preventDefault()
                          hideSlashDescription()
                          setSelectedSlashIndex((prev) => {
                            const next = prev === 0 ? filteredSlashCommands.length - 1 : prev - 1;
                            const container = slashListRef.current;
                            if (container) {
                              const el = container.querySelector(`button:nth-child(${next + 2})`) as HTMLElement;
                              if (el) el.scrollIntoView({ block: "nearest" });
                            }
                            return next;
                          })
                          return
                        }
                        if (e.key === "Enter" || e.key === "Tab") {
                          e.preventDefault()
                          const selected = filteredSlashCommands[selectedSlashIndex]
                          if (selected) {
                            insertSlashCommand(selected)
                          }
                          return
                        }
                        if (e.key === "Escape") {
                          e.preventDefault()
                          setSlashDismissed(true)
                          hideSlashDescription()
                          return
                        }
                      }

                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault()
                        if (!isRunningRef.current) {
                          void runAgent()
                        }
                      }
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (!isRunningRef.current) {
                        void runAgent()
                      }
                    }}
                    disabled={isRunning || !prompt.trim()}
                    className="flex h-9 w-9 shrink-0 items-center justify-center self-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:pointer-events-none disabled:opacity-40"
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

