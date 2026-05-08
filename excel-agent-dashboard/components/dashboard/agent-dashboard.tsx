"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react"
import Papa from "papaparse"
import * as XLSX from "xlsx"
import {
  Archive,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  FileSpreadsheet,
  Moon,
  Sun,
  Trash2,
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
  type ColumnRangeSelection,
  type DataGridSelection,
  type DataSelectionContext,
  type FullscreenPanel,
  type HighlightedColumn,
  type PivotTableTab,
  type RowRangeSelection,
  type SheetRow,
  type VisualizationPayload,
  type VizWidget,
  type WorkspaceDataset,
} from "@/components/dashboard/dashboard-shared"
import { CanvasBoard } from "@/components/dashboard/components/canvas-board"
import { ChatPanel } from "@/components/dashboard/components/chat-panel"
import { DataGrid } from "@/components/dashboard/components/data-grid"
import { API_BASE_URL, useAgentRunner } from "@/components/dashboard/hooks/use-agent-runner"

const initialDataset: WorkspaceDataset = {
  id: "seed-dataset",
  fileName: "gapminder-lite.xlsx",
  sheetName: "Data",
  displayName: "gapminder-lite.xlsx / Data",
  rows: seedRows,
  rowCount: seedRows.length,
  columnCount: Object.keys(seedRows[0] ?? {}).length,
  kind: "uploaded",
  modified: false,
}

const MAX_DATA_MUTATION_HISTORY = 50
const MAX_SELECTION_CONTEXT_INDICES = 500
const MAX_SELECTION_CONTEXT_CELLS = 200
const MAX_SELECTION_PREVIEW_ROWS = 6
const MAX_SELECTION_PREVIEW_COLUMNS = 8

const EMPTY_GRID_SELECTION: DataGridSelection = {
  rowRanges: [],
  columnRanges: [],
  cellRanges: [],
}

type DataMutationSnapshot = {
  datasets: WorkspaceDataset[]
  activeDatasetId: string
}

type DataMutationHistoryEntry = {
  before: DataMutationSnapshot
  after: DataMutationSnapshot
  revertHighlightIndex: number | null
}

function cloneRows(rows: SheetRow[]) {
  return rows.map((row) => ({ ...row }))
}

function cloneDatasets(datasets: WorkspaceDataset[]) {
  return datasets.map((dataset) => ({
    ...dataset,
    rows: cloneRows(dataset.rows),
    sourceDatasetIds: dataset.sourceDatasetIds ? [...dataset.sourceDatasetIds] : undefined,
  }))
}

function cloneMutationSnapshot(snapshot: DataMutationSnapshot): DataMutationSnapshot {
  return {
    activeDatasetId: snapshot.activeDatasetId,
    datasets: cloneDatasets(snapshot.datasets),
  }
}

function normalizeIndexRange(start: number, end: number, max: number) {
  const low = Math.max(0, Math.min(start, end))
  const high = Math.min(Math.max(0, max - 1), Math.max(start, end))
  return { start: low, end: high }
}

function expandIndexRanges(ranges: Array<{ start: number; end: number }>, maxItems: number) {
  const values: number[] = []
  const seen = new Set<number>()
  let compact = false

  ranges.forEach((range) => {
    for (let index = range.start; index <= range.end; index += 1) {
      if (seen.has(index)) continue
      if (values.length >= maxItems) {
        compact = true
        return
      }
      seen.add(index)
      values.push(index)
    }
  })

  return { values, compact }
}

function countIndexRanges(ranges: Array<{ start: number; end: number }>) {
  const seen = new Set<number>()
  ranges.forEach((range) => {
    for (let index = range.start; index <= range.end; index += 1) {
      seen.add(index)
    }
  })
  return seen.size
}

function selectionIsEmpty(selection: DataGridSelection) {
  return (
    selection.rowRanges.length === 0 &&
    selection.columnRanges.length === 0 &&
    selection.cellRanges.length === 0
  )
}

function selectionSummaryFromCounts({
  selectionType,
  rowCount,
  columnCount,
  cellCount,
}: {
  selectionType: DataSelectionContext["selection_type"]
  rowCount: number
  columnCount: number
  cellCount: number
}) {
  if (selectionType === "rows") return `${rowCount} row${rowCount === 1 ? "" : "s"}`
  if (selectionType === "columns") return `${columnCount} column${columnCount === 1 ? "" : "s"}`
  if (selectionType === "cells") return `${cellCount} cell${cellCount === 1 ? "" : "s"}`
  return `${rowCount} row${rowCount === 1 ? "" : "s"}, ${columnCount} column${columnCount === 1 ? "" : "s"}, ${cellCount} cell${cellCount === 1 ? "" : "s"}`
}

function detectRowIdColumn(columns: string[]) {
  return (
    columns.find((column) => column.toLowerCase() === "id") ??
    columns.find((column) => column.toLowerCase().endsWith("_id")) ??
    columns.find((column) => column.toLowerCase().endsWith(" id"))
  )
}

function buildDataSelectionContext({
  selection,
  dataset,
  rows,
  columns,
}: {
  selection: DataGridSelection
  dataset: WorkspaceDataset | undefined
  rows: SheetRow[]
  columns: string[]
}): DataSelectionContext | null {
  if (!dataset || rows.length === 0 || columns.length === 0 || selectionIsEmpty(selection)) return null

  const hasRowRanges = selection.rowRanges.length > 0
  const hasColumnRanges = selection.columnRanges.length > 0
  const hasCellRanges = selection.cellRanges.length > 0
  const selectedKinds = [hasRowRanges, hasColumnRanges, hasCellRanges].filter(Boolean).length
  const selectionType: DataSelectionContext["selection_type"] =
    selectedKinds > 1
      ? "mixed"
      : hasRowRanges
        ? "rows"
        : hasColumnRanges
          ? "columns"
          : "cells"

  const normalizedRowRanges = selection.rowRanges.map((range: RowRangeSelection) =>
    normalizeIndexRange(range.startRowIndex, range.endRowIndex, rows.length)
  )
  const normalizedCellRowRanges = selection.cellRanges.map((range) =>
    normalizeIndexRange(range.startRowIndex, range.endRowIndex, rows.length)
  )
  const normalizedColumnRanges = selection.columnRanges.map((range: ColumnRangeSelection) =>
    normalizeIndexRange(range.startColumnIndex, range.endColumnIndex, columns.length)
  )
  const normalizedCellColumnRanges = selection.cellRanges.map((range) =>
    normalizeIndexRange(range.startColumnIndex, range.endColumnIndex, columns.length)
  )
  const effectiveRowRanges = [...normalizedRowRanges, ...normalizedCellRowRanges]
  const effectiveColumnRanges = [...normalizedColumnRanges, ...normalizedCellColumnRanges]
  const appliesToAllRows = hasColumnRanges && !hasRowRanges && !hasCellRanges
  const appliesToAllColumns = hasRowRanges && !hasColumnRanges && !hasCellRanges
  const selectedRowExpansion = appliesToAllRows
    ? { values: [], compact: true }
    : expandIndexRanges(effectiveRowRanges, MAX_SELECTION_CONTEXT_INDICES)
  const selectedColumnIndices = appliesToAllColumns
    ? columns.map((_, index) => index)
    : expandIndexRanges(effectiveColumnRanges, columns.length).values
  const selectedColumns = selectedColumnIndices.map((index) => columns[index]).filter(Boolean)
  const rowCount = appliesToAllRows ? rows.length : countIndexRanges(effectiveRowRanges)
  const columnCount = appliesToAllColumns ? columns.length : selectedColumns.length
  const selectedCellCount = selection.cellRanges.reduce((total, range) => {
    const rowRange = normalizeIndexRange(range.startRowIndex, range.endRowIndex, rows.length)
    const columnRange = normalizeIndexRange(range.startColumnIndex, range.endColumnIndex, columns.length)
    return total + (rowRange.end - rowRange.start + 1) * (columnRange.end - columnRange.start + 1)
  }, 0)
  const cellCount = selectionType === "cells" ? selectedCellCount : rowCount * columnCount
  const rowIdColumn = detectRowIdColumn(columns)
  const rowIds = rowIdColumn && selectedRowExpansion.values.length > 0
    ? selectedRowExpansion.values.slice(0, MAX_SELECTION_CONTEXT_INDICES).map((rowIndex) => rows[rowIndex]?.[rowIdColumn] ?? null)
    : undefined
  const previewRowIndices = appliesToAllRows
    ? rows.slice(0, MAX_SELECTION_PREVIEW_ROWS).map((_, index) => index)
    : selectedRowExpansion.values.slice(0, MAX_SELECTION_PREVIEW_ROWS)
  const previewColumns = (appliesToAllColumns ? columns : selectedColumns).slice(0, MAX_SELECTION_PREVIEW_COLUMNS)
  const previewRows = previewRowIndices.map((rowIndex) => {
    const source = rows[rowIndex] ?? {}
    const values: SheetRow = {}
    previewColumns.forEach((column) => {
      values[column] = source[column] ?? null
    })
    return {
      row_index: rowIndex,
      row_id: rowIdColumn ? source[rowIdColumn] ?? null : undefined,
      values,
    }
  })
  const cellExpansion = selection.cellRanges.length === 0
    ? { values: [] as Array<{ row_index: number; column: string }>, compact: false }
    : (() => {
        const values: Array<{ row_index: number; column: string }> = []
        let compact = false
        for (const range of selection.cellRanges) {
          const rowRange = normalizeIndexRange(range.startRowIndex, range.endRowIndex, rows.length)
          const columnRange = normalizeIndexRange(range.startColumnIndex, range.endColumnIndex, columns.length)
          for (let rowIndex = rowRange.start; rowIndex <= rowRange.end; rowIndex += 1) {
            for (let columnIndex = columnRange.start; columnIndex <= columnRange.end; columnIndex += 1) {
              if (values.length >= MAX_SELECTION_CONTEXT_CELLS) {
                compact = true
                return { values, compact }
              }
              values.push({ row_index: rowIndex, column: columns[columnIndex] })
            }
          }
        }
        return { values, compact }
      })()

  return {
    dataset_id: dataset.id,
    dataset_name: dataset.displayName,
    selection_type: selectionType,
    row_count: rowCount,
    column_count: columnCount,
    cell_count: cellCount,
    applies_to_all_rows: appliesToAllRows,
    applies_to_all_columns: appliesToAllColumns,
    row_ranges: effectiveRowRanges,
    column_ranges: effectiveColumnRanges.map((range) => ({
      ...range,
      columns: columns.slice(range.start, range.end + 1),
    })),
    columns: selectedColumns,
    row_indices: selectedRowExpansion.compact ? undefined : selectedRowExpansion.values,
    cells: cellExpansion.compact ? undefined : cellExpansion.values,
    row_id_column: rowIdColumn,
    row_ids: rowIds,
    preview_rows: previewRows,
    compact: selectedRowExpansion.compact || cellExpansion.compact,
    summary: selectionSummaryFromCounts({ selectionType, rowCount, columnCount, cellCount }),
  }
}

function makeDatasetDisplayName(fileName: string, sheetName?: string) {
  return sheetName ? `${fileName} / ${sheetName}` : fileName
}

function resolvePromptDatasetIds(prompt: string, datasets: WorkspaceDataset[]) {
  const promptLower = prompt.toLowerCase()
  const matches: Array<{ id: string; index: number }> = []

  datasets.forEach((dataset) => {
    const candidates = [dataset.displayName, dataset.fileName]

    const matchIndex = candidates
      .map((candidate) => promptLower.indexOf(`@${candidate.toLowerCase()}`))
      .filter((index) => index >= 0)
      .sort((a, b) => a - b)[0]

    if (matchIndex !== undefined) {
      matches.push({ id: dataset.id, index: matchIndex })
    }
  })

  return matches
    .sort((a, b) => a.index - b.index)
    .map((match) => match.id)
}

function uniqueDatasetIds(ids: string[]) {
  return Array.from(new Set(ids))
}

async function registerDataset(name: string, rows: SheetRow[]) {
  const response = await fetch(`${API_BASE_URL}/dataset/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, rows }),
  })
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Dataset registration failed" }))
    throw new Error(errorData.detail ?? "Dataset registration failed")
  }
  return (await response.json()) as { dataset_id: string; row_count: number; column_count: number }
}

function rowsToWorkbookBlob(rows: SheetRow[], sheetName = "Data") {
  const worksheet = XLSX.utils.json_to_sheet(rows)
  const workbook = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(workbook, worksheet, sheetName.slice(0, 31) || "Data")
  const buffer = XLSX.write(workbook, { bookType: "xlsx", type: "array" }) as ArrayBuffer
  return new Blob([buffer], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" })
}

function datasetToWorkbookBlob(dataset: WorkspaceDataset) {
  return rowsToWorkbookBlob(dataset.rows, dataset.sheetName || "Data")
}

function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = fileName
  link.click()
  URL.revokeObjectURL(url)
}

function sanitizeFileName(value: string) {
  return value.replace(/[\\/:*?"<>|]+/g, "_").replace(/\s+/g, " ").trim() || "dataset"
}

function makeCrcTable() {
  const table: number[] = []
  for (let n = 0; n < 256; n += 1) {
    let c = n
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    }
    table[n] = c >>> 0
  }
  return table
}

const CRC_TABLE = makeCrcTable()

function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

function writeUint16(target: number[], value: number) {
  target.push(value & 0xff, (value >>> 8) & 0xff)
}

function writeUint32(target: number[], value: number) {
  target.push(value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff)
}

function bytesToBlobPart(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
}

async function buildZip(files: Array<{ name: string; blob: Blob }>) {
  const encoder = new TextEncoder()
  const chunks: BlobPart[] = []
  const centralDirectory: BlobPart[] = []
  let offset = 0
  let centralSize = 0

  for (const file of files) {
    const data = new Uint8Array(await file.blob.arrayBuffer())
    const name = encoder.encode(file.name)
    const crc = crc32(data)

    const local: number[] = []
    writeUint32(local, 0x04034b50)
    writeUint16(local, 20)
    writeUint16(local, 0)
    writeUint16(local, 0)
    writeUint16(local, 0)
    writeUint16(local, 0)
    writeUint32(local, crc)
    writeUint32(local, data.length)
    writeUint32(local, data.length)
    writeUint16(local, name.length)
    writeUint16(local, 0)
    chunks.push(bytesToBlobPart(new Uint8Array(local)), bytesToBlobPart(name), bytesToBlobPart(data))

    const central: number[] = []
    writeUint32(central, 0x02014b50)
    writeUint16(central, 20)
    writeUint16(central, 20)
    writeUint16(central, 0)
    writeUint16(central, 0)
    writeUint16(central, 0)
    writeUint16(central, 0)
    writeUint32(central, crc)
    writeUint32(central, data.length)
    writeUint32(central, data.length)
    writeUint16(central, name.length)
    writeUint16(central, 0)
    writeUint16(central, 0)
    writeUint16(central, 0)
    writeUint16(central, 0)
    writeUint32(central, 0)
    writeUint32(central, offset)
    centralDirectory.push(bytesToBlobPart(new Uint8Array(central)), bytesToBlobPart(name))
    centralSize += central.length + name.length

    offset += local.length + name.length + data.length
  }

  const centralOffset = offset
  const end: number[] = []
  writeUint32(end, 0x06054b50)
  writeUint16(end, 0)
  writeUint16(end, 0)
  writeUint16(end, files.length)
  writeUint16(end, files.length)
  writeUint32(end, centralSize)
  writeUint32(end, centralOffset)
  writeUint16(end, 0)

  return new Blob([...chunks, ...centralDirectory, bytesToBlobPart(new Uint8Array(end))], { type: "application/zip" })
}

export function AgentDashboard() {
  const [datasets, setDatasets] = useState<WorkspaceDataset[]>([initialDataset])
  const [activeDatasetId, setActiveDatasetId] = useState(initialDataset.id)
  const [attachedDatasetIds, setAttachedDatasetIds] = useState<string[]>([])
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
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0, total: 0 })
  const [queryCount, setQueryCount] = useState(0)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_CHAT_MESSAGES)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [thinkingMode, setThinkingMode] = useState(false)
  const [undoStack, setUndoStack] = useState<DataMutationHistoryEntry[]>([])
  const [redoStack, setRedoStack] = useState<DataMutationHistoryEntry[]>([])
  const [highlightedChatMessageIndex, setHighlightedChatMessageIndex] = useState<number | null>(null)
  const [gridSelection, setGridSelection] = useState<DataGridSelection>(EMPTY_GRID_SELECTION)
  const [attachedSelectionContext, setAttachedSelectionContext] = useState<DataSelectionContext | null>(null)
  const [datasetContextMenu, setDatasetContextMenu] = useState<{
    datasetId: string
    x: number
    y: number
  } | null>(null)

  const splitContainerRef = useRef<HTMLDivElement | null>(null)
  const mainContentRef = useRef<HTMLDivElement | null>(null)
  const newTableTimeoutsRef = useRef<number[]>([])
  const modelMenuRef = useRef<HTMLDivElement | null>(null)

  const { cancelRun, clearError, error, isRunning, runAgent, setError } = useAgentRunner()

  const activeDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === activeDatasetId) ?? datasets[0],
    [activeDatasetId, datasets]
  )
  const rows = useMemo(() => activeDataset?.rows ?? [], [activeDataset])
  const activeColumns = useMemo(() => Object.keys(rows[0] ?? {}), [rows])
  const activePivotTable = useMemo(
    () => pivotTables.find((table) => table.id === activeDataTab && table.open),
    [activeDataTab, pivotTables]
  )
  const pendingSelectionContext = useMemo(
    () =>
      activeDataTab === "base"
        ? buildDataSelectionContext({
            selection: gridSelection,
            dataset: activeDataset,
            rows,
            columns: activeColumns,
          })
        : null,
    [activeColumns, activeDataTab, activeDataset, gridSelection, rows]
  )

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
    return `${datasets.length} data | ${rows.length}X${baseColumnCount}`
  }, [datasets.length, rows])

  const visibleGridExport = useMemo(() => {
    if (activePivotTable) {
      return {
        fileName: `${sanitizeFileName(activePivotTable.title)}.xlsx`,
        rows: activePivotTable.rows,
        sheetName: activePivotTable.title,
      }
    }

    if (!activeDataset) return null
    return {
      fileName: `${sanitizeFileName(activeDataset.displayName)}.xlsx`,
      rows: activeDataset.rows,
      sheetName: activeDataset.sheetName || "Data",
    }
  }, [activeDataset, activePivotTable])
  const contextMenuDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === datasetContextMenu?.datasetId),
    [datasetContextMenu?.datasetId, datasets]
  )

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
    if (!datasetContextMenu) return

    const closeMenu = () => setDatasetContextMenu(null)
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeMenu()
      }
    }

    document.addEventListener("click", closeMenu)
    document.addEventListener("contextmenu", closeMenu)
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("click", closeMenu)
      document.removeEventListener("contextmenu", closeMenu)
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [datasetContextMenu])

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
    const files = Array.from(event.target.files ?? [])
    if (files.length === 0) return

    try {
      const nextDatasets: WorkspaceDataset[] = []

      for (const file of files) {
        const extension = file.name.split(".").pop()?.toLowerCase()

        if (extension === "xlsx" || extension === "xls") {
          const buffer = await file.arrayBuffer()
          const workbook = XLSX.read(buffer, { type: "array" })
          for (const sheetName of workbook.SheetNames) {
            const sheet = workbook.Sheets[sheetName]
            const parsedRows = XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, { defval: "" })
            const rowsForSheet = normalizeRows(parsedRows.filter((row) => Object.keys(row).length > 0))
            if (rowsForSheet.length === 0) continue
            const displayName = makeDatasetDisplayName(file.name, sheetName)
            const registered = await registerDataset(displayName, rowsForSheet)
            nextDatasets.push({
              id: registered.dataset_id,
              fileName: file.name,
              sheetName,
              displayName,
              rows: rowsForSheet,
              rowCount: registered.row_count,
              columnCount: registered.column_count,
              kind: "uploaded",
              modified: false,
            })
          }
        } else {
          const text = await file.text()
          const parsed = Papa.parse<Record<string, unknown>>(text, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
          })
          const rowsForFile = normalizeRows((parsed.data ?? []).filter((row) => Object.keys(row).length > 0))
          if (rowsForFile.length === 0) continue
          const displayName = makeDatasetDisplayName(file.name)
          const registered = await registerDataset(displayName, rowsForFile)
          nextDatasets.push({
            id: registered.dataset_id,
            fileName: file.name,
            displayName,
            rows: rowsForFile,
            rowCount: registered.row_count,
            columnCount: registered.column_count,
            kind: "uploaded",
            modified: false,
          })
        }
      }

      if (nextDatasets.length === 0) return

      setDatasets((prev) => {
        const keepSeed = prev.length === 1 && prev[0]?.id === initialDataset.id ? [] : prev
        return [...keepSeed, ...nextDatasets]
      })
      setActiveDatasetId(nextDatasets[0].id)
      setAttachedDatasetIds([])
      setWidgets([])
      setPivotTables([])
      setActiveDataTab("base")
      setHighlightedRows(new Set())
      setHighlightedColumns([])
      setUndoStack([])
      setRedoStack([])
      setHighlightedChatMessageIndex(null)
      setGridSelection(EMPTY_GRID_SELECTION)
      setAttachedSelectionContext(null)
      clearError()
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed")
    } finally {
      event.target.value = ""
    }
  }

  const handleDownloadExcel = () => {
    if (!visibleGridExport || visibleGridExport.rows.length === 0) return
    downloadBlob(rowsToWorkbookBlob(visibleGridExport.rows, visibleGridExport.sheetName), visibleGridExport.fileName)
  }

  const handleDownloadDataset = useCallback((dataset: WorkspaceDataset) => {
    downloadBlob(datasetToWorkbookBlob(dataset), `${sanitizeFileName(dataset.displayName)}.xlsx`)
  }, [])

  const handleDownloadWorkspaceZip = useCallback(async () => {
    const exportable = datasets.filter((dataset) => dataset.rows.length > 0)
    if (exportable.length === 0) return
    const files = exportable.map((dataset) => ({
      name: `${sanitizeFileName(dataset.displayName)}.xlsx`,
      blob: datasetToWorkbookBlob(dataset),
    }))
    const zip = await buildZip(files)
    downloadBlob(zip, "datapilot-workspace.zip")
  }, [datasets])

  const handleDeleteDataset = useCallback((datasetId: string) => {
    const nextDatasets = datasets.filter((dataset) => dataset.id !== datasetId)

    setDatasets(nextDatasets)
    if (activeDatasetId === datasetId) {
      setActiveDatasetId(nextDatasets[0]?.id ?? "")
      setActiveDataTab("base")
      setHighlightedRows(new Set())
      setHighlightedColumns([])
      setGridSelection(EMPTY_GRID_SELECTION)
    }
    setAttachedDatasetIds((prev) => prev.filter((id) => id !== datasetId))
    if (attachedSelectionContext?.dataset_id === datasetId) {
      setAttachedSelectionContext(null)
    }
    setUndoStack([])
    setRedoStack([])
    setDatasetContextMenu(null)
    clearError()
  }, [activeDatasetId, attachedSelectionContext?.dataset_id, clearError, datasets])

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

  const applyMutationSnapshot = useCallback((snapshot: DataMutationSnapshot) => {
    const nextSnapshot = cloneMutationSnapshot(snapshot)
    const nextActiveDatasetId = nextSnapshot.datasets.some((dataset) => dataset.id === nextSnapshot.activeDatasetId)
      ? nextSnapshot.activeDatasetId
      : nextSnapshot.datasets[0]?.id

    setDatasets(nextSnapshot.datasets)
    if (nextActiveDatasetId) {
      setActiveDatasetId(nextActiveDatasetId)
    }
    setActiveDataTab("base")
    setHighlightedRows(new Set())
    setHighlightedColumns([])
    setGridSelection(EMPTY_GRID_SELECTION)
    setAttachedSelectionContext(null)
  }, [])

  const handleRevertDataMutation = useCallback(() => {
    if (isRunning || undoStack.length === 0) return

    const entry = undoStack[undoStack.length - 1]
    applyMutationSnapshot(entry.before)
    setUndoStack(undoStack.slice(0, -1))
    setRedoStack((prev) => [...prev, entry].slice(-MAX_DATA_MUTATION_HISTORY))
    setHighlightedChatMessageIndex(entry.revertHighlightIndex)
    clearError()
  }, [applyMutationSnapshot, clearError, isRunning, undoStack])

  const handleRestoreDataMutation = useCallback(() => {
    if (isRunning || redoStack.length === 0) return

    const entry = redoStack[redoStack.length - 1]
    applyMutationSnapshot(entry.after)
    setRedoStack(redoStack.slice(0, -1))
    setUndoStack((prev) => [...prev, entry].slice(-MAX_DATA_MUTATION_HISTORY))
    setHighlightedChatMessageIndex(null)
    clearError()
  }, [applyMutationSnapshot, clearError, isRunning, redoStack])

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
    setHighlightedChatMessageIndex(null)

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

    const mentionedIds = resolvePromptDatasetIds(outgoingPrompt, datasets)
    const attachedIds = uniqueDatasetIds([
      ...mentionedIds,
      ...attachedDatasetIds,
    ]).filter((datasetId) => datasets.some((dataset) => dataset.id === datasetId))
    const requestDatasetIds = attachedIds.length > 0
      ? attachedIds
      : activeDataset && activeDataset.id !== initialDataset.id
        ? [activeDataset.id]
        : []
    const requestActiveDatasetId = requestDatasetIds[0] ?? null
    const requestRows = requestActiveDatasetId
      ? datasets.find((dataset) => dataset.id === requestActiveDatasetId)?.rows ?? rows
      : rows
    const datasetNames = Object.fromEntries(datasets.map((dataset) => [dataset.id, dataset.displayName]))

    const result = await runAgent({
      prompt: outgoingPrompt,
      rows: requestRows,
      activeDatasetId: requestActiveDatasetId,
      selectedDatasetIds: requestDatasetIds.slice(1),
      datasetNames,
      modelName,
      history: historyWithUser,
      thinkingMode,
      selectionContext: attachedSelectionContext,
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
      if (result.cancelled) {
        setChatMessages((prev) => {
          if (!thinkingMode || prev.length === 0) return prev
          const next = [...prev]
          const lastMessage = next[next.length - 1]
          if (lastMessage?.role === "assistant" && lastMessage.isStreaming) {
            next[next.length - 1] = {
              ...lastMessage,
              content: "Stopped.",
              isStreaming: false,
            }
            return next
          }
          return prev
        })
        return
      }

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

    const hasDatasetMutation =
      data.mutation ||
      data.updatedDatasets.length > 0 ||
      data.createdDatasets.length > 0

    if (hasDatasetMutation) {
      const beforeSnapshot: DataMutationSnapshot = {
        activeDatasetId,
        datasets: cloneDatasets(datasets),
      }
      const mutationTargetDatasetId = requestActiveDatasetId ?? activeDataset?.id
      let nextDatasets = cloneDatasets(datasets)
      let nextActiveDatasetId = activeDatasetId

      if (data.mutation && mutationTargetDatasetId) {
        nextDatasets = nextDatasets.map((dataset) =>
          dataset.id === mutationTargetDatasetId
            ? {
                ...dataset,
                rows: cloneRows(data.nextRows),
                rowCount: data.nextRows.length,
                columnCount: Object.keys(data.nextRows[0] ?? {}).length,
                modified: true,
              }
            : dataset
        )
      }

      if (data.updatedDatasets.length > 0 || data.createdDatasets.length > 0) {
        const byId = new Map(nextDatasets.map((dataset) => [dataset.id, dataset] as const))
        data.updatedDatasets.forEach((dataset) => {
          byId.set(dataset.id, {
            ...(byId.get(dataset.id) ?? dataset),
            ...dataset,
            rows: cloneRows(dataset.rows),
            sourceDatasetIds: dataset.sourceDatasetIds ? [...dataset.sourceDatasetIds] : undefined,
            modified: true,
          })
        })
        data.createdDatasets.forEach((dataset) => {
          byId.set(dataset.id, {
            ...dataset,
            rows: cloneRows(dataset.rows),
            sourceDatasetIds: dataset.sourceDatasetIds ? [...dataset.sourceDatasetIds] : undefined,
          })
        })
        nextDatasets = Array.from(byId.values())

        const firstCreated = data.createdDatasets[0]
        if (firstCreated) {
          nextActiveDatasetId = firstCreated.id
        }
      }

      const afterSnapshot: DataMutationSnapshot = {
        activeDatasetId: nextActiveDatasetId,
        datasets: cloneDatasets(nextDatasets),
      }

      setDatasets(nextDatasets)
      if (nextActiveDatasetId !== activeDatasetId) {
        setActiveDatasetId(nextActiveDatasetId)
      }
      setActiveDataTab("base")
      setGridSelection(EMPTY_GRID_SELECTION)
      setAttachedSelectionContext(null)
      setUndoStack((prev) => [
        ...prev,
        {
          before: beforeSnapshot,
          after: afterSnapshot,
          revertHighlightIndex: chatMessages.length > 0 ? chatMessages.length - 1 : null,
        },
      ].slice(-MAX_DATA_MUTATION_HISTORY))
      setRedoStack([])
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
    setAttachedDatasetIds([])
    setAttachedSelectionContext(null)
  }, [activeDataset, activeDatasetId, addVisualizationWidget, attachedDatasetIds, attachedSelectionContext, chatMessages, clearError, datasets, isRunning, modelName, prompt, rows, runAgent, selectedModelLabel, thinkingMode])

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
                multiple
                onChange={handleFileUpload}
              />
            </label>

            <button
              type="button"
              onClick={handleDownloadExcel}
              disabled={!visibleGridExport || visibleGridExport.rows.length === 0}
              className={`w-full group flex items-center gap-2.5 rounded-lg border border-dashed border-border p-2.5 text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground hover:bg-accent/50 disabled:opacity-50 disabled:cursor-not-allowed ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
              title="Download current data grid"
            >
              <Download className="size-4 shrink-0" />
              {!sidebarCollapsed && <span className="text-xs font-medium truncate">Download Excel</span>}
            </button>

            <button
              type="button"
              onClick={handleDownloadWorkspaceZip}
              disabled={datasets.length === 0}
              className={`w-full group flex items-center gap-2.5 rounded-lg border border-dashed border-border p-2.5 text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground hover:bg-accent/50 disabled:opacity-50 disabled:cursor-not-allowed ${
                sidebarCollapsed ? "justify-center" : ""
              }`}
              title="Download workspace ZIP"
            >
              <Archive className="size-4 shrink-0" />
              {!sidebarCollapsed && <span className="text-xs font-medium truncate">Download ZIP</span>}
            </button>
          </div>

          {!sidebarCollapsed && (
            <div className="min-h-0 border-y border-border/70 px-2 py-2">
              <div className="mb-1 px-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                Datasets
              </div>
              <div className="max-h-44 space-y-1 overflow-y-auto pr-1 scrollbar-thin">
                {datasets.map((dataset) => {
                  const isActive = dataset.id === activeDataset?.id
                  return (
                    <div
                      key={dataset.id}
                      onContextMenu={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        setDatasetContextMenu({
                          datasetId: dataset.id,
                          x: Math.max(8, Math.min(event.clientX, window.innerWidth - 190)),
                          y: Math.max(8, Math.min(event.clientY, window.innerHeight - 64)),
                        })
                      }}
                      className={`flex items-center gap-1 rounded-md px-1.5 py-1 transition-colors ${
                        isActive
                          ? "bg-emerald-500/15 text-emerald-700 ring-1 ring-emerald-500/25 dark:text-emerald-300"
                          : "text-muted-foreground hover:bg-accent hover:text-foreground"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setActiveDatasetId(dataset.id)
                          setActiveDataTab("base")
                          setHighlightedRows(new Set())
                          setHighlightedColumns([])
                          setGridSelection(EMPTY_GRID_SELECTION)
                          setAttachedSelectionContext(null)
                        }}
                        className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
                        title={dataset.displayName}
                      >
                        <Database className="size-3 shrink-0" />
                        <span className="truncate text-[11px] font-medium">{dataset.displayName}</span>
                        {dataset.modified && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400" />}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDownloadDataset(dataset)}
                        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded transition-colors ${
                          isActive
                            ? "text-emerald-700 hover:bg-emerald-500/15 dark:text-emerald-300"
                            : "text-muted-foreground hover:bg-accent hover:text-foreground"
                        }`}
                        title="Download dataset"
                      >
                        <Download className="size-3" />
                      </button>
                    </div>
                  )
                })}
              </div>
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
                  <div key={step.label} className="group relative">
                    <button
                      type="button"
                      className={`flex items-center w-full gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors ${
                        sidebarCollapsed ? "justify-center" : ""
                      } text-muted-foreground hover:bg-accent hover:text-foreground`}
                      title={step.label}
                    >
                      <Icon className="size-3.5 shrink-0" />
                      {!sidebarCollapsed && (
                        <div className="flex flex-1 items-center justify-between">
                          <span className="font-medium">{step.label}</span>
                          <ChevronRight className="size-3.5" />
                        </div>
                      )}
                    </button>

                    <div className="pointer-events-none absolute left-full top-0 z-[1000] ml-2 w-48 rounded-xl opacity-0 shadow-xl transition-all duration-200 ease-[var(--menu-ease)] group-hover:pointer-events-auto group-hover:translate-x-0 group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:translate-x-0 group-focus-within:opacity-100 translate-x-1">
                      <div className="menu-surface space-y-1.5 rounded-xl p-2.5 text-xs text-muted-foreground">
                        <div className="flex items-center justify-between rounded bg-secondary/50 px-2 py-1.5">
                          <span>Queries:</span>
                          <span className="font-medium text-foreground">{queryCount}</span>
                        </div>
                        <div className="flex items-center justify-between rounded bg-secondary/50 px-2 py-1.5">
                          <span>Prompt:</span>
                          <span className="font-medium text-foreground">{totalTokens.prompt.toLocaleString()}</span>
                        </div>
                        <div className="flex items-center justify-between rounded bg-secondary/50 px-2 py-1.5">
                          <span>Completion:</span>
                          <span className="font-medium text-foreground">{totalTokens.completion.toLocaleString()}</span>
                        </div>
                        <div className="flex items-center justify-between rounded bg-primary/10 px-2 py-1.5 text-primary">
                          <span className="font-semibold">Total Tokens:</span>
                          <span className="font-bold">{totalTokens.total.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
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

        {datasetContextMenu && contextMenuDataset && (
          <div
            className="menu-surface fixed z-[1200] min-w-44 overflow-hidden rounded-lg p-1.5 shadow-xl"
            style={{ left: datasetContextMenu.x, top: datasetContextMenu.y }}
            onClick={(event) => event.stopPropagation()}
            onContextMenu={(event) => {
              event.preventDefault()
              event.stopPropagation()
            }}
          >
            <div className="px-2.5 pb-1.5 pt-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              {contextMenuDataset.displayName}
            </div>
            <button
              type="button"
              onClick={() => handleDeleteDataset(contextMenuDataset.id)}
              className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs font-medium text-destructive transition-colors hover:bg-destructive/10"
            >
              <Trash2 className="size-3.5 shrink-0" />
              Delete dataset
            </button>
          </div>
        )}

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
                selection={gridSelection}
                fullscreen={fullscreenPanel === "data"}
                canRevert={!isRunning && undoStack.length > 0}
                canRestore={!isRunning && redoStack.length > 0}
                onSelectionChange={setGridSelection}
                onActivateBaseData={activateBaseData}
                onOpenPivotTable={openPivotTable}
                onClosePivotTable={closePivotTable}
                onRevert={handleRevertDataMutation}
                onRestore={handleRestoreDataMutation}
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
            highlightedMessageIndex={highlightedChatMessageIndex}
            pendingSelectionContext={pendingSelectionContext}
            attachedSelectionContext={attachedSelectionContext}
            prompt={prompt}
            isRunning={isRunning}
            error={error}
            thinkingMode={thinkingMode}
            modelLabel={selectedModelLabel}
            datasets={datasets}
            onPromptChange={(value) => setPrompt(value)}
            onSubmit={handleRunAgent}
            onCancel={cancelRun}
            onToggleThinkingMode={toggleThinkingMode}
            onClearError={clearError}
            onOpenTableLink={openPivotTable}
            onAttachSelection={() => {
              if (pendingSelectionContext) {
                setAttachedSelectionContext(pendingSelectionContext)
              }
            }}
            onClearSelection={() => {
              setGridSelection(EMPTY_GRID_SELECTION)
              setAttachedSelectionContext(null)
            }}
          />
        </div>
      </div>
    </div>
  )
}
