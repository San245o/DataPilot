"use client"

import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react"
import { ChevronDown, ChevronLeft, ChevronRight, Maximize2, Minimize2, Redo2, Undo2, X } from "lucide-react"

import {
  OVERSCAN,
  ROW_HEIGHT,
  type CellRangeSelection,
  type ColumnRangeSelection,
  type DataGridSelection,
  type HighlightedColumn,
  type PivotTableTab,
  type RowRangeSelection,
  type SheetRow,
} from "@/components/dashboard/dashboard-shared"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

const ROW_GUTTER_WIDTH = 48

type DataGridProps = {
  rows: SheetRow[]
  pivotTables: PivotTableTab[]
  activeDataTab: string
  highlightedRows: Set<number>
  highlightedColumns: HighlightedColumn[]
  selection: DataGridSelection
  fullscreen: boolean
  canRevert: boolean
  canRestore: boolean
  onSelectionChange: (selection: DataGridSelection) => void
  onActivateBaseData: () => void
  onOpenPivotTable: (pivotId: string) => void
  onClosePivotTable: (pivotId: string) => void
  onRevert: () => void
  onRestore: () => void
  onToggleFullscreen: () => void
}

export function DataGrid({
  rows,
  pivotTables,
  activeDataTab,
  highlightedRows,
  highlightedColumns,
  selection,
  fullscreen,
  canRevert,
  canRestore,
  onSelectionChange,
  onActivateBaseData,
  onOpenPivotTable,
  onClosePivotTable,
  onRevert,
  onRestore,
  onToggleFullscreen,
}: DataGridProps) {
  const [pivotMenuOpen, setPivotMenuOpen] = useState(false)
  const [scrollTop, setScrollTop] = useState(0)
  const [tableHeight, setTableHeight] = useState(400)
  const [colWidths, setColWidths] = useState<Record<string, number>>({})
  const [currentHighlightIndex, setCurrentHighlightIndex] = useState(0)

  const tableContainerRef = useRef<HTMLDivElement | null>(null)
  const resizingCol = useRef<{ key: string; startX: number; startWidth: number } | null>(null)
  const selectionAnchorRef = useRef<{
    kind: "cell" | "row" | "column"
    rowIndex?: number
    columnIndex?: number
  } | null>(null)
  const [selectionDrag, setSelectionDrag] = useState<{
    kind: "cell" | "row" | "column"
    anchorRowIndex?: number
    anchorColumnIndex?: number
    baseSelection: DataGridSelection
  } | null>(null)

  const activePivotTable = useMemo(
    () => pivotTables.find((table) => table.id === activeDataTab && table.open),
    [activeDataTab, pivotTables]
  )
  const isBaseDataTab = activeDataTab === "base" || !activePivotTable
  const selectionEnabled = isBaseDataTab
  const tableRows = isBaseDataTab ? rows : (activePivotTable?.rows ?? rows)
  const headers = useMemo(() => Object.keys(tableRows[0] ?? {}), [tableRows])
  const tableShapeLabel = useMemo(
    () => `${tableRows.length}X${headers.length}`,
    [headers.length, tableRows.length]
  )
  const highlightedRowsArray = useMemo(() => {
    if (!isBaseDataTab) return []
    return Array.from(highlightedRows).sort((a, b) => a - b)
  }, [highlightedRows, isBaseDataTab])
  const highlightedColumnSet = useMemo(() => {
    const set = new Set<string>()
    if (!isBaseDataTab) return set
    highlightedColumns.forEach((column) => {
      set.add(column.column)
    })
    return set
  }, [highlightedColumns, isBaseDataTab])
  const totalTableWidth = useMemo(
    () => Math.max(headers.reduce((sum, header) => sum + (colWidths[header] || 150), ROW_GUTTER_WIDTH), 100),
    [colWidths, headers]
  )

  const virtualizedData = useMemo(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
    const visibleCount = Math.ceil(tableHeight / ROW_HEIGHT) + OVERSCAN * 2
    const endIndex = Math.min(tableRows.length, startIndex + visibleCount)

    return {
      startIndex,
      endIndex,
      visibleRows: tableRows.slice(startIndex, endIndex),
      offsetTop: startIndex * ROW_HEIGHT,
    }
  }, [scrollTop, tableHeight, tableRows])

  useEffect(() => {
    if (!pivotMenuOpen) return

    const handleClickOutside = () => setPivotMenuOpen(false)
    document.addEventListener("click", handleClickOutside)
    return () => document.removeEventListener("click", handleClickOutside)
  }, [pivotMenuOpen])

  useEffect(() => {
    const updateTableHeight = () => {
      if (tableContainerRef.current) {
        setTableHeight(tableContainerRef.current.clientHeight)
      }
    }

    updateTableHeight()
    window.addEventListener("resize", updateTableHeight)
    return () => window.removeEventListener("resize", updateTableHeight)
  }, [fullscreen, tableRows.length])

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!resizingCol.current) return

      const { key, startX, startWidth } = resizingCol.current
      const newWidth = Math.max(50, startWidth + (event.clientX - startX))
      setColWidths((prev) => ({ ...prev, [key]: newWidth }))
    }

    const handleMouseUp = () => {
      if (!resizingCol.current) return
      resizingCol.current = null
      document.body.style.cursor = ""
    }

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
    }
  }, [])

  useEffect(() => {
    const tableContainer = tableContainerRef.current
    if (!tableContainer) return
    tableContainer.scrollTop = 0
  }, [activeDataTab])

  useEffect(() => {
    if (!selectionDrag) return

    const handleMouseUp = () => setSelectionDrag(null)
    window.addEventListener("mouseup", handleMouseUp)
    return () => window.removeEventListener("mouseup", handleMouseUp)
  }, [selectionDrag])

  const emptySelection = (): DataGridSelection => ({
    rowRanges: [],
    columnRanges: [],
    cellRanges: [],
  })

  const normalizeRowRange = (range: RowRangeSelection): RowRangeSelection => ({
    startRowIndex: Math.max(0, Math.min(range.startRowIndex, range.endRowIndex)),
    endRowIndex: Math.max(0, Math.max(range.startRowIndex, range.endRowIndex)),
  })

  const normalizeColumnRange = (range: ColumnRangeSelection): ColumnRangeSelection => ({
    startColumnIndex: Math.max(0, Math.min(range.startColumnIndex, range.endColumnIndex)),
    endColumnIndex: Math.min(headers.length - 1, Math.max(range.startColumnIndex, range.endColumnIndex)),
  })

  const normalizeCellRange = (range: CellRangeSelection): CellRangeSelection => ({
    startRowIndex: Math.max(0, Math.min(range.startRowIndex, range.endRowIndex)),
    endRowIndex: Math.max(0, Math.max(range.startRowIndex, range.endRowIndex)),
    startColumnIndex: Math.max(0, Math.min(range.startColumnIndex, range.endColumnIndex)),
    endColumnIndex: Math.min(headers.length - 1, Math.max(range.startColumnIndex, range.endColumnIndex)),
  })

  const applyRangeSelection = (
    baseSelection: DataGridSelection,
    kind: "cell" | "row" | "column",
    range: CellRangeSelection | RowRangeSelection | ColumnRangeSelection
  ) => {
    if (kind === "cell") {
      return {
        ...baseSelection,
        cellRanges: [...baseSelection.cellRanges, normalizeCellRange(range as CellRangeSelection)],
      }
    }

    if (kind === "row") {
      return {
        ...baseSelection,
        rowRanges: [...baseSelection.rowRanges, normalizeRowRange(range as RowRangeSelection)],
      }
    }

    return {
      ...baseSelection,
      columnRanges: [...baseSelection.columnRanges, normalizeColumnRange(range as ColumnRangeSelection)],
    }
  }

  const isAdditiveSelection = (event: ReactMouseEvent) => event.metaKey || event.ctrlKey

  const startCellSelection = (event: ReactMouseEvent, rowIndex: number, columnIndex: number) => {
    if (!selectionEnabled) return
    event.preventDefault()

    const existingAnchor = selectionAnchorRef.current
    const useExistingAnchor = event.shiftKey && existingAnchor?.kind === "cell"
    const anchorRowIndex = useExistingAnchor ? existingAnchor.rowIndex ?? rowIndex : rowIndex
    const anchorColumnIndex = useExistingAnchor ? existingAnchor.columnIndex ?? columnIndex : columnIndex
    const baseSelection = isAdditiveSelection(event) ? selection : emptySelection()

    if (!useExistingAnchor) {
      selectionAnchorRef.current = { kind: "cell", rowIndex, columnIndex }
    }

    setSelectionDrag({ kind: "cell", anchorRowIndex, anchorColumnIndex, baseSelection })
    onSelectionChange(
      applyRangeSelection(baseSelection, "cell", {
        startRowIndex: anchorRowIndex,
        endRowIndex: rowIndex,
        startColumnIndex: anchorColumnIndex,
        endColumnIndex: columnIndex,
      })
    )
  }

  const startRowSelection = (event: ReactMouseEvent, rowIndex: number) => {
    if (!selectionEnabled) return
    event.preventDefault()

    const existingAnchor = selectionAnchorRef.current
    const useExistingAnchor = event.shiftKey && existingAnchor?.kind === "row"
    const anchorRowIndex = useExistingAnchor ? existingAnchor.rowIndex ?? rowIndex : rowIndex
    const baseSelection = isAdditiveSelection(event) ? selection : emptySelection()

    if (!useExistingAnchor) {
      selectionAnchorRef.current = { kind: "row", rowIndex }
    }

    setSelectionDrag({ kind: "row", anchorRowIndex, baseSelection })
    onSelectionChange(
      applyRangeSelection(baseSelection, "row", {
        startRowIndex: anchorRowIndex,
        endRowIndex: rowIndex,
      })
    )
  }

  const startColumnSelection = (event: ReactMouseEvent, columnIndex: number) => {
    if (!selectionEnabled) return
    if ((event.target as HTMLElement).closest("[data-resize-handle]")) return
    event.preventDefault()

    const existingAnchor = selectionAnchorRef.current
    const useExistingAnchor = event.shiftKey && existingAnchor?.kind === "column"
    const anchorColumnIndex = useExistingAnchor ? existingAnchor.columnIndex ?? columnIndex : columnIndex
    const baseSelection = isAdditiveSelection(event) ? selection : emptySelection()

    if (!useExistingAnchor) {
      selectionAnchorRef.current = { kind: "column", columnIndex }
    }

    setSelectionDrag({ kind: "column", anchorColumnIndex, baseSelection })
    onSelectionChange(
      applyRangeSelection(baseSelection, "column", {
        startColumnIndex: anchorColumnIndex,
        endColumnIndex: columnIndex,
      })
    )
  }

  const extendCellSelection = (rowIndex: number, columnIndex: number) => {
    if (selectionDrag?.kind !== "cell") return
    onSelectionChange(
      applyRangeSelection(selectionDrag.baseSelection, "cell", {
        startRowIndex: selectionDrag.anchorRowIndex ?? rowIndex,
        endRowIndex: rowIndex,
        startColumnIndex: selectionDrag.anchorColumnIndex ?? columnIndex,
        endColumnIndex: columnIndex,
      })
    )
  }

  const extendRowSelection = (rowIndex: number) => {
    if (selectionDrag?.kind !== "row") return
    onSelectionChange(
      applyRangeSelection(selectionDrag.baseSelection, "row", {
        startRowIndex: selectionDrag.anchorRowIndex ?? rowIndex,
        endRowIndex: rowIndex,
      })
    )
  }

  const extendColumnSelection = (columnIndex: number) => {
    if (selectionDrag?.kind !== "column") return
    onSelectionChange(
      applyRangeSelection(selectionDrag.baseSelection, "column", {
        startColumnIndex: selectionDrag.anchorColumnIndex ?? columnIndex,
        endColumnIndex: columnIndex,
      })
    )
  }

  const isRowSelected = (rowIndex: number) =>
    selection.rowRanges.some((range) => {
      const normalized = normalizeRowRange(range)
      return rowIndex >= normalized.startRowIndex && rowIndex <= normalized.endRowIndex
    })

  const isColumnSelected = (columnIndex: number) =>
    selection.columnRanges.some((range) => {
      const normalized = normalizeColumnRange(range)
      return columnIndex >= normalized.startColumnIndex && columnIndex <= normalized.endColumnIndex
    })

  const isCellRangeSelected = (rowIndex: number, columnIndex: number) =>
    selection.cellRanges.some((range) => {
      const normalized = normalizeCellRange(range)
      return (
        rowIndex >= normalized.startRowIndex &&
        rowIndex <= normalized.endRowIndex &&
        columnIndex >= normalized.startColumnIndex &&
        columnIndex <= normalized.endColumnIndex
      )
    })

  const isCellSelected = (rowIndex: number, columnIndex: number) =>
    selectionEnabled && (isRowSelected(rowIndex) || isColumnSelected(columnIndex) || isCellRangeSelected(rowIndex, columnIndex))

  const activeHighlightIndex =
    highlightedRowsArray.length === 0
      ? 0
      : Math.min(currentHighlightIndex, highlightedRowsArray.length - 1)

  const navigateToHighlightedRow = (direction: "prev" | "next") => {
    if (highlightedRowsArray.length === 0) return

    const newIndex =
      direction === "prev"
        ? activeHighlightIndex > 0
          ? activeHighlightIndex - 1
          : highlightedRowsArray.length - 1
        : activeHighlightIndex < highlightedRowsArray.length - 1
          ? activeHighlightIndex + 1
          : 0

    setCurrentHighlightIndex(newIndex)

    const rowIndex = highlightedRowsArray[newIndex]
    const tableContainer = tableContainerRef.current
    if (!tableContainer) return

    const targetScrollTop = rowIndex * ROW_HEIGHT - tableHeight / 2 + ROW_HEIGHT / 2
    tableContainer.scrollTop = Math.max(0, targetScrollTop)
  }

  return (
    <div
      className={`flex flex-col min-h-[120px] shrink-0 overflow-hidden ${
        fullscreen ? "fullscreen-overlay" : ""
      }`}
      style={fullscreen ? {} : { height: "100%" }}
    >
      <div className="flex items-center justify-between h-10 px-4 border-b border-border bg-card/30 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 group relative overflow-hidden px-1.5 py-0.5 rounded-sm">
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
                onClick={() => onClosePivotTable(activePivotTable.id)}
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
                onClick={(event) => {
                  event.stopPropagation()
                  setPivotMenuOpen((prev) => !prev)
                }}
                className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                <span>{pivotTables.length} pivot{pivotTables.length !== 1 ? "s" : ""}</span>
                <ChevronDown className={`size-3 transition-transform ${pivotMenuOpen ? "rotate-180" : ""}`} />
              </button>

              <div
                data-state={pivotMenuOpen ? "open" : "closed"}
                className="menu-surface menu-popover absolute top-full left-0 mt-1 min-w-[220px] rounded-lg py-1 z-50"
                onClick={(event) => event.stopPropagation()}
              >
                  <button
                    type="button"
                    onClick={() => {
                      onActivateBaseData()
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
                          onOpenPivotTable(pivot.id)
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
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
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
                {activeHighlightIndex + 1} / {highlightedRowsArray.length}
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

          <div className="mr-1 flex items-center gap-0.5">
            <button
              type="button"
              onClick={onRevert}
              disabled={!canRevert}
              className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-35"
              title="Revert data change"
              aria-label="Revert data change"
            >
              <Undo2 className="size-3.5" />
            </button>
            <button
              type="button"
              onClick={onRestore}
              disabled={!canRestore}
              className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-35"
              title="Restore data change"
              aria-label="Restore data change"
            >
              <Redo2 className="size-3.5" />
            </button>
          </div>

          <span className="text-[10px] text-muted-foreground">{tableShapeLabel}</span>
          <button
            type="button"
            onClick={onToggleFullscreen}
            className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          >
            {fullscreen ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <div
          ref={tableContainerRef}
          className="flex-1 min-h-0 overflow-auto scrollbar-thin"
          onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        >
          <Table className="table-fixed" style={{ width: `max(100%, ${totalTableWidth}px)` }}>
            <TableHeader>
              <TableRow className="border-b-border hover:bg-transparent">
                <TableHead
                  style={{ width: ROW_GUTTER_WIDTH }}
                  className="sticky left-0 top-0 z-20 h-8 bg-card/95 px-2 text-center text-[10px] font-semibold text-muted-foreground backdrop-blur"
                >
                  <button
                    type="button"
                    onClick={() => onSelectionChange(emptySelection())}
                    disabled={!selectionEnabled}
                    className="h-5 w-5 rounded text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-35"
                    title="Clear selection"
                    aria-label="Clear selection"
                  >
                    #
                  </button>
                </TableHead>

                {headers.map((header, columnIndex) => (
                  <TableHead
                    key={header}
                    style={{ width: colWidths[header] || 150 }}
                    onMouseDown={(event) => startColumnSelection(event, columnIndex)}
                    onMouseEnter={() => extendColumnSelection(columnIndex)}
                    className={`relative sticky top-0 z-10 h-8 cursor-default bg-card/95 backdrop-blur px-3 text-[11px] font-semibold text-muted-foreground truncate group ${
                      isColumnSelected(columnIndex) ? "data-grid-selected-header" : ""
                    }`}
                  >
                    <div className={highlightedColumnSet.has(header) ? "column-highlight" : ""}>
                      <span className="relative z-10 block truncate">{header}</span>
                    </div>
                    <div
                      data-resize-handle="true"
                      className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize opacity-0 group-hover:opacity-100 hover:opacity-100 hover:bg-primary/50 transition-opacity"
                      onMouseDown={(event) => {
                        event.stopPropagation()
                        event.preventDefault()
                        resizingCol.current = {
                          key: header,
                          startX: event.clientX,
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
                <TableRow aria-hidden="true" className="hover:bg-transparent">
                  <TableCell colSpan={Math.max(headers.length + 1, 1)} className="p-0" style={{ height: virtualizedData.offsetTop }} />
                </TableRow>
              )}

              {virtualizedData.visibleRows.map((row, localIndex) => {
                const rowIndex = virtualizedData.startIndex + localIndex
                const isHighlighted = isBaseDataTab && highlightedRows.has(rowIndex)
                const isCurrentHighlight =
                  highlightedRowsArray.length > 0 && highlightedRowsArray[activeHighlightIndex] === rowIndex

                return (
                  <TableRow
                    key={rowIndex}
                    style={{ height: ROW_HEIGHT }}
                    className={`border-b-border/50 ${
                      isHighlighted
                        ? isCurrentHighlight
                          ? "bg-primary/25 border-l-[3px] border-l-primary"
                          : "bg-primary/15 border-l-[3px] border-l-primary"
                        : "hover:bg-accent/30"
                    }`}
                  >
                    <TableCell
                      style={{ width: ROW_GUTTER_WIDTH }}
                      onMouseDown={(event) => startRowSelection(event, rowIndex)}
                      onMouseEnter={() => extendRowSelection(rowIndex)}
                      className={`sticky left-0 z-[5] h-7 cursor-default border-r border-border/60 bg-card/95 px-2 py-1.5 text-center text-[10px] font-medium text-muted-foreground backdrop-blur ${
                        isRowSelected(rowIndex) ? "data-grid-selected-row-gutter" : ""
                      }`}
                    >
                      {rowIndex + 1}
                    </TableCell>

                    {headers.map((header, columnIndex) => (
                      <TableCell
                        key={`${rowIndex}-${header}`}
                        data-selected={isCellSelected(rowIndex, columnIndex) ? "true" : undefined}
                        onMouseDown={(event) => startCellSelection(event, rowIndex, columnIndex)}
                        onMouseEnter={() => extendCellSelection(rowIndex, columnIndex)}
                        className={`px-3 py-1.5 text-xs truncate ${
                          isHighlighted ? "text-foreground font-medium" : "text-foreground/80"
                        } ${isCellSelected(rowIndex, columnIndex) ? "data-grid-selected-cell" : ""}`}
                      >
                        {String(row[header] ?? "")}
                      </TableCell>
                    ))}
                  </TableRow>
                )
              })}

              {virtualizedData.endIndex < tableRows.length && (
                <TableRow aria-hidden="true" className="hover:bg-transparent">
                  <TableCell
                    colSpan={Math.max(headers.length + 1, 1)}
                    className="p-0"
                    style={{ height: (tableRows.length - virtualizedData.endIndex) * ROW_HEIGHT }}
                  />
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}
