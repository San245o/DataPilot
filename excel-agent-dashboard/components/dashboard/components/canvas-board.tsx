"use client"

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type MouseEvent as ReactMouseEvent,
  type SetStateAction,
} from "react"
import { ChevronDown, ExternalLink, Maximize2, Minimize2, X } from "lucide-react"

import { getNextWidgetZIndex, type VizWidget } from "@/components/dashboard/dashboard-shared"
import { PlotlyBoard } from "@/components/charts/plotly-board"

type CanvasBoardProps = {
  widgets: VizWidget[]
  setWidgets: Dispatch<SetStateAction<VizWidget[]>>
  isDark: boolean
  fullscreen: boolean
  onToggleFullscreen: () => void
  onError: (message: string | null) => void
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
  startWidgetX: number
  startWidgetY: number
  corner: "se" | "sw" | "ne" | "nw"
} | null

export function CanvasBoard({
  widgets,
  setWidgets,
  isDark,
  fullscreen,
  onToggleFullscreen,
  onError,
}: CanvasBoardProps) {
  const [chartMenuOpen, setChartMenuOpen] = useState(false)
  const [dragState, setDragState] = useState<DragState>(null)
  const [resizeState, setResizeState] = useState<ResizeState>(null)
  const boardRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!chartMenuOpen) return

    const handleClickOutside = () => setChartMenuOpen(false)
    document.addEventListener("click", handleClickOutside)
    return () => document.removeEventListener("click", handleClickOutside)
  }, [chartMenuOpen])

  const bringToFront = useCallback((widgetId: string) => {
    setWidgets((prev) => {
      const nextZIndex = getNextWidgetZIndex(prev)
      return prev.map((widget) => (widget.id === widgetId ? { ...widget, zIndex: nextZIndex } : widget))
    })
  }, [setWidgets])

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
          frames: widget.frames,
          isDark,
        })
      )

      if (popup) {
        popup.location.href = targetUrl
      } else {
        window.location.href = targetUrl
      }
    } catch {
      onError("Unable to open chart in a new tab.")
    }
  }, [isDark, onError])

  useEffect(() => {
    if (!dragState) return

    const handleMouseMove = (event: MouseEvent) => {
      const board = boardRef.current
      if (!board) return

      const rect = board.getBoundingClientRect()
      setWidgets((prev) =>
        prev.map((widget) => {
          if (widget.id !== dragState.id) return widget

          return {
            ...widget,
            x: Math.max(0, event.clientX - rect.left + board.scrollLeft - dragState.offsetX),
            y: Math.max(0, event.clientY - rect.top + board.scrollTop - dragState.offsetY),
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
  }, [dragState, setWidgets])

  const startWidgetDrag = (event: ReactMouseEvent<HTMLDivElement>, widget: VizWidget) => {
    const board = boardRef.current
    if (!board) return

    const rect = board.getBoundingClientRect()
    bringToFront(widget.id)
    setDragState({
      id: widget.id,
      offsetX: event.clientX - rect.left + board.scrollLeft - widget.x,
      offsetY: event.clientY - rect.top + board.scrollTop - widget.y,
    })
  }

  const startWidgetResize = (
    event: ReactMouseEvent,
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

          if (resizeState.corner === "se" || resizeState.corner === "ne") {
            newWidth = Math.max(MIN_WIDTH, resizeState.startWidth + deltaX)
          } else {
            newWidth = Math.max(MIN_WIDTH, resizeState.startWidth - deltaX)
            newX = Math.max(0, resizeState.startWidgetX + (resizeState.startWidth - newWidth))
          }

          if (resizeState.corner === "se" || resizeState.corner === "sw") {
            newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight + deltaY)
          } else {
            newHeight = Math.max(MIN_HEIGHT, resizeState.startHeight - deltaY)
            newY = Math.max(0, resizeState.startWidgetY + (resizeState.startHeight - newHeight))
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
  }, [resizeState, setWidgets])

  return (
    <div
      className={`flex flex-col flex-1 min-h-0 overflow-hidden ${
        fullscreen ? "fullscreen-overlay" : ""
      }`}
    >
      <div className="flex items-center justify-between h-10 px-4 border-b border-border bg-card/30">
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-foreground">Canvas</span>
          {widgets.length > 0 && (
            <div className="relative">
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  setChartMenuOpen((prev) => !prev)
                }}
                className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                <span>{widgets.length} chart{widgets.length !== 1 ? "s" : ""}</span>
                <ChevronDown className={`size-3 transition-transform ${chartMenuOpen ? "rotate-180" : ""}`} />
              </button>

              <div
                data-state={chartMenuOpen ? "open" : "closed"}
                className="menu-surface menu-popover absolute top-full left-0 mt-1 min-w-[160px] rounded-lg py-1 z-50"
                onClick={(event) => event.stopPropagation()}
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
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={onToggleFullscreen}
          className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
        >
          {fullscreen ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
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
                        onMouseDown={(event) => {
                          event.preventDefault()
                          event.stopPropagation()
                        }}
                        onClick={(event) => {
                          event.stopPropagation()
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
                      onMouseDown={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                      }}
                      onClick={(event) => {
                        event.stopPropagation()
                        setWidgets((prev) => prev.filter((item) => item.id !== widget.id))
                      }}
                      className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground hover:bg-accent"
                    >
                      <X className="size-3" />
                    </button>
                  </div>
                </div>

                <div className="min-h-0 flex-1 p-1">
                  <PlotlyBoard data={widget.data} layout={widget.layout} frames={widget.frames} isDark={isDark} />
                </div>

                <div
                  className="absolute top-0 left-0 w-4 h-4 cursor-nw-resize group/resize"
                  onMouseDown={(event) => startWidgetResize(event, widget, "nw")}
                >
                  <div className="absolute top-0.5 left-0.5 w-2.5 h-2.5 border-t-2 border-l-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-tl" />
                </div>
                <div
                  className="absolute top-0 right-0 w-4 h-4 cursor-ne-resize group/resize"
                  onMouseDown={(event) => startWidgetResize(event, widget, "ne")}
                >
                  <div className="absolute top-0.5 right-0.5 w-2.5 h-2.5 border-t-2 border-r-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-tr" />
                </div>
                <div
                  className="absolute bottom-0 left-0 w-4 h-4 cursor-sw-resize group/resize"
                  onMouseDown={(event) => startWidgetResize(event, widget, "sw")}
                >
                  <div className="absolute bottom-0.5 left-0.5 w-2.5 h-2.5 border-b-2 border-l-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-bl" />
                </div>
                <div
                  className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize group/resize"
                  onMouseDown={(event) => startWidgetResize(event, widget, "se")}
                >
                  <div className="absolute bottom-0.5 right-0.5 w-2.5 h-2.5 border-b-2 border-r-2 border-muted-foreground/30 group-hover/resize:border-primary transition-colors rounded-br" />
                </div>
              </div>
            ))
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-xs text-muted-foreground">Charts will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
