"use client"

import { Suspense, useEffect, useMemo, useState, useSyncExternalStore } from "react"
import { useSearchParams } from "next/navigation"
import { Moon, Sun } from "lucide-react"
import type { Data, Layout } from "plotly.js"

import { PlotlyBoard } from "@/components/charts/plotly-board"

type StoredChartPayload = {
  title: string
  data: Data[]
  layout?: Partial<Layout>
  isDark?: boolean
}

function ChartViewerContent() {
  const searchParams = useSearchParams()
  const chartKey = searchParams.get("chartKey")
  const [isDark, setIsDark] = useState(true)
  const isHydrated = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  )

  const { payload, error } = useMemo(() => {
    if (!isHydrated) {
      return { payload: null as StoredChartPayload | null, error: null as string | null }
    }

    if (!chartKey) {
      return { payload: null as StoredChartPayload | null, error: "Missing chart key." }
    }

    try {
      const raw = localStorage.getItem(chartKey)
      if (!raw) {
        return {
          payload: null as StoredChartPayload | null,
          error: "Chart data not found. Re-open this chart from the dashboard.",
        }
      }

      const parsed = JSON.parse(raw) as StoredChartPayload
      if (!parsed || !Array.isArray(parsed.data) || parsed.data.length === 0) {
        return { payload: null as StoredChartPayload | null, error: "Chart payload is invalid." }
      }

      return { payload: parsed, error: null as string | null }
    } catch {
      return { payload: null as StoredChartPayload | null, error: "Unable to load chart data." }
    }
  }, [chartKey, isHydrated])

  useEffect(() => {
    if (!isHydrated || !payload) return
    setIsDark(Boolean(payload.isDark))
  }, [isHydrated, payload])

  useEffect(() => {
    if (!isHydrated) return
    document.documentElement.classList.toggle("dark", isDark)
  }, [isHydrated, isDark])

  const chartLayout = useMemo(() => {
    if (!payload?.layout) return undefined
    return {
      ...payload.layout,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    } as Partial<Layout>
  }, [payload?.layout])

  if (error) {
    return (
      <div className="flex h-dvh w-full items-center justify-center bg-background px-4 text-foreground">
        <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm">{error}</div>
      </div>
    )
  }

  if (!payload) {
    return (
      <div className="flex h-dvh w-full items-center justify-center bg-background px-4 text-foreground">
        <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm">Loading chart...</div>
      </div>
    )
  }

  return (
    <main className="flex h-dvh w-full flex-col overflow-hidden bg-background text-foreground">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-card/80 px-4 backdrop-blur-sm">
        <div className="flex min-w-0 items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-emerald-500" />
          <h1 className="truncate text-sm font-semibold">{payload.title}</h1>
        </div>
        <button
          type="button"
          onClick={() => setIsDark((prev) => !prev)}
          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-secondary px-2 py-1 text-[11px] font-medium text-foreground transition-colors hover:bg-accent"
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          <span>{isDark ? "Light mode" : "Dark mode"}</span>
        </button>
      </header>
      <div className="min-h-0 flex-1 p-3">
        <div className="h-full w-full rounded-xl border border-border/80 bg-card/80 p-2 shadow-sm">
          <PlotlyBoard data={payload.data} layout={chartLayout ?? payload.layout} isDark={isDark} />
        </div>
      </div>
    </main>
  )
}

function ChartViewerFallback() {
  return (
    <div className="flex h-dvh w-full items-center justify-center bg-background px-4 text-foreground">
      <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm">Loading chart...</div>
    </div>
  )
}

export default function ChartViewerPage() {
  return (
    <Suspense fallback={<ChartViewerFallback />}>
      <ChartViewerContent />
    </Suspense>
  )
}
