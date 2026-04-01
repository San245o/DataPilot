"use client"

import { Suspense, useMemo, useSyncExternalStore } from "react"
import { useSearchParams } from "next/navigation"
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
    <div className={payload.isDark ? "dark" : ""}>
      <main className="flex h-dvh w-full flex-col bg-background text-foreground">
        <header className="flex h-12 shrink-0 items-center border-b border-border px-4">
          <h1 className="truncate text-sm font-semibold">{payload.title}</h1>
        </header>
        <div className="min-h-0 flex-1 p-2">
          <div className="h-full w-full rounded-lg border border-border bg-card p-1">
            <PlotlyBoard data={payload.data} layout={payload.layout} isDark={Boolean(payload.isDark)} />
          </div>
        </div>
      </main>
    </div>
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
