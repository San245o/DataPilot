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

type ChartViewerStateProps = {
  chartKey: string | null
  isHydrated: boolean
}

function ChartViewerState({ chartKey, isHydrated }: ChartViewerStateProps) {
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

  const [themeOverride, setThemeOverride] = useState<{ chartKey: string | null; value: boolean } | null>(null)
  const isDark =
    themeOverride?.chartKey === chartKey
      ? themeOverride.value
      : Boolean(payload?.isDark)

  useEffect(() => {
    if (!isHydrated) return
    document.documentElement.classList.toggle("dark", isDark)
  }, [isDark, isHydrated])

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
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: isDark
            ? "radial-gradient(circle at top, rgba(17,197,165,0.14), transparent 28%), radial-gradient(circle at bottom right, rgba(95,131,255,0.12), transparent 24%)"
            : "radial-gradient(circle at top, rgba(15,159,134,0.12), transparent 28%), radial-gradient(circle at bottom right, rgba(71,99,255,0.10), transparent 24%)",
        }}
      />
      <header className="relative z-10 flex h-14 shrink-0 items-center justify-between border-b border-border/70 bg-card/55 px-5 backdrop-blur-xl">
        <div className="flex min-w-0 items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-emerald-500" />
          <h1 className="truncate text-sm font-semibold tracking-[-0.02em]">{payload.title}</h1>
        </div>
        <button
          type="button"
          onClick={() => setThemeOverride({ chartKey, value: !isDark })}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border/80 bg-secondary/80 px-2.5 py-1.5 text-[11px] font-medium text-foreground transition-colors hover:bg-accent"
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          <span>{isDark ? "Light mode" : "Dark mode"}</span>
        </button>
      </header>
      <div className="relative z-10 min-h-0 flex-1 p-4">
        <div className="h-full w-full overflow-hidden rounded-[24px] border border-border/60 bg-card/58 p-4 shadow-[0_30px_80px_-44px_rgba(0,0,0,0.62)] backdrop-blur-2xl">
          <PlotlyBoard data={payload.data} layout={chartLayout ?? payload.layout} isDark={isDark} />
        </div>
      </div>
    </main>
  )
}

function ChartViewerContent() {
  const searchParams = useSearchParams()
  const isHydrated = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  )
  const chartKey = searchParams.get("chartKey")

  return <ChartViewerState key={chartKey ?? "missing"} chartKey={chartKey} isHydrated={isHydrated} />
}

function ChartViewerFallback() {
  return (
    <div className="flex h-dvh w-full items-center justify-center bg-background px-4 text-foreground">
      <div className="text-sm text-muted-foreground">Opening chart...</div>
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
