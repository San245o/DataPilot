"use client"

import Link from "next/link"
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Clock,
  Columns3,
  Compass,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileSpreadsheet,
  FileText,
  Hash,
  Info,
  Layers,
  Lightbulb,
  LineChart,
  Loader2,
  RotateCcw,
  Sparkles,
  Table2,
  TrendingUp,
} from "lucide-react"
import { PlotlyBoard } from "@/components/charts/plotly-board"
import { API_BASE_URL } from "@/components/dashboard/hooks/use-agent-runner"
import type { ThinkingTraceEntry } from "@/components/dashboard/dashboard-shared"
import type { AutoReport, ReportTarget } from "@/lib/report"
import { formatReportValue } from "@/lib/report-format"

type StreamEvent =
  | { type: "trace"; entry: ThinkingTraceEntry }
  | { type: "final"; payload: AutoReport }
  | { type: "error"; error: string }

export default function ReportPage() {
  const [target, setTarget] = useState<ReportTarget | null>(null)
  const [report, setReport] = useState<AutoReport | null>(null)
  const [trace, setTrace] = useState<ThinkingTraceEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [preparingPdf, setPreparingPdf] = useState(false)
  const [pdfError, setPdfError] = useState<string | null>(null)
  const [traceExpanded, setTraceExpanded] = useState(true)

  const abortRef = useRef<AbortController | null>(null)
  const startedRef = useRef(false)
  const tracePanelRef = useRef<HTMLDivElement | null>(null)
  const traceAtBottomRef = useRef(true)

  const generate = useCallback(async (reportTarget: ReportTarget) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    let stalled = false
    let stallTimer: ReturnType<typeof setTimeout> | undefined

    const resetStallTimer = () => {
      if (stallTimer) clearTimeout(stallTimer)
      stallTimer = setTimeout(() => {
        stalled = true
        controller.abort()
      }, 75_000)
    }

    traceAtBottomRef.current = true
    setRunning(true)
    setError(null)
    setTrace([])
    setReport(null)
    setTraceExpanded(true)
    resetStallTimer()

    try {
      const response = await fetch(`${API_BASE_URL}/agent/report/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ dataset_id: reportTarget.datasetId, model: reportTarget.model }),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || `Report generation failed (${response.status}).`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("The report stream was unavailable.")

      const decoder = new TextDecoder()
      let buffer = ""
      let receivedFinal = false

      const consume = (line: string) => {
        if (!line.trim()) return
        const event = JSON.parse(line) as StreamEvent
        resetStallTimer()
        if (event.type === "trace") {
          setTrace((current) => [...current, event.entry])
        }
        if (event.type === "final") {
          receivedFinal = true
          setReport(event.payload)
          setTraceExpanded(false)
        }
        if (event.type === "error") {
          throw new Error(event.error)
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""
        lines.forEach(consume)
      }

      if (buffer.trim()) consume(buffer)
      if (!receivedFinal) {
        throw new Error("Report generation ended before a final report was received. Please try again.")
      }
    } catch (reason) {
      setTrace((current) =>
        current.at(-1)?.status === "error"
          ? current
          : [
              ...current,
              {
                kind: "observation",
                content: "Report generation stopped before the final report was available.",
                status: "error",
                timestamp: new Date().toISOString(),
              },
            ]
      )
      if (stalled) {
        setError("Report generation took longer than expected. Please try again.")
      } else if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "Auto Report generation failed.")
      }
    } finally {
      if (stallTimer) clearTimeout(stallTimer)
      if (abortRef.current === controller) abortRef.current = null
      setRunning(false)
    }
  }, [])

  useEffect(() => {
    const panel = tracePanelRef.current
    if (!panel || !traceAtBottomRef.current) return
    const frame = requestAnimationFrame(() =>
      panel.scrollTo({ top: panel.scrollHeight, behavior: "smooth" })
    )
    return () => cancelAnimationFrame(frame)
  }, [trace])

  const downloadPdf = useCallback(async () => {
    if (!report || !target || preparingPdf) return
    setPreparingPdf(true)
    setPdfError(null)
    try {
      const { downloadReportPdf } = await import("@/lib/report-pdf")
      await downloadReportPdf(report, target)
    } catch (reason) {
      setPdfError(reason instanceof Error ? reason.message : "PDF generation failed.")
    } finally {
      setPreparingPdf(false)
    }
  }, [preparingPdf, report, target])

  useEffect(() => {
    const raw = sessionStorage.getItem("report_target")
    if (!raw) return
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      const saved = JSON.parse(raw) as ReportTarget
      if (!saved.datasetId) return
      timer = setTimeout(() => {
        setTarget(saved)
        if (!startedRef.current) {
          startedRef.current = true
          void generate(saved)
        }
      }, 0)
    } catch {
      timer = setTimeout(
        () => setError("The report target is invalid. Return to the dashboard and try again."),
        0
      )
    }
    return () => {
      if (timer) clearTimeout(timer)
      abortRef.current?.abort()
    }
  }, [generate])

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-100 antialiased selection:bg-emerald-500/20 selection:text-emerald-300">
      {/* Top Navigation Bar - Matches Dashboard Header Language */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
          {/* Left: Brand, Return Button & Breadcrumbs */}
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/dashboard"
              className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-2.5 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-slate-700 hover:bg-slate-800 hover:text-white"
              title="Return to Dashboard"
            >
              <ArrowLeft className="size-3.5" />
              <span className="hidden sm:inline">Dashboard</span>
            </Link>

            <span className="text-slate-700">/</span>

            <div className="flex min-w-0 items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-md bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20">
                <FileText className="size-3.5" />
              </span>
              <div className="min-w-0">
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
                  Auto Report
                </span>
                {target && (
                  <span className="ml-2 hidden text-xs font-medium text-slate-400 md:inline">
                    · {target.displayName}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-2.5">
            {target && (
              <>
                <button
                  type="button"
                  disabled={running || preparingPdf}
                  onClick={() => void generate(target)}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-slate-700 hover:bg-slate-800 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <RotateCcw className={`size-3.5 ${running ? "animate-spin text-emerald-400" : ""}`} />
                  <span className="hidden sm:inline">Regenerate</span>
                </button>

                {report && (
                  <button
                    type="button"
                    disabled={running || preparingPdf}
                    onClick={() => void downloadPdf()}
                    className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 shadow-sm transition-colors hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {preparingPdf ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Download className="size-3.5" />
                    )}
                    <span>{preparingPdf ? "Preparing..." : "Export PDF"}</span>
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Grid: Responsive 2-column layout */}
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 lg:grid lg:grid-cols-[290px_1fr] lg:gap-8">
        {/* Left Sticky Sidebar */}
        <aside className="mb-6 space-y-4 lg:mb-0">
          {/* Dataset Info Card */}
          <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800/60">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                <Database className="size-3.5 text-emerald-400" />
                <span>Dataset Overview</span>
              </div>
              <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400 ring-1 ring-emerald-500/20">
                <span className="size-1.5 rounded-full bg-emerald-400" />
                Active
              </span>
            </div>

            <div className="mt-3 space-y-2">
              <div>
                <p className="text-[11px] font-medium text-slate-500">File Name</p>
                <p className="truncate text-xs font-medium text-slate-200">
                  {target?.fileName || "gapminder-lite.xlsx"}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
                <div className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-2">
                  <p className="text-[10px] uppercase tracking-wider text-slate-500">Rows</p>
                  <p className="mt-0.5 font-mono text-sm font-semibold text-slate-200">
                    {target?.rowCount.toLocaleString() ?? "—"}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-2">
                  <p className="text-[10px] uppercase tracking-wider text-slate-500">Columns</p>
                  <p className="mt-0.5 font-mono text-sm font-semibold text-slate-200">
                    {target?.columnCount.toLocaleString() ?? "—"}
                  </p>
                </div>
              </div>

              {target?.sheetName && (
                <div className="pt-1 text-[11px] text-slate-400 flex items-center justify-between">
                  <span className="text-slate-500">Sheet:</span>
                  <span className="font-medium text-slate-300">{target.sheetName}</span>
                </div>
              )}
            </div>
          </div>

          {/* Quick Jump Links (When report is ready) */}
          {report && (
            <nav className="hidden lg:block rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                Report Structure
              </p>
              <ul className="space-y-1 text-xs font-medium text-slate-400">
                <li>
                  <a
                    href="#summary"
                    className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-slate-100"
                  >
                    <Compass className="size-3 text-emerald-400" />
                    <span>Executive Summary</span>
                  </a>
                </li>
                <li>
                  <a
                    href="#metrics"
                    className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-slate-100"
                  >
                    <Activity className="size-3 text-sky-400" />
                    <span>Key Metrics</span>
                  </a>
                </li>
                {report.charts.length > 0 && (
                  <li>
                    <a
                      href="#visualizations"
                      className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-slate-100"
                    >
                      <BarChart3 className="size-3 text-amber-400" />
                      <span>Visualizations ({report.charts.length})</span>
                    </a>
                  </li>
                )}
                {report.sections.length > 0 && (
                  <li>
                    <a
                      href="#sections"
                      className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-slate-100"
                    >
                      <Layers className="size-3 text-indigo-400" />
                      <span>Analytical Findings</span>
                    </a>
                  </li>
                )}
                {report.tables.length > 0 && (
                  <li>
                    <a
                      href="#tables"
                      className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-slate-100"
                    >
                      <Table2 className="size-3 text-teal-400" />
                      <span>Data Tables</span>
                    </a>
                  </li>
                )}
                <li>
                  <a
                    href="#conclusion"
                    className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-slate-100"
                  >
                    <CheckCircle2 className="size-3 text-emerald-400" />
                    <span>Conclusion & Next Steps</span>
                  </a>
                </li>
              </ul>
            </nav>
          )}

          {/* Thinking Mode Trace Card */}
          {(running || trace.length > 0) && (
            <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
              <button
                type="button"
                onClick={() => setTraceExpanded((prev) => !prev)}
                className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400"
              >
                <div className="flex items-center gap-2">
                  <Sparkles className="size-3.5 text-emerald-400" />
                  <span>Thinking Trace</span>
                  {running && <Loader2 className="size-3 animate-spin text-emerald-400" />}
                </div>
                {traceExpanded ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
              </button>

              {traceExpanded && (
                <div
                  ref={tracePanelRef}
                  onScroll={(event) => {
                    const panel = event.currentTarget
                    traceAtBottomRef.current =
                      panel.scrollHeight - panel.scrollTop - panel.clientHeight < 48
                  }}
                  className="mt-3 max-h-[50vh] space-y-2 overflow-y-auto pr-1 scrollbar-thin"
                >
                  {trace.map((entry, index) => {
                    const isError = entry.status === "error"
                    return (
                      <div
                        key={entry.sequence ?? index}
                        className={`rounded-lg border p-2.5 text-[11px] transition-all ${
                          isError
                            ? "border-red-900/50 bg-red-950/20 text-red-300"
                            : "border-slate-800/70 bg-slate-950/60 text-slate-300"
                        }`}
                      >
                        <div className="mb-1 flex items-center justify-between gap-1">
                          <span
                            className={`rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${
                              entry.kind === "action"
                                ? "bg-sky-500/10 text-sky-400"
                                : entry.kind === "thought"
                                ? "bg-indigo-500/10 text-indigo-400"
                                : "bg-emerald-500/10 text-emerald-400"
                            }`}
                          >
                            {entry.tool_name || entry.kind}
                          </span>
                          {entry.timestamp && (
                            <time className="text-[9px] text-slate-600">
                              {new Date(entry.timestamp).toLocaleTimeString([], {
                                hour: "2-digit",
                                minute: "2-digit",
                                second: "2-digit",
                              })}
                            </time>
                          )}
                        </div>
                        <p className="leading-relaxed text-slate-300">{entry.content}</p>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </aside>

        {/* Right: Main Report Display */}
        <main className="min-w-0">
          {!target && (
            <EmptyState
              icon={<Database className="size-8 text-slate-600" />}
              title="No active dataset selected"
              description="Please upload or choose a dataset from the dashboard before generating an automated analytical report."
            />
          )}

          {running && !report && (
            <div className="flex min-h-[460px] flex-col items-center justify-center rounded-2xl border border-slate-800/80 bg-slate-900/20 p-8 text-center">
              <div className="relative mb-5 flex size-14 items-center justify-center rounded-2xl bg-emerald-500/10 ring-1 ring-emerald-500/20">
                <Loader2 className="size-7 animate-spin text-emerald-400" />
              </div>
              <h2 className="text-base font-semibold text-slate-200">
                Generating Comprehensive Report
              </h2>
              <p className="mt-2 max-w-md text-xs leading-relaxed text-slate-400">
                Analyzing dataset patterns, generating distinct multi-perspective visualizations,
                profiling column health, and synthesizing verified analytical conclusions.
              </p>
            </div>
          )}

          {error && (
            <div className="mb-6 rounded-xl border border-red-900/60 bg-red-950/20 p-5 text-red-200">
              <div className="flex items-start gap-3">
                <AlertTriangle className="size-5 shrink-0 text-red-400" />
                <div>
                  <h3 className="text-sm font-semibold text-red-300">Report Generation Halted</h3>
                  <p className="mt-1 text-xs leading-relaxed text-red-400">{error}</p>
                </div>
              </div>
            </div>
          )}

          {pdfError && (
            <div className="mb-6 rounded-xl border border-amber-900/60 bg-amber-950/20 p-4 text-xs text-amber-300">
              <div className="flex items-center gap-2">
                <AlertCircle className="size-4 shrink-0 text-amber-400" />
                <span>{pdfError}</span>
              </div>
            </div>
          )}

          {report && <ReportContent report={report} target={target} />}
        </main>
      </div>
    </div>
  )
}

function ReportContent({ report, target }: { report: AutoReport; target: ReportTarget | null }) {
  return (
    <div className="space-y-10">
      {/* 1. Executive Summary */}
      <section id="summary" className="scroll-mt-20">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-400">
          <FileText className="size-4" />
          <span>Executive Summary</span>
        </div>

        <div className="mt-3 rounded-2xl border border-slate-800/80 bg-slate-900/40 p-6 md:p-8">
          <div className="border-l-2 border-emerald-500 pl-4">
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100">
              {target?.displayName || "Dataset Analysis"}
            </h1>
            <p className="mt-1 text-xs text-slate-400">
              Synthesized by DataPilot Analytics Engine · {new Date().toLocaleDateString(undefined, { dateStyle: "long" })}
            </p>
          </div>

          <div className="mt-5 space-y-4 text-sm leading-relaxed text-slate-300">
            {report.summary.split(/\n\n+/).map((para, i) => (
              <p key={i} className="whitespace-pre-wrap">
                {para}
              </p>
            ))}
          </div>
        </div>
      </section>

      {/* 2. Key Metrics & Dataset Health */}
      <section id="metrics" className="scroll-mt-20">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <Activity className="size-4 text-sky-400" />
            <span>Dataset Vital Signs & Profile</span>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {report.metrics.map((metric) => {
            const Icon = getMetricIcon(metric.label)
            return (
              <div
                key={metric.label}
                className="group rounded-xl border border-slate-800/80 bg-slate-900/30 p-4 transition-colors hover:border-slate-700/80 hover:bg-slate-900/50"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400">{metric.label}</span>
                  <span className="flex size-7 items-center justify-center rounded-lg bg-slate-800/60 text-slate-400 group-hover:text-emerald-400">
                    <Icon className="size-3.5" />
                  </span>
                </div>
                <p className="mt-2 font-mono text-xl font-bold tracking-tight text-emerald-400">
                  {metric.value}
                </p>
              </div>
            )
          })}
        </div>
      </section>

      {/* 3. Visualizations Gallery */}
      {report.charts && report.charts.length > 0 && (
        <section id="visualizations" className="scroll-mt-20">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
              <BarChart3 className="size-4 text-amber-400" />
              <span>Multi-Perspective Visual Analysis</span>
            </div>
            <span className="text-xs text-slate-500 font-mono">
              {report.charts.length} {report.charts.length === 1 ? "Chart" : "Charts"} Generated
            </span>
          </div>

          <div
            className={`grid gap-6 ${
              report.charts.length > 1 ? "xl:grid-cols-2" : "grid-cols-1"
            }`}
          >
            {report.charts.map((chart, index) => {
              const badge = getChartBadge(chart.title, chart.figure)
              return (
                <div
                  key={`${chart.title}-${index}`}
                  className="flex flex-col rounded-2xl border border-slate-800/80 bg-slate-900/40 overflow-hidden"
                >
                  {/* Card Header */}
                  <div className="flex items-center justify-between border-b border-slate-800/60 px-5 py-3.5 bg-slate-950/40">
                    <div className="min-w-0 pr-2">
                      <h3 className="truncate text-sm font-semibold text-slate-200">
                        {chart.title}
                      </h3>
                    </div>
                    <span className="shrink-0 rounded-full border border-slate-800 bg-slate-900 px-2 py-0.5 text-[10px] font-medium text-slate-300">
                      {badge}
                    </span>
                  </div>

                  {/* Interactive Plotly Container with data-report-chart attribute for PDF */}
                  <div
                    data-report-chart
                    className="h-[360px] w-full p-2 bg-slate-950/20"
                  >
                    <PlotlyBoard
                      data={chart.figure.data || []}
                      layout={chart.figure.layout}
                      frames={chart.figure.frames}
                      isDark
                    />
                  </div>

                  {/* Interpretation Callout */}
                  {chart.interpretation && (
                    <div className="border-t border-slate-800/60 bg-slate-950/50 px-5 py-3 text-xs leading-relaxed text-slate-400 flex items-start gap-2">
                      <Lightbulb className="size-3.5 shrink-0 text-amber-400 mt-0.5" />
                      <span>{chart.interpretation}</span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* 4. Structured Analytical Findings */}
      {report.sections && report.sections.length > 0 && (
        <section id="sections" className="space-y-6 scroll-mt-20">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <Layers className="size-4 text-indigo-400" />
            <span>Analytical Findings & Observations</span>
          </div>

          <div className="space-y-6">
            {report.sections
              .filter((section) => section.title !== "Executive Summary")
              .map((section) => (
                <div
                  key={section.title}
                  className="rounded-2xl border border-slate-800/80 bg-slate-900/40 p-6"
                >
                  <h3 className="text-base font-semibold text-slate-100">{section.title}</h3>

                  {section.content && (
                    <p className="mt-2 text-xs leading-relaxed text-slate-300 whitespace-pre-wrap">
                      {section.content}
                    </p>
                  )}

                  {section.items && section.items.length > 0 && (
                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                      {section.items.map((item, idx) => (
                        <div
                          key={`${item.headline}-${idx}`}
                          className="rounded-xl border border-slate-800/70 bg-slate-950/50 p-4"
                        >
                          <h4 className="text-xs font-semibold text-emerald-300">
                            {item.headline}
                          </h4>
                          <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                            {item.content}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
          </div>
        </section>
      )}

      {/* 5. Key Evidence Tables */}
      {report.tables && report.tables.length > 0 && (
        <section id="tables" className="scroll-mt-20">
          <div className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <Table2 className="size-4 text-teal-400" />
            <span>Structured Data Summaries</span>
          </div>

          <div className="space-y-6">
            {report.tables.map((table, index) => (
              <div
                key={`${table.title}-${index}`}
                className="overflow-hidden rounded-2xl border border-slate-800/80 bg-slate-900/40"
              >
                <div className="border-b border-slate-800/60 px-5 py-3.5 bg-slate-950/40">
                  <h3 className="text-sm font-semibold text-slate-200">{table.title}</h3>
                </div>

                <div className="max-h-80 overflow-auto scrollbar-thin">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                      <tr>
                        {table.columns.map((column) => (
                          <th key={column} className="px-4 py-2.5 whitespace-nowrap">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-slate-300">
                      {table.rows.map((row, rowIndex) => (
                        <tr
                          key={rowIndex}
                          className="hover:bg-slate-800/30 transition-colors font-mono text-[11px]"
                        >
                          {table.columns.map((column) => (
                            <td key={column} className="px-4 py-2 whitespace-nowrap text-slate-300">
                              {formatReportValue(row[column])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {table.interpretation && (
                  <div className="border-t border-slate-800/60 bg-slate-950/50 px-5 py-3 text-xs text-slate-400 flex items-start gap-2">
                    <Info className="size-3.5 shrink-0 text-teal-400 mt-0.5" />
                    <span>{table.interpretation}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 6. Strategic Recommendations */}
      {report.recommendations && report.recommendations.length > 0 && (
        <section id="recommendations" className="scroll-mt-20">
          <div className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <Lightbulb className="size-4 text-amber-400" />
            <span>Key Recommendations</span>
          </div>

          <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 p-6">
            <ul className="space-y-3">
              {report.recommendations.map((item, index) => (
                <li key={index} className="flex items-start gap-3 text-xs leading-relaxed text-slate-300">
                  <CheckCircle2 className="size-4 shrink-0 text-emerald-400 mt-0.5" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {/* 7. Conclusion */}
      <section id="conclusion" className="scroll-mt-20">
        <div className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
          <CheckCircle2 className="size-4 text-emerald-400" />
          <span>Synthesis & Conclusion</span>
        </div>

        <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 p-6 md:p-8">
          <div className="space-y-3 text-xs leading-relaxed text-slate-300">
            {report.conclusion.split(/\n\n+/).map((para, i) => (
              <p key={i} className="whitespace-pre-wrap">
                {para}
              </p>
            ))}
          </div>
        </div>
      </section>

      {/* 8. Sources (if present) */}
      {report.sources && report.sources.length > 0 && (
        <section className="text-xs text-slate-500 pt-4 border-t border-slate-800/60">
          <p className="font-semibold uppercase tracking-wider text-[10px] text-slate-400 mb-2">
            Referenced Sources
          </p>
          <ul className="space-y-1.5">
            {report.sources.map((source) => (
              <li key={source.url}>
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-emerald-400 hover:underline"
                >
                  <ExternalLink className="size-3" />
                  <span>{source.title}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

function EmptyState({
  icon,
  title,
  description,
}: {
  icon: ReactNode
  title: string
  description: string
}) {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center rounded-2xl border border-slate-800/80 bg-slate-900/20 p-8 text-center text-slate-400">
      <div className="mb-4 flex size-12 items-center justify-center rounded-xl bg-slate-900 ring-1 ring-slate-800">
        {icon}
      </div>
      <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
      <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-slate-500">{description}</p>
    </div>
  )
}

function getMetricIcon(label: string) {
  const lowered = label.toLowerCase()
  if (lowered.includes("row")) return Hash
  if (lowered.includes("col")) return Columns3
  if (lowered.includes("numeric")) return Activity
  if (lowered.includes("categor")) return Layers
  if (lowered.includes("date") || lowered.includes("time") || lowered.includes("year")) return Clock
  if (lowered.includes("miss")) return AlertCircle
  if (lowered.includes("dup")) return Copy
  return BarChart3
}

function getChartBadge(title: string, figure: any): string {
  const loweredTitle = (title || "").toLowerCase()
  const traceType = figure?.data?.[0]?.type || ""
  if (traceType === "bar") return "Category Breakdown"
  if (traceType === "histogram" || traceType === "box") return "Distribution"
  if (traceType === "scatter" && figure?.data?.[0]?.mode === "markers") return "Correlation"
  if (loweredTitle.includes("over") || loweredTitle.includes("trend") || traceType === "scatter")
    return "Trend Analysis"
  return "Data Perspective"
}
