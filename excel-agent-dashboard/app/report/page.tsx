"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { ArrowLeft, BarChart3, Check, Download, ListChecks, Loader2, Maximize2, RotateCcw, Square, X } from "lucide-react"
import { PlotlyBoard } from "@/components/charts/plotly-board"
import { Button } from "@/components/ui/button"
import type { ThinkingTraceEntry } from "@/components/dashboard/dashboard-shared"
import { apiFetch } from "@/lib/api"
import type { AutoReport, ReportChart, ReportPlan, ReportTarget } from "@/lib/report"
import { formatReportValue } from "@/lib/report-format"

type StreamEvent =
  | { type: "trace"; entry: ThinkingTraceEntry }
  | { type: "planner"; message: string }
  | { type: "plan"; plan: ReportPlan }
  | { type: "chart"; chart: ReportChart }
  | { type: "chart_skipped"; id: number; reason: string }
  | { type: "synthesizing" }
  | { type: "final"; payload: AutoReport }
  | { type: "error"; error: string }

export default function ReportPage() {
  const [target, setTarget] = useState<ReportTarget | null>(null)
  const [plan, setPlan] = useState<ReportPlan | null>(null)
  const [plannerMessage, setPlannerMessage] = useState("")
  const [streamedCharts, setStreamedCharts] = useState<ReportChart[]>([])
  const [skipped, setSkipped] = useState<Record<number, string>>({})
  const [report, setReport] = useState<AutoReport | null>(null)
  const [writing, setWriting] = useState(false)
  const [trace, setTrace] = useState<ThinkingTraceEntry[]>([])
  const [error, setError] = useState("")
  const [running, setRunning] = useState(false)
  const [preparingPdf, setPreparingPdf] = useState(false)
  const [focusedChart, setFocusedChart] = useState<ReportChart | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const generate = useCallback(async (reportTarget: ReportTarget) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    let stalled = false
    let stallTimer: ReturnType<typeof setTimeout> | undefined
    let receivedFinal = false

    function resetStallTimer() {
      if (stallTimer) clearTimeout(stallTimer)
      stallTimer = setTimeout(() => {
        stalled = true
        controller.abort()
      }, 75_000)
    }

    setRunning(true)
    setError("")
    setTrace([])
    setReport(null)
    setPlan(null)
    setPlannerMessage("")
    setStreamedCharts([])
    setSkipped({})
    setWriting(false)
    setFocusedChart(null)
    resetStallTimer()

    try {
      const response = await apiFetch("/agent/report/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ dataset_id: reportTarget.datasetId }),
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(typeof payload?.detail === "string" ? payload.detail : `Report generation failed (${response.status}).`)
      }
      const reader = response.body?.getReader()
      if (!reader) throw new Error("The report stream was unavailable.")

      const decoder = new TextDecoder()
      let buffer = ""
      function consume(line: string) {
        if (!line.trim() || receivedFinal || controller.signal.aborted) return
        const event = JSON.parse(line) as StreamEvent
        resetStallTimer()
        if (event.type === "trace") setTrace((current) => [...current, event.entry])
        if (event.type === "planner") setPlannerMessage(event.message)
        if (event.type === "plan") setPlan(event.plan)
        if (event.type === "chart") setStreamedCharts((current) => [...current, event.chart])
        if (event.type === "chart_skipped") setSkipped((current) => ({ ...current, [event.id]: event.reason }))
        if (event.type === "synthesizing") setWriting(true)
        if (event.type === "final") {
          receivedFinal = true
          setReport(event.payload)
        }
        if (event.type === "error") throw new Error(event.error)
      }

      try {
        while (!receivedFinal) {
          const { done, value } = await reader.read()
          buffer += decoder.decode(value, { stream: !done })
          const lines = buffer.split("\n")
          buffer = done ? "" : lines.pop() ?? ""
          lines.forEach(consume)
          if (done) break
        }
      } finally {
        void reader.cancel().catch(() => {})
        reader.releaseLock()
      }
      if (!receivedFinal) throw new Error("The connection ended before the finished report arrived.")
    } catch (reason) {
      if (abortRef.current !== controller) return
      if (stalled) setError("The report timed out. You can run it again.")
      else if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Auto Report failed.")
    } finally {
      if (stallTimer) clearTimeout(stallTimer)
      if (abortRef.current === controller) {
        abortRef.current = null
        setRunning(false)
      }
    }
  }, [])

  useEffect(() => {
    const raw = sessionStorage.getItem("report_target")
    if (!raw) return
    const timer = setTimeout(() => {
      try {
        const saved = JSON.parse(raw) as ReportTarget
        if (!saved.datasetId) throw new Error("Missing dataset")
        setTarget(saved)
        void generate(saved)
      } catch {
        setError("Choose a dataset from the dashboard to create a report.")
      }
    }, 0)
    return () => {
      clearTimeout(timer)
      const controller = abortRef.current
      abortRef.current = null
      controller?.abort()
    }
  }, [generate])

  async function downloadPdf() {
    if (!report || !target || preparingPdf) return
    setPreparingPdf(true)
    setError("")
    try {
      const { downloadReportPdf } = await import("@/lib/report-pdf")
      await downloadReportPdf(report, target)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "PDF export failed.")
    } finally {
      setPreparingPdf(false)
    }
  }

  const datasetName = report?.dataset.name ?? target?.displayName ?? "Dataset report"
  const charts = report?.charts ?? streamedCharts
  const status = error ? "Run ended" : report ? "Ready" : writing ? "Writing takeaways" : plan ? "Building charts" : "Planning report"

  return (
    <div className="min-h-screen bg-[#0b1018] text-slate-100 antialiased">
      <header className="sticky top-0 z-30 border-b border-white/8 bg-[#0b1018]/90 backdrop-blur-xl">
        <div className="mx-auto flex min-h-15 max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-2 sm:px-7">
          <Link href="/dashboard" className="inline-flex items-center gap-2 text-sm text-slate-300 hover:text-white">
            <ArrowLeft className="size-4" aria-hidden="true" /> Dashboard
          </Link>
          {target && (
            <div className="flex items-center gap-2">
              {running ? (
                <Button variant="outline" onClick={() => {
                  setError("Report generation stopped.")
                  abortRef.current?.abort()
                }}><Square aria-hidden="true" /> Stop</Button>
              ) : (
                <Button variant="outline" disabled={preparingPdf} onClick={() => void generate(target)}>
                  <RotateCcw aria-hidden="true" /> Run again
                </Button>
              )}
              {report && (
                <Button disabled={preparingPdf} onClick={() => void downloadPdf()}>
                  {preparingPdf ? <Loader2 className="animate-spin" aria-hidden="true" /> : <Download aria-hidden="true" />}
                  {preparingPdf ? "Exporting…" : "Export PDF"}
                </Button>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-8 px-4 py-7 sm:px-7 sm:py-10">
        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-semibold tracking-wide text-emerald-300">AUTO REPORT</span>
            {target && <span role="status" className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300">
              {running && <Loader2 className="size-3 animate-spin" aria-hidden="true" />}{status}
            </span>}
          </div>
          <h1 className="break-words text-3xl font-semibold tracking-tight sm:text-4xl">{datasetName}</h1>
          {target && <p className="text-xs text-slate-400">
            {(report?.dataset.rows ?? target.rowCount).toLocaleString()} rows · {(report?.dataset.columns ?? target.columnCount).toLocaleString()} columns
          </p>}
        </section>

        {error && <p role="alert" className="rounded-xl border border-amber-400/20 bg-amber-400/8 p-3 text-sm text-amber-200">{error}</p>}
        {plan ? (
          <AnalysisPlan plan={plan} charts={charts} skipped={skipped} running={running} />
        ) : !error && (
          <section className="space-y-2 rounded-2xl border border-emerald-400/15 bg-emerald-400/5 p-5 sm:p-6">
            <h2 className="flex items-center gap-2 text-base font-semibold">
              <ListChecks className="size-5 text-emerald-300" aria-hidden="true" /> Analysis plan
            </h2>
            <p role="status" className="text-sm text-slate-300">
              {plannerMessage || (target ? "Preparing a dataset sample for the planner." : "Open Auto Report from a dataset in the dashboard.")}
            </p>
          </section>
        )}

        {!!charts.length && (
          <section aria-labelledby="report-charts" className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <h2 id="report-charts" className="text-xl font-semibold tracking-tight">Visualizations</h2>
              <span role="status" className="text-xs text-slate-400">{charts.length} {charts.length === 1 ? "chart" : "charts"} ready</span>
            </div>
            <div className="grid gap-4 xl:grid-cols-2">
              {charts.map((chart) => (
                <ChartCard key={chart.id} chart={chart} onExpand={() => setFocusedChart(chart)} />
              ))}
            </div>
          </section>
        )}

        {report?.summary && (
          <section aria-label="Report summary" className="space-y-2 border-l border-emerald-400/60 pl-4 text-sm leading-relaxed text-slate-200">
            <h2 className="font-semibold">What stands out</h2>
            {report.summary.split(/\n\n+/).map((paragraph, index) => <p key={index}>{paragraph}</p>)}
          </section>
        )}
        {report?.generation_mode === "fallback" && <p className="text-sm text-slate-400">The charts include their calculated takeaways. The report summary was unavailable.</p>}
        {report && <ReportDetails report={report} />}
        {!!trace.length && (
          <details className="rounded-xl border border-white/8 px-4 py-3 text-xs text-slate-400">
            <summary className="cursor-pointer font-medium text-slate-200">Report activity</summary>
            <ol className="mt-3 max-h-56 space-y-2 overflow-y-auto">
              {trace.map((entry, index) => <li key={index}>{entry.content}</li>)}
            </ol>
          </details>
        )}
      </main>
      {focusedChart && <ChartDialog chart={focusedChart} onClose={() => setFocusedChart(null)} />}
    </div>
  )
}

function AnalysisPlan({ plan, charts, skipped, running }: {
  plan: ReportPlan; charts: ReportChart[]; skipped: Record<number, string>; running: boolean
}) {
  const ready = new Set(charts.map((chart) => chart.id))
  return (
    <section aria-labelledby="analysis-plan" className="space-y-4 rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-4 sm:p-6 motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="analysis-plan" className="flex items-center gap-2 text-lg font-semibold">
          <ListChecks className="size-5 text-emerald-300" aria-hidden="true" /> Analysis plan
        </h2>
        <span className="text-xs text-emerald-200">{plan.model}</span>
      </header>
      {plan.focus && <p className="max-w-3xl text-sm leading-relaxed text-slate-200">{plan.focus}</p>}
      {!!plan.charts.length && (
        <ol className="grid gap-2 lg:grid-cols-2">
          {plan.charts.map((item) => (
            <li key={item.id} className="rounded-xl border border-white/10 bg-[#0b1018]/70 px-3 py-2.5">
              <details>
                <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
                  <span className="mb-1 flex items-center justify-between gap-2 text-xs">
                    <span className="font-medium capitalize text-emerald-200">{item.type}</span>
                    <span className="inline-flex items-center gap-1 text-slate-300">
                      {ready.has(item.id) && <Check className="size-3 text-emerald-300" aria-hidden="true" />}
                      {ready.has(item.id) ? "Ready" : skipped[item.id] ? "Skipped" : running ? "Pending" : "Incomplete"}
                    </span>
                  </span>
                  <span className="text-sm font-medium text-slate-100">{item.question}</span>
                  <span className="ml-2 whitespace-nowrap text-xs text-slate-400">View method</span>
                </summary>
                <p className="mt-2 border-t border-white/10 pt-2 text-xs leading-relaxed text-slate-300">{item.method}</p>
              </details>
              {skipped[item.id] && <p className="mt-1 text-xs leading-relaxed text-amber-200">{skipped[item.id]}</p>}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function ChartCard({ chart, onExpand }: { chart: ReportChart; onExpand: () => void }) {
  const wide = chart.id === 1 || ["heatmap", "flow", "parallel coordinates", "treemap"].includes(chart.kind)
  return (
    <article className={`min-w-0 overflow-hidden rounded-2xl border border-white/10 bg-[#111923] shadow-lg shadow-black/20 transition hover:border-white/20 motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-300 ${wide ? "xl:col-span-2" : ""}`}>
      <div className="flex items-start justify-between gap-3 border-b border-white/8 px-4 py-3.5 sm:px-5">
        <div className="flex min-w-0 items-start gap-2">
          <BarChart3 aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-emerald-300" />
          <div>
            <span className="text-xs capitalize text-emerald-300">{chart.kind}</span>
            <h3 className="break-words text-sm font-semibold text-slate-100">{chart.title}</h3>
          </div>
        </div>
        <Button variant="ghost" size="icon-sm" aria-label={`Expand chart: ${chart.title}`} title="Expand chart" onClick={onExpand}
          className="transition-transform hover:bg-white/10 motion-safe:hover:scale-110">
          <Maximize2 aria-hidden="true" />
        </Button>
      </div>
      <div data-report-chart className={`min-w-0 px-2 ${wide ? "h-[340px] sm:h-[430px]" : "h-[310px]"}`}>
        <ChartPlot chart={chart} />
      </div>
      {chart.interpretation && <p className="border-t border-white/8 px-4 py-3 text-sm leading-relaxed text-slate-300 sm:px-5">{chart.interpretation}</p>}
    </article>
  )
}

function ChartPlot({ chart }: { chart: ReportChart }) {
  return <PlotlyBoard data={chart.figure.data ?? []} layout={chart.figure.layout} frames={chart.figure.frames} showTitle={false} />
}

function ChartDialog({ chart, onClose }: { chart: ReportChart; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    if (ref.current && !ref.current.open) ref.current.showModal()
  }, [])
  return (
    <dialog ref={ref} onClose={onClose} aria-labelledby="expanded-chart-title"
      className="fixed inset-0 m-auto h-[min(87dvh,820px)] w-[min(95vw,1250px)] max-w-none overflow-hidden rounded-2xl border border-white/15 bg-[#111923] text-slate-100 shadow-2xl backdrop:bg-black/75 motion-safe:animate-in motion-safe:fade-in motion-safe:zoom-in-95 motion-safe:duration-200">
      <div className="flex h-full flex-col">
        <header className="flex items-center justify-between gap-3 border-b border-white/8 px-5 py-3">
          <h2 id="expanded-chart-title" className="min-w-0 break-words text-base font-semibold">{chart.title}</h2>
          <Button variant="ghost" size="icon" aria-label="Close expanded chart" onClick={() => ref.current?.close()}><X aria-hidden="true" /></Button>
        </header>
        <div className="min-h-0 flex-1"><ChartPlot chart={chart} /></div>
        {chart.interpretation && <p className="border-t border-white/8 px-5 py-3 text-sm text-slate-300">{chart.interpretation}</p>}
      </div>
    </dialog>
  )
}

function ReportDetails({ report }: { report: AutoReport }) {
  return (
    <div className="space-y-7">
      {!!report.sections.length && (
        <section aria-label="Findings" className="space-y-3">
          <h2 className="text-xl font-semibold tracking-tight">Findings</h2>
          <div className="grid gap-3 lg:grid-cols-2">
            {report.sections.map((section, index) => (
              <article key={index} className="space-y-2 rounded-xl border border-white/8 bg-white/3 p-4">
                <h3 className="text-sm font-semibold">{section.title}</h3>
                {section.content && <p className="text-sm leading-relaxed text-slate-300">{section.content}</p>}
                {!!section.items?.length && <ul className="space-y-2">
                  {section.items.map((item, i) => (
                    <li key={i} className="text-sm leading-relaxed text-slate-300">
                      <span className="font-medium text-emerald-200">{item.headline}</span>
                      <p>{item.content}</p>
                    </li>
                  ))}
                </ul>}
              </article>
            ))}
          </div>
        </section>
      )}
      {!!report.tables.length && (
        <section className="space-y-2" aria-label="Evidence tables">
          <h2 className="text-lg font-semibold">Evidence tables</h2>
          {report.tables.map((table, index) => (
            <details key={index} className="rounded-xl border border-white/8 bg-white/3 p-3">
              <summary className="cursor-pointer text-sm font-medium">{table.title} <span className="ml-2 text-xs text-slate-400">{table.rows.length} rows</span></summary>
              <div className="mt-3 max-h-80 overflow-auto">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-[#182231]">
                    <tr>{table.columns.map((column) => <th key={column} scope="col" className="whitespace-nowrap px-3 py-2">{column}</th>)}</tr>
                  </thead>
                  <tbody>{table.rows.map((row, rowIndex) => (
                    <tr key={rowIndex} className="border-t border-white/8">
                      {table.columns.map((column) => <td key={column} className="whitespace-nowrap px-3 py-2 text-slate-300">{formatReportValue(row[column])}</td>)}
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              {table.interpretation && <p className="mt-2 text-xs text-slate-400">{table.interpretation}</p>}
            </details>
          ))}
        </section>
      )}
      <details className="rounded-xl border border-white/8 p-3">
        <summary className="cursor-pointer text-sm font-medium">Dataset checks</summary>
        <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {report.metrics.map((metric) => (
            <div key={metric.label} className="rounded-lg bg-white/4 p-3">
              <dt className="text-xs text-slate-400">{metric.label}</dt>
              <dd className="mt-1 text-lg font-semibold tabular-nums">{metric.value}</dd>
            </div>
          ))}
        </dl>
      </details>
      {!!report.recommendations.length && <section className="space-y-2">
        <h2 className="text-lg font-semibold">Next steps</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">
          {report.recommendations.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      </section>}
      {report.conclusion && <p className="border-t border-white/8 pt-3 text-sm text-slate-400">{report.conclusion}</p>}
      {!!report.sources.length && <ul className="space-y-1 text-sm text-emerald-300">
        {report.sources.map((source) => <li key={source.url}><a href={source.url} target="_blank" rel="noreferrer" className="underline">{source.title}</a></li>)}
      </ul>}
    </div>
  )
}
