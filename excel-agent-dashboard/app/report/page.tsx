"use client"

import Link from "next/link"
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { AlertTriangle, ArrowLeft, Database, Download, Loader2, RotateCcw, Sparkles } from "lucide-react"
import { PlotlyBoard } from "@/components/charts/plotly-board"
import { API_BASE_URL } from "@/components/dashboard/hooks/use-agent-runner"
import type { ThinkingTraceEntry } from "@/components/dashboard/dashboard-shared"
import type { AutoReport, ReportTarget } from "@/lib/report"
import { formatReportValue } from "@/lib/report-format"

type StreamEvent = { type: "trace"; entry: ThinkingTraceEntry } | { type: "final"; payload: AutoReport } | { type: "error"; error: string }

export default function ReportPage() {
  const [target, setTarget] = useState<ReportTarget | null>(null)
  const [report, setReport] = useState<AutoReport | null>(null)
  const [trace, setTrace] = useState<ThinkingTraceEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [preparingPdf, setPreparingPdf] = useState(false)
  const [pdfError, setPdfError] = useState<string | null>(null)
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
      }, 60_000)
    }
    traceAtBottomRef.current = true
    setRunning(true); setError(null); setTrace([]); setReport(null)
    resetStallTimer()
    try {
      const response = await fetch(`${API_BASE_URL}/agent/report/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ dataset_id: reportTarget.datasetId, model: reportTarget.model }),
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || `Report generation failed (${response.status}).`)
      }
      const reader = response.body?.getReader()
      if (!reader) throw new Error("The report stream was unavailable.")
      const decoder = new TextDecoder(); let buffer = ""
      let receivedFinal = false
      const consume = (line: string) => {
        if (!line.trim()) return
        const event = JSON.parse(line) as StreamEvent
        resetStallTimer()
        if (event.type === "trace") setTrace((current) => [...current, event.entry])
        if (event.type === "final") { receivedFinal = true; setReport(event.payload) }
        if (event.type === "error") throw new Error(event.error)
      }
      while (true) {
        const { done, value } = await reader.read(); if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n"); buffer = lines.pop() || ""; lines.forEach(consume)
      }
      if (buffer.trim()) consume(buffer)
      if (!receivedFinal) throw new Error("Report generation ended before a final report was received. Please try again.")
    } catch (reason) {
      setTrace((current) => current.at(-1)?.status === "error" ? current : [...current, {
        kind: "observation",
        content: "Report generation stopped before the final report was available.",
        status: "error",
        timestamp: new Date().toISOString(),
      }])
      if (stalled) setError("Report generation is taking longer than expected. Please try again.")
      else if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Auto Report failed.")
    } finally {
      if (stallTimer) clearTimeout(stallTimer)
      if (abortRef.current === controller) abortRef.current = null
      setRunning(false)
    }
  }, [])

  useEffect(() => {
    const panel = tracePanelRef.current
    if (!panel || !traceAtBottomRef.current) return
    const frame = requestAnimationFrame(() => panel.scrollTo({ top: panel.scrollHeight, behavior: "smooth" }))
    return () => cancelAnimationFrame(frame)
  }, [trace])

  const downloadPdf = useCallback(async () => {
    if (!report || !target || preparingPdf) return
    setPreparingPdf(true); setPdfError(null)
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
        if (!startedRef.current) { startedRef.current = true; void generate(saved) }
      }, 0)
    } catch { timer = setTimeout(() => setError("The report target is invalid. Return to the dashboard and try again."), 0) }
    return () => { if (timer) clearTimeout(timer); abortRef.current?.abort() }
  }, [generate])

  return <main className="min-h-screen bg-slate-950 text-slate-100">
    <header className="border-b border-slate-800 bg-slate-950/90 px-6 py-4"><div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
      <div className="flex min-w-0 items-center gap-4"><Link href="/dashboard" className="rounded-lg border border-slate-800 p-2 text-slate-400 hover:text-white" aria-label="Return to dashboard"><ArrowLeft className="size-4" /></Link><div className="min-w-0"><p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-400">Auto Report</p><h1 className="truncate text-xl font-bold">{target?.displayName || "No active dataset"}</h1>{target && <p className="text-xs text-slate-400">{target.sheetName || "Data"} · {target.rowCount.toLocaleString()} rows × {target.columnCount} columns</p>}</div></div>
      {target && <div className="flex items-center gap-2">{report && <button disabled={running || preparingPdf} onClick={() => void downloadPdf()} className="flex items-center gap-2 rounded-lg bg-emerald-500 px-3 py-2 text-xs font-bold text-slate-950 hover:bg-emerald-400 disabled:opacity-50">{preparingPdf ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />} {preparingPdf ? "Preparing PDF..." : "Download PDF"}</button>}<button disabled={running || preparingPdf} onClick={() => void generate(target)} className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-xs font-semibold hover:bg-slate-800 disabled:opacity-50"><RotateCcw className="size-4" /> Regenerate</button></div>}
    </div></header>
    <div className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[280px_1fr]">
      <aside className="space-y-4"><div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4"><div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400"><Database className="size-4 text-emerald-400" /> Report target</div><p className="break-words text-sm font-semibold">{target?.displayName || "None"}</p>{target && <p className="mt-1 text-xs text-slate-500">{target.rowCount.toLocaleString()} rows × {target.columnCount} columns</p>}</div>
      {(running || trace.length > 0) && <div ref={tracePanelRef} onScroll={(event) => { const panel = event.currentTarget; traceAtBottomRef.current = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 48 }} className="max-h-[65vh] overflow-y-auto rounded-2xl border border-slate-800 bg-slate-900/50 p-4"><div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400"><Sparkles className="size-4 text-emerald-400" /> Thinking Mode {running && <Loader2 className="ml-auto size-4 animate-spin" />}</div><div className="space-y-3">{trace.map((entry, index) => <div key={entry.sequence ?? index} className="rounded-lg border border-slate-800 bg-slate-950/40 p-2 text-[11px]"><div className="mb-1 flex items-center justify-between gap-2"><p className="uppercase tracking-wide text-emerald-400">{entry.tool_name || entry.kind}</p>{entry.timestamp && <time className="text-[10px] text-slate-600">{new Date(entry.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time>}</div><p className="text-slate-300">{entry.content}</p></div>)}</div></div>}</aside>
      <section className="min-w-0 rounded-3xl border border-slate-800 bg-slate-900/30 p-6 md:p-8">
        {!target && <State icon={<AlertTriangle className="size-6" />} title="No active dataset" text="Upload a dataset in the dashboard before generating an Auto Report." />}
        {running && !report && <State icon={<Loader2 className="size-7 animate-spin" />} title="Analyzing the active dataset" text="Thinking Mode is calculating evidence, selecting useful tables, and deciding whether visualizations add value." />}
        {error && <State icon={<AlertTriangle className="size-6" />} title="Report generation failed" text={error} />}
        {pdfError && report && <div className="mb-6 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-300">{pdfError}</div>}
        {report && <ReportContent report={report} />}
      </section>
    </div>
  </main>
}

function ReportContent({ report }: { report: AutoReport }) {
  return <div className="space-y-10">
    <article><h2 className="text-2xl font-bold">Executive Summary</h2><p className="mt-3 whitespace-pre-wrap leading-7 text-slate-300">{report.summary}</p></article>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{report.metrics.map((metric) => <div key={metric.label} className="rounded-xl border border-slate-800 bg-slate-950/50 p-4"><p className="text-xs text-slate-500">{metric.label}</p><p className="mt-1 text-xl font-bold text-emerald-400">{metric.value}</p></div>)}</div>
    {report.sections.filter((section) => section.title !== "Executive Summary").map((section) => <article key={section.title} className="max-w-4xl"><h2 className="text-xl font-bold">{section.title}</h2>{section.content && <p className="mt-3 whitespace-pre-wrap leading-7 text-slate-300">{section.content}</p>}{section.items && section.items.length > 0 && <div className="mt-5 grid gap-4">{section.items.map((item, index) => <div key={`${item.headline}-${index}`} className="rounded-xl border border-slate-800 bg-slate-950/35 p-4"><h3 className="font-semibold text-slate-100">{item.headline}</h3><p className="mt-2 leading-6 text-slate-400">{item.content}</p></div>)}</div>}</article>)}
    {report.tables.map((table, index) => <article key={`${table.title}-${index}`}><h2 className="mb-3 text-xl font-bold">{table.title}</h2><div className="max-h-96 overflow-auto rounded-xl border border-slate-800"><table className="w-full text-left text-xs"><thead className="sticky top-0 bg-slate-900"><tr>{table.columns.map((column) => <th key={column} className="border-b border-slate-800 px-3 py-2">{column}</th>)}</tr></thead><tbody>{table.rows.map((row, rowIndex) => <tr key={rowIndex} className="border-b border-slate-900">{table.columns.map((column) => <td key={column} className="px-3 py-2 text-slate-400">{formatReportValue(row[column])}</td>)}</tr>)}</tbody></table></div>{table.interpretation && <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-400">{table.interpretation}</p>}</article>)}
    {report.charts.map((chart, index) => <article key={`${chart.title}-${index}`}><h2 className="text-xl font-bold">{chart.title}</h2><div data-report-chart className="mt-3 h-[420px] rounded-xl border border-slate-800 bg-slate-950/50 p-3"><PlotlyBoard data={chart.figure.data || []} layout={chart.figure.layout} frames={chart.figure.frames} isDark /></div><p className="mt-3 text-sm leading-6 text-slate-400">{chart.interpretation}</p></article>)}
    <article><h2 className="text-xl font-bold">Conclusion</h2><p className="mt-3 whitespace-pre-wrap leading-7 text-slate-300">{report.conclusion}</p></article>
    {report.recommendations?.length > 0 && <article><h2 className="text-xl font-bold">Recommendations</h2><ul className="mt-3 space-y-2 text-slate-300">{report.recommendations.map((item, index) => <li key={index} className="flex gap-3"><span className="text-emerald-400">•</span><span>{item}</span></li>)}</ul></article>}
    {report.sources.length > 0 && <article><h2 className="text-xl font-bold">Sources</h2><ul className="mt-3 space-y-2">{report.sources.map((source) => <li key={source.url}><a className="text-emerald-400 hover:underline" href={source.url} target="_blank" rel="noreferrer">{source.title}</a></li>)}</ul></article>}
  </div>
}

<<<<<<< HEAD
function State({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return <div className="flex min-h-[420px] flex-col items-center justify-center gap-3 text-center text-slate-400"><div className="text-emerald-400">{icon}</div><h2 className="text-lg font-bold text-slate-200">{title}</h2><p className="max-w-lg text-sm">{text}</p></div>
=======
    const blob = new Blob([htmlContent], { type: "text/html;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${dataset?.fileName.replace(/\.[^/.]+$/, "")}-report.html`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // Print PDF helper
  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans overflow-y-auto selection:bg-emerald-500/30">
      {/* Header */}
      <header className="no-print sticky top-0 z-50 bg-slate-900/60 backdrop-blur-md border-b border-slate-800/80 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            href="/dashboard"
            className="flex items-center justify-center h-8 w-8 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700 transition-all active:scale-95"
          >
            <ArrowLeft className="size-4" />
          </Link>
          <div className="flex items-center gap-2">
            <Sparkles className="size-5 text-emerald-400" />
            <span className="font-extrabold text-lg text-slate-100 tracking-tight">DataPilot Report Hub</span>
          </div>
        </div>

        {reportMarkdown && (
          <div className="flex items-center gap-2.5">
            <button
              onClick={downloadMarkdown}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white transition-all active:scale-95"
            >
              <Download className="size-3.5" />
              <span>MD</span>
            </button>
            <button
              onClick={downloadHtml}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white transition-all active:scale-95"
            >
              <Download className="size-3.5" />
              <span>HTML</span>
            </button>
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-emerald-500 text-slate-950 text-xs font-extrabold hover:bg-emerald-400 transition-all active:scale-95 shadow-[0_0_15px_rgba(16,185,129,0.3)] hover:shadow-[0_0_20px_rgba(16,185,129,0.5)]"
            >
              <Printer className="size-3.5" />
              <span>Print / PDF</span>
            </button>
          </div>
        )}
      </header>

      {/* Main Layout */}
      <main className="flex-1 max-w-[1400px] w-full mx-auto p-4 md:p-8 space-y-8 print-card relative z-10">
        {!dataset ? (
          <div className="no-print flex flex-col items-center justify-center py-20 text-center space-y-4">
            <div className="p-4 rounded-full bg-slate-900 border border-slate-800 text-amber-500 animate-bounce">
              <AlertTriangle className="size-8" />
            </div>
            <h2 className="text-xl font-bold text-slate-200">No Active Dataset Selected</h2>
            <p className="text-sm text-slate-400 max-w-md">
              Auto Report requires spreadsheet data to analyze. Please go to the dashboard and upload a CSV or Excel sheet.
            </p>
            <Link
              href="/dashboard"
              className="mt-2 inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-emerald-500 text-slate-950 font-bold hover:bg-emerald-400 transition-all active:scale-95"
            >
              <Database className="size-4" />
              <span>Go to Dashboard</span>
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 print-card">
            {/* Sidebar Controls & Running Traces */}
            <div className="no-print lg:col-span-3 space-y-6">
              {/* Dataset Details Card */}
              <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 backdrop-blur-md p-5 space-y-4">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Database className="size-3.5 text-emerald-400" />
                  Analyzing Dataset
                </h3>
                <div className="min-w-0">
                  <p className="text-sm font-bold text-slate-200 truncate">{dataset.displayName}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-semibold border border-slate-700/50">
                      Rows: {dataset.rowCount.toLocaleString()}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-semibold border border-slate-700/50">
                      Cols: {dataset.columnCount}
                    </span>
                  </div>
                </div>

                {!isRunning && (
                  <button
                    onClick={() => generateReport()}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-slate-850 hover:bg-slate-800 border border-slate-750 text-xs font-bold text-slate-200 hover:text-white transition-all active:scale-95"
                  >
                    <RotateCcw className="size-3.5 text-emerald-400 animate-spin-hover" />
                    <span>Regenerate Report</span>
                  </button>
                )}
              </div>

              {/* Progress & Live Thinking Logs */}
              {(isRunning || thinkingTrace.length > 0) && (
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 backdrop-blur-md p-5 space-y-4 max-h-[480px] flex flex-col min-h-[160px]">
                  <div className="flex items-center justify-between shrink-0">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                      <Sparkles className="size-3.5 text-emerald-400" />
                      Agent Log
                    </h3>
                    {isRunning && (
                      <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-bold uppercase tracking-widest animate-pulse">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                        Thinking...
                      </div>
                    )}
                  </div>

                  <div className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin">
                    {thinkingTrace.map((entry, idx) => {
                      const isThought = entry.kind === "thought"
                      const isAction = entry.kind === "action"
                      const isObservation = entry.kind === "observation"

                      return (
                        <div
                          key={idx}
                          className={`text-xs p-3 rounded-xl border border-slate-800/40 flex flex-col space-y-1.5 ${
                            isThought
                              ? "bg-slate-900/60"
                              : isAction
                              ? "bg-emerald-950/10 border-emerald-900/30"
                              : "bg-slate-900/30"
                          }`}
                        >
                          <div className="flex items-center gap-2 font-bold uppercase text-[9px] tracking-wider shrink-0">
                            {isThought && <span className="text-slate-400">Thought Process</span>}
                            {isAction && (
                              <span className="text-emerald-400 flex items-center gap-1">
                                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                                Tool Execution
                              </span>
                            )}
                            {isObservation && <span className="text-sky-400">Observation Result</span>}
                          </div>
                          
                          <p className="text-slate-300 break-words leading-relaxed font-mono">
                            {entry.content}
                          </p>

                          {isAction && entry.tool_input && (
                            <pre className="text-[10px] p-2 bg-slate-950 rounded-lg text-slate-400 border border-slate-850 overflow-x-auto max-w-full font-mono max-h-24">
                              {entry.tool_input}
                            </pre>
                          )}

                          {isObservation && entry.details && (
                            <pre className="text-[10px] p-2 bg-slate-950/40 rounded-lg text-slate-400 border border-slate-850/40 overflow-x-auto max-w-full font-mono max-h-24">
                              {entry.details}
                            </pre>
                          )}
                        </div>
                      )
                    })}
                  </div>

                  {isRunning && (
                    <button
                      onClick={cancelRun}
                      className="w-full shrink-0 flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-red-950/20 hover:bg-red-950/40 border border-red-900/30 text-xs font-bold text-red-400 hover:text-red-300 transition-all active:scale-95"
                    >
                      <span>Stop Agent execution</span>
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Generated Report Output Panel */}
            <div className={`${(isRunning || thinkingTrace.length > 0) ? "lg:col-span-9" : "lg:col-span-12"} print-card`}>
              <div className="rounded-3xl border border-slate-800/80 bg-slate-900/20 backdrop-blur-md p-6 md:p-10 shadow-2xl min-h-[400px] space-y-8 print-card">
                
                {/* Generation Loading State overlay if running and no output yet */}
                {isRunning && !reportMarkdown && (
                  <div className="flex flex-col items-center justify-center py-24 text-center space-y-6">
                    <Loader2 className="size-10 text-emerald-400 animate-spin" />
                    <div className="space-y-1 max-w-md">
                      <h3 className="font-bold text-slate-200">Analyzing dataset semantics</h3>
                      <p className="text-xs text-slate-400">
                        Our agent is executing descriptive calculations, finding correlations, and drawing custom charts in real-time.
                      </p>
                    </div>
                  </div>
                )}


                {/* Error State display */}
                {error && !reportMarkdown && (
                  <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
                    <div className="p-3.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-400">
                      <AlertTriangle className="size-6" />
                    </div>
                    <h3 className="font-bold text-slate-200">Analysis Failed</h3>
                    <p className="text-xs text-red-400 max-w-md">
                      {error}
                    </p>
                    <button
                      onClick={() => generateReport()}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-bold transition-all active:scale-95 cursor-pointer"
                    >
                      <RotateCcw className="size-4 text-emerald-400" />
                      <span>Try Again</span>
                    </button>
                  </div>
                )}

                {/* Report Content */}
                {reportMarkdown ? (
                  <div className="space-y-8 print-card">
                    {/* Rendered Text Report */}
                    <div className="prose prose-slate prose-invert max-w-none print-card">
                      <MarkdownRenderer content={reportMarkdown} />
                    </div>

                    {/* Rendered Chart visualization if available */}
                    {visualization && (
                      <div className="space-y-4 pt-6 border-t border-slate-800/60 break-inside-avoid">
                        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2 no-print">
                          <Sparkles className="size-4.5 text-emerald-400" />
                          Generated Dashboard Visualization
                        </h3>
                        <div className="h-[380px] w-full rounded-2xl border border-slate-800 bg-slate-950/30 p-4 shadow-inner">
                          <PlotlyBoard 
                            data={visualization.data || []} 
                            layout={visualization.layout} 
                            frames={visualization.frames} 
                            isDark={true} 
                          />
                        </div>
                      </div>
                    )}

                    {/* Logged Tables from execute_python */}
                    {loggedTables.map((table, index) => (
                      <div key={table.id || index} className="space-y-4 pt-6 border-t border-slate-800/60 break-inside-avoid">
                        <h3 className="text-base font-bold text-slate-200">
                          {table.title || `Logged Data Table ${index + 1}`}
                        </h3>
                        <div className="overflow-x-auto rounded-xl border border-slate-800 max-h-[300px]">
                          <table className="w-full text-left text-xs border-collapse">
                            <thead>
                              <tr className="bg-slate-900 text-slate-300 font-semibold border-b border-slate-800">
                                {table.rows.length > 0 && Object.keys(table.rows[0]).map((col) => (
                                  <th key={col} className="px-4 py-2 border-r border-slate-800">{col}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {table.rows.map((row: Record<string, unknown>, rIdx: number) => (
                                <tr 
                                  key={rIdx} 
                                  className="border-b border-slate-850 hover:bg-slate-900/40 text-slate-400 transition-colors"
                                >
                                  {Object.values(row).map((val: unknown, cIdx: number) => (
                                    <td key={cIdx} className="px-4 py-2 border-r border-slate-850 truncate max-w-xs">
                                      {val !== null && val !== undefined ? String(val) : ""}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  !isRunning && (
                    <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
                      <div className="p-3.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                        <Sparkles className="size-6" />
                      </div>
                      <h3 className="font-bold text-slate-200">Ready to Analyze</h3>
                      <p className="text-xs text-slate-400 max-w-md">
                        Press generate to let the AI agent clean, parse stats, search variables, and build the business report of your dataset.
                      </p>
                      <button
                        onClick={() => generateReport()}
                        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-500 text-slate-950 font-bold hover:bg-emerald-400 transition-all active:scale-95 shadow-[0_0_15px_rgba(16,185,129,0.35)]"
                      >
                        <Play className="size-4" />
                        <span>Run Report Generator</span>
                      </button>
                    </div>
                  )
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Global Background Shader */}
      <div className="fixed inset-0 z-0 pointer-events-none no-print">
        <div className="w-full h-full opacity-40">
          <canvas ref={canvasRef} className="block w-full h-full"></canvas>
        </div>
        <div className="absolute inset-0 data-grid-overlay"></div>
      </div>
    </div>
  )
>>>>>>> 388280c (Fix SSR hydration mismatch and remove unused datapilot-ai-suite prototype)
}
