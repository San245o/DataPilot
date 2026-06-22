"use client"

import { useEffect, useState, useMemo, useCallback } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import {
  ArrowLeft,
  Download,
  Loader2,
  FileText,
  CheckCircle2,
  Play,
  Printer,
  RotateCcw,
  Sparkles,
  Search,
  Database,
  Grid,
  AlertTriangle
} from "lucide-react"

import { useAgentRunner } from "@/components/dashboard/hooks/use-agent-runner"
import { PlotlyBoard } from "@/components/charts/plotly-board"
import { DEFAULT_THINKING_MODEL, type SheetRow, type ThinkingTraceEntry } from "@/components/dashboard/dashboard-shared"

interface SavedDataset {
  id: string
  fileName: string
  displayName: string
  sheetName?: string
  rows: SheetRow[]
  rowCount: number
  columnCount: number
}

// Markdown-to-React Helper Component
function MarkdownRenderer({ content }: { content: string }) {
  const lines = content.split("\n")
  return (
    <div className="space-y-4 text-sm md:text-base text-slate-300 leading-relaxed font-sans">
      {lines.map((line, idx) => {
        const trimmed = line.trim()
        if (!trimmed) return <div key={idx} className="h-2" />

        // Headers
        if (trimmed.startsWith("### ")) {
          return (
            <h3 key={idx} className="text-base md:text-lg font-bold text-emerald-400 mt-6 mb-2 flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              {trimmed.slice(4)}
            </h3>
          )
        }
        if (trimmed.startsWith("## ")) {
          return (
            <h2 key={idx} className="text-lg md:text-xl font-extrabold text-slate-100 border-b border-slate-700/60 pb-2 mt-8 mb-4">
              {trimmed.slice(3)}
            </h2>
          )
        }
        if (trimmed.startsWith("# ")) {
          return (
            <h1 key={idx} className="text-2xl md:text-3xl font-black text-slate-50 tracking-tight mt-10 mb-6 pb-2 border-b-2 border-emerald-500/30">
              {trimmed.slice(2)}
            </h1>
          )
        }

        // Horizontal Rules
        if (trimmed === "---") {
          return <hr key={idx} className="border-slate-800 my-6" />
        }

        // Bullet Lists
        if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
          return (
            <ul key={idx} className="list-disc pl-6 space-y-1.5 my-1 text-slate-300">
              <li>{parseInlineMarkdown(trimmed.slice(2))}</li>
            </ul>
          )
        }

        // Numbered Lists
        const numMatch = trimmed.match(/^(\d+)\.\s+(.*)$/)
        if (numMatch) {
          return (
            <ol key={idx} className="list-decimal pl-6 space-y-1.5 my-1 text-slate-300">
              <li value={parseInt(numMatch[1], 10)}>{parseInlineMarkdown(numMatch[2])}</li>
            </ol>
          )
        }

        // Blockquotes
        if (trimmed.startsWith("> ")) {
          return (
            <blockquote key={idx} className="border-l-4 border-emerald-500/50 bg-emerald-950/20 px-4 py-2.5 rounded-r-lg my-4 italic text-slate-400">
              {parseInlineMarkdown(trimmed.slice(2))}
            </blockquote>
          )
        }

        // Standard Paragraph
        return <p key={idx} className="my-2 text-slate-300">{parseInlineMarkdown(trimmed)}</p>
      })}
    </div>
  )
}

function parseInlineMarkdown(text: string) {
  const boldParts = text.split("**")
  if (boldParts.length === 1) return text
  return (
    <>
      {boldParts.map((part, index) => {
        if (index % 2 === 1) {
          return (
            <strong key={index} className="font-bold text-slate-100">
              {part}
            </strong>
          )
        }
        return part
      })}
    </>
  )
}

export default function ReportPage() {
  const router = useRouter()
  const [dataset, setDataset] = useState<SavedDataset | null>(null)
  const [reportHtml, setReportHtml] = useState<string>("")
  const [reportMarkdown, setReportMarkdown] = useState<string>("")
  const [visualization, setVisualization] = useState<any | null>(null)
  const [loggedTables, setLoggedTables] = useState<any[]>([])
  const [thinkingTrace, setThinkingTrace] = useState<ThinkingTraceEntry[]>([])
  const [reportTitle, setReportTitle] = useState("Dataset Analysis Report")

  const { runAgent, isRunning, error, cancelRun } = useAgentRunner()

  // Retrieve dataset from sessionStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = sessionStorage.getItem("report_dataset")
      if (stored) {
        try {
          setDataset(JSON.parse(stored))
        } catch (e) {
          console.error("Failed to parse stored dataset", e)
        }
      }
    }
  }, [])

  // Enable scrolling for report page overriding global styling constraints
  useEffect(() => {
    document.documentElement.style.overflow = "auto"
    document.body.style.overflow = "auto"
    return () => {
      document.documentElement.style.overflow = ""
      document.body.style.overflow = ""
    }
  }, [])

  // Trigger Report Generation using Thinking Mode
  const generateReport = useCallback(async (customPrompt?: string) => {
    if (!dataset || isRunning) return

    setReportMarkdown("")
    setVisualization(null)
    setLoggedTables([])
    setThinkingTrace([])

    const promptText = customPrompt || 
      `Generate a highly detailed and comprehensive analytical business report of the dataset.
Please structure your response as follows:
# ${dataset.displayName.split(" / ")[0]} Data Analysis Report
## 1. Executive Summary
Provide a high-level overview of the dataset, target goals, key metrics (row count: ${dataset.rowCount}, column count: ${dataset.columnCount}), and summary of key findings.

## 2. Key Metrics & Descriptive Statistics
Deep dive into columns, averages, distributions, data quality check, and summary tables.

## 3. Key Findings & Trend Analysis
Identify interesting correlations, trends over time, anomaly patterns, or categories.

## 4. Actionable Recommendations
List 3-5 concrete business suggestions based on this data.

Crucial: In your planning step, you MUST generate at least one high-quality Plotly visualization by assigning the Plotly figure to the variable \`fig\` in your execute_python step, and also log a summary dataframe using \`log_output(df)\`. The chart should be beautiful, clean, and directly related to the key insights.`

    const datasetNames = { [dataset.id]: dataset.displayName }

    const result = await runAgent({
      prompt: promptText,
      rows: dataset.rows,
      activeDatasetId: dataset.id,
      selectedDatasetIds: [dataset.id],
      datasetNames,
      modelName: DEFAULT_THINKING_MODEL,
      history: [],
      thinkingMode: true,
      onThinkingTrace: (entry) => {
        setThinkingTrace((prev) => [...prev, entry])
      }
    })

    if (result.ok) {
      const responseData = result.data
      setReportMarkdown(responseData.assistantMessage.content)
      setVisualization(responseData.visualizationPayload)
      if (responseData.responseTables && responseData.responseTables.length > 0) {
        setLoggedTables(responseData.responseTables)
      }
      setReportTitle(`${dataset.displayName.split(" / ")[0]} Analysis Report`)
    }
  }, [dataset, isRunning, runAgent])

  // Generate automatically once dataset is loaded
  useEffect(() => {
    if (dataset) {
      generateReport()
    }
  }, [dataset])

  // Download Report as Markdown
  const downloadMarkdown = () => {
    if (!reportMarkdown) return
    const blob = new Blob([reportMarkdown], { type: "text/markdown;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${dataset?.fileName.replace(/\.[^/.]+$/, "")}-report.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // Download Report as Self-contained HTML Page
  const downloadHtml = () => {
    if (!reportMarkdown) return
    
    // Create clean HTML wrapper with Tailwind stylesheet and CSS grid styles
    const htmlContent = `<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${reportTitle}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      background-color: #0b1326;
      color: #dae2fd;
      font-family: system-ui, -apple-system, sans-serif;
    }
    .glass-card {
      background: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(51, 65, 85, 0.5);
    }
  </style>
</head>
<body class="p-8 max-w-4xl mx-auto">
  <div class="glass-card p-8 rounded-2xl shadow-2xl space-y-6">
    <div class="border-b border-slate-700/60 pb-6 mb-6">
      <h1 class="text-3xl font-extrabold tracking-tight text-white">${reportTitle}</h1>
      <p class="text-sm text-slate-400 mt-2">Generated by DataPilot AI Agent</p>
    </div>
    
    <div class="prose prose-invert max-w-none space-y-4">
      ${reportMarkdown
        .split("\n")
        .map(line => {
          const trimmed = line.trim()
          if (!trimmed) return "<br/>"
          if (trimmed.startsWith("### ")) return `<h3 class="text-lg font-bold text-emerald-400 mt-4 mb-2">${trimmed.slice(4)}</h3>`
          if (trimmed.startsWith("## ")) return `<h2 class="text-xl font-extrabold text-white border-b border-slate-700/40 pb-1 mt-6 mb-3">${trimmed.slice(3)}</h2>`
          if (trimmed.startsWith("# ")) return `<h1 class="text-2xl font-black text-white mt-8 mb-4">${trimmed.slice(2)}</h1>`
          if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) return `<li class="ml-6 list-disc text-slate-300">${trimmed.slice(2).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}</li>`
          return `<p class="text-slate-300">${trimmed.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}</p>`
        })
        .join("\n")}
    </div>
  </div>
</body>
</html>`

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
      {/* Global CSS for Print-styling layout */}
      <style jsx global>{`
        @media print {
          body {
            background-color: white !important;
            color: black !important;
          }
          .no-print {
            display: none !important;
          }
          .print-card {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            backdrop-filter: none !important;
          }
          .prose-invert {
            color: black !important;
          }
          h1, h2, h3, strong {
            color: black !important;
          }
          p, li {
            color: #334155 !important;
          }
        }
      `}</style>

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
      <main className="flex-1 max-w-5xl w-full mx-auto p-4 md:p-8 space-y-8 print-card">
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
            <div className="no-print lg:col-span-4 space-y-6">
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
            <div className={`lg:col-span-${isRunning || thinkingTrace.length > 0 ? "8" : "12"} print-card`}>
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
                            data={visualization.data} 
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
                              {table.rows.map((row: any, rIdx: number) => (
                                <tr 
                                  key={rIdx} 
                                  className="border-b border-slate-850 hover:bg-slate-900/40 text-slate-400 transition-colors"
                                >
                                  {Object.values(row).map((val: any, cIdx: number) => (
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
    </div>
  )
}
