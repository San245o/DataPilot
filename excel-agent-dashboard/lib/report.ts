import type { ThinkingTraceEntry, VisualizationPayload } from "@/components/dashboard/dashboard-shared"

export type ReportPlan = {
  model: string
  focus: string
  charts: Array<{ id: number; type: string; question: string; method: string; fields: string[] }>
}

export type ReportChart = {
  id: number
  kind: string
  title: string
  figure: VisualizationPayload
  interpretation: string
}

export type AutoReport = {
  generation_mode?: "model" | "fallback"
  dataset: { id: string; name: string; rows: number; columns: number }
  summary: string
  metrics: Array<{ label: string; value: string }>
  sections: Array<{ title: string; content: string; items?: Array<{ headline: string; content: string }> }>
  tables: Array<{ title: string; columns: string[]; rows: Array<Record<string, unknown>>; interpretation?: string }>
  charts: ReportChart[]
  conclusion: string
  recommendations: string[]
  sources: Array<{ title: string; url: string }>
  thinking_trace: ThinkingTraceEntry[]
}

export type ReportTarget = {
  datasetId: string
  displayName: string
  fileName: string
  sheetName?: string
  rowCount: number
  columnCount: number
}
