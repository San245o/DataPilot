import type { ThinkingTraceEntry, VisualizationPayload } from "@/components/dashboard/dashboard-shared"

export type AutoReport = {
  generation_mode?: "model" | "fallback"
  dataset: { id: string; name: string; rows: number; columns: number }
  summary: string
  metrics: Array<{ label: string; value: string }>
  sections: Array<{ title: string; content: string; items?: Array<{ headline: string; content: string }> }>
  tables: Array<{ title: string; columns: string[]; rows: Array<Record<string, unknown>>; interpretation?: string }>
  charts: Array<{ title: string; figure: VisualizationPayload; interpretation: string }>
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
  model: string
}
