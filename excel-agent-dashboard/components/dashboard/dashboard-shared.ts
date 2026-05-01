import { Layers, Upload, type LucideIcon } from "lucide-react"
import type { Data, Frame, Layout } from "plotly.js"

export type CellValue = string | number | boolean | null
export type SheetRow = Record<string, CellValue>

export type VisualizationPayload = {
  data?: Data[]
  layout?: Partial<Layout>
  frames?: Frame[]
}

export type QueryTablePayload = {
  id: string
  title: string
  rows: SheetRow[]
}

export type HighlightedColumn = {
  column: string
}

export type ThinkingTraceEntry = {
  kind: "thought" | "action" | "observation"
  content: string
  tool_name?: string
  tool_input?: string
  details?: string
  status?: "completed" | "error"
}

export type TokenUsageSummary = {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export type AgentExecuteResponse = {
  rows?: SheetRow[]
  visualization?: VisualizationPayload | null
  query_output?: string | null
  query_tables?: QueryTablePayload[]
  code?: string
  assistant_reply?: string
  mutation?: boolean
  highlight_indices?: number[]
  highlighted_columns?: HighlightedColumn[]
  thinking_trace?: ThinkingTraceEntry[]
  token_usage?: TokenUsageSummary
  detail?: string
}

export type ChatMessage = {
  role: "user" | "assistant"
  content: string
  code?: string
  query_output?: string | null
  table_links?: Array<{ id: string; title: string }>
  thinkingTrace?: ThinkingTraceEntry[]
  modelLabel?: string
  isStreaming?: boolean
}

export type VizWidget = {
  id: string
  title: string
  data: Data[]
  layout?: Partial<Layout>
  frames?: Frame[]
  x: number
  y: number
  width: number
  height: number
  zIndex: number
}

export type PivotTableTab = {
  id: string
  title: string
  rows: SheetRow[]
  open: boolean
  isNew?: boolean
  insertions?: number
  deletions?: number
}

export type SlashCommandOption = {
  command: string
  description: string
  rewritePrefix: string
}

export type FullscreenPanel = "data" | "canvas" | null

export type WorkflowStep = {
  label: string
  icon: LucideIcon
  isContextButton?: boolean
}

export const seedRows: SheetRow[] = [
  { year: 2000, country: "Kenya", fertility: 4.9, life_expectancy: 52.1, pop_size: 3.7 },
  { year: 2005, country: "Kenya", fertility: 4.6, life_expectancy: 56.7, pop_size: 4.1 },
  { year: 2010, country: "Kenya", fertility: 4.3, life_expectancy: 60.8, pop_size: 4.9 },
  { year: 2000, country: "Japan", fertility: 1.4, life_expectancy: 81.2, pop_size: 3.1 },
  { year: 2005, country: "Japan", fertility: 1.3, life_expectancy: 82.1, pop_size: 2.9 },
  { year: 2010, country: "Japan", fertility: 1.4, life_expectancy: 83.1, pop_size: 3.0 },
  { year: 2000, country: "Brazil", fertility: 2.4, life_expectancy: 70.1, pop_size: 5.9 },
  { year: 2005, country: "Brazil", fertility: 2.1, life_expectancy: 72.4, pop_size: 6.3 },
  { year: 2010, country: "Brazil", fertility: 1.9, life_expectancy: 74.2, pop_size: 6.8 },
]

export const INITIAL_CHAT_MESSAGES: ChatMessage[] = [
  {
    role: "assistant",
    content: "Upload a sheet and ask for transformations or charts. I can generate Plotly 2D and 3D visualizations.",
  },
]

export const transformationSteps: WorkflowStep[] = [
  { label: "Upload", icon: Upload },
  { label: "Context", icon: Layers, isContextButton: true },
]

export const MODEL_OPTIONS = [
  { value: "gemini-3-flash-preview", label: "Gemini 3 Flash" },
  { value: "gemini-3.1-flash-lite-preview", label: "Gemini 3.1 Flash Lite" },
  { value: "models/gemma-4-31b-it", label: "Gemma 4 31B IT" },
  { value: "gpt-4o-mini", label: "GPT-4o Mini (Fast)" },
  { value: "gpt-4o", label: "GPT-4o (GitHub Models)" },
  { value: "minimaxai/minimax-m2.5", label: "Minimax m2.5" },
  { value: "meta/llama-3.1-405b-instruct", label: "Llama 3.1 405B (NVIDIA)" },
] as const

export const DEFAULT_THINKING_MODEL = "models/gemma-4-31b-it" as const

export const THINKING_MODEL_OPTIONS = MODEL_OPTIONS.filter(
  (option) =>
    option.value === DEFAULT_THINKING_MODEL ||
    option.value === "minimaxai/minimax-m2.5"
)

export const SLASH_COMMANDS: SlashCommandOption[] = [
  { command: "/visualize", description: "Build a chart (bar, line, scatter, pie, heatmap).", rewritePrefix: "Create a visualization." },
  { command: "/modify", description: "Update rows/columns in the main dataset.", rewritePrefix: "Modify the dataset." },
  { command: "/extract", description: "Create a separate result table from selected rows/columns.", rewritePrefix: "[FORCE_EXTRACT_TABLE] Create a separate extracted table. Use a new result table instead of answering only in chat." },
  { command: "/filter", description: "Find matching rows without mutating source data.", rewritePrefix: "Filter the dataset to find matching rows." },
  { command: "/summarize", description: "Get concise KPIs, totals, and summary stats.", rewritePrefix: "Summarize the key metrics concisely." },
  { command: "/pivot", description: "Create a pivot/matrix view from categorical dimensions.", rewritePrefix: "Create a pivot-style matrix." },
  { command: "/compare", description: "Compare categories, segments, or time periods.", rewritePrefix: "Compare groups or categories." },
  { command: "/correlate", description: "Analyze numeric relationships/correlation.", rewritePrefix: "Analyze correlation between relevant numeric columns." },
  { command: "/trend", description: "Analyze changes over time.", rewritePrefix: "Analyze the trend over time." },
  { command: "/clean", description: "Standardize values and clean missing/dirty data.", rewritePrefix: "Clean and standardize the data." },
  { command: "/help", description: "Show what this dataset assistant can do.", rewritePrefix: "Show a concise list of things you can do with this dataset." },
]

export const SLASH_ROW_HEIGHT = 32
export const SLASH_ROW_GAP = 4
export const SLASH_DESCRIPTION_DELAY_MS = 1000
export const ROW_HEIGHT = 28
export const OVERSCAN = 5

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(value, max))
}

export function getActiveSlashQuery(text: string): string | null {
  const trimmed = text.trimStart()
  if (!trimmed.startsWith("/")) return null
  const firstToken = trimmed.split(/\s+/, 1)[0]
  if (firstToken.length < 1 || trimmed.includes(" ")) return null
  return firstToken.slice(1).toLowerCase()
}

export function expandSlashPrompt(rawPrompt: string): string {
  const trimmed = rawPrompt.trim()
  if (!trimmed.startsWith("/")) return rawPrompt
  const [commandToken, ...restParts] = trimmed.split(/\s+/)
  const rest = restParts.join(" ").trim()
  const matched = SLASH_COMMANDS.find((item) => item.command === commandToken.toLowerCase())
  if (!matched) return rawPrompt
  return rest ? `${matched.rewritePrefix} ${rest}` : matched.rewritePrefix
}

export function normalizeRows(rows: ReadonlyArray<Record<string, unknown>>): SheetRow[] {
  return rows.map((row) => {
    const normalized: SheetRow = {}
    Object.entries(row).forEach(([key, value]) => {
      if (
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean" ||
        value === null
      ) {
        normalized[key] = value
      } else {
        normalized[key] = value == null ? "" : String(value)
      }
    })
    return normalized
  })
}

export function getNextWidgetZIndex(widgets: Array<Pick<VizWidget, "zIndex">>) {
  return widgets.reduce((max, widget) => Math.max(max, widget.zIndex), 0) + 1
}
