export function humanizeFieldLabel(value: string): string {
  const acronyms = new Set(["id", "url", "api", "sku", "kpi", "usd", "gbp", "eur"])
  const words = value.replace(/[_-]+/g, " ").trim().split(/\s+/).filter(Boolean)
  return words.map((word, index) => {
    const normalized = acronyms.has(word.toLowerCase()) ? word.toUpperCase() : word.toLowerCase()
    return index === 0 && !acronyms.has(word.toLowerCase())
      ? normalized.charAt(0).toUpperCase() + normalized.slice(1)
      : normalized
  }).join(" ")
}

export function formatReportValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—"
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value)
  const magnitude = Math.abs(value)
  if (magnitude >= 1_000_000) return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value)
  if (Number.isInteger(value)) return new Intl.NumberFormat("en").format(value)
  return new Intl.NumberFormat("en", { maximumFractionDigits: magnitude < 1 ? 3 : 2 }).format(value)
}

export function safeReportFilename(fileName: string, sheetName?: string): string {
  const base = fileName.replace(/\.[^/.]+$/, "")
  const sheet = sheetName && sheetName.toLowerCase() !== "data" ? `-${sheetName}` : ""
  const normalized = `${base}${sheet}-auto-report`
    .normalize("NFKD").replace(/[<>:"/\\|?*\u0000-\u001F]/g, "-")
    .replace(/\s+/g, "-").replace(/-+/g, "-").replace(/^[-.]+|[-.]+$/g, "")
  return `${normalized || "datapilot-auto-report"}.pdf`
}
