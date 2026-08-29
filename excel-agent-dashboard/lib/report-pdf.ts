import { jsPDF } from "jspdf"
import autoTable from "jspdf-autotable"
import type { AutoReport, ReportTarget } from "@/lib/report"
import { formatReportValue, safeReportFilename } from "@/lib/report-format"

const PAGE_WIDTH = 210
const PAGE_HEIGHT = 297
const MARGIN = 18
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2

function ensureSpace(doc: jsPDF, y: number, required: number): number {
  if (y + required <= PAGE_HEIGHT - MARGIN) return y
  doc.addPage()
  return MARGIN
}

function addHeading(doc: jsPDF, title: string, y: number, size = 15): number {
  y = ensureSpace(doc, y, 14)
  doc.setFont("helvetica", "bold"); doc.setFontSize(size); doc.setTextColor(24, 36, 48)
  doc.text(title, MARGIN, y)
  return y + 8
}

function addParagraph(doc: jsPDF, text: string, y: number): number {
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(55, 65, 81)
  const paragraphs = text.split(/\n+/).map((item) => item.trim()).filter(Boolean)
  for (const paragraph of paragraphs) {
    const lines = doc.splitTextToSize(paragraph, CONTENT_WIDTH) as string[]
    y = ensureSpace(doc, y, lines.length * 5 + 4)
    doc.text(lines, MARGIN, y, { lineHeightFactor: 1.35 })
    y += lines.length * 5 + 4
  }
  return y
}

async function figureToPng(index: number): Promise<string> {
  const Plotly = (await import("plotly.js-dist-min")).default
  const chart = document.querySelectorAll<HTMLElement>("[data-report-chart] .js-plotly-plot")[index]
  if (!chart) throw new Error(`Report chart ${index + 1} is not ready for export.`)
  return Plotly.toImage(chart, { format: "png", width: 1000, height: 520 })
}

export async function downloadReportPdf(report: AutoReport, target: ReportTarget): Promise<void> {
  const doc = new jsPDF({ unit: "mm", format: "a4", orientation: "portrait" })
  let y = MARGIN
  doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.setTextColor(5, 150, 105); doc.text("DATAPILOT · AUTO REPORT", MARGIN, y)
  y += 9; doc.setFontSize(22); doc.setTextColor(15, 23, 42); doc.text(target.displayName, MARGIN, y)
  y += 7; doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(100, 116, 139)
  doc.text(`${target.sheetName || "Data"} · ${report.dataset.rows.toLocaleString()} rows × ${report.dataset.columns} columns · Generated ${new Date().toLocaleString()}`, MARGIN, y)
  y += 12

  y = addHeading(doc, "Executive Summary", y, 17)
  y = addParagraph(doc, report.summary, y)
  y = addHeading(doc, "Dataset Overview", y)
  const metricWidth = CONTENT_WIDTH / 3
  report.metrics.forEach((metric, index) => {
    if (index > 0 && index % 3 === 0) y += 17
    y = ensureSpace(doc, y, 17)
    const x = MARGIN + (index % 3) * metricWidth
    doc.setDrawColor(226, 232, 240); doc.roundedRect(x, y, metricWidth - 3, 14, 1.5, 1.5)
    doc.setFontSize(8); doc.setTextColor(100, 116, 139); doc.text(metric.label, x + 3, y + 5)
    doc.setFont("helvetica", "bold"); doc.setFontSize(11); doc.setTextColor(5, 150, 105); doc.text(metric.value, x + 3, y + 11)
    doc.setFont("helvetica", "normal")
  })
  y += 23

  for (const section of report.sections) {
    y = addHeading(doc, section.title, y)
    if (section.content) y = addParagraph(doc, section.content, y)
    for (const item of section.items || []) {
      y = addHeading(doc, item.headline, y, 11)
      y = addParagraph(doc, item.content, y)
    }
  }

  for (const table of report.tables) {
    y = addHeading(doc, table.title, y)
    autoTable(doc, {
      startY: y, head: [table.columns], body: table.rows.map((row) => table.columns.map((column) => formatReportValue(row[column]))),
      margin: { left: MARGIN, right: MARGIN }, theme: "grid", pageBreak: "auto", rowPageBreak: "avoid",
      styles: { font: "helvetica", fontSize: 8, cellPadding: 2, textColor: [55, 65, 81], lineColor: [226, 232, 240] },
      headStyles: { fillColor: [241, 245, 249], textColor: [30, 41, 59], fontStyle: "bold" },
    })
    y = ((doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY || y) + 9
    if (table.interpretation) y = addParagraph(doc, table.interpretation, y)
  }

  for (const [chartIndex, chart] of report.charts.entries()) {
    y = ensureSpace(doc, y, 108)
    y = addHeading(doc, chart.title, y)
    const image = await figureToPng(chartIndex)
    doc.addImage(image, "PNG", MARGIN, y, CONTENT_WIDTH, CONTENT_WIDTH * 0.52, undefined, "FAST")
    y += CONTENT_WIDTH * 0.52 + 5
    y = addParagraph(doc, chart.interpretation, y)
  }

  y = addHeading(doc, "Conclusion", y)
  y = addParagraph(doc, report.conclusion, y)
  if (report.recommendations?.length) {
    y = addHeading(doc, "Recommendations", y)
    y = addParagraph(doc, report.recommendations.map((item) => `• ${item}`).join("\n"), y)
  }
  if (report.sources.length) {
    y = addHeading(doc, "Sources", y)
    y = addParagraph(doc, report.sources.map((source) => `${source.title}\n${source.url}`).join("\n\n"), y)
  }
  doc.save(safeReportFilename(target.fileName, target.sheetName))
}
