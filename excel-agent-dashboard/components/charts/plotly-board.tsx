"use client"

import dynamic from "next/dynamic"
import type { Data, Layout } from "plotly.js"

const Plot = dynamic(async () => {
  const createPlotlyComponent = (await import("react-plotly.js/factory")).default
  const Plotly = (await import("plotly.js-dist-min")).default
  return createPlotlyComponent(Plotly)
}, { ssr: false })

type PlotlyBoardProps = {
  data: Data[]
  layout?: Partial<Layout>
  isDark?: boolean
}

export function PlotlyBoard({ data, layout, isDark = true }: PlotlyBoardProps) {
  const fontColor = isDark ? "#94a3b8" : "#334155"
  const gridColor = isDark ? "rgba(148,163,184,0.12)" : "rgba(0,0,0,0.08)"
  const bgColor = "rgba(0,0,0,0)"

  const mergedLayout: Partial<Layout> = {
    autosize: true,
    paper_bgcolor: bgColor,
    plot_bgcolor: bgColor,
    margin: { l: 48, r: 24, t: 32, b: 48 },
    font: { size: 12, color: fontColor, family: "Manrope, system-ui, sans-serif" },
    xaxis: {
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      automargin: true,
    },
    yaxis: {
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      automargin: true,
    },
    legend: {
      bgcolor: bgColor,
      font: { color: fontColor, size: 11 },
    },
    hoverlabel: {
      bgcolor: isDark ? "#1e293b" : "#fff",
      bordercolor: isDark ? "#334155" : "#e2e8f0",
      font: { color: isDark ? "#e2e8f0" : "#1e293b", size: 12 },
    },
    ...layout,
  }

  return (
    <Plot
      data={data}
      layout={mergedLayout}
      config={{
        displaylogo: false,
        responsive: true,
        displayModeBar: "hover",
        modeBarButtonsToRemove: ["lasso2d", "select2d", "sendDataToCloud"],
        toImageButtonOptions: {
          format: "png",
          filename: "data-pilot-chart",
          scale: 2,
        },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  )
}
