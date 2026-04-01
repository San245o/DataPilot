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
  const gridColor = isDark ? "rgba(148,163,184,0.06)" : "rgba(0,0,0,0.035)"
  const bgColor = "rgba(0,0,0,0)"
  const sceneLayout = (layout?.scene ?? {}) as Partial<Layout["scene"]>

  const mergedLayout: Partial<Layout> = {
    ...layout,
    autosize: true,
    paper_bgcolor: bgColor,
    plot_bgcolor: bgColor,
    margin: { l: 36, r: 14, t: 24, b: 34 },
    font: { size: 12, color: fontColor, family: "Manrope, system-ui, sans-serif" },
    xaxis: {
      ...(layout?.xaxis ?? {}),
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      automargin: true,
    },
    yaxis: {
      ...(layout?.yaxis ?? {}),
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      automargin: true,
    },
    scene: {
      ...sceneLayout,
      xaxis: {
        ...(sceneLayout?.xaxis ?? {}),
        gridcolor: gridColor,
        zerolinecolor: gridColor,
      },
      yaxis: {
        ...(sceneLayout?.yaxis ?? {}),
        gridcolor: gridColor,
        zerolinecolor: gridColor,
      },
      zaxis: {
        ...(sceneLayout?.zaxis ?? {}),
        gridcolor: gridColor,
        zerolinecolor: gridColor,
      },
    },
    legend: {
      ...(layout?.legend ?? {}),
      bgcolor: bgColor,
      font: { color: fontColor, size: 11 },
    },
    hoverlabel: {
      ...(layout?.hoverlabel ?? {}),
      bgcolor: isDark ? "#1e293b" : "#f6ede1",
      bordercolor: isDark ? "#334155" : "#d8c8af",
      font: { color: isDark ? "#e2e8f0" : "#1e293b", size: 12 },
    },
  }

  return (
    <Plot
      data={data}
      layout={mergedLayout}
      config={{
        displaylogo: false,
        responsive: true,
        scrollZoom: true,
        doubleClick: "reset+autosize",
        displayModeBar: "hover",
        modeBarButtonsToAdd: [
          "zoom2d",
          "pan2d",
          "zoomIn2d",
          "zoomOut2d",
          "autoScale2d",
          "resetScale2d",
          "zoom3d",
          "pan3d",
          "orbitRotation",
          "tableRotation",
          "resetCameraDefault3d",
          "resetCameraLastSave3d",
          "hoverClosest3d",
        ],
        modeBarButtonsToRemove: ["lasso2d", "select2d", "sendDataToCloud", "toggleSpikelines"],
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
