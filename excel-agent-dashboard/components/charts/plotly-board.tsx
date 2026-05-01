"use client"

import dynamic from "next/dynamic"
import type { Data, Frame, Layout } from "plotly.js"

const Plot = dynamic(async () => {
  const createPlotlyComponent = (await import("react-plotly.js/factory")).default
  const Plotly = (await import("plotly.js-dist-min")).default
  return createPlotlyComponent(Plotly)
}, { ssr: false })

type PlotlyBoardProps = {
  data: Data[]
  layout?: Partial<Layout>
  frames?: Frame[]
  isDark?: boolean
}

export function PlotlyBoard({ data, layout, frames, isDark = true }: PlotlyBoardProps) {
  const fontColor = isDark ? "#aab8cb" : "#334155"
  const titleColor = isDark ? "#e6edf5" : "#1f2d3d"
  const gridColor = isDark ? "rgba(148,163,184,0.08)" : "rgba(0,0,0,0.05)"
  const bgColor = "rgba(0,0,0,0)"
  const colorway = isDark
    ? ["#11c5a5", "#5f83ff", "#ff8a5b", "#8cdb72", "#c18cff", "#ffd166"]
    : ["#0f9f86", "#4763ff", "#f26b43", "#5fa83b", "#9757e8", "#d7a21f"]
  const sceneLayout = (layout?.scene ?? {}) as Partial<Layout["scene"]>

  const mergedLayout: Partial<Layout> = {
    ...layout,
    autosize: true,
    paper_bgcolor: bgColor,
    plot_bgcolor: bgColor,
    colorway,
    margin: { l: 42, r: 18, t: 52, b: 44 },
    font: { size: 12, color: fontColor, family: "Manrope, system-ui, sans-serif" },
    title: layout?.title
      ? {
          ...(typeof layout.title === "object" ? layout.title : { text: String(layout.title) }),
          x: 0.03,
          xanchor: "left",
          font: { color: titleColor, size: 22, family: "Manrope, system-ui, sans-serif" },
        }
      : undefined,
    xaxis: {
      ...(layout?.xaxis ?? {}),
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      automargin: true,
      tickfont: { color: fontColor, size: 12 },
      title: layout?.xaxis?.title
        ? {
            ...(typeof layout.xaxis.title === "object" ? layout.xaxis.title : { text: String(layout.xaxis.title) }),
            font: { color: fontColor, size: 13, family: "Manrope, system-ui, sans-serif" },
          }
        : undefined,
    },
    yaxis: {
      ...(layout?.yaxis ?? {}),
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      automargin: true,
      tickfont: { color: fontColor, size: 12 },
      title: layout?.yaxis?.title
        ? {
            ...(typeof layout.yaxis.title === "object" ? layout.yaxis.title : { text: String(layout.yaxis.title) }),
            font: { color: fontColor, size: 13, family: "Manrope, system-ui, sans-serif" },
          }
        : undefined,
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
      borderwidth: 0,
      font: { color: fontColor, size: 12, family: "Manrope, system-ui, sans-serif" },
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
      frames={frames}
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
