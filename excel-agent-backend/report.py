from __future__ import annotations

import json
import logging
import re
import threading
import time
from queue import Empty, Queue
from typing import Any, Callable

import pandas as pd

from agent import invoke_model_json
from thinking import run_thinking_agent


TraceCallback = Callable[[dict[str, Any]], None]
logger = logging.getLogger("excel-agent-backend.report")
ANALYSIS_TIMEOUT_SECONDS = 20
SYNTHESIS_TIMEOUT_SECONDS = 30
SYNTHESIS_HEARTBEAT_SECONDS = 5

FORBIDDEN_REPORT_MARKERS = (
    "created the visualization",
    "the tool created",
    "execute_python",
    "dtype:",
    "non_null_count:",
    "null_count:",
    "unique_values:",
)

SYNTHESIS_SYSTEM_PROMPT = """You are the final report writer for DataPilot.
You receive compact evidence calculated by a separate dataset-analysis agent. Write a polished report for a non-technical reader using ONLY that evidence.

Return one JSON object with this exact shape:
{
  "executive_summary": "2-4 meaningful natural-language paragraphs",
  "sections": [{"title": "Data Quality|Key Findings|Patterns and Relationships|other useful title", "content": "short prose", "items": [{"headline": "clear finding", "content": "evidence and interpretation"}]}],
  "table_presentations": [{"index": 0, "title": "human title", "column_labels": {"raw_key": "Readable label"}, "interpretation": "what the table shows"}],
  "chart_presentations": [{"index": 0, "title": "human title", "interpretation": "1-3 evidence-based sentences"}],
  "conclusion": "2-4 substantive paragraphs when evidence supports that depth",
  "recommendations": ["optional evidence-appropriate recommendation"]
}

Rules:
- Translate technical evidence into clear analytical prose; never dump raw key/value profiles or tool observations.
- Never mention tools, Python, execution, schemas, dtypes, internal keys, or that a visualization was created.
- Humanize field names. For example life_expectancy becomes Life expectancy and customer_id becomes Customer ID.
- Round ordinary decimal values to sensible precision, generally no more than two decimals. Use commas and readable percentages.
- Every specific claim must be supported by the supplied calculations, table rows, or chart data. Do not invent values.
- Distinguish association from causation. Never claim causality from descriptive or correlational evidence.
- For small samples, explicitly use appropriately cautious language and avoid statistical overstatement.
- Avoid repeating metric-card values in prose unless they support a meaningful quality note.
- Include only useful sections. Omit recommendations when they are not appropriate.
- When supported, provide several distinct findings covering change over time, group differences, and numerical relationships rather than one generic observation.
- Explain start and end values, absolute or percentage change, comparative rank, and analytical significance when those calculations are available.
- table_presentations selects which evidence tables belong in the report. It may be empty. Never change table values.
- chart_presentations must include every supplied chart index, with a meaningful title and interpretation.
- Conclusion must synthesize findings, never repeat a status message.
- Produce a comprehensive, professional, human-readable analytical report. Do not optimize for excessive concision.
- Return JSON only.
"""


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,}" if isinstance(value, int) else str(value)


def _humanize(value: str) -> str:
    words = re.sub(r"[_\-]+", " ", str(value)).strip().split()
    acronyms = {"id", "url", "api", "sku", "kpi", "usd", "gbp", "eur"}
    return " ".join(word.upper() if word.lower() in acronyms else word.lower() for word in words).capitalize()


def _is_polished_text(value: Any) -> bool:
    text = str(value or "").strip()
    lowered = text.lower()
    return bool(text) and not any(marker in lowered for marker in FORBIDDEN_REPORT_MARKERS)


def _compact_figure(figure: dict[str, Any]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    for trace in (figure.get("data") or [])[:6]:
        compact: dict[str, Any] = {}
        for key in ("type", "name", "x", "y", "labels", "values"):
            value = trace.get(key)
            if isinstance(value, list):
                compact[key] = value[:30]
            elif value is not None:
                compact[key] = value
        traces.append(compact)
    return {"title": (figure.get("layout") or {}).get("title"), "traces": traces}


def _synthesize_report(
    *, model_name: str, dataset_name: str, evidence: dict[str, Any], emit: TraceCallback | None = None,
) -> dict[str, Any]:
    user_message = "\n".join([
        f"Dataset: {dataset_name}",
        "Calculated evidence follows. Technical names are internal and must be translated for readers:",
        json.dumps(evidence, default=str, ensure_ascii=False),
    ])
    result_queue: Queue[tuple[dict[str, Any] | None, Exception | None]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            payload, _usage, _raw = invoke_model_json(
                model_name=model_name,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_message=user_message,
            )
            result_queue.put((payload, None))
        except Exception as exc:
            result_queue.put((None, exc))

    threading.Thread(target=invoke, daemon=True).start()
    deadline = time.monotonic() + SYNTHESIS_TIMEOUT_SECONDS
    progress_messages = (
        "Organizing the key findings into the final report.",
        "Writing the executive summary and analytical sections.",
        "Formatting the tables and visual analysis.",
        "Completing the conclusion and final report checks.",
    )
    heartbeat_index = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Final report writing timed out.")
        try:
            payload, error = result_queue.get(timeout=min(SYNTHESIS_HEARTBEAT_SECONDS, remaining))
            break
        except Empty:
            if emit:
                emit({"kind": "thought", "content": progress_messages[heartbeat_index % len(progress_messages)]})
            heartbeat_index += 1
    if error:
        raise error
    if payload is None:
        raise ValueError("Final synthesis returned no report.")
    if not _is_polished_text(payload.get("executive_summary")) or not _is_polished_text(payload.get("conclusion")):
        raise ValueError("Final synthesis contained internal execution language.")
    return payload


def _fallback_synthesis(*, dataset_name: str, evidence: dict[str, Any]) -> dict[str, Any]:
    dataset = evidence["dataset"]
    profile = evidence.get("column_quality_profile") or []
    missing = sum(int(item.get("Missing") or 0) for item in profile)
    duplicates = next((metric["value"] for metric in evidence.get("overview_metrics", []) if metric["label"] == "Duplicate rows"), "0")
    quality = (
        "The dataset is complete, with no missing values or duplicate records detected."
        if missing == 0 and duplicates == "0"
        else f"The profile identified {missing:,} missing cells and {duplicates} duplicate rows; findings involving affected fields should be interpreted with care."
    )
    safe_notes = [
        str(item.get("analysis_note") or "").strip() for item in evidence.get("analyses") or []
        if _is_polished_text(item.get("analysis_note"))
    ]
    deterministic_findings = evidence.get("deterministic_findings") or []

    def finding_text(item: dict[str, Any]) -> str:
        headline = str(item.get("headline") or "").strip()
        content = str(item.get("content") or "").strip()
        return f"{headline}. {content}" if headline and content else headline or content

    leading_evidence = " ".join(
        finding_text(item) for item in deterministic_findings[:2] if finding_text(item)
    )
    relationship_findings = [
        item for item in deterministic_findings
        if "relationship" in str(item.get("headline") or "").lower()
    ]
    trend_findings = [item for item in deterministic_findings if item not in relationship_findings]
    relationship_evidence = " ".join(finding_text(item) for item in relationship_findings[:1])

    def chart_interpretation(chart: dict[str, Any]) -> str:
        raw_title = chart.get("title")
        title = str(raw_title.get("text") if isinstance(raw_title, dict) else raw_title or "").lower()
        matching = next(
            (item for item in deterministic_findings if str(item.get("headline") or "").lower().split(" changed")[0] in title),
            None,
        )
        return finding_text(matching) if matching else leading_evidence or "This visualization summarizes calculated evidence from the active dataset."
    summary = (
        f"This report examines {dataset_name}, containing {dataset['rows']:,} records across {dataset['columns']} fields. "
        "The analysis covers data quality, group differences, changes over the observed period, and numerical relationships supported by the active dataset.\n\n"
        f"{quality}"
        + (f"\n\n{leading_evidence}" if leading_evidence else "")
        + (f"\n\n{relationship_evidence}" if relationship_evidence else "")
    )
    sections: list[dict[str, Any]] = [{"title": "Data Quality", "content": quality, "items": []}]
    if trend_findings or safe_notes:
        sections.append({"title": "Trend and Comparative Analysis", "content": safe_notes[0] if safe_notes else "", "items": trend_findings})
    if relationship_findings:
        sections.append({"title": "Patterns and Relationships", "content": "These relationships describe the observed data and do not establish causation.", "items": relationship_findings})
    return {
        "executive_summary": summary,
        "sections": sections,
        "table_presentations": [
            {"index": index, "title": str(table.get("title") or "Analysis Table"), "column_labels": {}, "interpretation": str(table.get("interpretation") or leading_evidence or "This table summarizes calculated comparisons from the active dataset.")}
            for index, table in enumerate(evidence.get("available_tables") or [])
        ],
        "chart_presentations": [
            {
                "index": index,
                "title": str(
                    (chart.get("title") or {}).get("text")
                    if isinstance(chart.get("title"), dict)
                    else chart.get("title") or "Report Visualization"
                ),
                "interpretation": chart_interpretation(chart),
            }
            for index, chart in enumerate(evidence.get("available_charts") or [])
        ],
        "conclusion": (
            "The strongest trends and group comparisons are summarized in the findings above. "
            + leading_evidence
            + "\n\n"
            + (relationship_evidence + " " if relationship_evidence else "")
            + "These results are descriptive; numerical associations should not be treated as causal, and the dataset's size and quality should remain part of any decision based on them."
        ) if leading_evidence else "The available evidence provides a descriptive view of the dataset. Review the calculated comparisons above while keeping the dataset size and any identified quality limitations in mind.",
        "recommendations": [],
    }


def _deterministic_evidence(frame: pd.DataFrame) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    findings: list[dict[str, str]] = []
    tables: list[dict[str, Any]] = []
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric and frame[column].nunique(dropna=True) <= 20]
    time_columns = [column for column in frame.columns if any(token in str(column).lower() for token in ("year", "date", "time"))]

    if time_columns and categorical and numeric:
        time_column = time_columns[0]
        group_column = categorical[0]
        ordered = frame.copy()
        ordered[time_column] = pd.to_numeric(ordered[time_column], errors="coerce")
        ordered = ordered.dropna(subset=[time_column]).sort_values(time_column)
        change_rows: list[dict[str, Any]] = []
        for group, group_frame in ordered.groupby(group_column, dropna=False):
            for value_column in [column for column in numeric if column != time_column][:3]:
                values = pd.to_numeric(group_frame[value_column], errors="coerce")
                valid = group_frame.loc[values.notna()]
                if len(valid) < 2:
                    continue
                start = float(valid[value_column].iloc[0])
                end = float(valid[value_column].iloc[-1])
                change_rows.append({
                    _humanize(group_column): group,
                    "Measure": _humanize(value_column),
                    "Start": round(start, 2),
                    "End": round(end, 2),
                    "Change": round(end - start, 2),
                })
        if change_rows:
            tables.append({"title": "Change over the observed period", "columns": list(change_rows[0]), "rows": change_rows, "interpretation": "This comparison shows how each measured value changed from its first to last observation within each group."})
            for measure in dict.fromkeys(row["Measure"] for row in change_rows):
                strongest = max(
                    (row for row in change_rows if row["Measure"] == measure),
                    key=lambda row: abs(float(row["Change"])),
                )
                direction = "increased" if strongest["Change"] >= 0 else "decreased"
                percent_change = abs(float(strongest["Change"]) / float(strongest["Start"]) * 100) if strongest["Start"] else None
                percentage_text = f", or {percent_change:.1f}%" if percent_change is not None else ""
                findings.append({
                    "headline": f"{measure} changed most for {strongest[_humanize(group_column)]}",
                    "content": f"It {direction} from {_display(float(strongest['Start']))} to {_display(float(strongest['End']))}, an absolute change of {_display(abs(float(strongest['Change'])))}{percentage_text} over the observed period. This was the largest absolute change for this measure among the represented groups.",
                })

    if not findings:
        for column in [item for item in numeric if item not in time_columns][:3]:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if series.empty:
                continue
            findings.append({
                "headline": f"{_humanize(column)} ranges from {_display(float(series.min()))} to {_display(float(series.max()))}",
                "content": f"Across {len(series):,} recorded values, the average {_humanize(column).lower()} is {_display(float(series.mean()))}. This describes the observed sample without implying values beyond it.",
            })

    relationship_columns = [column for column in numeric if column not in time_columns]
    if len(relationship_columns) >= 2 and len(frame) >= 3:
        correlations = frame[relationship_columns].corr(numeric_only=True)
        pairs = [
            (left, right, float(correlations.loc[left, right]))
            for index, left in enumerate(relationship_columns)
            for right in relationship_columns[index + 1:]
            if pd.notna(correlations.loc[left, right])
        ]
        if pairs:
            left, right, correlation = max(pairs, key=lambda item: abs(item[2]))
            direction = "positive" if correlation >= 0 else "inverse"
            findings.append({
                "headline": f"{_humanize(left)} and {_humanize(right)} show an {direction} relationship",
                "content": f"Their correlation within this dataset is {correlation:.2f}. This indicates association in the observed sample, not causation, and should be interpreted cautiously when the sample is small.",
            })
    return findings[:6], tables


def _deterministic_chart(frame: pd.DataFrame) -> dict[str, Any] | None:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    time_columns = [column for column in frame.columns if any(token in str(column).lower() for token in ("year", "date", "time"))]
    groups = [column for column in frame.columns if column not in numeric and frame[column].nunique(dropna=True) <= 12]
    values = [column for column in numeric if column not in time_columns]
    if not time_columns or not groups or not values:
        return None
    value_column = next((column for column in values if any(token in str(column).lower() for token in ("life", "revenue", "sales", "value"))), values[0])
    time_column, group_column = time_columns[0], groups[0]
    traces = []
    for group, subset in frame.sort_values(time_column).groupby(group_column, dropna=False):
        traces.append({
            "type": "scatter", "mode": "lines+markers", "name": str(group),
            "x": subset[time_column].tolist(), "y": subset[value_column].tolist(),
        })
    return {
        "title": f"{_humanize(value_column)} over {_humanize(time_column).lower()}",
        "figure": {
            "data": traces,
            "layout": {
                "title": {"text": f"{_humanize(value_column)} by {_humanize(group_column)}"},
                "xaxis": {"title": {"text": _humanize(time_column)}},
                "yaxis": {"title": {"text": _humanize(value_column)}},
            },
        },
        "interpretation": "",
    }


def _profile(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    date_columns = [
        column for column in frame.columns
        if any(token in str(column).lower() for token in ("date", "time", "year", "month"))
        or (
            column in categorical and len(frame) > 0
            and pd.to_datetime(frame[column], errors="coerce", format="mixed").notna().mean() >= 0.8
        )
    ]
    missing = frame.isna().sum()
    missing_total = int(missing.sum())
    metrics = [
        {"label": "Rows", "value": _display(len(frame))},
        {"label": "Columns", "value": _display(len(frame.columns))},
        {"label": "Numeric fields", "value": _display(len(numeric))},
        {"label": "Categorical fields", "value": _display(len(categorical))},
        {"label": "Date/time fields", "value": _display(len(date_columns))},
        {"label": "Missing cells", "value": _display(missing_total)},
        {"label": "Duplicate rows", "value": _display(int(frame.duplicated().sum()))},
    ]
    quality_rows = [
        {
            "Column": str(column),
            "Type": str(frame[column].dtype),
            "Missing": int(missing[column]),
            "Missing %": round(float(missing[column]) * 100 / max(len(frame), 1), 2),
            "Unique": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    ]
    return metrics, quality_rows


def _run_analysis(
    *, prompt: str, rows: list[dict[str, Any]], dataset_id: str, dataset_name: str,
    model_name: str, emit: TraceCallback | None,
) -> dict[str, Any]:
    return run_thinking_agent(
        prompt=prompt,
        rows=rows,
        model_name=model_name,
        history=[],
        datasets={dataset_id: {"name": dataset_name, "rows": rows, "kind": "uploaded"}},
        active_dataset_id=dataset_id,
        selected_dataset_ids=[dataset_id],
        dataset_names={dataset_id: dataset_name},
        event_callback=emit,
    )


def _run_analysis_bounded(**kwargs: Any) -> dict[str, Any]:
    result_queue: Queue[tuple[dict[str, Any] | None, BaseException | None]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put((_run_analysis(**kwargs), None))
        except BaseException as exc:
            result_queue.put((None, exc))

    threading.Thread(target=invoke, daemon=True).start()
    try:
        payload, error = result_queue.get(timeout=ANALYSIS_TIMEOUT_SECONDS)
    except Empty as exc:
        raise TimeoutError("Thinking analysis timed out; deterministic evidence will be used.") from exc
    if error:
        raise error
    if payload is None:
        raise ValueError("Thinking analysis returned no evidence.")
    return payload


def generate_auto_report(
    *, dataset_id: str, dataset_name: str, rows: list[dict[str, Any]], model_name: str,
    event_callback: TraceCallback | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("The active dataset is empty.")

    traces: list[dict[str, Any]] = []

    def emit(entry: dict[str, Any]) -> None:
        traces.append(entry)
        if event_callback:
            event_callback(entry)

    emit({"kind": "thought", "content": "Inspecting the dataset structure and calculating deterministic quality checks."})
    logger.info("report metrics started dataset=%s rows=%s", dataset_id, len(rows))
    metrics, _quality_profile = _profile(rows)
    deterministic_findings, deterministic_tables = _deterministic_evidence(pd.DataFrame(rows))
    logger.info("report metrics completed dataset=%s", dataset_id)
    missing_cells = next((metric["value"] for metric in metrics if metric["label"] == "Missing cells"), "0")
    duplicate_rows = next((metric["value"] for metric in metrics if metric["label"] == "Duplicate rows"), "0")
    emit({
        "kind": "observation",
        "content": f"Dataset checks completed: {len(rows):,} rows, {len(pd.DataFrame(rows).columns)} columns, {missing_cells} missing cells, and {duplicate_rows} duplicate rows.",
        "status": "completed",
    })
    # Retain deterministic profiling as internal evidence. Displayed tables are
    # added only when Thinking Mode decides they improve the report.
    tables: list[dict[str, Any]] = list(deterministic_tables)
    charts: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []

    prompts = [(
        "analysis",
        "Perform one bounded, evidence-focused analysis for a professional Auto Report. Use the active dataset only; never use "
        "web_search and never mutate data. In one execute_python step where practical, calculate the most useful supported evidence: "
        "data quality, descriptive metrics, category/group comparisons, rankings, and—when ordered or date-like data exists—start/end "
        "values, absolute and percentage changes, strongest increase/decrease, and stability. Investigate useful numerical relationships "
        "without implying causation and use cautious language for small samples. Log one polished evidence table when it adds value. "
        "If one Plotly chart materially improves understanding, assign it to fig with readable axis labels and a meaningful title; otherwise "
        "do not create a chart. Preserve actual calculated values for final synthesis."
    )]

    analysis_evidence: list[dict[str, Any]] = []
    for title, prompt in prompts:
        emit({"kind": "thought", "content": "Comparing groups, trends, and numerical relationships in the active dataset."})
        logger.info("report thinking started dataset=%s stage=%s", dataset_id, title)
        try:
            result = _run_analysis_bounded(
                prompt=prompt, rows=rows, dataset_id=dataset_id, dataset_name=dataset_name,
                model_name=model_name, emit=emit,
            )
        except Exception as exc:
            logger.warning("report thinking failed dataset=%s stage=%s error=%s", dataset_id, title, type(exc).__name__)
            emit({"kind": "observation", "content": "The extended analysis took too long, so the report will continue with the calculated deterministic evidence.", "status": "error"})
            continue
        logger.info("report thinking completed dataset=%s stage=%s", dataset_id, title)

        answer = str(result.get("assistant_reply") or "").strip()
        evidence_tables: list[dict[str, Any]] = []
        for table in result.get("query_tables") or []:
            table_rows = (table.get("rows") or [])[:50]
            if table_rows:
                evidence_table = {
                    "title": table.get("title") or "Analysis Table",
                    "columns": list(table_rows[0].keys()),
                    "rows": table_rows,
                }
                tables.append(evidence_table)
                evidence_tables.append(evidence_table)
        figure = result.get("visualization")
        if figure and figure.get("data"):
            layout = figure.get("layout") or {}
            raw_title = layout.get("title")
            chart_title = raw_title.get("text") if isinstance(raw_title, dict) else raw_title
            charts.append({
                "title": str(chart_title or "Report Visualization"),
                "figure": figure,
                "interpretation": "",
            })
        analysis_evidence.append({
            "purpose": title,
            "calculated_output": str(result.get("query_output") or "")[:3000],
            "evidence_tables": evidence_tables,
            "analysis_note": answer[:1800],
        })
        for source in result.get("sources") or []:
            if source not in sources:
                sources.append(source)

    if not charts:
        emit({"kind": "thought", "content": "Checking whether a visualization would make the strongest pattern easier to understand."})
        deterministic_chart = _deterministic_chart(pd.DataFrame(rows))
        if deterministic_chart:
            charts.append(deterministic_chart)
            emit({"kind": "observation", "content": "A trend visualization was prepared from the active dataset.", "status": "completed"})

    synthesis_evidence = {
        "dataset": {"name": dataset_name, "rows": len(rows), "columns": len(pd.DataFrame(rows).columns)},
        "overview_metrics": metrics,
        "column_quality_profile": _quality_profile,
        "deterministic_findings": deterministic_findings,
        "analyses": analysis_evidence,
        "available_tables": tables,
        "available_charts": [_compact_figure(chart["figure"]) for chart in charts],
    }
    emit({"kind": "thought", "content": "The analytical evidence is ready. I’m preparing a comprehensive professional report."})
    logger.info("report synthesis started dataset=%s", dataset_id)
    generation_mode = "model"
    try:
        synthesis = _synthesize_report(
            model_name=model_name, dataset_name=dataset_name, evidence=synthesis_evidence, emit=emit,
        )
    except Exception as exc:
        emit({"kind": "observation", "content": "The narrative synthesis took too long, so the final report was completed from the verified deterministic evidence.", "status": "error"})
        logger.warning("report synthesis fallback dataset=%s error=%s", dataset_id, type(exc).__name__)
        synthesis = _fallback_synthesis(dataset_name=dataset_name, evidence=synthesis_evidence)
        generation_mode = "fallback"
    else:
        logger.info("report synthesis completed dataset=%s", dataset_id)

    selected_tables: list[dict[str, Any]] = []
    for presentation in synthesis.get("table_presentations") or []:
        index = presentation.get("index")
        if not isinstance(index, int) or not 0 <= index < len(tables):
            continue
        original = tables[index]
        labels = presentation.get("column_labels") if isinstance(presentation.get("column_labels"), dict) else {}
        selected_tables.append({
            "title": str(presentation.get("title") or original["title"]),
            "columns": [str(labels.get(column) or _humanize(column)) for column in original["columns"]],
            "rows": [
                {str(labels.get(column) or _humanize(column)): row.get(column) for column in original["columns"]}
                for row in original["rows"]
            ],
            "interpretation": str(presentation.get("interpretation") or original.get("interpretation") or ""),
        })

    chart_presentations = {
        item.get("index"): item for item in synthesis.get("chart_presentations") or []
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    final_charts: list[dict[str, Any]] = []
    for index, chart in enumerate(charts):
        presentation = chart_presentations.get(index)
        if not presentation or not _is_polished_text(presentation.get("interpretation")):
            continue
        final_charts.append({
            **chart,
            "title": str(presentation.get("title") or chart["title"]),
            "interpretation": str(presentation["interpretation"]),
        })

    final_sections: list[dict[str, Any]] = []
    for section in synthesis.get("sections") or []:
        if not isinstance(section, dict):
            continue
        content = str(section.get("content") or "").strip()
        items = [
            {"headline": str(item.get("headline") or "").strip(), "content": str(item.get("content") or "").strip()}
            for item in (section.get("items") or []) if isinstance(item, dict)
            and _is_polished_text(item.get("headline")) and _is_polished_text(item.get("content"))
        ]
        if _is_polished_text(content) or items:
            final_sections.append({"title": str(section.get("title") or "Analysis"), "content": content, "items": items})

    final_report = {
        "dataset": {"id": dataset_id, "name": dataset_name, "rows": len(rows), "columns": len(pd.DataFrame(rows).columns)},
        "summary": str(synthesis["executive_summary"]),
        "metrics": metrics,
        "sections": final_sections,
        "tables": selected_tables,
        "charts": final_charts,
        "conclusion": str(synthesis["conclusion"]),
        "recommendations": [str(item) for item in (synthesis.get("recommendations") or []) if _is_polished_text(item)],
        "sources": sources,
        "thinking_trace": traces,
        "generation_mode": generation_mode,
    }
    logger.info(
        "report completed dataset=%s sections=%s tables=%s charts=%s",
        dataset_id, len(final_sections), len(selected_tables), len(final_charts),
    )
    return final_report
