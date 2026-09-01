from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import patch

from report import FORBIDDEN_REPORT_MARKERS, _is_polished_text, _run_analysis_bounded, _synthesize_report, generate_auto_report
from main import _cache_dataset, execute_auto_report_stream
from schemas import ReportRequest
from thinking import ToolExecution, run_thinking_agent


GAPMINDER_ROWS = [
    {"year": 2000, "country": "Kenya", "fertility": 4.9, "life_expectancy": 52.1, "pop_size": 3.7},
    {"year": 2005, "country": "Kenya", "fertility": 4.6, "life_expectancy": 56.7, "pop_size": 4.1},
    {"year": 2010, "country": "Kenya", "fertility": 4.3, "life_expectancy": 60.8, "pop_size": 4.9},
    {"year": 2000, "country": "Japan", "fertility": 1.4, "life_expectancy": 81.2, "pop_size": 3.1},
    {"year": 2005, "country": "Japan", "fertility": 1.3, "life_expectancy": 82.1, "pop_size": 2.9},
    {"year": 2010, "country": "Japan", "fertility": 1.4, "life_expectancy": 83.1, "pop_size": 3.0},
    {"year": 2000, "country": "Brazil", "fertility": 2.4, "life_expectancy": 70.1, "pop_size": 5.9},
    {"year": 2005, "country": "Brazil", "fertility": 2.1, "life_expectancy": 72.4, "pop_size": 6.3},
    {"year": 2010, "country": "Brazil", "fertility": 1.9, "life_expectancy": 74.2, "pop_size": 6.8},
]


class AutoReportSynthesisTests(unittest.TestCase):
    @staticmethod
    def _stream_events(response) -> list[dict]:
        async def collect() -> str:
            chunks: list[str] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        return [json.loads(line) for line in asyncio.run(collect()).splitlines() if line.strip()]

    def test_gapminder_tool_text_never_becomes_report_prose(self) -> None:
        base_result = {
            "assistant_reply": "Created the visualization.",
            "query_output": "life_exp_avg: 56.533333333333333; dtype: float64",
            "query_tables": [{"title": "raw", "rows": [{"life_exp_avg": 56.533333333333333}]}],
            "visualization": None,
            "sources": [],
        }
        chart_result = {**base_result, "query_tables": [], "visualization": {"data": [{"type": "scatter", "x": [4.9, 1.4], "y": [52.1, 81.2]}], "layout": {}}}
        synthesis = {
            "executive_summary": "This small dataset tracks fertility, life expectancy and population for three countries between 2000 and 2010.",
            "sections": [{"title": "Key Findings", "content": "", "items": [{"headline": "Life expectancy improved", "content": "Kenya's life expectancy rose by 8.7 years over the observed decade."}]}],
            "table_presentations": [{"index": 1, "title": "Country comparison", "column_labels": {"life_exp_avg": "Average life expectancy"}}],
            "chart_presentations": [{"index": 0, "title": "Fertility and life expectancy", "interpretation": "Within this small sample, higher fertility is associated with lower life expectancy."}],
            "conclusion": "Life expectancy rose over time while fertility generally declined across the observed countries.",
            "recommendations": [],
        }
        chart_result["query_tables"] = base_result["query_tables"]
        with patch("report._run_analysis", return_value=chart_result), patch("report._synthesize_report", return_value=synthesis):
            generated = generate_auto_report(dataset_id="gap", dataset_name="gapminder-lite.xlsx / Data", rows=GAPMINDER_ROWS, model_name="test")

        visible = json.dumps({
            "summary": generated["summary"], "sections": generated["sections"], "tables": generated["tables"],
            "charts": [{"title": chart["title"], "interpretation": chart["interpretation"]} for chart in generated["charts"]],
            "conclusion": generated["conclusion"],
        }).lower()
        for marker in FORBIDDEN_REPORT_MARKERS:
            self.assertNotIn(marker, visible)
        self.assertEqual(generated["tables"][0]["columns"], ["Average life expectancy"])
        self.assertIn("associated", generated["charts"][0]["interpretation"])

    def test_leak_guard_rejects_tool_status(self) -> None:
        self.assertFalse(_is_polished_text("The tool created the requested visualization."))
        self.assertFalse(_is_polished_text("execute_python"))
        self.assertTrue(_is_polished_text("The dataset contains no duplicate records."))

    def test_synthesis_failure_returns_substantive_deterministic_report(self) -> None:
        raw_result = {"assistant_reply": "Created the visualization.", "query_output": "", "query_tables": [], "visualization": None, "sources": []}
        with patch("report._run_analysis", return_value=raw_result), patch("report._synthesize_report", side_effect=TimeoutError("timed out")):
            generated = generate_auto_report(dataset_id="gap", dataset_name="gapminder-lite.xlsx / Data", rows=GAPMINDER_ROWS, model_name="test")
        findings = [item for section in generated["sections"] for item in section.get("items") or []]
        self.assertGreaterEqual(len(findings), 3)
        self.assertTrue(generated["tables"])
        self.assertTrue(generated["charts"])
        self.assertIn("small", json.dumps(findings).lower())
        self.assertEqual(generated["generation_mode"], "fallback")

    def test_model_synthesis_emits_a_normal_final_event(self) -> None:
        synthesis = {
            "executive_summary": "The dataset covers three groups over an observed period.\n\nThe calculated evidence shows meaningful changes and comparisons.",
            "sections": [{"title": "Key Findings", "content": "", "items": []}],
            "table_presentations": [], "chart_presentations": [],
            "conclusion": "The observed changes provide a supported descriptive summary of this small dataset.",
            "recommendations": [],
        }
        dataset_id = _cache_dataset(GAPMINDER_ROWS, "gapminder-lite.xlsx / Data", user_id="test-user")
        with patch("report._run_analysis", return_value={"assistant_reply": "Done.", "query_output": "", "query_tables": [], "visualization": None, "sources": []}), patch("report._synthesize_report", return_value=synthesis):
            response = execute_auto_report_stream(
                ReportRequest(dataset_id=dataset_id, model="models/gemma-4-31b-it"),
                current_user={"user_id": "test-user"},
            )
            events = self._stream_events(response)
        self.assertEqual(events[-1]["type"], "final")
        self.assertFalse(any(event["type"] == "error" for event in events))
        self.assertEqual(events[-1]["payload"]["generation_mode"], "model")

    def test_synthesis_timeout_emits_fallback_as_a_normal_final_event(self) -> None:
        dataset_id = _cache_dataset(GAPMINDER_ROWS, "gapminder-lite.xlsx / Data", user_id="test-user")
        with patch("report._run_analysis", return_value={"assistant_reply": "Done.", "query_output": "", "query_tables": [], "visualization": None, "sources": []}), patch("report._synthesize_report", side_effect=TimeoutError("timed out")):
            response = execute_auto_report_stream(
                ReportRequest(dataset_id=dataset_id, model="models/gemma-4-31b-it"),
                current_user={"user_id": "test-user"},
            )
            events = self._stream_events(response)
        self.assertEqual(events[-1]["type"], "final")
        self.assertFalse(any(event["type"] == "error" for event in events))
        self.assertEqual(events[-1]["payload"]["generation_mode"], "fallback")
        self.assertGreaterEqual(sum(len(section.get("items") or []) for section in events[-1]["payload"]["sections"]), 3)

    def test_thinking_analysis_has_an_orchestration_timeout(self) -> None:
        with patch("report.ANALYSIS_TIMEOUT_SECONDS", 0.01), patch("report._run_analysis", side_effect=lambda **_: time.sleep(0.1)):
            with self.assertRaisesRegex(TimeoutError, "deterministic evidence"):
                _run_analysis_bounded(
                    prompt="analyze", rows=GAPMINDER_ROWS, dataset_id="gap",
                    dataset_name="gapminder-lite.xlsx / Data", model_name="test", emit=None,
                )

    def test_synthesis_emits_safe_heartbeats_until_its_deadline(self) -> None:
        events: list[dict] = []
        with patch("report.SYNTHESIS_TIMEOUT_SECONDS", 0.03), patch("report.SYNTHESIS_HEARTBEAT_SECONDS", 0.005), patch("report.invoke_model_json", side_effect=lambda **_: time.sleep(0.1)):
            with self.assertRaises(TimeoutError):
                _synthesize_report(model_name="test", dataset_name="Data", evidence={"dataset": {}}, emit=events.append)
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(all(event["kind"] == "thought" for event in events))

    def test_tool_action_streams_before_its_observation(self) -> None:
        events: list[dict] = []

        def execute_schema(*_args, **_kwargs):
            self.assertEqual(events[-1]["kind"], "action")
            return ToolExecution(
                rows=GAPMINDER_ROWS, visualization=None, query_output=None, query_table_rows=None,
                mutation=False, highlight_indices=[], highlighted_columns=[], observation="Schema checked.",
                raw_observation="5 columns", code="inspect_schema(sample_rows=2)",
            )

        plan = ({
            "kind": "plan", "thought": "I will inspect the available fields.",
            "steps": [{"tool": "inspect_schema", "args": {"sample_rows": 2}}],
            "final_answer": "The schema was inspected.",
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        with patch("thinking._invoke_planner_step", return_value=plan), patch("thinking._execute_schema_tool", side_effect=execute_schema):
            run_thinking_agent(
                prompt="inspect", rows=GAPMINDER_ROWS, model_name="test", history=[],
                event_callback=events.append,
            )

        action_index = next(index for index, event in enumerate(events) if event["kind"] == "action")
        observation_index = next(index for index, event in enumerate(events) if event["content"] == "Schema checked.")
        self.assertLess(action_index, observation_index)


if __name__ == "__main__":
    unittest.main()
