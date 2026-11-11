from __future__ import annotations

import asyncio
import io
import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd

from main import _cache_dataset, execute_auto_report_stream
from report import REPORT_PLANNERS, _generate_chart, _plan_report, _report_context, _safe_index, _sanitize_for_report_json, _synthesize_report
from schemas import ReportRequest
from thinking import ToolExecution, run_thinking_agent


ROWS = [
    {"position": n + 1, "player": f"Player {n + 1}", "runs": n * 10, "strike_rate": 85 + n * 2, "team": "East" if n % 2 else "West"}
    for n in range(20)
]
TYPES = ["scatter", "heatmap", "treemap", "range", "ranking", "parallel coordinates", "distribution"]


def proposal(kind: str, **overrides) -> dict:
    return {"type": kind, "question": f"What can the {kind} view show?", "fields": ["runs", "strike_rate"], "method": "Compare the calculated values.", **overrides}


def completed_chart(item: dict) -> dict:
    return {"chart": {
        "id": item["id"], "kind": item["type"], "title": item["question"],
        "figure": {"data": [{"type": "scatter", "x": [1, 2], "y": [3, 4]}]},
        "interpretation": "The calculated values differ by 1.",
    }}


class AutoReportTests(unittest.TestCase):
    def _stream(self):
        dataset_id = _cache_dataset(ROWS, "Batting leaderboard", user_id="test-user")
        return execute_auto_report_stream(
            ReportRequest(dataset_id=dataset_id, model="models/gemma-4-31b-it"), current_user={"user_id": "test-user"},
        ).body_iterator

    def test_one_planner_request_and_leaderboard_position_excluded(self) -> None:
        context = _report_context(pd.DataFrame(ROWS), "Batting leaderboard")
        payload = {
            "focus": "Compare the performance of the listed batters.",
            "charts": [
                proposal("scatter", fields=["position", "runs"]),
                proposal("treemap", fields=["missing", "runs"]),
                *[proposal(kind) for kind in TYPES],
                proposal("ranking"),
            ],
        }
        response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(payload)}]}}]}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch(
            "report.urlopen", return_value=io.BytesIO(json.dumps(response).encode()),
        ) as request:
            plan = _plan_report(context, "gemini-3.8-flash", 12)
        request.assert_called_once()
        call = request.call_args.args[0]
        self.assertIn("/gemini-3.8-flash:generateContent", call.full_url)
        self.assertEqual(request.call_args.kwargs["timeout"], 12)
        body = json.loads(call.data)
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "LOW")
        model_input = json.loads(body["contents"][0]["parts"][0]["text"])
        self.assertEqual(len(model_input["sample"]), 5)
        self.assertEqual(model_input["rows"], 20)
        self.assertTrue(model_input["columns"]["position"]["identifier_or_rank"])
        self.assertEqual(plan["model"], "Gemini 3.8 Flash")
        self.assertEqual([item["type"] for item in plan["charts"]], TYPES)

    def test_plan_precedes_seven_individual_chart_events_and_summary(self) -> None:
        context = _report_context(pd.DataFrame(ROWS), "Data")
        with patch("report._request_report_plan", return_value={"focus": "Batting performance", "charts": [proposal(kind) for kind in TYPES]}):
            plan = _plan_report(context, "gemini-3.8-flash", 12)
        release = threading.Event()

        def generate(item, rows, context, model, stop):
            self.assertEqual(model, "gemini-3.5-flash-lite")
            self.assertEqual(len(rows), 20)
            if item["id"] != 1 and not release.wait(5):
                raise TimeoutError("Expected first chart to arrive before the other views.")
            return completed_chart(item)

        async def collect():
            events = []
            async for chunk in self._stream():
                event = json.loads(chunk)
                events.append(event)
                if event["type"] == "chart":
                    self.assertTrue(any(entry["type"] == "plan" for entry in events))
                    release.set()
            return events

        synthesis = {"executive_summary": "Computed summary.", "conclusion": "Computed conclusion."}
        with patch("report._plan_report", return_value=plan) as planner, patch(
            "report._generate_chart", side_effect=generate,
        ), patch("report._synthesize_report", return_value=synthesis) as summarize:
            events = asyncio.run(collect())
        planner.assert_called_once()
        public = [event for event in events if event["type"] != "trace"]
        self.assertEqual([event["type"] for event in public], ["planner", "plan", *["chart"] * 7, "synthesizing", "final"])
        self.assertEqual(public[2]["chart"]["id"], 1)
        self.assertEqual(public[-1]["payload"]["charts"], [event["chart"] for event in public[2:9]])
        self.assertEqual(summarize.call_args.kwargs["model_name"], "gemini-3.5-flash-lite")

    def test_planner_exhausts_fallbacks_once_in_order_and_produces_no_charts(self) -> None:
        async def collect():
            return [json.loads(chunk) async for chunk in self._stream()]

        with patch("report._request_report_plan", side_effect=TimeoutError) as request, patch("report._generate_chart") as generate:
            events = asyncio.run(collect())
        expected = [
            "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3-flash-preview",
            "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
        ]
        self.assertEqual([call.args[1] for call in request.call_args_list], expected)
        generate.assert_not_called()
        public = [event for event in events if event["type"] != "trace"]
        self.assertEqual([event["type"] for event in public], ["planner"] * len(expected) + ["error"])
        self.assertIn("Gemini 3.8 Flash timed out. Trying Gemini 3.6 Flash.", public[1]["message"])
        self.assertIn("Gemini 3.5 Flash-Lite timed out", public[-1]["error"])

    def test_planner_skips_unavailable_or_invalid_models_and_stops_on_success(self) -> None:
        async def collect():
            return [json.loads(chunk) async for chunk in self._stream()]

        responses = [
            HTTPError("https://example.test", 503, "Unavailable", {}, None),
            {"focus": "Invalid fields", "charts": [proposal("scatter", fields=["not a column"])]},
            {"focus": "Batting performance", "charts": [proposal("scatter")]},
        ]
        synthesis = {"executive_summary": "Computed summary.", "conclusion": "Computed conclusion."}
        with patch("report._request_report_plan", side_effect=responses) as request, patch(
            "report._generate_chart", side_effect=lambda item, *_args: completed_chart(item),
        ), patch("report._synthesize_report", return_value=synthesis):
            events = asyncio.run(collect())

        self.assertEqual([call.args[1] for call in request.call_args_list], list(REPORT_PLANNERS)[:3])
        self.assertEqual([call.args[2] for call in request.call_args_list], [12, 15, 15])
        planner_events = [event["message"] for event in events if event["type"] == "planner"]
        self.assertIn("returned HTTP 503", planner_events[1])
        self.assertIn("returned an unusable plan", planner_events[2])
        plan = next(event["plan"] for event in events if event["type"] == "plan")
        self.assertEqual(plan["model"], "Gemini 3 Flash Preview")
        self.assertEqual(events[-1]["type"], "final")

    def test_incomplete_charts_are_marked_and_existing_charts_survive_summary_failure(self) -> None:
        plan = {"model": "Gemini 3.8 Flash", "focus": "Batting performance", "charts": [proposal(kind, id=i) for i, kind in enumerate(TYPES[:2], 1)]}

        def generate(item, *_args):
            return {"reason": "The grouping does not have enough records."} if item["id"] == 2 else completed_chart(item)

        async def collect():
            return [json.loads(chunk) async for chunk in self._stream()]

        with patch("report._plan_report", return_value=plan), patch(
            "report._generate_chart", side_effect=generate,
        ), patch("report._synthesize_report", side_effect=TimeoutError):
            events = asyncio.run(collect())
        skipped = [event for event in events if event["type"] == "chart_skipped"]
        self.assertEqual([event["id"] for event in skipped], [2])
        self.assertEqual(events[-1]["payload"]["generation_mode"], "fallback")
        self.assertEqual(events[-1]["payload"]["charts"], [completed_chart(plan["charts"][0])["chart"]])

    def test_chart_agent_retries_wrong_plot_and_executes_against_full_dataset(self) -> None:
        context = _report_context(pd.DataFrame(ROWS), "Data")
        wrong = "fig = px.bar(df, x='team', y='runs'); assistant_reply = 'A chart caption.'; result_df = df"
        correct = (
            "fig = px.scatter(df, x='runs', y='strike_rate', title='Scoring volume and pace')\n"
            "assistant_reply = f'{len(df)} batters are included.'\nresult_df = df"
        )
        with patch("report.invoke_model_json", side_effect=[({"code": wrong}, {}, ""), ({"code": correct}, {}, "")]) as invoke:
            result = _generate_chart(proposal("scatter", id=1), ROWS, context, "gemini-3.5-flash-lite", threading.Event())
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("requested chart type", invoke.call_args.kwargs["user_message"])
        self.assertEqual(result["chart"]["title"], "Scoring volume and pace")
        self.assertEqual(result["chart"]["interpretation"], "20 batters are included.")

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
                rows=ROWS, visualization=None, query_output=None, query_table_rows=None,
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
                prompt="inspect", rows=ROWS, history=[],
                event_callback=events.append,
            )

        action_index = next(index for index, event in enumerate(events) if event["kind"] == "action")
        observation_index = next(index for index, event in enumerate(events) if event["content"] == "Schema checked.")
        self.assertLess(action_index, observation_index)

    def test_safe_index_parsing(self) -> None:
        self.assertEqual(_safe_index(0), 0)
        self.assertEqual(_safe_index("0"), 0)
        self.assertEqual(_safe_index(" 2 "), 2)
        self.assertEqual(_safe_index(1.0), 1)
        self.assertIsNone(_safe_index(None))
        self.assertIsNone(_safe_index(True))
        self.assertIsNone(_safe_index(False))
        self.assertIsNone(_safe_index("invalid"))

    def test_sanitize_for_report_json_eliminates_nan_and_inf(self) -> None:
        payload = {
            "title": "Test",
            "values": [1.0, float("nan"), float("inf"), -float("inf")],
            "nested": {"val": float("nan")},
        }
        sanitized = _sanitize_for_report_json(payload)
        self.assertEqual(sanitized["values"], [1.0, None, None, None])
        self.assertIsNone(sanitized["nested"]["val"])
        # Standard RFC-compliant JSON serialization without allow_nan=True must succeed
        encoded = json.dumps(sanitized, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

if __name__ == "__main__":
    unittest.main()
