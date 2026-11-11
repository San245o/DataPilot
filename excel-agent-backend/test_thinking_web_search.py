from __future__ import annotations

import asyncio
import io
import json
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.api_core.exceptions import InternalServerError

from agent import _extract_json
from thinking import ToolExecution, run_thinking_agent
from web_search import WebSearchError


ROWS = [
    {"year": 2000, "country": "Brazil", "fertility": 2.4},
    {"year": 2010, "country": "Brazil", "fertility": 1.9},
]


def execution(*, observation: str, query_rows=None, query_output: str | None = None) -> ToolExecution:
    return ToolExecution(
        rows=ROWS, visualization=None, query_output=query_output, query_table_rows=query_rows,
        mutation=False, highlight_indices=[], highlighted_columns=[], observation=observation,
        raw_observation=observation, code="result_df=df",
    )


WEB_RESULT = ToolExecution(
    rows=ROWS, visualization=None, query_output=None, query_table_rows=None,
    mutation=False, highlight_indices=[], highlighted_columns=[],
    observation="Result 1: World Bank fertility rate (https://data.worldbank.org/indicator/SP.DYN.TFRT.IN)\nSummary: Brazil 2020 value: 1.65 births per woman.",
    raw_observation="World Bank result", code="web_search(query='Brazil fertility 2020')",
    sources=[{"title": "World Bank fertility rate", "url": "https://data.worldbank.org/indicator/SP.DYN.TFRT.IN"}],
)


class ThinkingWebFallbackTests(unittest.TestCase):
    def test_explicit_lookup_sends_late_source_text_without_a_model_plan(self) -> None:
        winner = "East City won the 2026 Autumn Cup final, defeating West City by two goals."
        results = [
            {"title": f"Preview {index}", "url": f"https://example.test/preview-{index}",
             "content": "This page lists the Autumn Cup squads and schedule. " * 35}
            for index in range(4)
        ] + [
            {"title": "Autumn Cup final report", "url": "https://league.test/2026-final", "content": "Tournament final coverage.",
             "raw_content": "Fixtures and news.\n" + winner},
        ]
        prompt = "search the web and tell me who won the Autumn Cup. this is currently unfinished match data"
        response = SimpleNamespace(text=json.dumps({"final_answer": "East City won the 2026 Autumn Cup.", "source_ids": [5]}))
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key", "GEMINI_API_KEY": "test-key"}), patch(
            "web_search.urllib.request.urlopen", return_value=io.BytesIO(json.dumps({"results": results}).encode()),
        ) as request, patch("thinking._invoke_planner_step") as plan, patch("agent.genai.configure"), patch(
            "agent.genai.GenerativeModel",
        ) as model:
            summarize = model.return_value.generate_content
            summarize.return_value = response
            result = run_thinking_agent(prompt=prompt, rows=ROWS, history=[])

        plan.assert_not_called()
        summarize.assert_called_once()
        self.assertEqual(summarize.call_args.kwargs["request_options"], {"timeout": 18, "retry": None})
        model.assert_called_once_with("gemini-3.5-flash-lite")
        search_payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(search_payload["include_raw_content"], "markdown")
        self.assertEqual(search_payload["query"], "who won the Autumn Cup latest completed final result match report")
        evidence = summarize.call_args.args[0][1].split("Retrieved source evidence:\n", 1)[1].splitlines()[0]
        self.assertGreater(evidence.index(winner), 1600)
        self.assertIn(winner, json.loads(evidence)[4]["excerpt"])
        self.assertNotIn("https://league.test/2026-final", evidence)
        self.assertEqual(result["sources"], [{"title": "Autumn Cup final report", "url": "https://league.test/2026-final"}])
        self.assertIn("East City", result["assistant_reply"])
        self.assertFalse(result["mutation"])

    def test_failed_synthesis_keeps_the_retrieved_sources(self) -> None:
        with patch("thinking._execute_web_search_tool", return_value=WEB_RESULT), patch(
            "thinking._write_external_fallback_answer", side_effect=InternalServerError("Internal error"),
        ) as summarize:
            result = run_thinking_agent(
                prompt="search the web for Brazil's fertility rate in 2020", rows=ROWS, history=[],
            )
        summarize.assert_called_once()
        self.assertIn("retrieved the sources", result["assistant_reply"])
        self.assertEqual(result["sources"], WEB_RESULT.sources)

    def test_failed_explicit_web_lookup_does_not_trigger_python_repair(self) -> None:
        with patch("thinking.search_web", side_effect=WebSearchError("Tavily returned HTTP 503.")), patch(
            "thinking._repair_python_code",
        ) as repair:
            result = run_thinking_agent(
                prompt="search the web for the Autumn Cup winner", rows=ROWS, history=[],
            )
        repair.assert_not_called()
        self.assertIn("Tavily returned HTTP 503", result["assistant_reply"])
        self.assertFalse(result["sources"])

    def test_json_parser_preserves_braces_in_python_and_fences_in_replies(self) -> None:
        responses = [
            {
                "kind": "plan",
                "steps": [{"tool": "execute_python", "args": {
                    "code": "log_output({'kind': 'summary', 'tool': 'count'})\nresult_df = df",
                }}],
            },
            {"kind": "final", "final_answer": "Previous result:\n```python\nprint(10)\n```"},
        ]
        for response in responses:
            with self.subTest(kind=response["kind"]):
                raw = json.dumps(response)
                self.assertEqual(_extract_json(raw), response)
                self.assertEqual(_extract_json(f"```json\n{raw}\n```"), response)

    def test_provider_error_is_reported_without_switching_models(self) -> None:
        error = InternalServerError("Internal error encountered")
        with patch("thinking.invoke_model_json", side_effect=error) as invoke:
            result = run_thinking_agent(prompt="Summarize this dataset", rows=ROWS, history=[])
        invoke.assert_called_once()
        self.assertEqual(invoke.call_args.kwargs["model_name"], "gemini-3.5-flash-lite")
        self.assertIn("Gemini 3.5 Flash-Lite returned HTTP 500", result["assistant_reply"])
        self.assertFalse(result["mutation"])

    def test_dataset_count_and_follow_up_work_without_a_planner(self) -> None:
        rows = [
            {"id": 1, "venue": "M. Chinnaswamy Stadium", "team1": "Royal Challengers Bengaluru", "team2": "CSK", "winner": "RCB"},
            {"id": 2, "venue": "M Chinnaswamy Stadium", "team1": "RCB", "team2": "KKR", "winner": "KKR"},
            {"id": 3, "venue": "M Chinnaswamy Stadium", "team1": "RCB", "team2": "CSK", "winner": None},
            {"id": 4, "venue": "Other Stadium", "team1": "RCB", "team2": "CSK", "winner": "RCB"},
        ]
        prompt = "how many matches has rcb won in chinnaswamy"
        with patch("thinking._invoke_planner_step", side_effect=InternalServerError("Offline")) as invoke:
            first = run_thinking_agent(prompt=prompt, rows=rows, history=[])
            follow_up = run_thinking_agent(
                prompt="and how many did they lose there?", rows=rows,
                history=[{"role": "user", "content": prompt}, {"role": "assistant", "content": first["assistant_reply"]}],
            )
        invoke.assert_not_called()
        self.assertIn("RCB won **1** match", first["assistant_reply"])
        self.assertIn("RCB lost **1** match", follow_up["assistant_reply"])
        self.assertEqual([entry["kind"] for entry in first["thinking_trace"]], ["action", "observation"])
        self.assertEqual(follow_up["result_rows"], rows)

    def test_general_cleanup_checks_all_rows_and_preserves_unknown_award_counts(self) -> None:
        rows = [{"Title": f"Film {index}", "Oscars Won": 0} for index in range(12)]
        rows.extend([
            {"Title": "  Film 12  ", "Oscars Won": "  "},
            {"Title": "Film 13", "Oscars Won": "unknown"},
        ])
        with patch("thinking._invoke_planner_step") as planner:
            result = run_thinking_agent(
                prompt="clean this dataset", rows=rows,
                history=[], active_dataset_id="films",
            )
        planner.assert_not_called()
        self.assertTrue(result["mutation"])
        self.assertEqual(result["updated_datasets"][0]["dataset_id"], "films")
        self.assertEqual(result["result_rows"][12], {"Title": "Film 12", "Oscars Won": None})
        self.assertEqual(result["result_rows"][13]["Oscars Won"], "unknown")
        self.assertEqual(rows[12]["Oscars Won"], "  ")
        self.assertEqual(result["highlight_indices"], [12])
        self.assertIn("Updated 1 of 14 rows", result["assistant_reply"])
        self.assertEqual(result["query_tables"], [])

    def test_general_cleanup_reports_a_clean_dataset_without_saving_an_update(self) -> None:
        with patch("thinking._invoke_planner_step") as planner:
            result = run_thinking_agent(
                prompt="clean this dataset", rows=ROWS,
                history=[], active_dataset_id="clean",
            )
        planner.assert_not_called()
        self.assertFalse(result["mutation"])
        self.assertEqual(result["updated_datasets"], [])
        self.assertIn("Found 0 cells to clean", result["assistant_reply"])

    def test_specific_cleanup_reports_no_saved_changes_instead_of_a_preview(self) -> None:
        rows = [{"Title": "Film", "Oscars Won": 1}]
        action = {"kind": "action", "tool": "execute_python", "code": (
            "result_df = df.copy()\nresult_df['Oscars Won'] = pd.to_numeric(result_df['Oscars Won'])\n"
            "log_output(result_df.head(10))"
        )}
        with patch("thinking._invoke_planner_step", return_value=(action, {})):
            result = run_thinking_agent(
                prompt="Clean the Oscars Won column by converting existing values to numbers.",
                rows=rows, history=[],
            )
        self.assertFalse(result["mutation"])
        self.assertIn("0 saved changes", result["assistant_reply"])

    def test_incomplete_action_retries_once_with_flash_lite(self) -> None:
        action = {"kind": "action", "tool": "execute_python", "code": "log_output(df['fertility'].mean())\nresult_df=df"}
        with patch("thinking.invoke_model_json", side_effect=[({"kind": "action"}, {}, ""), (action, {}, "")]) as invoke:
            result = run_thinking_agent(prompt="Calculate average fertility", rows=ROWS, history=[])
        self.assertEqual([call.kwargs["model_name"] for call in invoke.call_args_list], ["gemini-3.5-flash-lite"] * 2)
        self.assertIn("response_schema", invoke.call_args.kwargs)
        self.assertIn("2.15", result["assistant_reply"])

    def test_planner_keeps_prior_turns_and_omits_duplicate_current_prompt(self) -> None:
        earlier = {"role": "assistant", "content": "The prior result was 10."}
        prompt = "Summarize this dataset"
        history = [earlier, {"role": "user", "content": prompt}]
        with patch("thinking._invoke_planner_step", return_value=({"kind": "final", "final_answer": "Summary."}, {})) as planner:
            run_thinking_agent(prompt=prompt, rows=ROWS, history=history)
        message = planner.call_args.kwargs["planner_message"]
        self.assertIn(earlier["content"], message)
        self.assertEqual(message.count(prompt), 1)

    def test_final_answer_preserves_paragraphs_and_list(self) -> None:
        answer = "Fertility fell over the decade.\n\n- 2000: 2.4\n- 2010: 1.9\n\nThis covers Brazil only."
        with patch("thinking._invoke_planner_step", return_value=({"kind": "final", "final_answer": answer}, {})):
            result = run_thinking_agent(prompt="List the findings", rows=ROWS, history=[])
        self.assertEqual(result["assistant_reply"], answer)

    def test_schema_followup_preserves_final_answer(self) -> None:
        plan = {"kind": "plan", "steps": [{"tool": "inspect_schema", "args": {}}]}
        answer = "The dataset starts in 2000.\n\nIt ends in 2010."
        final = {"kind": "final", "final_answer": answer}
        with patch("thinking._invoke_planner_step", side_effect=[(plan, {}), (final, {})]):
            result = run_thinking_agent(prompt="What period does this cover?", rows=ROWS, history=[])
        self.assertEqual(result["assistant_reply"], answer)

    def test_multiline_tool_output_takes_priority_over_unverified_plan(self) -> None:
        plan = {
            "kind": "plan",
            "final_answer": "The total is 42.",
            "steps": [{"tool": "execute_python", "args": {
                "code": "log_output('Minimum: 1.9')\nlog_output('Maximum: 2.4')\nresult_df = df",
            }}],
        }
        with patch("thinking._invoke_planner_step", return_value=(plan, {})):
            result = run_thinking_agent(prompt="Calculate the minimum and maximum", rows=ROWS, history=[])
        self.assertEqual(result["assistant_reply"], "Minimum: 1.9\n\nMaximum: 2.4")

    def test_large_result_preview_links_to_complete_table(self) -> None:
        rows = [{"number": number} for number in range(12)]
        plan = {"kind": "plan", "steps": [{"tool": "execute_python", "args": {
            "code": "log_output(df)\nresult_df = df",
        }}]}
        with patch("thinking._invoke_planner_step", return_value=(plan, {})):
            result = run_thinking_agent(prompt="Show the results", rows=rows, history=[])
        self.assertIn("and 4 more", result["assistant_reply"])
        self.assertEqual(result["query_tables"][0]["rows"], rows)

    def run_case(self, prompt: str, dataset_execution: ToolExecution):
        initial_plan = ({
            "kind": "plan", "thought": "I will check the active dataset first.",
            "steps": [{"tool": "execute_python", "args": {"code": "result_df=df"}}],
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        with patch("thinking._invoke_planner_step", return_value=initial_plan), \
             patch("thinking._execute_sandbox_tool", return_value=dataset_execution), \
             patch("thinking._execute_web_search_tool", return_value=WEB_RESULT) as web_search, \
             patch("thinking._write_external_fallback_answer", return_value=(
                 "The uploaded dataset does not include Brazil for 2020. According to the World Bank, the external result is 1.65 births per woman.",
                 {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                 [],
             )):
            result = run_thinking_agent(prompt=prompt, rows=ROWS, history=[])
        return result, web_search

    def test_public_fact_dataset_miss_uses_web_search(self) -> None:
        result, web_search = self.run_case(
            "What was Brazil's fertility rate in 2020?",
            execution(observation="No data found for Brazil in 2020.", query_rows=[]),
        )
        web_search.assert_called_once()
        self.assertIn("uploaded dataset does not include", result["assistant_reply"].lower())
        self.assertEqual(result["sources"][0]["title"], "World Bank fertility rate")
        self.assertIn("web_search", [entry.get("tool_name") for entry in result["thinking_trace"]])

    def test_dataset_scoped_miss_does_not_use_web_search(self) -> None:
        result, web_search = self.run_case(
            "What was Brazil's fertility rate in 2020 according to this dataset?",
            execution(observation="No data found for Brazil in 2020.", query_rows=[]),
        )
        web_search.assert_not_called()
        self.assertIn("no data found", result["assistant_reply"].lower())

    def test_dataset_average_does_not_use_web_search(self) -> None:
        result, web_search = self.run_case(
            "What is the average fertility rate in this dataset?",
            execution(observation="The average fertility rate is 2.15.", query_output="2.15"),
        )
        web_search.assert_not_called()
        self.assertIn("2.15", result["assistant_reply"])

    def test_current_public_fact_uses_web_search(self) -> None:
        _result, web_search = self.run_case(
            "What is Brazil's current fertility rate?",
            execution(observation="The latest uploaded value is 1.9.", query_rows=[ROWS[-1]]),
        )
        web_search.assert_called_once()

    def test_missing_customer_record_does_not_use_web_search(self) -> None:
        result, web_search = self.run_case(
            "Show customer 12345 from this spreadsheet.",
            execution(observation="No matching record exists for customer 12345.", query_rows=[]),
        )
        web_search.assert_not_called()
        self.assertIn("no matching record", result["assistant_reply"].lower())

    def test_exact_dataset_value_does_not_use_web_search(self) -> None:
        exact_row = {"year": 2020, "country": "Brazil", "fertility": 1.65}
        result, web_search = self.run_case(
            "What was Brazil's fertility rate in 2020?",
            execution(observation="One matching row was found.", query_rows=[exact_row]),
        )
        web_search.assert_not_called()
        self.assertIn("1.65", result["assistant_reply"])

    def test_external_context_combines_dataset_and_web_sources(self) -> None:
        result, web_search = self.run_case(
            "What recent external factors could explain this trend?",
            execution(observation="Fertility declined from 2.4 to 1.9.", query_output="Fertility declined from 2.4 to 1.9."),
        )
        web_search.assert_called_once()
        self.assertTrue(result["sources"])

    def test_schema_tool_handles_null_args(self) -> None:
        from thinking import _execute_schema_tool, _safe_int
        self.assertEqual(_safe_int(None, 2), 2)
        self.assertEqual(_safe_int("invalid", 10), 10)
        self.assertEqual(_safe_int("50", 10, max_val=20), 20)
        exec_result = _execute_schema_tool(ROWS, {"sample_rows": None})
        self.assertIsNone(exec_result.error)
        self.assertIn("inspect_schema", exec_result.code)

    def test_final_answer_fallback_avoids_hallucination_on_error(self) -> None:
        from thinking import _final_answer_fallback
        # When error occurs, do not return speculative planned_answer
        reply = _final_answer_fallback(
            query_output=None,
            query_table_rows=None,
            visualization=None,
            created_output_rows=None,
            mutation_applied=False,
            fallback_text="Tool failed with: division by zero",
            planned_answer="The average is 42.",
            has_error=True,
        )
        self.assertIn("Tool failed", reply)
        self.assertNotIn("42", reply)

    def test_execute_sandbox_tool_catches_invalid_tool_args(self) -> None:
        from thinking import _execute_sandbox_tool
        exec_result = _execute_sandbox_tool(
            tool="delete_row",
            args={"index": "not_a_number"},
            prompt="delete this row",
            rows=ROWS,
        )
        self.assertIsNotNone(exec_result.error)
        self.assertIn("delete_row", exec_result.code)

    def test_heal_code_escaping_glitches(self) -> None:
        from sandbox import heal_code_escaping_glitches, _validate_code
        glitchy_code = (
            "new_rows = [\n"
            "    {'year': 2000, 'pop': 1056},n {'year': 2005, 'pop': 1147},n {'year': 2010, 'pop': 1234}\n"
            "]\n"
            "df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)\n"
            "result_df = df"
        )
        healed = heal_code_escaping_glitches(glitchy_code)
        self.assertNotIn(",n {", healed)
        self.assertIn(",\n {", healed)
        # Verify it passes AST validation cleanly
        _validate_code(healed)

        # Check that normal variable names 'n' are not broken
        normal_code = "for i, n in enumerate([1, 2, 3]):\n    pass"
        self.assertEqual(heal_code_escaping_glitches(normal_code), normal_code)

    def test_syntax_self_correction_retry_on_syntax_error(self) -> None:
        initial_plan = ({
            "kind": "plan",
            "thought": "I will calculate the sum.",
            "steps": [{"tool": "execute_python", "args": {"code": "result_df = df\ninvalid syntax here ("}}],
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        repair_response = ({
            "thought": "Fixed unclosed parenthesis.",
            "code": "result_df = df\nx = 10\nlog_output(x)",
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        with patch("thinking._invoke_planner_step", side_effect=[initial_plan, repair_response]):
            result = run_thinking_agent(
                prompt="calculate something",
                rows=ROWS,
                history=[],
            )

        trace_contents = [entry.get("content", "") for entry in result["thinking_trace"]]
        self.assertTrue(any("Requesting syntax self-correction" in c for c in trace_contents))
        self.assertTrue(any("Retrying with self-corrected code" in c for c in trace_contents))
        self.assertIn("execute_python (self-corrected)", result["code"])
        self.assertNotIn("Tool selection failed", result["assistant_reply"])

    def test_two_phase_web_search_to_execution_adds_rows(self) -> None:
        phase1_plan = ({
            "kind": "plan",
            "thought": "I will search the web for demographic data on India.",
            "steps": [{"tool": "web_search", "args": {"query": "India fertility rate 2000 2005 2010"}}],
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        phase2_plan = ({
            "kind": "plan",
            "thought": "Adding 3 rows for India using real data from the web search.",
            "steps": [{
                "tool": "execute_python",
                "args": {
                    "code": (
                        "new_rows = pd.DataFrame([\n"
                        "    {'year': 2000, 'country': 'India', 'fertility': 3.3},\n"
                        "    {'year': 2005, 'country': 'India', 'fertility': 2.9},\n"
                        "    {'year': 2010, 'country': 'India', 'fertility': 2.6}\n"
                        "])\n"
                        "df = pd.concat([df, new_rows], ignore_index=True)\n"
                        "result_df = df"
                    )
                },
                "reason": "Add retrieved rows.",
            }],
            "final_answer": "Added 3 rows for India with fertility rates from 2000 to 2010.",
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        with patch("thinking._invoke_planner_step", side_effect=[phase1_plan, phase2_plan]), \
             patch("thinking._execute_web_search_tool", return_value=WEB_RESULT) as web_mock:
            result = run_thinking_agent(
                prompt="using web search add 3 rows of india",
                rows=ROWS,
                history=[],
                active_dataset_id="active_1",
            )

        web_mock.assert_called_once()
        self.assertEqual(len(result["result_rows"]), len(ROWS) + 3)
        self.assertTrue(result["mutation"])
        self.assertEqual(len(result["updated_datasets"]), 1)
        self.assertIn("Added 3 rows for India", result["assistant_reply"])
        self.assertTrue(result["sources"])
        trace_contents = [entry.get("content", "") for entry in result["thinking_trace"]]
        self.assertTrue(any("Web search completed with real external data" in c for c in trace_contents))

    def test_sanitize_execute_python_heals_glitches(self) -> None:
        from thinking import _sanitize_execute_python
        glitchy_code = (
            "new_rows = [\n"
            "    {'year': 2000, 'pop': 1056},n {'year': 2005, 'pop': 1147}\n"
            "]\n"
            "df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)\n"
            "result_df = df"
        )
        sanitized = _sanitize_execute_python(glitchy_code)
        self.assertNotIn(",n {", sanitized)
        self.assertIn(",\n {", sanitized)

    def test_speculative_mutation_truncated_for_web_search_mutation(self) -> None:
        # If the model tried to plan both web_search AND speculative execute_python in step 1,
        # ensure it runs web_search first and discards the speculative step in favor of Phase 2.
        speculative_step1 = ({
            "kind": "plan",
            "thought": "I will search web and guess the code.",
            "steps": [
                {"tool": "web_search", "args": {"query": "India population"}},
                {"tool": "execute_python", "args": {"code": "df = df # speculative guess"}},
            ],
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        phase2_real = ({
            "kind": "plan",
            "thought": "Adding real rows from search.",
            "steps": [{
                "tool": "execute_python",
                "args": {
                    "code": (
                        "new_rows = pd.DataFrame([{'year': 2020, 'country': 'India', 'fertility': 2.05}])\n"
                        "df = pd.concat([df, new_rows], ignore_index=True)\n"
                        "result_df = df"
                    )
                },
            }],
            "final_answer": "Added 1 real row for India.",
        }, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        with patch("thinking._invoke_planner_step", side_effect=[speculative_step1, phase2_real]), \
             patch("thinking._execute_web_search_tool", return_value=WEB_RESULT) as web_mock:
            result = run_thinking_agent(
                prompt="search the web and add 1 row for india",
                rows=ROWS,
                history=[],
                active_dataset_id="active_1",
            )

        web_mock.assert_called_once()
        self.assertEqual(len(result["result_rows"]), len(ROWS) + 1)
        self.assertTrue(result["mutation"])
        self.assertIn("Added 1 real row for India", result["assistant_reply"])


class ThinkingStreamTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def open_stream():
        from main import execute_thinking_agent_stream
        from schemas import AgentRequest

        response = await execute_thinking_agent_stream(
            AgentRequest(prompt="Analyze", rows=ROWS, model="models/gemma-4-31b-it"),
            current_user={"user_id": "test-user"},
        )
        return response.body_iterator

    async def test_legacy_model_selection_still_runs_flash_lite(self) -> None:
        with patch("thinking.invoke_model_json", return_value=({"kind": "final", "final_answer": "Summary."}, {}, "")) as invoke:
            stream = await self.open_stream()
            events = [json.loads(line) async for line in stream]
        invoke.assert_called_once()
        self.assertEqual(invoke.call_args.kwargs["model_name"], "gemini-3.5-flash-lite")
        self.assertEqual(events[-1]["payload"]["assistant_reply"], "Summary.")

    async def test_heartbeat_and_complete_final_payload(self) -> None:
        ready = Event()
        answer = "First finding.\n\nSecond finding."

        def execute(**kwargs):
            kwargs["event_callback"]({"kind": "thought", "content": "Analyzing the data"})
            ready.wait(timeout=3)
            return {"assistant_reply": answer, "result_rows": [{"missing": float("nan")}]}

        with patch("main.run_thinking_agent", side_effect=execute):
            stream = await self.open_stream()
            try:
                self.assertEqual(json.loads(await asyncio.wait_for(anext(stream), timeout=2))["type"], "trace")
                with patch("main.THINKING_STREAM_HEARTBEAT_SECONDS", 0.01):
                    self.assertEqual(json.loads(await anext(stream))["type"], "heartbeat")
                ready.set()
                events = [json.loads(line) async for line in stream]
            finally:
                ready.set()
                await stream.aclose()

        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["payload"]["assistant_reply"], answer)
        self.assertEqual(events[-1]["payload"]["rows"], [{"missing": None}])

    async def test_closed_stream_rejects_late_work(self) -> None:
        ready = Event()
        rejected = Event()

        def execute(**kwargs):
            kwargs["event_callback"]({"kind": "thought", "content": "Analyzing the data"})
            ready.wait(timeout=3)
            try:
                kwargs["event_callback"]({"kind": "action", "content": "Updating rows"})
            except RuntimeError:
                rejected.set()
                raise

        with patch("main.run_thinking_agent", side_effect=execute), patch("main._persist_response_datasets") as persist:
            stream = await self.open_stream()
            try:
                self.assertEqual(json.loads(await anext(stream))["type"], "trace")
            finally:
                await stream.aclose()
                ready.set()
            self.assertTrue(await asyncio.to_thread(rejected.wait, 3))
            persist.assert_not_called()


if __name__ == "__main__":
    unittest.main()
