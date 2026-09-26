from __future__ import annotations

import unittest
from unittest.mock import patch

from thinking import ToolExecution, run_thinking_agent


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
             )):
            result = run_thinking_agent(prompt=prompt, rows=ROWS, model_name="test", history=[])
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
                model_name="test",
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
                model_name="test",
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
                model_name="test",
                history=[],
                active_dataset_id="active_1",
            )

        web_mock.assert_called_once()
        self.assertEqual(len(result["result_rows"]), len(ROWS) + 1)
        self.assertTrue(result["mutation"])
        self.assertIn("Added 1 real row for India", result["assistant_reply"])


if __name__ == "__main__":
    unittest.main()
