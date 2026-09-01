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


if __name__ == "__main__":
    unittest.main()
