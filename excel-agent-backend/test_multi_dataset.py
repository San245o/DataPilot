import asyncio
import json
import unittest
from unittest.mock import patch

from agent import clean_final_reply_text
from main import _cache_dataset, execute_agent, execute_thinking_agent_stream
from schemas import AgentRequest


class MultiDatasetAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.user = {"user_id": "multi-dataset-test"}
        self.left = _cache_dataset([{"player": "A", "runs": 10}], "batting.csv", user_id=self.user["user_id"])
        self.right = _cache_dataset([{"bowler": "B", "wickets": 3}], "bowling.csv", user_id=self.user["user_id"])
        self.prompt = "Summarize all files with visualizations."
        self.code = (
            "batting = dfs['batting.csv']\n"
            "bowling = dfs.get('bowling.csv')\n"
            "for frame in dfs.values():\n"
            "    metric = 'runs' if 'runs' in frame.columns else 'wickets'\n"
            "    log_output(f'- {frame.attrs[\"name\"]}: {frame[metric].sum()} {metric}')\n"
            "log_chart(px.bar(batting, x='player', y='runs', title='Batting'))\n"
            "log_chart(px.bar(bowling, x='bowler', y='wickets', title='Bowling'))\n"
            "result_df = df"
        )

    def request(self, *, thinking=False):
        return AgentRequest(
            prompt=self.prompt,
            active_dataset_id=self.left,
            selected_dataset_ids=[self.right],
            model="models/gemma-4-31b-it",
            thinking_mode=thinking,
        )

    def assert_analysis(self, result):
        self.assertEqual(len(result["visualizations"]), 2)
        self.assertIn("batting.csv: 10 runs", result["assistant_reply"])
        self.assertIn("bowling.csv: 3 wickets", result["assistant_reply"])
        self.assertEqual(result["assistant_reply"].count("batting.csv:"), 1)
        self.assertEqual(result["assistant_reply"].count("bowling.csv:"), 1)
        self.assertEqual(result["created_datasets"], [])
        self.assertFalse(result["mutation"])

    def test_standard_request_returns_both_charts_and_findings(self):
        generated = {"code": self.code, "assistant_reply": "Analysis complete.", "action_type": "visualization"}
        with patch("workflow.generate_code", return_value=generated):
            response = execute_agent(self.request(), current_user=self.user)
        self.assert_analysis(response.model_dump())

    def test_thinking_stream_returns_both_charts_and_findings(self):
        async def collect():
            response = await execute_thinking_agent_stream(self.request(thinking=True), current_user=self.user)
            return [json.loads(line) async for line in response.body_iterator]

        plan = {"kind": "plan", "steps": [{"tool": "execute_python", "args": {"code": self.code}}]}
        with patch("thinking._invoke_planner_step", return_value=(plan, {})):
            events = asyncio.run(collect())
        self.assertEqual(events[-1]["type"], "final")
        self.assert_analysis(events[-1]["payload"])

    def test_joined_result_is_visible_without_a_deletion_warning(self):
        batting = [
            {"player": "A", "runs": 10},
            {"player": "B", "runs": 21},
            {"player": "C", "runs": 7},
        ]
        self.left = _cache_dataset(batting, "batting.csv", user_id=self.user["user_id"])
        self.right = _cache_dataset(
            [{"player": "A", "wickets": 3}, {"player": "B", "wickets": 4}],
            "bowling.csv", user_id=self.user["user_id"],
        )
        self.prompt = "Use these files and tell me the best player who did well in batting and bowling."
        code = (
            "joined = dfs['batting.csv'].merge(dfs['bowling.csv'], on='player')\n"
            "joined['score'] = joined['runs'] / joined['runs'].max() + joined['wickets'] / joined['wickets'].max()\n"
            "joined = joined.sort_values('score', ascending=False)\n"
            "winner = joined.iloc[0]\n"
            "assistant_reply = f'{winner[\"player\"]} ranked first with {winner[\"runs\"]} runs and {winner[\"wickets\"]} wickets.'\n"
            "log_output(joined)\n"
            "result_df = joined"
        )
        generated = {"code": code, "assistant_reply": "The best player can be identified.", "action_type": "combined"}
        with patch("workflow.generate_code", return_value=generated):
            standard = execute_agent(self.request(), current_user=self.user).model_dump()

        async def collect():
            response = await execute_thinking_agent_stream(self.request(thinking=True), current_user=self.user)
            return [json.loads(line) async for line in response.body_iterator][-1]["payload"]

        plan = {"kind": "plan", "steps": [{"tool": "execute_python", "args": {"code": code}}]}
        with patch("thinking._invoke_planner_step", return_value=(plan, {})):
            thinking = asyncio.run(collect())

        for result in (standard, thinking):
            self.assertEqual(result["assistant_reply"], "B ranked first with 21 runs and 4 wickets.")
            self.assertNotIn("Safety guard", result.get("query_output") or "")
            self.assertEqual(result["rows"], batting)
            self.assertFalse(result["mutation"])
            self.assertEqual([row["player"] for row in result["query_tables"][0]["rows"]], ["B", "A"])

    def test_unrequested_source_deletion_is_still_blocked(self):
        rows = [{"player": "A"}, {"player": "B"}]
        self.left = _cache_dataset(rows, "batting.csv", user_id=self.user["user_id"])
        request = AgentRequest(
            prompt="Update the player names", active_dataset_id=self.left, model="models/gemma-4-31b-it",
        )
        generated = {"code": "df = df.head(1)\nresult_df = df", "action_type": "mutation"}
        with patch("workflow.generate_code", return_value=generated), patch("workflow.draft_final_reply", return_value={}):
            result = execute_agent(request, current_user=self.user)
        self.assertEqual(result.rows, rows)
        self.assertFalse(result.mutation)
        self.assertIn("Safety guard", result.query_output)

    def test_standard_reply_keeps_multiple_findings(self):
        answer = "B leads on both measures.\n\n- Runs: 21\n- Wickets: 4\n\nScores use normalized totals."
        self.assertEqual(clean_final_reply_text(answer, initial_reply="", wants_list=True), answer)
