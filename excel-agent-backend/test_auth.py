from unittest import TestCase

from fastapi.testclient import TestClient

from auth import SESSION_COOKIE
from main import app


class LocalAuthTests(TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 41000))
        self.addCleanup(self.client.close)

    def login(self, email: str = "analyst@example.test"):
        return self.client.post("/auth/login", json={"email": email, "password": "datapilot-test"})

    def test_sign_in_and_sign_out_revoke_api_access(self) -> None:
        dataset = {"name": "Synthetic", "rows": [{"value": 5}]}
        self.assertEqual(self.client.post("/dataset/register", json=dataset).status_code, 401)
        self.assertEqual(self.client.post(
            "/dataset/register", json=dataset, headers={"Authorization": "Bearer dev_token_analyst"}
        ).status_code, 401)
        self.assertEqual(self.client.post(
            "/auth/login", json={"email": "analyst@example.test", "password": "wrong"}
        ).status_code, 401)
        response = self.login()
        self.assertEqual(response.status_code, 200)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        token = self.client.cookies[SESSION_COOKIE]
        self.assertEqual(self.client.get("/auth/session").json()["user"]["email"], "analyst@example.test")
        self.assertEqual(self.client.post("/dataset/register", json=dataset).status_code, 200)
        self.assertEqual(self.client.post("/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/auth/session", headers={"Cookie": f"{SESSION_COOKIE}={token}"}).status_code, 401)

    def test_test_accounts_have_separate_dataset_access(self) -> None:
        self.assertEqual(self.login().status_code, 200)
        dataset = self.client.post("/dataset/register", json={"rows": [{"value": 5}]}).json()
        self.assertEqual(self.login("reviewer@example.test").status_code, 200)
        response = self.client.post("/agent/think/stream", json={
            "prompt": "Summarize", "model": "models/gemma-4-31b-it", "dataset_id": dataset["dataset_id"],
        })
        self.assertEqual(response.status_code, 403)

    def test_local_dashboard_can_send_credentials(self) -> None:
        origin = "http://127.0.0.1:3000"
        headers = {"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
        response = self.client.options("/auth/login", headers=headers)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")
        response = self.client.post(
            "/auth/login", headers={"Origin": "https://untrusted.example"},
            json={"email": "analyst@example.test", "password": "datapilot-test"},
        )
        self.assertEqual(response.status_code, 403)
