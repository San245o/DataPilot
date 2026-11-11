import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from image_import import _checked_extraction
from main import app


def extracted_table(**changes):
    return {
        "status": "ready",
        "reason": "",
        "columns": [{"name": "Item", "kind": "text"}, {"name": "Quantity", "kind": "number"}],
        "rows": [["001", "12"], ["002", "9"]],
        "uncertain_cells": [],
        **changes,
    }


class ImageImportTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 41001))
        self.addCleanup(self.client.close)

    def test_unclear_cells_are_blank_and_marked_for_review(self):
        result = _checked_extraction(json.dumps(extracted_table(uncertain_cells=[{"row": 1, "column": 1}])))
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.rows, [["001", "12"], ["002", None]])
        self.assertEqual(result.uncertain_cells[0].row, 1)

    def test_rejected_photos_cannot_produce_a_table(self):
        result = _checked_extraction(json.dumps(extracted_table(status="cannot_convert", reason="This is a landscape.")))
        self.assertEqual(result.reason, "This is a landscape.")
        self.assertEqual(result.columns, [])
        self.assertEqual(result.rows, [])

    def test_invalid_model_tables_are_rejected(self):
        for changes in ({"rows": [["001"]]}, {"uncertain_cells": [{"row": 3, "column": 0}]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _checked_extraction(json.dumps(extracted_table(**changes)))

    def test_auth_and_file_checks_run_before_extraction(self):
        path = "/dataset/extract-image"
        headers = {"Content-Type": "image/png"}
        png = b"\x89PNG\r\n\x1a\n"
        with patch("image_import._extract_from_image") as extract:
            self.assertEqual(self.client.post(path, headers=headers, content=png).status_code, 401)
            self.assertEqual(self.client.post(
                "/auth/login", json={"email": "analyst@example.test", "password": "datapilot-test"},
            ).status_code, 200)
            self.assertEqual(self.client.post(path, content=b"hello").status_code, 415)
            self.assertEqual(self.client.post(path, headers=headers, content=b"hello").status_code, 422)
            with patch("image_import.MAX_IMAGE_BYTES", len(png)):
                self.assertEqual(self.client.post(path, headers=headers, content=png + b"more").status_code, 413)
            extract.assert_not_called()
            extract.return_value = _checked_extraction(json.dumps(extracted_table()))
            response = self.client.post(path, headers=headers, content=png)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["rows"][0], ["001", "12"])
            extract.assert_called_once_with(png, "image/png")
