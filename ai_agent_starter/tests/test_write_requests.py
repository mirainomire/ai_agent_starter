import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai_agent.core.memory import Memory, TestCase
from ai_agent.core.orchestrator import execute
from ai_agent.core.tools_http import HttpClient
from ai_agent.core.utils import extract_request_schema, generate_payload_from_schema


def specification(method, path, media):
    return {"paths": {path: {method.lower(): {
        "requestBody": {"content": {"application/json": media}}
    }}}}


class WriteRequestTests(unittest.TestCase):
    def test_enabled_writes_use_memory_and_openapi_example(self):
        auth = SimpleNamespace(
            roles={"account": {}}, get_token=lambda role: "test-token",
            get_user_id=lambda role: None, get_policy_role=lambda role: role,
        )
        media = {
            "schema": {"type": "object", "required": ["title"],
                       "properties": {"title": {"type": "string"}}},
            "examples": {"sample": {"value": {"title": "Documented value"}}},
        }
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient("http://example.invalid", artifacts_dir=folder)
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                with self.subTest(method=method):
                    path = "/records/" if method == "POST" else "/records/{record_id}/"
                    memory = Memory()
                    memory.store_resource_id("account", "record_id", 37)
                    tc = TestCase(method, path, "account", True)
                    with patch.object(http, "request", return_value={"status_code": 200, "body": {}}) as request:
                        execute(memory, http, auth, {}, tests=[tc],
                                openapi=specification(method, path, media),
                                allow_state_changes=True)
                    request.assert_called_once()
                    sent = request.call_args.kwargs
                    self.assertEqual(sent["path"], "/records/" if method == "POST" else "/records/37/")
                    if method != "DELETE":
                        self.assertEqual(sent["json_body"], {"title": "Documented value"})
                    self.assertEqual(memory.results[0].status_code, 200)

    def test_disabled_writes_still_skip_without_request(self):
        memory = Memory()
        http = SimpleNamespace(compatibility_profile="generic")
        execute(memory, http, None, {}, tests=[TestCase("DELETE", "/records/37/", "account", True)])
        self.assertEqual(memory.results[0].status_code, 0)
        self.assertTrue(memory.results[0].body["skipped"])

    def test_examples_are_payload_values_and_not_example_names(self):
        for media, expected in (
            ({"example": {"title": "Inline"}}, {"title": "Inline"}),
            ({"examples": {"named": {"value": {"title": "Named"}}}}, {"title": "Named"}),
            ({"schema": {"type": "object", "example": {"title": "Schema"}}}, {"title": "Schema"}),
            ({"example": [{"title": "Array"}]}, [{"title": "Array"}]),
            ({"example": {}}, {}),
        ):
            with self.subTest(media=media):
                info = extract_request_schema(specification("POST", "/records/", media), "POST", "/records/")
                self.assertEqual(generate_payload_from_schema(info), expected)

    def test_required_id_uses_discovered_memory_without_example(self):
        schema = {"type": "object", "required": ["record_id"],
                  "properties": {"record_id": {"type": "integer"}}}
        self.assertEqual(generate_payload_from_schema(schema, {"record_id": 37}), {"record_id": 37})


if __name__ == "__main__":
    unittest.main()
