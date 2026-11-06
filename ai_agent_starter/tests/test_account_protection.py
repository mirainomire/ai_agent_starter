import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

from ai_agent.core.id_selection import accountRecordTemplate
from ai_agent.core.memory import Memory, TestCase
from ai_agent.core.orchestrator import execute
from ai_agent.core.tools_http import HttpClient

ACCOUNT_IDS = {"admin": 46, "staff": 47}
OPENAPI = {"paths": {
    "/users/": {"post": {"requestBody": {"content": {"application/json": {"schema": {
        "type": "object", "required": ["username"], "properties": {"username": {"type": "string"}}}}}}}},
    "/users/{id}/": {"get": {}, "delete": {}},
    "/users/{id}/role/": {"put": {"requestBody": {"content": {"application/json": {"schema": {
        "type": "object", "properties": {"role": {"type": "string"}}}}}}}},
}}


def fakeAuth() -> SimpleNamespace:
    return SimpleNamespace(roles={alias: {} for alias in ACCOUNT_IDS}, get_token=lambda alias: f"{alias}-token",
                           get_user_id=lambda alias: ACCOUNT_IDS.get(alias), get_policy_role=lambda alias: alias)


def sentCalls(request_mock: Any) -> List[Dict[str, Any]]:
    calls = []
    for call in request_mock.call_args_list:
        method = call.args[0] if call.args else call.kwargs.get("method")
        path = call.args[1] if len(call.args) > 1 else call.kwargs.get("path")
        calls.append({"method": method, "path": path, **call.kwargs})
    return calls


class AccountRecordTemplateTests(unittest.TestCase):
    def test_only_the_account_record_itself_is_protected(self):
        self.assertEqual(accountRecordTemplate("/users/{id}/"), "/users/{id}/")
        self.assertEqual(accountRecordTemplate("/users/{id}/role/"), "/users/{id}/")
        self.assertIsNone(accountRecordTemplate("/payroll/{user_id}/"))
        self.assertIsNone(accountRecordTemplate("/attendance/{attendance_id}/"))
        self.assertIsNone(accountRecordTemplate("/admin/audit-logs/{user_id}/"))


class AccountProtectionTests(unittest.TestCase):
    def run_case(self, tc: TestCase, responses: Dict[str, Dict[str, Any]]):
        memory = Memory()
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient("http://example.invalid", artifacts_dir=folder)

            def respond(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                method = args[0] if args else kwargs.get("method")
                return responses.get(str(method).upper(), {"status_code": 200, "body": {}})

            with patch.object(http, "request", side_effect=respond) as request:
                execute(memory, http, fakeAuth(), {}, tests=[tc], openapi=OPENAPI, allow_state_changes=True)
        return memory, sentCalls(request)

    def test_role_change_on_own_record_repeats_current_values(self):
        tc = TestCase("PUT", "/users/{id}/role/", "admin", True, mutation={"json": {"role": "employee"}})
        memory, calls = self.run_case(tc, {"GET": {"status_code": 200, "body": {"id": 46, "role": "admin"}}})
        snapshot = next(call for call in calls if call["method"] == "GET")
        self.assertEqual((snapshot["path"], snapshot["token"]), ("/users/46/", "admin-token"))
        write = next(call for call in calls if call["method"] == "PUT")
        self.assertEqual(write["path"], "/users/46/role/")
        self.assertEqual(write["json_body"], {"role": "admin"})
        self.assertEqual(memory.results[0].status_code, 200)

    def test_delete_of_own_account_is_skipped_without_request(self):
        memory, calls = self.run_case(TestCase("DELETE", "/users/{id}/", "admin", True), {})
        self.assertFalse([call for call in calls if call["method"] == "DELETE"])
        self.assertEqual(memory.results[0].status_code, 0)
        self.assertTrue(memory.results[0].body["skipped"])

    def test_delete_of_other_account_targets_a_disposable_record(self):
        memory, calls = self.run_case(TestCase("DELETE", "/users/{id}/", "admin", False),
                                      {"POST": {"status_code": 201, "body": {"id": 900}},
                                       "DELETE": {"status_code": 204, "body": {}}})
        deleted = [call["path"] for call in calls if call["method"] == "DELETE"]
        self.assertEqual(deleted, ["/users/900/"])
        self.assertEqual(memory.results[0].status_code, 204)

    def test_method_override_to_delete_is_treated_as_delete(self):
        tc = TestCase("GET", "/users/{id}/", "admin", True,
                      mutation={"type": "METHOD_OVERRIDE", "headers": {"X-HTTP-Method-Override": "DELETE"}})
        memory, calls = self.run_case(tc, {})
        self.assertFalse(calls)
        self.assertTrue(memory.results[0].body["skipped"])


if __name__ == "__main__":
    unittest.main()
