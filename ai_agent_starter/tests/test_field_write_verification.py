import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

from ai_agent.core.evaluators import classifyResult
from ai_agent.core.memory import Memory, Result, TestCase
from ai_agent.core.orchestrator import execute
from ai_agent.core.tools_http import HttpClient

POLICY = {
    "account_role_map": {"admin": "Admin", "staff": "Employee"},
    "field_rules": [{"method": "PUT", "path": "/users/{id}/", "field": "role", "allow_roles": ["Admin"]}],
}
OPENAPI = {"paths": {"/users/{id}/": {"get": {}, "put": {}}}}


def escalationCase() -> TestCase:
    return TestCase("PUT", "/users/{id}/", "staff", True,
                    mutation={"type": "ADMIN_FIELD_WRITE", "json": {"role": "admin"}, "disposable_actor": True})


class ClassifyResultTests(unittest.TestCase):
    def test_accepted_write_is_judged_by_the_stored_value(self):
        for applied, expected in ((False, "TN"), (True, "FP"), (None, "FP")):
            with self.subTest(applied=applied):
                result = Result(tc=escalationCase(), status_code=200, body={}, write_applied=applied)
                self.assertEqual(classifyResult(POLICY, result), expected)


class FieldWriteVerificationTests(unittest.TestCase):
    def run_escalation(self, stored_role: str) -> Result:
        auth = SimpleNamespace(roles={"admin": {}, "staff": {}}, get_token=lambda alias: f"{alias}-token",
                               get_user_id=lambda alias: None,
                               get_policy_role=lambda alias: POLICY["account_role_map"][alias])
        responses: Dict[Any, Dict[str, Any]] = {
            ("POST", "/admin/users/"): {"status_code": 201, "body": {"id": 553}},
            ("POST", "/auth/login/"): {"status_code": 200, "body": {"access": "temp-token"}},
            ("PUT", "/users/553/"): {"status_code": 200, "body": {}},
            ("GET", "/users/553/"): {"status_code": 200, "body": {"id": 553, "role": stored_role}},
        }

        def respond(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            method = args[0] if args else kwargs.get("method")
            path = args[1] if len(args) > 1 else kwargs.get("path")
            return responses.get((method, path), {"status_code": 200, "body": {}})

        memory = Memory()
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient("http://example.invalid", artifacts_dir=folder)
            with patch.object(http, "request", side_effect=respond):
                execute(memory, http, auth, POLICY, tests=[escalationCase()], openapi=OPENAPI,
                        allow_state_changes=True)
        return memory.results[0]

    def test_ignored_read_only_field_is_a_correct_denial(self):
        result = self.run_escalation(stored_role="employee")
        self.assertIs(result.write_applied, False)
        self.assertEqual(classifyResult(POLICY, result), "TN")

    def test_stored_privileged_value_is_an_escalation(self):
        result = self.run_escalation(stored_role="admin")
        self.assertIs(result.write_applied, True)
        self.assertEqual(classifyResult(POLICY, result), "FP")


if __name__ == "__main__":
    unittest.main()
