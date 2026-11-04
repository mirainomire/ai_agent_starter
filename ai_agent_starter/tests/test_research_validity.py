import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from ai_agent.core.evaluators import (
    classify,
    effective_account_alias,
    effective_policy_role,
    expected_status,
    metrics,
)
from ai_agent.core.memory import Memory, Result, TestCase
from ai_agent.core.orchestrator import (
    AgentOrchestrator,
    _discover_ids,
    _gemini_generate_content,
    execute,
)
from ai_agent.core.reporters import save_json_report


POLICY = {
    "account_role_map": {
        "Admin_HC": "Admin",
        "Employee": "Employee",
        "Employee_2": "Employee",
    },
    "rbac_rules": {
        "Admin": {
            "permissions": ["rbac_admin"],
            "allowed_endpoints": ["GET:/users/{user_id}/"],
        },
        "Employee": {
            "permissions": ["view_own_profile"],
            "allowed_endpoints": ["GET:/users/{user_id}/"],
        },
    },
}


class ClassificationTests(unittest.TestCase):
    def test_allow_positive_confusion_labels(self):
        allow = {"status_in": [200], "status_not_in": [401, 403]}
        deny = {"status_in": [401, 403, 404], "status_not_in": [200]}

        self.assertEqual(classify(allow, 200), "TP_ALLOW")
        self.assertEqual(classify(allow, 403), "FN")
        self.assertEqual(classify(deny, 403), "TN")
        self.assertEqual(classify(deny, 200), "FP")

    def test_not_found_is_not_counted_as_authorization_success(self):
        allow = {"status_in": [200], "status_not_in": [401, 403]}
        deny = {"status_in": [401, 403, 404], "status_not_in": [200]}

        self.assertEqual(classify(allow, 404), "NOT_FOUND")
        self.assertEqual(classify(deny, 404), "NOT_FOUND")

    def test_metric_formulas(self):
        calculated = metrics({"TP": 8, "FP": 2, "FN": 4, "TN": 6})

        self.assertAlmostEqual(calculated["precision"], 0.8)
        self.assertEqual(calculated["recall"], 0.667)
        self.assertEqual(calculated["f1"], 0.727)
        self.assertAlmostEqual(calculated["accuracy"], 0.7)


class IdentityTests(unittest.TestCase):
    def test_employee_2_is_an_account_alias_not_a_distinct_role(self):
        tc = TestCase("GET", "/users/{user_id}/", "Employee_2", True)

        self.assertEqual(effective_account_alias(tc), "Employee_2")
        self.assertEqual(effective_policy_role(POLICY, tc), "Employee")
        self.assertEqual(classify(expected_status(POLICY, tc), 200), "TP_ALLOW")

    def test_mutation_account_controls_expected_role(self):
        tc = TestCase(
            "GET",
            "/users/{user_id}/",
            "Employee",
            False,
            mutation={"as_role": "Admin_HC"},
        )

        self.assertEqual(effective_account_alias(tc), "Admin_HC")
        self.assertEqual(effective_policy_role(POLICY, tc), "Admin")
        self.assertEqual(classify(expected_status(POLICY, tc), 200), "TP_ALLOW")


class PlanningTests(unittest.TestCase):
    def test_other_principal_cases_only_for_path_id_operations(self):
        root = Path(__file__).resolve().parents[1]
        policy = yaml.safe_load(
            (root / "ai_agent" / "config" / "policy.yaml").read_text(encoding="utf-8")
        )
        openapi = json.loads(
            (root / "ai_agent" / "data" / "openapi.json").read_text(encoding="utf-8")
        )
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._max_endpoints = 0

        plan, endpoints = orchestrator._plan_from_policy(
            policy,
            ["Admin_HC", "Employee", "Employee_2"],
            openapi,
        )

        id_operations = sum("{" in e["path"] for e in endpoints)
        expected_cases = 3 * (len(endpoints) + id_operations)
        self.assertEqual(len(plan), expected_cases)
        self.assertEqual(
            len(
                {
                    (item["method"], item["path"], item["role"], item["self_access"])
                    for item in plan
                }
            ),
            expected_cases,
        )
        self.assertTrue(
            all("{" in item["path"] for item in plan if not item["self_access"])
        )


class LlmOpenApiContextTests(unittest.TestCase):
    def test_generation_reads_descriptions_schemas_and_id_memory(self):
        prompts = []

        class Completions:
            def create(self, **kwargs):
                prompts.append(kwargs["messages"][-1]["content"])
                content = json.dumps({
                    "tests": [{
                        "method": "GET",
                        "path": "/documents/{doc_id}/",
                        "role": "Employee",
                        "mutations": [{"type": "IDOR", "variant": "other"}],
                    }]
                })
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        orchestrator.llm_provider = "openai"
        orchestrator._llm_generation_batch_size = 10
        memory = Memory()
        memory.store_resource_id("Employee", "doc", 42)
        openapi = {
            "paths": {
                "/documents/{doc_id}/": {
                    "get": {
                        "summary": "Get owned document",
                        "description": "Owner or admin only",
                        "responses": {"200": {"description": "Allowed"}},
                    }
                }
            }
        }

        cases = orchestrator._generate_llm(
            [{
                "method": "GET",
                "path": "/documents/{doc_id}/",
                "role": "Employee",
                "self_access": True,
            }],
            POLICY,
            openapi,
            memory,
        )

        self.assertEqual(len(cases), 2)
        self.assertIn("Owner or admin only", prompts[0])
        self.assertIn('"doc": [42]', prompts[0])


class FixtureExecutionTests(unittest.TestCase):
    class FakeAuth:
        roles = {
            "Admin_HC": {"policy_role": "Admin"},
            "Employee": {"policy_role": "Employee"},
            "Employee_2": {"policy_role": "Employee"},
        }

        def get_token(self, role):
            return f"token-{role}"

        def get_policy_role(self, role):
            return self.roles[role]["policy_role"]

        def get_user_id(self, role):
            return {"Admin_HC": 1, "Employee": 2, "Employee_2": 3}[role]

    def test_get_discovery_stores_trailing_slash_detail_id_in_memory(self):
        class DiscoveryHttp:
            def request(self, method, path, **kwargs):
                return {"status_code": 200, "body": {"data": [{"id": 77}]}}

        memory = Memory()
        openapi = {
            "paths": {
                "/documents/": {"get": {"summary": "List documents"}},
                "/documents/{doc_id}/": {"get": {"summary": "Get document"}},
            }
        }

        _discover_ids(
            DiscoveryHttp(),
            self.FakeAuth(),
            openapi,
            memory,
            ["Employee"],
        )

        self.assertEqual(memory.resource_ids["Employee"]["doc_id"], 77)
        self.assertEqual(memory.resource_id_pools["Employee"]["doc_id"], [77])


class GeminiCompatibilityTests(unittest.TestCase):
    def test_current_sdk_models_interface(self):
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(text="new-sdk-response")

        client = SimpleNamespace(models=Models())
        response = _gemini_generate_content(
            client,
            "gemini-test",
            "prompt",
            "application/json",
        )

        self.assertEqual(response, "new-sdk-response")
        self.assertEqual(calls[0]["model"], "models/gemini-test")
        self.assertEqual(calls[0]["contents"], "prompt")
        self.assertEqual(calls[0]["config"]["temperature"], 0.0)

    def test_legacy_sdk_interface(self):
        calls = []

        class LegacyClient:
            def generate_content(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return SimpleNamespace(text="legacy-response")

        response = _gemini_generate_content(
            LegacyClient(),
            "models/gemini-test",
            "prompt",
            "application/json",
        )

        self.assertEqual(response, "legacy-response")
        self.assertEqual(calls[0][0], "prompt")


class ReportingTests(unittest.TestCase):
    def test_report_separates_case_counts_and_actor_identity(self):
        tests = [
            TestCase("GET", "/users/{user_id}/", "Employee_2", True),
            TestCase("GET", "/users/{user_id}/", "Employee", False),
        ]
        results = [Result(tests[0], 200)]

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "ai_agent" / "runs" / "report.json"
            save_json_report(
                str(report_path),
                results,
                POLICY,
                tests,
                ["Employee_2", "Employee"],
                [{"method": "GET", "path": "/users/{user_id}/"}],
                run_metadata={"provider": "test"},
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            scenarios = (
                Path(temp_dir) / "ai_agent" / "data" / "test_scenarios.csv"
            ).read_text(encoding="utf-8")

        self.assertEqual(report["summary"]["generated_cases"], 2)
        self.assertEqual(report["summary"]["result_records"], 1)
        self.assertEqual(report["summary"]["evaluated_cases"], 1)
        self.assertEqual(report["results"][0]["actor_account"], "Employee_2")
        self.assertEqual(report["results"][0]["actor_role"], "Employee")
        self.assertEqual(report["label_schema"]["positive_class"], "ALLOW")
        self.assertIn("actor_account,actor_role", scenarios.splitlines()[0])
        self.assertIn("Employee_2,Employee", scenarios.splitlines()[1])


if __name__ == "__main__":
    unittest.main()
