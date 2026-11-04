import json
import unittest
from types import SimpleNamespace

from ai_agent.core.memory import Result, TestCase
from ai_agent.core.orchestrator import (
    _followups_llm,
    _friendly_llm_error,
    _llm_call_with_retry,
    generate_followups,
    observe,
)


POLICY = {
    "account_role_map": {"Employee": "Employee"},
    "rbac_rules": {
        "Employee": {
            "permissions": ["view_own_document"],
            "allowed_endpoints": ["GET:/documents/{doc_id}/"],
        }
    },
}


class SuspiciousFollowupTests(unittest.TestCase):
    def test_transient_provider_error_is_retried_with_a_bound(self):
        attempts = []

        def call():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503 UNAVAILABLE high demand")
            return "ok"

        self.assertEqual(
            _llm_call_with_retry(call, retries=2, delay_s=0),
            "ok",
        )
        self.assertEqual(len(attempts), 3)

    def test_dns_failure_is_retried(self):
        attempts = []

        def call():
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("[Errno 11001] getaddrinfo failed")
            return "ok"

        self.assertEqual(
            _llm_call_with_retry(call, retries=2, delay_s=0),
            "ok",
        )
        self.assertEqual(len(attempts), 2)

    def test_dropped_connection_is_retried(self):
        attempts = []

        def call():
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("Server disconnected without sending a response.")
            return "ok"

        self.assertEqual(_llm_call_with_retry(call, retries=2, delay_s=0), "ok")
        self.assertEqual(len(attempts), 2)

    def test_bad_request_is_not_retried(self):
        attempts = []

        def call():
            attempts.append(1)
            raise RuntimeError("400 INVALID_ARGUMENT")

        with self.assertRaises(RuntimeError):
            _llm_call_with_retry(call, retries=2, delay_s=0)
        self.assertEqual(len(attempts), 1)

    def test_provider_quota_error_has_actionable_notification(self):
        message = _friendly_llm_error(
            "429 RESOURCE_EXHAUSTED quota exceeded. Please retry in 2h18m."
        )

        self.assertIn("LLM quota exhausted", message)
        self.assertIn("2h18m", message)

    def test_not_found_is_selected_for_bounded_followup(self):
        result = Result(
            TestCase("GET", "/documents/{doc_id}/", "Employee", True),
            status_code=404,
        )

        observations = observe([result], POLICY)
        followups = generate_followups(observations, ["Employee"], 0, 2)

        self.assertEqual(observations[0]["classification"], "NOT_FOUND")
        self.assertTrue(observations[0]["followup_recommended"])
        self.assertEqual(len(followups), 4)
        self.assertTrue(all(item.depth == 1 for item in followups))

    def test_llm_cannot_invent_endpoint_or_actor(self):
        payload = {
            "tests": [{
                "seed_index": 0,
                "path": "/invented/",
                "role": "Admin",
                "variant": "other",
                "rationale": "Confirm the suspicious 404 with another ownership variant.",
            }]
        }

        class Completions:
            def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        observations = [{
            "method": "GET",
            "path": "/documents/{doc_id}/",
            "role": "Employee",
            "self_access": True,
            "actual": 404,
            "classification": "NOT_FOUND",
            "followup_recommended": True,
            "mutation": {},
            "body_excerpt": "",
        }]

        generated = _followups_llm(
            client, "openai", observations, current_depth=0
        )

        self.assertEqual(generated[0]["path"], "/documents/{doc_id}/")
        self.assertEqual(generated[0]["role"], "Employee")
        self.assertEqual(generated[0]["mutation"]["type"], "LLM_FOLLOWUP")
        self.assertEqual(generated[0]["depth"], 1)


if __name__ == "__main__":
    unittest.main()
