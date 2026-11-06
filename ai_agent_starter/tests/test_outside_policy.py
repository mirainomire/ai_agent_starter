import unittest

from ai_agent.core.evaluators import SESSION_INVALIDATION_SOURCE, classify, confusion_counts, expected_status, metrics
from ai_agent.core.memory import Result, TestCase

POLICY = {"rules": [{"method": "GET", "path": "/users/{id}/", "allow": [{"role": "Employee", "self": True}]}]}


def logoutCase() -> TestCase:
    return TestCase("GET", "/users/{id}/", "Employee", True, mutation={"type": "JWT_AFTER_LOGOUT"})


class OutsidePolicyTests(unittest.TestCase):
    def test_token_reuse_after_logout_is_a_builtin_check(self):
        expectation = expected_status(POLICY, logoutCase())
        self.assertEqual(expectation["expectation_source"], SESSION_INVALIDATION_SOURCE)
        self.assertEqual(classify(expectation, 200), "OUTSIDE_POLICY")
        self.assertEqual(classify(expectation, 401), "OUTSIDE_POLICY_PASS")

    def test_unauthenticated_requests_still_follow_the_policy(self):
        tc = TestCase("GET", "/users/{id}/", "Employee", True, mutation={"type": "NO_AUTH", "no_auth": True})
        self.assertEqual(classify(expected_status(POLICY, tc), 200), "FP")

    def test_outside_policy_outcomes_do_not_change_metrics(self):
        allowed = Result(tc=TestCase("GET", "/users/{id}/", "Employee", True), status_code=200, body={})
        baseline = metrics(confusion_counts([allowed], POLICY))
        counts = confusion_counts([allowed, Result(tc=logoutCase(), status_code=200, body={})], POLICY)
        self.assertEqual((counts["FP"], counts["OUTSIDE_POLICY"]), (0, 1))
        self.assertEqual(metrics(counts), baseline)


if __name__ == "__main__":
    unittest.main()
