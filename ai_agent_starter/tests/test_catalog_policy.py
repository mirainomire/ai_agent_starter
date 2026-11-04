import unittest

from ai_agent.core.catalog_policy import catalogPolicy, findRequestExample, makeUnique, resolveExample
from ai_agent.core.evaluators import expected_status
from ai_agent.core.memory import TestCase

CATALOG = {"endpoints": [
    {"method": "GET", "path": "/api/expenses/summary/", "access_policy": "authenticated; summary is owner-filtered",
     "admin_access": True},
    {"method": "GET", "path": "/api/expenses/{expense_id}/", "access_policy": "owner_or_admin", "admin_access": True},
    {"method": "POST", "path": "/api/benefits/", "access_policy": "authenticated for GET; admin_only for POST",
     "admin_access": True, "request_example": {"name": "Dental Plus"}},
    {"method": "PUT", "path": "/api/users/{id}/", "admin_access": True,
     "access_policy": "owner_or_admin; role changes within the profile payload are admin_only",
     "request_example": {"role": "admin"}},
    {"method": "POST", "path": "/api/auth/login/", "access_policy": "public", "admin_access": True},
    {"method": "GET", "path": "/api/odd/", "access_policy": "something new", "admin_access": True},
]}
POLICY = {**catalogPolicy(CATALOG), "account_role_map": {"Admin_HC": "Admin", "Employee": "Employee"}}


def expectedAccess(method: str, path: str, account: str, self_access: bool) -> str:
    return expected_status(POLICY, TestCase(method, path, account, self_access))["expected_access"]


class CatalogRulesTests(unittest.TestCase):
    def test_literal_route_wins_over_id_route(self):
        self.assertEqual(expectedAccess("GET", "/expenses/summary/", "Employee", True), "allow")

    def test_owner_scope(self):
        self.assertEqual(expectedAccess("GET", "/expenses/{expense_id}/", "Employee", True), "allow")
        self.assertEqual(expectedAccess("GET", "/expenses/{expense_id}/", "Employee", False), "deny")
        self.assertEqual(expectedAccess("GET", "/expenses/{expense_id}/", "Admin_HC", False), "allow")

    def test_method_specific_admin_only(self):
        self.assertEqual(expectedAccess("POST", "/benefits/", "Employee", True), "deny")

    def test_unmapped_policy_text_is_not_guessed(self):
        self.assertEqual(POLICY["catalog_unmapped"], ["GET /odd/"])
        self.assertEqual(POLICY["public_endpoints"], ["POST:/auth/login/"])


class FieldRuleTests(unittest.TestCase):
    def test_owner_writing_admin_only_field_is_denied(self):
        tc = TestCase("PUT", "/users/{id}/", "Employee", True, mutation={"json": {"role": "admin"}})
        self.assertEqual(expected_status(POLICY, tc)["expected_access"], "deny")
        self.assertEqual(expected_status(POLICY, tc)["expectation_source"], "policy.field_rules")

    def test_owner_without_the_field_keeps_owner_rule(self):
        tc = TestCase("PUT", "/users/{id}/", "Employee", True, mutation={"json": {"first_name": "A"}})
        self.assertEqual(expected_status(POLICY, tc)["expected_access"], "allow")

    def test_admin_may_write_the_field(self):
        tc = TestCase("PUT", "/users/{id}/", "Admin_HC", False, mutation={"json": {"role": "admin"}})
        self.assertEqual(expected_status(POLICY, tc)["expected_access"], "allow")


class RequestExampleTests(unittest.TestCase):
    def test_admin_only_field_is_removed_from_owner_payload(self):
        self.assertEqual(findRequestExample(POLICY, "PUT", "/users/{user_pk}/"), {})

    def test_unique_values_change_per_sequence(self):
        body = {"username": "new", "work_date": "2026-10-19", "month": 1, "year": 2026}
        first = makeUnique(body, "a")
        second = makeUnique(body, "b")
        self.assertNotEqual(first["work_date"], second["work_date"])
        self.assertNotEqual((first["month"], first["year"]), (second["month"], second["year"]))
        self.assertEqual(first["username"], "new_a")

    def test_unresolved_marker_is_reported(self):
        body, unresolved = resolveExample({"plan_id": "resolve_plan"}, {})
        self.assertEqual((body, unresolved), ({}, ["resolve_plan"]))


if __name__ == "__main__":
    unittest.main()
