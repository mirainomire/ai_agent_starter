import json
import unittest

from ai_agent.core.id_selection import buildIdSlots, selectIdsWithLlm

SPEC = {
    "paths": {
        "/leaves/": {"get": {}},
        "/leaves/{id}/approve/": {"put": {"summary": "Approve a pending leave"}},
        "/leaves/{id}/": {"delete": {}},
        "/users/{id}/": {"get": {}},
    }
}
USER_IDS = {"Admin_HC": 1, "Employee": 2, "Employee_2": 3}
LEAVES = [
    {"id": 10, "user_id": 2, "status": "approved", "reason": "private text"},
    {"id": 11, "user_id": 2, "status": "pending"},
    {"id": 20, "user_id": 3, "status": "pending"},
]


def fetchLeaves(collection: str, account: str) -> dict:
    return {"data": LEAVES if collection == "/leaves/" else []}


class BuildIdSlotsTests(unittest.TestCase):
    def setUp(self):
        self.slots, self.single = buildIdSlots(SPEC, list(USER_IDS), fetchLeaves, USER_IDS.get)

    def test_only_owned_records_are_offered(self):
        slot = next(s for s in self.slots if s["owner_account"] == "Employee")
        self.assertEqual([c["candidate"] for c in slot["candidates"]], ["11", "10"])

    def test_single_candidate_needs_no_model_call(self):
        self.assertEqual(self.single[("PUT", "/leaves/{id}/approve/", "Employee_2")]["id"], 20)
        self.assertNotIn("Employee_2", [s["owner_account"] for s in self.slots])

    def test_identity_and_delete_operations_are_excluded(self):
        templates = {(s["method"], s["path_template"]) for s in self.slots}
        templates |= {(key[0], key[1]) for key in self.single}
        self.assertEqual(templates, {("PUT", "/leaves/{id}/approve/")})

    def test_admin_without_own_records_gets_every_record(self):
        slots, _ = buildIdSlots(SPEC, list(USER_IDS), fetchLeaves, USER_IDS.get, admin_accounts=["Admin_HC"])
        admin_slot = next(s for s in slots if s["owner_account"] == "Admin_HC")
        self.assertEqual([c["candidate"] for c in admin_slot["candidates"]], ["20", "11", "10"])

    def test_free_text_fields_never_reach_the_model(self):
        fields = [c["fields"] for s in self.slots for c in s["candidates"]]
        self.assertFalse(any("reason" in f for f in fields))


class SelectIdsWithLlmTests(unittest.TestCase):
    def test_one_call_and_invented_ids_are_rejected(self):
        slots, _ = buildIdSlots(SPEC, list(USER_IDS), fetchLeaves, USER_IDS.get)
        calls = []

        def fakeModel(prompt: str, schema: dict, max_tokens: int) -> str:
            calls.append(prompt)
            return json.dumps({"choices": [
                {"slot": slots[0]["slot"], "candidate": "11", "reason": "status pending"},
                {"slot": slots[0]["slot"], "candidate": "999", "reason": "invented"},
            ]})

        choices, rejected = selectIdsWithLlm(fakeModel, slots, "prompt")

        self.assertEqual(len(calls), 1)
        self.assertEqual(choices[("PUT", "/leaves/{id}/approve/", "Employee")]["id"], 11)
        self.assertEqual(rejected, [{"slot": slots[0]["slot"], "candidate": "999"}])


if __name__ == "__main__":
    unittest.main()
