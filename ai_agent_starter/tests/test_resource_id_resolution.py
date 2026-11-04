import unittest

from ai_agent.core.memory import Memory
from ai_agent.core.orchestrator import resourceKeyAliases, resourceKeyForPlaceholder


class ResourceKeyTests(unittest.TestCase):
    def test_bare_id_resolves_to_its_collection(self):
        self.assertEqual(resourceKeyForPlaceholder("/users/{id}/", "id"), "user")
        self.assertEqual(resourceKeyForPlaceholder("/leaves/{id}/approve/", "id"), "leave")
        self.assertEqual(resourceKeyForPlaceholder("/documents/{id}/download/", "id"), "document")

    def test_named_placeholder_keeps_its_resource(self):
        self.assertEqual(resourceKeyForPlaceholder("/expenses/{expense_id}/", "expense_id"), "expense")

    def test_document_aliases_include_doc(self):
        self.assertIn("doc", resourceKeyAliases("document"))


class DiscardResourceIdTests(unittest.TestCase):
    def test_deleted_id_falls_back_to_previous_live_id(self):
        memory = Memory()
        memory.store_resource_id("Employee", "expense", 24)
        memory.store_resource_id("Employee", "expense", 30)

        memory.discardResourceId(resourceKeyAliases("expense"), 30)

        self.assertEqual(memory.resource_ids["Employee"]["expense_id"], 24)
        self.assertNotIn(30, memory.resource_id_pools["Employee"]["expense"])

    def test_last_deleted_id_is_removed(self):
        memory = Memory()
        memory.store_resource_id("Employee", "expense", 24)

        memory.discardResourceId(resourceKeyAliases("expense"), 24)

        self.assertNotIn("expense", memory.resource_ids["Employee"])


if __name__ == "__main__":
    unittest.main()
