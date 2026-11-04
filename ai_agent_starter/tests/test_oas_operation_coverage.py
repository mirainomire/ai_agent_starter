import json
import unittest
from unittest.mock import patch

from ai_agent.core.memory import Memory, TestCase
from ai_agent.core.orchestrator import (
    AgentOrchestrator,
    merge_planned_and_llm_cases,
    validate_operation_coverage,
)


class OasOperationCoverageTests(unittest.TestCase):
    def test_all_oas_operations_are_sent_to_model_for_every_account(self):
        spec = {'paths': {f'/resource/{i}/': {'get': {'description': 'Read record'}} for i in range(28)}}
        accounts = ['Admin_HC', 'Employee', 'Employee_2']
        agent = AgentOrchestrator.__new__(AgentOrchestrator)
        agent.client = object()
        agent.llm_provider, agent.llm_model = 'gemini', 'configured-model'
        agent._max_endpoints = 0
        agent._llm_generation_batch_size = 28
        plan, _ = agent._plan_from_policy({}, accounts, spec)
        received = []

        def respond(client, model, prompt, mime, **kwargs):
            batch = json.loads(prompt.split('CandidatePairs:\n', 1)[1].split('\n\nAuthenticationAccounts:', 1)[0])
            received.extend(batch)
            schema = kwargs['response_json_schema']['properties']['tests']
            self.assertNotIn('minItems', schema)
            self.assertNotIn('maxItems', schema)
            self.assertNotIn('maxItems', schema['items']['properties']['mutations'])
            self.assertGreater(kwargs['max_output_tokens'], 3000)
            return json.dumps({'tests': [{**p, 'mutations': []} for p in batch]})

        with patch('ai_agent.core.orchestrator._gemini_generate_content', side_effect=respond) as model:
            cases = agent._generate_llm(plan, {}, spec, Memory())
        coverage = validate_operation_coverage(plan, cases)
        self.assertEqual(model.call_count, 3)
        self.assertEqual(len(received), 84)
        self.assertEqual(len(cases), 84)
        self.assertEqual(coverage['operations'], 28)
        self.assertEqual(coverage['covered_pairs'], 84)
        for account in accounts:
            self.assertEqual(sum(tc.role == account for tc in cases), 28)

    def test_llm_variants_cannot_remove_planned_access_modes(self):
        plan = [
            {"method": "GET", "path": "/documents/{doc_id}/", "role": "Employee", "self_access": True},
            {"method": "GET", "path": "/documents/{doc_id}/", "role": "Employee", "self_access": False},
        ]
        llm_cases = [
            TestCase("GET", "/documents/{doc_id}/", "Employee", True, mutation={"type": "oas_operation"}),
            TestCase("GET", "/documents/{doc_id}/", "Employee", False, mutation={"type": "escalate"}),
        ]

        merged = merge_planned_and_llm_cases(plan, llm_cases)

        self.assertEqual(
            [(tc.self_access, (tc.mutation or {}).get("type")) for tc in merged],
            [(True, None), (False, None), (False, "escalate")],
        )

    def test_partial_selection_is_rejected_without_adding_cases(self):
        plan = [dict(method='GET', path='/resource/', role=role) for role in ('a', 'b', 'c')]
        selected = [TestCase('GET', '/resource/', 'a', True)]
        with self.assertRaisesRegex(ValueError, '2 of 3'):
            validate_operation_coverage(plan, selected)
        self.assertEqual(len(selected), 1)

    def test_operations_are_method_path_pairs_not_a_fixed_count(self):
        spec = {'paths': {'/resource/': {'get': {}, 'post': {}, 'parameters': []},
                          '/resource/{id}/': {'put': {}, 'delete': {}}}}
        agent = AgentOrchestrator.__new__(AgentOrchestrator)
        agent._max_endpoints = 0
        for count in (1, 3, 6):
            with self.subTest(accounts=count):
                plan, operations = agent._plan_from_policy({}, [f'actor_{i}' for i in range(count)], spec)
                cases = [TestCase(**pair) for pair in plan]
                coverage = validate_operation_coverage(plan, cases)
                self.assertEqual(len(operations), 4)
                self.assertEqual(coverage['operation_account_pairs'], 4 * count)

    def test_duplicate_variants_do_not_cover_a_missing_operation(self):
        plan = [dict(method=method, path='/resource/', role='actor') for method in ('GET', 'POST')]
        cases = [TestCase('GET', '/resource/', 'actor', True),
                 TestCase('GET', '/resource/', 'actor', False)]
        with self.assertRaisesRegex(ValueError, 'POST /resource/'):
            validate_operation_coverage(plan, cases)


if __name__ == '__main__':
    unittest.main()
