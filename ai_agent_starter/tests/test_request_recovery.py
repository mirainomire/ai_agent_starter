import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_agent.core.evaluators import classify
from ai_agent.core.memory import Memory, Result, TestCase
from ai_agent.core.orchestrator import execute, _state_compatible_resource
from ai_agent.core.reporters import save_json_report
from ai_agent.core.tools_http import HttpClient


class RequestRecoveryTests(unittest.TestCase):
    def test_duplicate_create_retries_with_a_unique_schema_field(self):
        schema = {'type': 'object', 'required': ['handle'], 'properties': {
            'handle': {'type': 'string', 'maxLength': 30, 'example': 'existing'}}}
        spec = {'paths': {'/accounts/': {'post': {'requestBody': {
            'content': {'application/json': {'schema': schema}}}}}}}
        sent = []
        def request(**kwargs):
            sent.append(copy.deepcopy(kwargs['json_body']))
            return ({'status_code': 400, 'body': {'error': 'Handle already exists'}}
                    if len(sent) == 1 else {'status_code': 201, 'body': {}})
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient('http://example.invalid', artifacts_dir=folder)
            with patch.object(http, 'request', side_effect=request):
                result = http.smart_request('POST', '/accounts/', None, spec)
        self.assertEqual(result['status_code'], 201)
        self.assertNotEqual(sent[0]['handle'], sent[1]['handle'])
        self.assertLessEqual(len(sent[1]['handle']), 30)

    def test_update_omits_optional_duplicate_field(self):
        payload = {'handle': 'existing', 'description': 'update'}
        repaired = HttpClient._repair_unique_conflict('PUT', payload, {
            'details': {'handle': ['A record with that handle already exists.']}},
            {'schema': {'properties': {'handle': {'type': 'string'}}}})
        self.assertTrue(repaired)
        self.assertEqual(payload, {'description': 'update'})

    def test_state_retry_uses_observed_record_of_same_owner(self):
        auth = SimpleNamespace(roles={'actor': {}}, get_token=lambda role: 'token',
                               get_user_id=lambda role: None, get_policy_role=lambda role: role)
        memory = Memory()
        memory.store_resource_id('other', 'record_id', 99, record={'status': 'pending'})
        discovery = TestCase('GET', '/records/', 'actor', True)
        tc = TestCase('PUT', '/records/{record_id}/', 'actor', True)
        spec = {'paths': {'/records/': {'get': {'summary': 'List records'}},
                          '/records/{record_id}/': {'put': {'summary': 'Update record'}}}}
        responses = [
            {'status_code': 200, 'body': {'data': [
                {'id': 1, 'status': 'pending'}, {'id': 2, 'status': 'approved'}]}},
            {'status_code': 400, 'body': {'error': 'Cannot edit record that is not in pending status'}},
            {'status_code': 200, 'body': {}},
        ]
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient('http://example.invalid', artifacts_dir=folder)
            with patch.object(http, 'request', side_effect=responses) as request:
                execute(memory, http, auth, {}, tests=[discovery, tc], openapi=spec, allow_state_changes=True)
        self.assertEqual([c.kwargs['path'] for c in request.call_args_list],
                         ['/records/', '/records/2/', '/records/1/'])
        self.assertEqual(memory.results[-1].status_code, 200)
        self.assertNotIn(1, memory.resource_records['actor']['record_id'])

    def test_state_recovery_does_not_retry_auth_denial_or_guess(self):
        memory = Memory()
        memory.store_resource_id('actor', 'record_id', 1, record={'status': 'pending'})
        self.assertIsNone(_state_compatible_resource(memory, 'actor', 'record_id',
            {'status_code': 403, 'body': {'error': 'not in pending status'}}, set()))
        self.assertIsNone(_state_compatible_resource(memory, 'actor', 'record_id',
            {'status_code': 400, 'body': {'error': 'Invalid input'}}, set()))

    def test_report_distinguishes_rejected_request_from_not_sent(self):
        tc = TestCase('PUT', '/records/1/', 'actor', True)
        results = [Result(tc, 400, {'error': 'Invalid state'}), Result(tc, 0, {'reason': 'disabled'})]
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'report.json'
            save_json_report(str(report), results, {}, [tc], ['actor'], [])
            markdown = report.with_suffix('.md').read_text(encoding='utf-8')
        self.assertIn('INCONCLUSIVE — rejected by validation or resource state (1)', markdown)
        self.assertIn('SKIP — skipped (1)', markdown)
        self.assertIn('Invalid state', markdown)
        self.assertEqual(classify({'status_in': [200]}, 400), 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
