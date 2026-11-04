import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_agent.core.memory import Memory, Result, TestCase
from ai_agent.core.orchestrator import AgentOrchestrator, execute
from ai_agent.core.reporters import save_json_report
from ai_agent.core.resource_recovery import recover_resource
from ai_agent.core.tools_auth import AuthManager
from ai_agent.core.tools_http import HttpClient
from ai_agent.core.utils import resource_id_field


class AgentResourceSelectionTests(unittest.TestCase):
    def test_foreign_key_is_not_replaced_by_primary_record_id(self):
        self.assertEqual(resource_id_field({'id': 74, 'user_id': 5}, 'user_id', '/payroll/'), ('user_id', 5))
        self.assertEqual(resource_id_field({'id': 6934, 'actor_id': 5}, 'user_id', '/audit-logs/'), (None, None))
        self.assertEqual(resource_id_field({'id': 5}, 'user_id', '/users/'), ('id', 5))

    def test_login_identity_overrides_polluted_memory_and_env_id(self):
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient('http://example.invalid', artifacts_dir=folder)
            auth = AuthManager(http, {'user_id_json_path': 'user_id', 'roles': {
                'tester': {'endpoint': '/login/', 'token_json_path': 'access'}}})
            with patch.object(auth, '_env_credentials_for_role', return_value=('u', 'p')), \
                 patch.object(auth.http_login, 'request', return_value={
                     'status_code': 200, 'body': {'access': 'fake-token', 'user_id': 37}}):
                auth.get_token('tester')
            with patch.object(auth, '_env_user_id_for_role', return_value=597):
                self.assertEqual(auth.get_user_id('tester'), 37)
                memory = Memory()
                memory.store_resource_id('tester', 'user_id', 597)
                tc = TestCase('GET', '/users/{user_id}/', 'tester', True)
                with patch.object(http, 'request', return_value={'status_code': 200, 'body': {}}) as request:
                    execute(memory, http, auth, {}, tests=[tc])
                self.assertEqual(request.call_args.kwargs['path'], '/users/37/')

    def test_llm_response_loop_selects_only_observed_candidate(self):
        memory = Memory()
        memory.store_resource_id('actor', 'record_id', 7, {'id': 7, 'status': 'ready'}, '/records/')
        memory.store_resource_id('stranger', 'record_id', 9, {'id': 9}, '/records/')
        tc = TestCase('GET', '/records/{record_id}/', 'actor', True)
        decide = Mock(return_value={'action': 'retry', 'candidate': '7', 'reason': 'Observed ready record'})
        send = Mock(return_value={'status_code': 200, 'body': {}})
        response, path, attempts, decisions = recover_resource(tc, 'actor', 'actor', '/records/1/',
            {'status_code': 404, 'body': {'error': 'Missing'}}, memory, {}, decide, send)
        self.assertEqual(path, '/records/7/')
        self.assertEqual([a['status'] for a in attempts], [404, 200])
        context = decide.call_args.args[0]
        self.assertEqual([c['candidate'] for c in context['candidates']], ['7'])
        self.assertEqual(context['observations'][0]['body'], {'error': 'Missing'})
        self.assertEqual(decisions[0]['reason'], 'Observed ready record')
        self.assertEqual(response['status_code'], 200)

    def test_invented_candidate_and_auth_denial_do_not_retry(self):
        memory = Memory()
        memory.store_resource_id('actor', 'record_id', 7, {'id': 7}, '/records/')
        tc = TestCase('GET', '/records/{record_id}/', 'actor', True)
        send = Mock()
        decide = Mock(return_value={'action': 'retry', 'candidate': '999'})
        _, _, _, decisions = recover_resource(tc, 'actor', 'actor', '/records/1/',
            {'status_code': 404}, memory, {}, decide, send)
        self.assertIn('Rejected selection', decisions[0]['reason'])
        send.assert_not_called()
        decide.reset_mock()
        recover_resource(tc, 'actor', 'actor', '/records/1/', {'status_code': 403}, memory, {}, decide, send)
        decide.assert_not_called()

    def test_six_accounts_can_target_a_specific_peer(self):
        aliases = [f'account_{i}' for i in range(6)]
        auth = SimpleNamespace(roles=dict.fromkeys(aliases, {}), get_token=lambda role: 'token',
                               get_user_id=lambda role: aliases.index(role) + 10,
                               get_policy_role=lambda role: 'Employee')
        memory = Memory()
        tc = TestCase('GET', '/users/{user_id}/', aliases[1], False,
                      mutation={'target_account': aliases[5]})
        with tempfile.TemporaryDirectory() as folder:
            http = HttpClient('http://example.invalid', artifacts_dir=folder)
            with patch.object(http, 'request', return_value={'status_code': 403, 'body': {}}) as request:
                execute(memory, http, auth, {}, tests=[tc])
            self.assertEqual(request.call_args.kwargs['path'], '/users/15/')
            self.assertEqual(request.call_args.kwargs['role'], aliases[1])
            self.assertEqual(memory.results[0].target_account, aliases[5])

    def test_report_displays_each_account_role_and_actual_path(self):
        first = TestCase('GET', '/users/{user_id}/', 'Employee', True)
        second = TestCase('GET', '/users/{user_id}/', 'Employee_2', True)
        policy = {'account_role_map': {'Employee': 'Employee', 'Employee_2': 'Employee'}}
        rows = [Result(first, 404, resolved_path='/users/37/'), Result(second, 404, resolved_path='/users/38/')]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            save_json_report(str(path), rows, policy, [first, second], ['Employee', 'Employee_2'], [],
                             run_metadata={'account_labels': {'Employee': 'employee1', 'Employee_2': 'employee2'}})
            markdown = path.with_suffix('.md').read_text(encoding='utf-8')
        self.assertIn('| employee1 | Employee |', markdown)
        self.assertIn('| employee2 | Employee |', markdown)
        self.assertIn('`/users/37/`', markdown)
        self.assertIn('`/users/38/`', markdown)

    def test_configured_model_is_used_for_resource_decision(self):
        agent = AgentOrchestrator.__new__(AgentOrchestrator)
        agent.client = object()
        agent.llm_provider, agent.llm_model = 'gemini', 'configured-model'
        with patch('ai_agent.core.orchestrator._gemini_generate_content', return_value=
                   '{"action":"stop","reason":"No alternate ID with ownership evidence"}') as generate:
            decision = agent._decide_resource_retry({'current_path': '/records/1/', 'candidates': []})
        self.assertEqual(decision['action'], 'stop')
        self.assertEqual(generate.call_args.args[1], 'configured-model')
        self.assertIn('/records/1/', generate.call_args.args[2])


if __name__ == '__main__':
    unittest.main()
