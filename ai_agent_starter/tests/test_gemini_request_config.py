import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ai_agent.core.orchestrator import _gemini_generate_content, LLM_TEST_RESPONSE_SCHEMA


class GeminiRequestConfigTests(unittest.TestCase):
    def test_schema_keeps_structure_without_nested_array_count_constraints(self):
        response = SimpleNamespace(text='{"tests": []}', candidates=[SimpleNamespace(finish_reason='STOP')])
        generate = Mock(return_value=response)
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        content = _gemini_generate_content(client, 'gemini-3-flash-preview', 'Synthetic test',
            'application/json', response_json_schema=LLM_TEST_RESPONSE_SCHEMA,
            max_output_tokens=6160, thinking_level='MINIMAL')
        self.assertEqual(content, '{"tests": []}')
        config = generate.call_args.kwargs['config']
        schema = config['response_json_schema']
        self.assertEqual(schema['required'], ['tests'])
        arrays = [schema['properties']['tests'],
                  schema['properties']['tests']['items']['properties']['mutations']]
        for node in arrays:
            self.assertEqual(node['type'], 'array')
            self.assertNotIn('minItems', node)
            self.assertNotIn('maxItems', node)
        self.assertEqual(config['max_output_tokens'], 6160)

    def test_truncated_json_reports_token_limit_before_json_parsing(self):
        response = SimpleNamespace(text='{"tests": [{"path": "',
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name='MAX_TOKENS'))])
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=response)))
        with self.assertRaisesRegex(ValueError, 'output truncated.*MAX_TOKENS'):
            _gemini_generate_content(client, 'gemini-3-flash-preview', 'test', 'application/json')


if __name__ == '__main__':
    unittest.main()
