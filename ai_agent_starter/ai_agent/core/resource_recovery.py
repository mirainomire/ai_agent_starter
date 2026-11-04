"""Bounded response -> model decision -> validated resource selection loop."""
import re

from .reporters import _redact_sensitive
from .utils import normalize_path


def recover_resource(tc, actor, owner, path, response, memory, openapi, decide,
                     send, identity_id=None, max_attempts=2):
    attempts, decisions = [], []

    def observe(current_path, result):
        attempts.append({'path': current_path, 'status': result.get('status_code'),
                         'body': _redact_sensitive(result.get('body')),
                         'artifact': result.get('artifact')})

    observe(path, response)
    if response.get('status_code') not in (400, 404, 409, 422):
        return response, path, attempts, decisions
    if decide is None:
        decisions.append({'action': 'stop', 'reason': 'LLM resource recovery disabled'})
        return response, path, attempts, decisions
    placeholders = re.findall(r'\{([^}/]+)\}', tc.path)
    if len(placeholders) != 1:
        decisions.append({'action': 'stop', 'reason': 'Recovery requires one unambiguous path parameter'})
        return response, path, attempts, decisions
    ph = placeholders[0]
    source = normalize_path(tc.path.split('{', 1)[0]).rstrip('/')
    operation = next((item.get(tc.method.lower(), {}) for route, item in openapi.get('paths', {}).items()
                      if normalize_path(route).rstrip('/') == normalize_path(tc.path).rstrip('/')), {})
    candidates = {}
    for observation in memory.resource_observations:
        if (observation['owner_account'] != owner or observation['placeholder'] != ph
                or observation['source_path'].rstrip('/') != source):
            continue
        rid = observation['id']
        if ph in ('user_id', 'id_user') and rid != identity_id:
            continue  # A different user would change the self/other scenario.
        candidate_path = tc.path.replace('{' + ph + '}', str(rid))
        record = observation['record']
        candidates[str(rid)] = {
            'candidate': str(rid), 'path': candidate_path, 'owner_account': owner,
            'source_path': observation['source_path'],
            'observed_fields': {k: v for k, v in record.items()
                                if k in (ph, 'id', 'owner_id', 'user_id', 'status', 'state')},
        }
    for _ in range(max_attempts):
        if response.get('status_code') not in (400, 404, 409, 422):
            break
        tried = {normalize_path(a['path']).rstrip('/') for a in attempts}
        available = {key: value for key, value in candidates.items()
                     if normalize_path(value['path']).rstrip('/') not in tried}
        # Bound context size; prefer recently observed candidates.
        available = dict(list(available.items())[-40:])
        context = {
            'actor_account': actor, 'target_owner_account': owner,
            'self_access': tc.self_access, 'method': tc.method, 'path_template': tc.path,
            'operation': operation, 'current_path': path,
            'observations': attempts, 'candidates': list(available.values()),
        }
        try:
            decision = decide(context)
        except Exception as exc:
            decisions.append({'action': 'stop', 'reason': 'LLM recovery failed: ' + type(exc).__name__})
            break
        if not isinstance(decision, dict):
            decisions.append({'action': 'stop', 'reason': 'Invalid LLM decision'})
            break
        reason = str(decision.get('reason', ''))[:700]
        key = str(decision.get('candidate', ''))
        if decision.get('action') == 'stop':
            decisions.append({'action': 'stop', 'reason': reason})
            break
        if decision.get('action') != 'retry' or key not in available:
            decisions.append({'action': 'stop', 'reason': 'Rejected selection outside observed candidates',
                              'model_reason': reason})
            break
        chosen = available[key]
        decisions.append({'action': 'retry', 'candidate': key, 'reason': reason,
                          'source_path': chosen['source_path'], 'owner_account': owner})
        path = chosen['path']
        response = send(path)
        observe(path, response)
    else:
        if response.get('status_code') in (400, 404, 409, 422):
            decisions.append({'action': 'stop', 'reason': 'Resource retry budget exhausted'})
    return response, path, attempts, decisions
