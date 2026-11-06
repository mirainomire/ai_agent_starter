from typing import List, Dict, Any, Tuple
from collections import Counter
from .memory import TestCase, Result
from .utils import normalize_path
import os
from pathlib import Path

# Expectations from the agent's own checks rather than the configured policy (the ground truth).
# Their outcomes are reported separately and never enter the confusion matrix.
BUILTIN_SOURCE_PREFIX = "builtin."
SESSION_INVALIDATION_SOURCE = BUILTIN_SOURCE_PREFIX + "session_invalidation"


def _locate_rules_file() -> Path | None:
    # Priority: ENV override, then common locations upward from CWD
    env = os.getenv("RULES_FILE") or os.getenv("RULE_FILE")
    if env:
        p = Path(env)
        if p.exists():
            return p
    # Try repo root (cwd), then parent dirs, then ai_agent_starter/rule.txt
    cand_names = ["rule.txt", "rules.txt"]
    start = Path.cwd()
    for up in [start] + list(start.parents)[:3]:
        for name in cand_names:
            p = up / name
            if p.exists():
                return p
    # Specific common path for this repo layout
    p = Path("ai_agent_starter") / "rule.txt"
    return p if p.exists() else None


def _parse_rule_line(line: str, out: Dict[str, set]):
    import re
    ls = line.lower()
    nums = set(int(n) for n in re.findall(r"\b(\d{3})\b", ls))
    if not nums:
        # Keywords without explicit code
        if "conflict" in ls:
            out["non_finding"].add(409)
        if "not found" in ls:
            out["not_found"].add(404)
        return
    if "not found" in ls:
        out["not_found"].update(nums)
    elif "bukan temuan" in ls or "not a finding" in ls or "non finding" in ls:
        out["non_finding"].update(nums)
    elif "error" in ls and any(n >= 500 for n in nums):
        out["error_like"].update({n for n in nums if n >= 500})


def _load_status_rules() -> Dict[str, set]:
    # Defaults aligned with best-practice and project rules
    rules = {
        "non_finding": set([400, 409, 422]),  # validation/conflict/unprocessable
        "not_found": set([404]),         # not found is not a BAC finding
        "error_like": set(),            # 5xx handled separately
    }
    try:
        rf = _locate_rules_file()
        if rf and rf.exists():
            for line in rf.read_text(encoding="utf-8").splitlines():
                _parse_rule_line(line, rules)
    except Exception:
        pass
    return rules


_STATUS_RULES = _load_status_rules()


def _success_statuses_for_method(method: str) -> List[int]:
    """Return acceptable success HTTP status codes for a given method.
    Keeps semantics simple and safe without parsing OpenAPI responses.
    """
    m = (method or "").strip().upper()
    mapping = {
        "GET": [200],
        "HEAD": [200],
        "OPTIONS": [200],
        "POST": [200, 201, 202],
        "PUT": [200, 204],
        "PATCH": [200, 204],
        "DELETE": [200, 204],
    }
    # Fallback: treat common 2xx as success
    return mapping.get(m, [200, 201, 202, 204])

def _policy_allowed_endpoints(policy, role: str):
    rules = policy.get("rbac_rules", {})
    role_rules = rules.get(role) or rules.get(str(role)) or {}
    allowed = set(role_rules.get("allowed_endpoints", []) or [])
    denied = set(role_rules.get("denied_endpoints", []) or [])
    critical_deny = set(role_rules.get("critical_deny", []) or [])
    self_only = set(role_rules.get("self_only_endpoints", []) or [])
    return allowed, denied, critical_deny, self_only


def effective_account_alias(tc: TestCase) -> str:
    """Return the account alias whose token is used for this test."""
    mutation = tc.mutation or {}
    return str(mutation.get("as_role") or tc.role)


def effective_policy_role(policy: Dict[str, Any], tc: TestCase) -> str:
    """Resolve an auth account alias to the backend authorization role."""
    alias = effective_account_alias(tc)
    role_map = (policy or {}).get("account_role_map", {}) or {}
    return str(role_map.get(alias) or alias)

def _match_endpoint_pattern(method: str, path: str, pattern: str) -> bool:
    """Return True if METHOD:path matches METHOD:/path with placeholders in pattern.
    Pattern examples: "GET:/user/{user_id}", "POST:/role/{id_role}/permissions"
    """
    try:
        if not isinstance(pattern, str) or ':' not in pattern:
            return False
        pm, pp = pattern.split(':', 1)
        if str(pm).strip().upper() != str(method).strip().upper():
            return False
        p = normalize_path(pp)
        # Escape regex special chars except placeholder braces
        import re
        # Replace placeholders {...} with segment matcher (no slash)
        rx = re.sub(r"\{[^}]+\}", r"[^/]+", re.escape(p).replace(r"\{", "{").replace(r"\}", "}"))
        # Ensure full path match
        return re.fullmatch(rx, normalize_path(path)) is not None
    except Exception:
        return False

def _in_endpoints(method: str, path: str, patterns: set) -> bool:
    for pat in list(patterns or []):
        try:
            if _match_endpoint_pattern(method, path, pat):
                return True
        except Exception:
            continue
    return False


def expected_status(policy, tc: TestCase) -> Dict[str, Any]:
    """Resolve expected access from explicit policy; never infer it from URL names."""
    policy = policy or {}
    policy_role = effective_policy_role(policy, tc)
    method = str(tc.method or "GET").upper()
    path = normalize_path(tc.path)
    success = _success_statuses_for_method(method)

    def expectation(access: str, source: str) -> Dict[str, Any]:
        if access == "allow":
            return {"status_in": success, "status_not_in": [401, 403],
                    "expected_access": access, "expectation_source": source}
        if access == "deny":
            return {"status_in": [401, 403], "status_not_in": success,
                    "expected_access": access, "expectation_source": source}
        return {"status_in": [], "status_not_in": [], "expected_access": "unknown",
                "expectation_source": source}

    # Public routes and their accepted outcomes are policy data, not URL heuristics.
    for item in policy.get("public_endpoints", []) or []:
        if isinstance(item, str) and ":" in item:
            route_method, route_path = item.split(":", 1)
            matches = route_method.strip().upper() == method and normalize_path(route_path) == path
            allowed_statuses, denied_statuses = success + [401], [403]
        elif isinstance(item, dict):
            matches = str(item.get("method", "GET")).upper() == method and normalize_path(item.get("path", "")) == path
            allowed_statuses = item.get("allow_statuses", success)
            denied_statuses = item.get("deny_statuses", [401, 403])
        else:
            continue
        if matches:
            return {"status_in": list(allowed_statuses), "status_not_in": list(denied_statuses),
                    "expected_access": "public", "expectation_source": "policy.public_endpoints"}

    # An unauthenticated request, or one reusing a token after logout, must be
    # denied on any non-public route regardless of the account's nominal role.
    mutation = tc.mutation or {}
    mutation_type = str(mutation.get("type") or "").upper()
    if mutation_type == "JWT_AFTER_LOGOUT":
        return expectation("deny", SESSION_INVALIDATION_SOURCE)
    if (mutation.get("no_auth") or mutation.get("without_auth")
            or mutation_type in ("NO_AUTH", "NEGATIVE_AUTH")):
        return expectation("deny", "policy.unauthenticated")

    # A field that only some roles may write decides the case when the request sends it.
    sent_body = (tc.mutation or {}).get("json")
    if isinstance(sent_body, dict):
        for field_rule in policy.get("field_rules", []) or []:
            if (not isinstance(field_rule, dict) or field_rule.get("field") not in sent_body
                    or not _match_endpoint_pattern(method, path, f"{str(field_rule.get('method', '')).upper()}:{field_rule.get('path', '')}")):
                continue
            access = "allow" if policy_role in (field_rule.get("allow_roles") or []) else "deny"
            return expectation(access, "policy.field_rules")

    # Rules can express role access at any scope, self scope, or other-resource scope.
    for rule in policy.get("rules", []) or []:
        if not isinstance(rule, dict) or str(rule.get("method", "")).upper() != method:
            continue
        pattern = f"{method}:{rule.get('path', '')}"
        if not _match_endpoint_pattern(method, path, pattern):
            continue
        grants = [grant for grant in (rule.get("allow", []) or [])
                  if isinstance(grant, dict) and grant.get("role") == policy_role]
        for grant in grants:
            scope = grant.get("self", "any")
            if scope == "any" or (scope is True and tc.self_access) or (scope is False and not tc.self_access):
                return expectation("allow", "policy.rules")
        return expectation("deny", "policy.rules")

    # Allow/deny patterns and ownership scope are all explicit policy inputs.
    if "rbac_rules" in policy:
        allowed, denied, critical_deny, self_only = _policy_allowed_endpoints(policy, policy_role)
        if (_in_endpoints(method, path, critical_deny) or
                _in_endpoints(method, path, denied)):
            return expectation("deny", "policy.rbac_rules")
        if _in_endpoints(method, path, allowed):
            if _in_endpoints(method, path, self_only) and not tc.self_access:
                return expectation("deny", "policy.rbac_rules.self_only_endpoints")
            return expectation("allow", "policy.rbac_rules")

    default_action = str(policy.get("default_action", "unknown")).strip().lower()
    if default_action in ("allow", "deny"):
        return expectation(default_action, "policy.default_action")
    return expectation("unknown", "no_matching_policy")


def classify(exp: Dict[str, Any], actual_status: int) -> str:
    def _as_int(v, default=-1):
        try:
            return int(v)
        except Exception:
            return default

    def _is_2xx(v: int) -> bool:
        return 200 <= int(v) < 300

    def _expected_mode(status_in: set) -> str:
        has_allow = any(_is_2xx(s) for s in status_in)
        has_deny = any(s in {401, 403, 404} for s in status_in)
        if has_allow and not has_deny:
            return "allow"
        if has_deny and not has_allow:
            return "deny"
        return "mixed"

    sc = _as_int(actual_status, default=-1)
    if sc == 0:
        return "SKIP"
    if 500 <= sc <= 599:
        return "ERROR"

    status_in = set(_as_int(s) for s in (exp or {}).get("status_in", []))
    status_in.discard(-1)
    mode = _expected_mode(status_in)

    # Input/data quality errors are inconclusive for authz labeling.
    if sc in _STATUS_RULES.get("non_finding", set([400, 409, 422])):
        return "INCONCLUSIVE"
    if str((exp or {}).get("expectation_source", "")).startswith(BUILTIN_SOURCE_PREFIX):
        if _is_2xx(sc):
            return "OUTSIDE_POLICY"
        if sc in {401, 403}:
            return "OUTSIDE_POLICY_PASS"
    if (exp or {}).get("expected_access") == "public":
        return "TP_ALLOW" if _is_2xx(sc) else "SKIP"
    if not status_in:
        return "UNVERIFIED"

    if mode == "allow":
        if _is_2xx(sc):
            return "TP_ALLOW"
        if sc in {401, 403}:
            return "FN"
        if sc in _STATUS_RULES.get("not_found", set([404])):
            return "NOT_FOUND"
        return "SKIP"

    if mode == "deny":
        # A 404 cannot prove authorization enforcement unless resource existence
        # was established independently. Keep it outside the confusion matrix.
        if sc in _STATUS_RULES.get("not_found", set([404])):
            return "NOT_FOUND"
        if sc in {401, 403}:
            return "TN"
        if _is_2xx(sc):
            return "FP"
        return "SKIP"

    # Mixed/unknown expectation fallback
    if sc in _STATUS_RULES.get("not_found", set([404])):
        return "NOT_FOUND"
    if sc in status_in:
        if _is_2xx(sc):
            return "TP_ALLOW"
        if sc in {401, 403}:
            return "TN"
        return "SKIP"
    if _is_2xx(sc) and any(s in {401, 403, 404} for s in status_in):
        return "FP"
    if sc in {401, 403} and any(_is_2xx(s) for s in status_in):
        return "FN"
    return "SKIP"

def classifyResult(policy, result: Result) -> str:
    """Classify one result. A 2xx write whose fields read back unchanged was ignored by the
    server, so the denied write was enforced (TN) even though the request was accepted."""
    label = classify(expected_status(policy, result.tc), result.status_code)
    if label == "FP" and result.write_applied is False:
        return "TN"
    return label


def confusion_counts(results: List[Result], policy) -> Dict[str, int]:
    c = Counter()
    for r in results:
        lab = classifyResult(policy, r)
        c[lab] += 1
    # Map TP_ALLOW to TP: allowed endpoints returning 200 are True Positives
    return {
        "TP": c.get("TP",0) + c.get("TP_ALLOW",0),
        "FP": c.get("FP",0),
        "FN": c.get("FN",0),
        "TN": c.get("TN",0),
        "ERR": c.get("ERROR",0),
        "NF": c.get("NOT_FOUND",0),
        "SKIP": c.get("SKIP",0),
        "INCONCLUSIVE": c.get("INCONCLUSIVE",0),
        "UNVERIFIED": c.get("UNVERIFIED",0),
        "OUTSIDE_POLICY": c.get("OUTSIDE_POLICY", 0),
        "OUTSIDE_POLICY_PASS": c.get("OUTSIDE_POLICY_PASS", 0),
    }

def metrics(cf: Dict[str, int]) -> Dict[str, float]:
    TP, FP, FN, TN = cf["TP"], cf["FP"], cf["FN"], cf["TN"]

    # Standard classification metrics on evaluated labels only (TP/FP/FN/TN).
    precision = TP / (TP + FP) if (TP+FP)>0 else 0.0
    recall = TP / (TP + FN) if (TP+FN)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    evaluated = TP + TN + FP + FN
    acc = (TP + TN) / max(1, evaluated)

    # Security-oriented companion metrics with DENY/BLOCK as the positive class.
    # Conventional FN (allowed request denied) is a false block; conventional FP
    # (denied request allowed) is a missed block.
    block_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    block_recall = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "evaluated_cases": int(evaluated),
        "block_precision": round(block_precision, 3),
        "block_recall": round(block_recall, 3),
        "false_alarm_rate": round(false_alarm_rate, 3),
    }

def coverage(tests: List[TestCase], roles: List[str], endpoints: List[Dict[str,Any]]) -> Dict[str, Any]:
    total_pairs = len(roles) * len(endpoints)
    tested_pairs = len({(t.role, t.method, t.path) for t in tests})
    pct = int(round((tested_pairs/max(1,total_pairs))*100))
    return {"endpoints": len(endpoints), "roles": len(roles), "total_pairs": total_pairs,
            "tested_pairs": tested_pairs, "coverage_pct": pct}

def time_to_detect(results: List[Result], policy, start_ts: float) -> Dict[str, Any]:
    """Measure the time from start_ts until the first false-allow finding (FP).
    Return a dict with seconds (float) or None, including the 1-based test index.
    """
    first_ts = None
    first_idx = None
    for idx, r in enumerate(results, start=1):
        if classifyResult(policy, r) == "FP":
            first_ts = r.ts or None
            first_idx = idx
            break
    return {
        "seconds": round((first_ts - start_ts), 3) if (first_ts and start_ts) else None,
        "test_index": first_idx,
    }

def bac_type(policy: dict, tc: TestCase) -> str:
    """
    Determine BAC type based on test case and policy context.
    
    Returns:
        - 'horizontal': IDOR - same privilege level, accessing other user's resources
        - 'vertical': BOLA - privilege escalation attempt
        - 'baseline': Normal expected operations (self access)
        - 'auth': Authentication related tests
    """
    # Check mutation for explicit type hints
    mut = tc.mutation or {}
    mut_type = str(mut.get("type", "")).upper()
    
    # Vertical escalation (privilege escalation)
    if mut_type in ("BOLA", "VERTICAL", "ESCALATION"):
        return 'vertical'
    
    # Check if accessing with different role than original
    if mut.get("as_role") and mut.get("as_role") != tc.role:
        return 'vertical'
    
    # Horizontal access (IDOR - other user's resource at same level)
    if mut_type == "IDOR" or mut.get("variant") == "other":
        return 'horizontal'
    
    # Self access = false indicates testing access to other users' resources
    if not tc.self_access:
        return 'horizontal'
    
    # Check for no-auth attempts (treat as vertical since it's trying to bypass auth)
    if mut.get("no_auth") or mut.get("without_auth") or mut_type in ("NO_AUTH", "NEGATIVE_AUTH"):
        return 'vertical'
    
    # Do not infer a category from route names; use explicit mutation intent.
    return 'baseline'
