"""Translate the backend endpoint catalog into evaluator rules and valid request bodies."""
import json
import re
import secrets
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import normalize_path

API_PREFIX = "/api"
# Fields the catalog text marks as admin-only inside an owner-editable payload.
# They are removed from the owner's request so the case tests ownership only.
ADMIN_ONLY_PAYLOAD_FIELDS = {
    "role changes within the profile payload are admin_only": "role",
    "employee put is limited to submitted claim fields": "status",
}
FIRST_UNIQUE_DATE = date(2100, 1, 1)
UNIQUE_DATE_SPAN_DAYS = 2_000_000
UNIQUE_YEAR_SPAN = 7_800


def stripApiPrefix(path: str) -> str:
    normalized = normalize_path(path)
    if normalized.startswith(API_PREFIX + "/"):
        normalized = normalized[len(API_PREFIX):]
    return normalized


def templateKey(method: str, path: str) -> str:
    """Placeholder names differ between documents; compare their positions only."""
    generic_path = re.sub(r"\{[^}/]+\}", "{}", stripApiPrefix(path))
    return f"{method.upper()} {generic_path}"


def employeeScope(entry: Dict[str, Any]) -> Optional[Union[str, bool]]:
    """Return the Employee grant: "any", True (own records), "deny", or None if unmapped."""
    policy_text = str(entry.get("access_policy") or "").strip().lower()
    method = str(entry.get("method") or "").lower()
    if f"admin_only for {method}" in policy_text or policy_text.startswith("admin_only"):
        return "deny"
    if policy_text.startswith("owner_or_admin"):
        return True
    if policy_text.startswith("authenticated"):
        return "any"
    return None


def ownerExample(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    example = entry.get("request_example")
    if not isinstance(example, dict):
        return None
    policy_text = str(entry.get("access_policy") or "").lower()
    cleaned = deepcopy(example)
    for clause, field_name in ADMIN_ONLY_PAYLOAD_FIELDS.items():
        if clause in policy_text:
            cleaned.pop(field_name, None)
    return cleaned


def catalogPolicy(catalog: Dict[str, Any], admin_role: str = "Admin",
                  employee_role: str = "Employee") -> Dict[str, Any]:
    """Build `rules`, `public_endpoints` and `request_examples` from the catalog."""
    rules: List[Dict[str, Any]] = []
    field_rules: List[Dict[str, Any]] = []
    public_endpoints: List[str] = []
    request_examples: Dict[str, Dict[str, Any]] = {}
    unmapped: List[str] = []
    for entry in catalog.get("endpoints") or []:
        method = str(entry.get("method") or "").upper()
        path = stripApiPrefix(entry.get("path") or "")
        policy_text = str(entry.get("access_policy") or "").lower()
        raw_example = entry.get("request_example") if isinstance(entry.get("request_example"), dict) else {}
        for clause, field_name in ADMIN_ONLY_PAYLOAD_FIELDS.items():
            if clause in policy_text and field_name in raw_example:
                field_rules.append({"method": method, "path": path, "field": field_name,
                                    "value": raw_example[field_name], "allow_roles": [admin_role]})
        example = ownerExample(entry)
        if example is not None:
            request_examples[templateKey(method, path)] = example
        if str(entry.get("access_policy") or "").strip().lower() == "public":
            public_endpoints.append(f"{method}:{path}")
            continue
        grants = []
        if entry.get("admin_access") is True:
            grants.append({"role": admin_role, "self": "any"})
        scope = employeeScope(entry)
        if scope is None:
            unmapped.append(f"{method} {path}")
            continue
        if scope != "deny":
            grants.append({"role": employee_role, "self": scope})
        rules.append({"method": method, "path": path, "allow": grants})
    # Literal routes such as /expenses/summary/ must win over /expenses/{id}/.
    rules.sort(key=lambda rule: rule["path"].count("{"))
    return {
        "rules": rules,
        "field_rules": field_rules,
        "public_endpoints": public_endpoints,
        "request_examples": request_examples,
        "catalog_unmapped": unmapped,
    }


def loadCatalogPolicy(policy: Dict[str, Any], agent_dir: Path,
                      account_roles: Dict[str, str]) -> Dict[str, Any]:
    """Merge catalog-derived rules into a policy that names a catalog but has no rules."""
    catalog_ref = policy.get("catalog")
    if not catalog_ref or policy.get("rules") or policy.get("rbac_rules"):
        return policy
    catalog_path = agent_dir / str(catalog_ref)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    merged = dict(policy)
    merged.update(catalogPolicy(catalog))
    merged.setdefault("account_role_map", account_roles)
    return merged


def findRequestExample(policy: Dict[str, Any], method: str, path_template: str) -> Optional[Dict[str, Any]]:
    example = ((policy or {}).get("request_examples") or {}).get(templateKey(method, path_template))
    return deepcopy(example) if isinstance(example, dict) else None


def uniqueDate() -> date:
    """Random far-future day; deterministic offsets collided between consecutive requests."""
    return FIRST_UNIQUE_DATE + timedelta(days=secrets.randbelow(UNIQUE_DATE_SPAN_DAYS))


def makeUnique(body: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Replace values the backend enforces as unique so repeated runs never collide."""
    result = dict(body)
    if "username" in result:
        result["username"] = f"{result['username']}_{suffix}"
    if "email" in result and "@" in str(result["email"]):
        local_part, domain = str(result["email"]).split("@", 1)
        result["email"] = f"{local_part}_{suffix}@{domain}"
    if "name" in result:
        result["name"] = f"{result['name']} {suffix}"
    if "work_date" in result:
        result["work_date"] = uniqueDate().isoformat()
    if "start_date" in result:
        start = uniqueDate()
        result["start_date"] = start.isoformat()
        if "end_date" in result:
            result["end_date"] = (start + timedelta(days=1)).isoformat()
    if "month" in result and "year" in result:
        result["month"] = secrets.randbelow(12) + 1
        result["year"] = 2100 + secrets.randbelow(UNIQUE_YEAR_SPAN)
    return result


def resolveExample(body: Dict[str, Any], resolvers: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Replace catalog markers such as resolve_employee_id; report markers left unresolved."""
    resolved: Dict[str, Any] = {}
    unresolved: List[str] = []
    for key, value in body.items():
        if isinstance(value, str) and value.startswith("resolve_"):
            resolver = resolvers.get(value)
            resolved_value = resolver() if callable(resolver) else None
            if resolved_value is None:
                unresolved.append(value)
                continue
            resolved[key] = resolved_value
        else:
            resolved[key] = value
    return resolved, unresolved
