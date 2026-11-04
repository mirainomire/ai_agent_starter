"""Deterministic ID candidate collection plus one LLM call that picks path IDs."""
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import normalize_path

ID_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string"},
                    "candidate": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["slot", "candidate", "reason"],
            },
        }
    },
    "required": ["choices"],
}
OWNER_FIELDS = ("employee_id", "user_id", "owner_id", "employee", "user", "owner")
STATE_FIELDS = ("status", "state", "is_active", "role", "type")
SELECTABLE_METHODS = ("get", "put", "patch", "post")
MAX_FIELD_CHARS = 60


def resourceKeyForPlaceholder(path: str, placeholder: str) -> str:
    """Name the resource a path placeholder refers to, e.g. /users/{id}/ -> user."""
    if placeholder.lower() != 'id':
        return re.sub(r"(^id_|_id$)", "", placeholder)
    match = re.search(r"/([^/{}]+)/\{id\}", path or "")
    if not match:
        return placeholder
    segment = match.group(1).lower()
    if segment.endswith('ies'):
        return segment[:-3] + 'y'
    if segment.endswith('s'):
        return segment[:-1]
    return segment


def resourceKeyAliases(resource_key: str) -> List[str]:
    aliases = [resource_key, f"{resource_key}_id", f"id_{resource_key}"]
    if resource_key == 'document':
        aliases.extend(['doc', 'doc_id', 'id_doc'])
    return aliases


def responseItems(body: Any) -> List[Dict[str, Any]]:
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict):
        for key in ("data", "results", "items", "list"):
            value = body.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested_items = responseItems(value)
                if nested_items:
                    return nested_items
    return []


def compactRecord(item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only ID, ownership and state fields; other content never reaches the model."""
    compact = {}
    for key, value in item.items():
        key_lower = str(key).lower()
        if not (key_lower == "id" or key_lower.endswith("_id") or key_lower in STATE_FIELDS):
            continue
        if isinstance(value, str):
            compact[key] = value[:MAX_FIELD_CHARS]
        elif isinstance(value, (int, float, bool)) or value is None:
            compact[key] = value
    return compact


def recordOwnerId(item: Dict[str, Any]) -> Optional[int]:
    for field_name in OWNER_FIELDS:
        value = item.get(field_name)
        if isinstance(value, dict):
            value = value.get("id")
        try:
            owner_id = int(value)
        except (TypeError, ValueError):
            continue
        if owner_id > 0:
            return owner_id
    return None


def collectCollectionRecords(
    collection: str,
    accounts: List[str],
    fetch_collection: Callable[[str, str], Any],
    user_id_of: Callable[[str], Optional[int]],
) -> List[Dict[str, Any]]:
    """Return visible records with their owner account, or None when not user-owned."""
    owner_by_user_id = {user_id_of(account): account for account in accounts if user_id_of(account)}
    records: Dict[int, Dict[str, Any]] = {}
    for account in accounts:
        try:
            body = fetch_collection(collection, account)
        except Exception:
            continue
        for item in responseItems(body):
            try:
                record_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if record_id <= 0 or record_id in records:
                continue
            owner_id = recordOwnerId(item)
            records[record_id] = {
                "id": record_id,
                "owner_account": owner_by_user_id.get(owner_id) if owner_id else None,
                "owned": owner_id is not None,
                "fields": compactRecord(item),
            }
    return list(records.values())


def buildIdSlots(
    openapi: dict,
    accounts: List[str],
    fetch_collection: Callable[[str, str], Any],
    user_id_of: Callable[[str], Optional[int]],
    max_candidates: int = 8,
    admin_accounts: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    """Build model slots for operations with several candidates.

    Identity paths use the account's own user ID and DELETE uses disposable
    fixtures, so neither is offered to the model. An admin that owns no record
    (e.g. reviews belong to employees) is offered every record, since its
    access does not depend on ownership.
    """
    paths = (openapi or {}).get("paths") or {}
    get_collections = {
        normalize_path(path) for path, item in paths.items()
        if isinstance(item, dict) and any(str(method).lower() == "get" for method in item)
    }
    records_by_collection: Dict[str, List[Dict[str, Any]]] = {}
    llm_slots: List[Dict[str, Any]] = []
    single_choices: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for path, item in paths.items():
        template = normalize_path(path)
        placeholders = re.findall(r"\{([^}/]+)\}", template)
        if len(placeholders) != 1 or not isinstance(item, dict):
            continue
        placeholder = placeholders[0]
        if placeholder in ("user_id", "id_user") or resourceKeyForPlaceholder(template, placeholder) == "user":
            continue
        collection = normalize_path(template.split("{", 1)[0])
        if collection not in get_collections:
            continue
        if collection not in records_by_collection:
            records_by_collection[collection] = collectCollectionRecords(
                collection, accounts, fetch_collection, user_id_of
            )
        records = records_by_collection[collection]
        for method_name, operation in item.items():
            if str(method_name).lower() not in SELECTABLE_METHODS:
                continue
            method = str(method_name).upper()
            operation_text = ""
            if isinstance(operation, dict):
                operation_text = str(operation.get("summary") or operation.get("description") or "")[:300]
            for owner in accounts:
                usable = [
                    record for record in records
                    if record["owner_account"] == owner or not record["owned"]
                ]
                if not usable and owner in (admin_accounts or []):
                    usable = list(records)
                usable = sorted(usable, key=lambda record: record["id"], reverse=True)[:max_candidates]
                if not usable:
                    continue
                key = (method, template, owner)
                if len(usable) == 1:
                    single_choices[key] = {
                        "id": usable[0]["id"],
                        "source": "single_candidate",
                        "reason": "Only one observed record for this owner",
                    }
                    continue
                llm_slots.append({
                    "slot": f"{method} {template} owner={owner}",
                    "method": method,
                    "path_template": template,
                    "owner_account": owner,
                    "operation": operation_text,
                    "candidates": [
                        {"candidate": str(record["id"]), "fields": record["fields"]}
                        for record in usable
                    ],
                })
    return llm_slots, single_choices


def selectIdsWithLlm(
    llm_call: Callable[[str, dict, int], str],
    slots: List[Dict[str, Any]],
    prompt_template: str,
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    """Ask the model once for every slot; reject any ID outside a slot's candidates."""
    model_slots = [
        {key: slot[key] for key in ("slot", "operation", "candidates")}
        for slot in slots
    ]
    prompt = prompt_template + "\n\nSlots:\n" + json.dumps(model_slots, ensure_ascii=False, default=str)
    content = llm_call(prompt, ID_SELECTION_SCHEMA, max(2000, len(slots) * 120))
    parsed = json.loads(content or "{}")
    slot_by_name = {slot["slot"]: slot for slot in slots}
    choices: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    for choice in parsed.get("choices") or []:
        slot = slot_by_name.get(str(choice.get("slot")))
        candidate = str(choice.get("candidate"))
        allowed = {item["candidate"] for item in slot["candidates"]} if slot else set()
        if slot is None or candidate not in allowed:
            rejected.append({"slot": str(choice.get("slot")), "candidate": candidate})
            continue
        choices[(slot["method"], slot["path_template"], slot["owner_account"])] = {
            "id": int(candidate),
            "source": "llm",
            "reason": str(choice.get("reason") or "")[:300],
        }
    return choices, rejected
