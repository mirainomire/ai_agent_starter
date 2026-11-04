from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TestCase:
    method: str
    path: str
    role: str
    self_access: bool
    depth: int = 0
    mutation: Optional[Dict[str, Any]] = None

@dataclass
class Result:
    tc: TestCase
    status_code: int
    body: Dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0
    artifact: Optional[str] = None
    resolved_path: Optional[str] = None
    target_account: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    recovery: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Memory:
    tests: List[TestCase] = field(default_factory=list)
    results: List[Result] = field(default_factory=list)
    # Seeded resource IDs: role -> { key/placeholder: id }
    resource_ids: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Preserve every observed ID instead of overwriting discovery history.
    resource_id_pools: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    # Response snapshots support selecting IDs whose current state fits a request.
    resource_records: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = field(default_factory=dict)
    resource_observations: List[Dict[str, Any]] = field(default_factory=list)
    # Optional: created resources to allow safe cleanup decisions (by resource key)
    created: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record_test(self, tc: TestCase):
        self.tests.append(tc)

    def record_result(self, res: Result):
        self.results.append(res)

    def store_resource_id(self, role: str, key: str, rid: int, record: Optional[Dict[str, Any]] = None, source_path: Optional[str] = None):
        if record is not None and source_path:
            from copy import deepcopy
            self.resource_observations = [o for o in self.resource_observations if
                (o['owner_account'], o['placeholder'], o['id'], o['source_path']) != (role, key, rid, source_path)]
            self.resource_observations.append(dict(owner_account=role, placeholder=key, id=rid,
                                                   source_path=source_path, record=deepcopy(record)))
        if role not in self.resource_ids:
            self.resource_ids[role] = {}
        aliases = {key}
        # Add common alias patterns for convenience
        if not key.endswith("_id"):
            aliases.add(f"{key}_id")
        if not key.startswith("id_"):
            aliases.add(f"id_{key}")
        for alias in aliases:
            self.resource_ids[role][alias] = rid
            pool = self.resource_id_pools.setdefault(role, {}).setdefault(alias, [])
            if rid not in pool:
                pool.append(rid)
            if record is not None:
                from copy import deepcopy
                self.resource_records.setdefault(role, {}).setdefault(alias, {})[rid] = deepcopy(record)

    def discardResourceId(self, keys: List[str], rid: int):
        """Forget a deleted ID so later cases do not target a missing record."""
        for role, rid_map in self.resource_ids.items():
            role_pools = self.resource_id_pools.get(role, {})
            role_records = self.resource_records.get(role, {})
            for alias in keys:
                pool = role_pools.get(alias, [])
                if rid in pool:
                    pool.remove(rid)
                role_records.get(alias, {}).pop(rid, None)
                if rid_map.get(alias) == rid:
                    if pool:
                        rid_map[alias] = pool[-1]
                    else:
                        del rid_map[alias]

    def mark_created(self, resource_key: str, role: str, rid: int):
        if resource_key not in self.created:
            self.created[resource_key] = {}
        self.created[resource_key][str(rid)] = rid
