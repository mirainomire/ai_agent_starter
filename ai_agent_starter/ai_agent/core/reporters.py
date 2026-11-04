"""Generic report serialization for API access-control assessments."""

import datetime
import json
import os
import time
from typing import Any, Dict, List

from .evaluators import (
    classify,
    confusion_counts,
    coverage,
    effective_account_alias,
    effective_policy_role,
    expected_status,
    metrics,
    time_to_detect,
)
from .memory import Result, TestCase


_SENSITIVE_FIELDS = (
    "password", "passwd", "secret", "token", "authorization", "api_key",
    "apikey", "private_key", "client_secret", "session_id", "cookie",
)


def _redact_sensitive(value: Any) -> Any:
    """Redact common credentials before writing response bodies into reports."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(marker in str(key).lower() for marker in _SENSITIVE_FIELDS)
            else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _llm_expectation_hypothesis(mutation: Any) -> Dict[str, Any] | None:
    """Normalize advisory model output while keeping it separate from policy verdicts."""
    if not isinstance(mutation, dict):
        return None
    value = mutation.get("expectation_hypothesis")
    if not isinstance(value, dict):
        return None
    access = str(value.get("access", "unknown")).strip().lower()
    if access not in {"allow", "deny", "unknown"}:
        access = "unknown"
    try:
        confidence = float(value.get("confidence"))
        confidence = min(1.0, max(0.0, confidence))
    except (TypeError, ValueError):
        confidence = None
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    return {
        "access": access,
        "confidence": confidence,
        "rationale": str(value.get("rationale", ""))[:1000],
        "evidence": [str(item)[:300] for item in evidence[:8]],
        "source": "llm",
        "used_for_verdict": False,
    }


def _fallback_summary(
    counts: Dict[str, int], metric_values: Dict[str, float],
    coverage_data: Dict[str, Any], classified: Dict[str, list], total: int,
) -> str:
    """Build neutral fallback text; unknown policy is never described as secure."""
    fp = counts.get("FP", 0)
    fn = counts.get("FN", 0)
    errors = counts.get("ERR", 0)
    unverified = counts.get("UNVERIFIED", 0)
    if not metric_values.get("evaluated_cases", 0):
        assessment = "Incomplete: no cases could be evaluated against explicit access expectations."
    elif fp:
        assessment = "Review required: access was allowed where the configured policy expected denial."
    elif unverified:
        assessment = "Incomplete: some operations have no matching access expectation."
    elif counts.get("INCONCLUSIVE", 0):
        assessment = "Incomplete: some requests were rejected by validation or resource state checks."
    elif fn or errors:
        assessment = "Review required: unexpected denials or server errors were observed."
    else:
        assessment = "No policy violations were classified in evaluated cases."

    lines = [
        "# Security Assessment Summary", "", assessment, "",
        "Results compare observed responses with configured expectations. "
        "Missing expectations are unverified and are not treated as findings.", "",
        f"- Outcomes: {total}",
        f"- Unexpected allowed outcomes (FP): {fp}",
        f"- Unexpected denied outcomes (FN): {fn}",
        f"- Server errors: {errors}",
        f"- Not found / inconclusive: {counts.get('NF', 0)}",
        f"- Skipped: {counts.get('SKIP', 0)}",
        f"- Request rejected / inconclusive: {counts.get('INCONCLUSIVE', 0)}",
        f"- Unverified (no matching policy): {unverified}",
        f"- Operation coverage: {coverage_data.get('coverage_pct', 0)}% "
        f"({coverage_data.get('tested_pairs', 0)}/{coverage_data.get('total_pairs', 0)} role-operation pairs)",
        "", "## Cases for review", "",
    ]
    review_items = (
        classified.get("FP", []) + classified.get("FN", [])
        + classified.get("INCONCLUSIVE", [])
        + classified.get("UNVERIFIED", [])
    )
    if not review_items:
        lines.append("No cases require review based on available expectations.")
    else:
        for item in review_items[:50]:
            hypothesis = item.get("ai_expectation_hypothesis") or {}
            hypothesis_text = (
                f"; AI hypothesis={hypothesis.get('access')} "
                f"(confidence={hypothesis.get('confidence')})"
                if hypothesis else ""
            )
            lines.append(
                f"- {item.get('method')} {item.get('resolved_path') or item.get('path')} "
                f"[account={item.get('account_label') or item.get('account')}; role={item.get('role')}] "
                f"status={item.get('status')}; expectation={item.get('expected_access', 'unknown')} "
                f"(source={item.get('expectation_source', 'unknown')}){hypothesis_text}"
            )
    lines.extend([
        "", "## Metrics", "",
        f"- Accuracy: {metric_values.get('accuracy', 0)}",
        f"- Precision: {metric_values.get('precision', 0)}",
        f"- Recall: {metric_values.get('recall', 0)}",
        f"- F1 score: {metric_values.get('f1', 0)}", "",
        "Review each candidate against API documentation and policy before confirming an issue.",
    ])
    return "\n".join(lines)


def save_json_report(
    path: str,
    results: List[Result],
    policy,
    tests: List[TestCase],
    roles: list,
    endpoints: list,
    start_ts: float = 0.0,
    reflection: Dict[str, Any] = None,
    llm_summary: str = None,
    run_metadata: Dict[str, Any] = None,
):
    """Write JSON and Markdown reports beside the run, without touching input data."""
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    total_duration = round(time.time() - start_ts, 3) if start_ts > 0 else None

    counts = confusion_counts(results, policy)
    metric_values = metrics(counts)
    coverage_data = coverage(tests, roles, endpoints)
    detection = time_to_detect(results, policy, start_ts)
    classified: Dict[str, list] = {
        key: [] for key in ("TP", "TN", "FP", "FN", "NF", "ERR", "SKIP", "INCONCLUSIVE", "UNVERIFIED")
    }

    serialized_results = []
    artifacts = []
    for result in results:
        tc = result.tc
        expectation = expected_status(policy, tc)
        label = classify(expectation, result.status_code)
        key = {"TP_ALLOW": "TP", "NOT_FOUND": "NF", "ERROR": "ERR"}.get(label, label)
        row = {
            "method": tc.method,
            "path": tc.path,
            "account": effective_account_alias(tc),
            "account_label": (run_metadata or {}).get('account_labels', {}).get(effective_account_alias(tc), effective_account_alias(tc)),
            "resolved_path": result.resolved_path,
            "target_account": result.target_account,
            "attempts": _redact_sensitive(result.attempts),
            "recovery": _redact_sensitive(result.recovery),
            "role": effective_policy_role(policy, tc),
            "actor_account": effective_account_alias(tc),
            "actor_role": effective_policy_role(policy, tc),
            "self_access": tc.self_access,
            "status": result.status_code,
            "classification": label,
            "expected_access": expectation.get("expected_access", "unknown"),
            "expectation_source": expectation.get("expectation_source", "unknown"),
            "ai_expectation_hypothesis": _llm_expectation_hypothesis(tc.mutation),
            "mutation": _redact_sensitive(tc.mutation),
            "body": _redact_sensitive(result.body),
            "timestamp": result.ts,
            "artifact": result.artifact,
        }
        serialized_results.append(row)
        artifacts.append({
            "method": tc.method, "path": tc.path,
            "role": effective_policy_role(policy, tc), "artifact": result.artifact,
        })
        if key in classified:
            classified[key].append(row)

    endpoint_inventory = [
        {"method": str(endpoint.get("method", "GET")).upper(), "path": str(endpoint.get("path", ""))}
        for endpoint in (endpoints or [])
        if isinstance(endpoint, dict) and endpoint.get("path")
    ]
    planned_cases = [
        {
            "method": tc.method, "path": tc.path,
            "account": effective_account_alias(tc),
            "role": effective_policy_role(policy, tc),
            "self_access": tc.self_access, "depth": tc.depth,
            "mutation": _redact_sensitive(tc.mutation),
        }
        for tc in (tests or [])
    ]
    llm_text = (llm_summary or "").strip()
    llm_unavailable = not llm_text or llm_text.startswith("⚠️")
    summary = _fallback_summary(counts, metric_values, coverage_data, classified, len(results)) if llm_unavailable else llm_text
    data = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_metadata": run_metadata or {},
        "label_schema": {
            "version": "3.1",
            "positive_class": "ALLOW",
            "TP": "expected ALLOW, observed 2xx",
            "FN": "expected ALLOW, observed deny",
            "TN": "expected DENY, observed 401/403",
            "FP": "expected DENY, observed 2xx",
            "UNVERIFIED": "no explicit access expectation matched this case",
            "INCONCLUSIVE": "request sent but validation or resource state prevents an authorization verdict",
        },
        "confusion": counts,
        "metrics": metric_values,
        "coverage": coverage_data,
        "endpoint_inventory": endpoint_inventory,
        "planned_cases": planned_cases,
        "time_to_detect": detection,
        "total_duration_seconds": total_duration,
        "summary": {
            "generated_cases": len(tests or []),
            "result_records": len(results),
            "evaluated_cases": int(metric_values.get("evaluated_cases", 0)),
            "potential_vulnerabilities": int(counts.get("FP", 0)),
            "unverified_cases": int(counts.get("UNVERIFIED", 0)),
        },
        "llm_summary": summary,
        "llm_summary_raw": llm_text,
        "llm_summary_fallback": llm_unavailable,
        "artifacts": artifacts,
        "reflection": reflection or {},
        "classified": classified,
        "results": serialized_results,
    }
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump(data, report_file, indent=2, ensure_ascii=False)

    markdown_path = os.path.splitext(path)[0] + ".md"
    with open(markdown_path, "w", encoding="utf-8") as report_file:
        report_file.write(render_markdown_report(data))
    return path


def _markdown_cell(text: Any) -> str:
    """Make a value safe for a single markdown table cell."""
    return str(text).replace("\n", " ").replace("\r", " ").replace("|", "\\|").strip()


def render_markdown_report(data: Dict[str, Any]) -> str:
    """Render the human-readable markdown report from a serialized report `data`
    dict (the same structure written to the JSON file). Results are shown as
    scannable tables grouped by classification, findings first."""
    summary = data.get("llm_summary") or ""
    endpoint_inventory = data.get("endpoint_inventory", []) or []
    classified = data.get("classified", {}) or {}

    lines = [f"# Security Assessment Report", "", f"Generated: {data.get('generated_at', '')}", "", summary, ""]

    lines.append(f"## Operation inventory ({len(endpoint_inventory)})")
    lines.append("")
    lines.append("| Method | Path |")
    lines.append("| --- | --- |")
    for endpoint in endpoint_inventory:
        lines.append(f"| {endpoint.get('method', '')} | `{_markdown_cell(endpoint.get('path', ''))}` |")

    label_titles = {
        "FP": "FP — allowed, but policy expected deny (review required)",
        "FN": "FN — denied, but policy expected allow",
        "INCONCLUSIVE": "INCONCLUSIVE — rejected by validation or resource state",
        "UNVERIFIED": "UNVERIFIED — no matching policy expectation",
        "NF": "NF — not found; authorization unverified",
        "ERR": "ERR — fixture/setup error; not evaluated",
        "SKIP": "SKIP — skipped",
        "TN": "TN — denied as expected",
        "TP": "TP — allowed as expected",
    }
    label_order = ["FP", "FN", "INCONCLUSIVE", "UNVERIFIED", "NF", "ERR", "SKIP", "TN", "TP"]

    lines.extend(["", "## Results", ""])
    for label in label_order:
        items = classified.get(label) or []
        if not items:
            continue
        lines.append(f"### {label_titles.get(label, label)} ({len(items)})")
        lines.append("")
        lines.append("| Method | Path | Account | Role | Scope | Status | Expected | Source | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in items[:200]:
            notes = []
            if item.get('resolved_path') and item['resolved_path'] != item.get('path'):
                notes.append(f"template `{item.get('path')}`")
            if item.get('status') in (0, 400, 404, 409, 422):
                body = item.get('body') or {}
                reason = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
                notes.append("reason: " + reason.replace("\n", " ")[:200])
            attempts = [attempt for attempt in (item.get('attempts') or []) if attempt.get('path')]
            if len(attempts) > 1:
                notes.append("steps: " + ", ".join(f"{attempt['path']}→{attempt['status']}" for attempt in attempts[:4]))
            for decision in (item.get('recovery') or []):
                notes.append(f"recovery: {decision.get('action')}")
            scope = 'self' if item.get('self_access') else 'other'
            lines.append(
                f"| {item.get('method', '')} | `{_markdown_cell(item.get('resolved_path') or item.get('path'))}` "
                f"| {_markdown_cell(item.get('account_label'))} | {_markdown_cell(item.get('role'))} "
                f"| {scope} | {item.get('status')} "
                f"| {_markdown_cell(item.get('expected_access'))} | {_markdown_cell(item.get('expectation_source'))} "
                f"| {_markdown_cell('; '.join(notes))} |"
            )
        lines.append("")
    return "\n".join(lines)
