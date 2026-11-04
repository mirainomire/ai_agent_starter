"""Convert a Postman v2.1 collection into a minimal OpenAPI 3.0 document.

The agent consumes OpenAPI (`paths`), not Postman collections. This converter
walks the collection, normalizes path parameters (`:id` and literal id segments
become `{...}`), de-duplicates method/path operations, and captures a request
body example and bearer security when present.

Usage:
  python ai_agent/scripts/postman_to_openapi.py <collection.json> <openapi_out.json>
"""
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# A path segment is treated as a resource identifier when it is all digits or a
# long id token (e.g. ULID). Those become a `{...}` placeholder so the planner
# and id-selection can target the operation generically.
ID_NUMERIC = re.compile(r"^\d+$")
ID_TOKEN = re.compile(r"^(?=.*\d)[0-9A-Za-z]{16,}$")


def singularize(segment: str) -> str:
    word = segment.replace("-", "_")
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def isIdSegment(segment: str) -> bool:
    return bool(ID_NUMERIC.match(segment) or ID_TOKEN.match(segment))


def templatePath(segments: List[str]) -> Tuple[str, List[str]]:
    """Return a templated path and its ordered parameter names."""
    out: List[str] = []
    params: List[str] = []
    last_resource = "item"
    for segment in segments:
        if not segment:
            continue
        if segment.startswith(":"):
            name = segment[1:].replace("-", "_") or "id"
        elif segment.startswith("{") and segment.endswith("}"):
            name = segment[1:-1].replace("-", "_") or "id"
        elif isIdSegment(segment):
            name = f"{singularize(last_resource)}_id"
        else:
            out.append(segment)
            last_resource = segment
            continue
        base_name = name
        counter = 2
        while name in params:
            name = f"{base_name}_{counter}"
            counter += 1
        params.append(name)
        out.append("{" + name + "}")
    return "/" + "/".join(out), params


def urlSegments(url: Any) -> Optional[List[str]]:
    if isinstance(url, dict):
        path = url.get("path")
        if isinstance(path, list):
            return [str(p) for p in path]
        raw = url.get("raw")
    else:
        raw = url
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"\{\{[^}]+\}\}", "", raw)              # drop {{baseurl}} etc.
    cleaned = re.sub(r"^https?://[^/]+", "", cleaned)         # drop absolute host
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]      # drop query/fragment
    return [seg for seg in cleaned.split("/") if seg]


def bodyExample(body: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Any]:
    if not isinstance(body, dict):
        return None, None
    mode = body.get("mode")
    if mode == "raw":
        raw = body.get("raw")
        try:
            return "application/json", json.loads(raw)
        except Exception:
            return None, None
    if mode in ("urlencoded", "formdata"):
        items = body.get(mode) or []
        example = {str(i.get("key")): i.get("value") for i in items if isinstance(i, dict) and i.get("key")}
        ctype = "application/x-www-form-urlencoded" if mode == "urlencoded" else "multipart/form-data"
        return ctype, example
    return None, None


def requestHasBearer(request: Dict[str, Any]) -> bool:
    auth = request.get("auth")
    return isinstance(auth, dict) and auth.get("type") == "bearer"


def walk(items: List[Dict[str, Any]], out: List[Dict[str, Any]]) -> None:
    for entry in items:
        if isinstance(entry.get("item"), list):
            walk(entry["item"], out)
        elif isinstance(entry.get("request"), dict):
            out.append(entry)


def convert(collection: Dict[str, Any]) -> Dict[str, Any]:
    requests: List[Dict[str, Any]] = []
    walk(collection.get("item") or [], requests)
    paths: Dict[str, Dict[str, Any]] = {}
    uses_bearer = False
    for entry in requests:
        request = entry["request"]
        method = str(request.get("method") or "GET").lower()
        segments = urlSegments(request.get("url"))
        if not segments:
            continue
        path, params = templatePath(segments)
        operation = paths.setdefault(path, {})
        if method in operation:
            continue  # keep the first example of each method/path
        ctype, example = bodyExample(request.get("body"))
        op: Dict[str, Any] = {
            "summary": entry.get("name") or f"{method.upper()} {path}",
            "operationId": re.sub(r"[^a-zA-Z0-9]+", "_", f"{method}_{path}").strip("_"),
            "responses": {"default": {"description": "Converted from Postman; response schema unknown"}},
        }
        if params:
            op["parameters"] = [
                {"name": name, "in": "path", "required": True, "schema": {"type": "string"}}
                for name in params
            ]
        if ctype and example is not None:
            op["requestBody"] = {"content": {ctype: {"example": example}}}
        if requestHasBearer(request):
            uses_bearer = True
            op["security"] = [{"bearerAuth": []}]
        operation[method] = op
    document: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {
            "title": (collection.get("info") or {}).get("name") or "Converted API",
            "version": "1.0.0",
            "description": "Generated from a Postman collection by postman_to_openapi.py.",
        },
        "paths": paths,
    }
    if uses_bearer:
        document["components"] = {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}}
        }
    return document


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    collection = json.loads(open(argv[1], encoding="utf-8").read())
    document = convert(collection)
    with open(argv[2], "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {argv[2]}: {len(document['paths'])} paths, "
          f"{sum(len(v) for v in document['paths'].values())} operations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
