import os, time, json, pathlib, requests, random
from urllib.parse import urlsplit
from typing import Dict, Any, Optional

class HttpClient:
    def __init__(self, base_url: str, timeout_s: int=20, retries: int=1, artifacts_dir: str="ai_agent/runs/artifacts", dry_run: bool=False,
                 token_header: str = "Authorization", token_type: str = "Bearer"):
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s
        self.retries = retries
        self.artifacts_dir = artifacts_dir
        self.dry_run = dry_run
        self.token_header = token_header or "Authorization"
        self.token_type = token_type or "Bearer"
        pathlib.Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        self.session = None if dry_run else requests.Session()

    @staticmethod
    def _safe_segment(value: str, fallback: str = "unknown", max_len: int = 72) -> str:
        """Create filesystem-safe path segment with bounded length."""
        try:
            s = str(value or "").strip().lower()
            if not s:
                s = fallback
            # Keep alnum + separators only.
            cleaned = []
            for ch in s:
                if ch.isalnum() or ch in ("_", "-", "."):
                    cleaned.append(ch)
                elif ch in ("/", "\\", " ", ":", "?", "&", "="):
                    cleaned.append("_")
            s = "".join(cleaned).strip("._-")
            while "__" in s:
                s = s.replace("__", "_")
            if not s:
                s = fallback
            return s[:max_len]
        except Exception:
            return fallback

    def _artifact_path(
        self,
        name: str,
        role: str = None,
        bac_type: str = None,
        target_label: str | None = None,
        method: str | None = None,
        path: str | None = None,
    ) -> str:
        """
        Generate organized artifact path structure:
        artifacts/
          {role}/
            horizontal/  (IDOR - same privilege level, different user)
            vertical/    (BOLA - privilege escalation)
            baseline/    (normal expected operations)
            auth/        (authentication related)
        """
        # Group by date to keep runs tidy and easy to inspect.
        day_folder = time.strftime("%Y_%m_%d")
        if role:
            # Normalize role name for folder (lowercase, replace spaces/special chars)
            role_folder = self._safe_segment(role, fallback="unknown_role")
            
            # Determine subfolder based on BAC type, mapped to expected labels
            bt = (bac_type or "baseline").strip().lower()
            label = (
                "cross_principal" if bt == "horizontal" else
                "role_scope" if bt == "vertical" else
                "authentication" if bt == "auth" else
                "baseline"
            )
            
            # Optional target label (e.g., to_admin, to_employee, to_same_role)
            parts = [self.artifacts_dir, day_folder, role_folder, label]
            if target_label:
                safe_target = self._safe_segment(target_label, fallback="to_unknown")
                parts.append(safe_target)
            # Method split (GET/POST/PUT/etc.)
            if method:
                parts.append(self._safe_segment(method.upper(), fallback="UNKNOWN_METHOD", max_len=16))
            # Endpoint split (users_id, payroll, auth_login, etc.)
            if path:
                endpoint_key = str(path or "").strip().strip("/")
                endpoint_key = endpoint_key.replace("{", "").replace("}", "")
                endpoint_key = endpoint_key.replace("/", "_")
                parts.append(self._safe_segment(endpoint_key, fallback="root_endpoint"))

            # Create directory structure
            full_dir = os.path.join(*parts)
            pathlib.Path(full_dir).mkdir(parents=True, exist_ok=True)
            return os.path.join(full_dir, name)
        else:
            # Fallback to flat structure if no role provided
            return os.path.join(self.artifacts_dir, day_folder, name)

    def _mask_headers_for_artifact(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        masked = dict(headers or {})
        # Mask common and configured auth header
        header_keys = {"authorization", "Authorization", self.token_header, self.token_header.lower()}
        auth = None
        for k in header_keys:
            if k in masked and isinstance(masked.get(k), str):
                auth = masked.get(k)
                target_key = k
                break
        if auth and isinstance(auth, str):
            try:
                parts = auth.split()
                if len(parts) == 2:
                    scheme, tok = parts
                    if len(tok) > 12:
                        tok = tok[:6] + "..." + tok[-4:]
                    masked[target_key] = f"{scheme} {tok}"
                else:
                    masked[target_key] = "***masked***"
            except Exception:
                masked[target_key] = "***masked***"
        return masked
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """
        Recursively mask sensitive information in JSON data.
        Masks: password, access_token, refresh_token, token, secret, api_key, etc.
        """
        if data is None:
            return None
        
        # Sensitive field keywords to mask
        sensitive_keywords = {
            'password', 'passwd', 'pwd', 
            'access_token', 'refresh_token', 'token', 'bearer',
            'secret', 'api_key', 'apikey', 'private_key',
            'client_secret', 'auth_token', 'session_id',
            'jwt', 'authorization'
        }
        
        def _should_mask(key: str) -> bool:
            """Check if field should be masked based on keywords"""
            key_lower = str(key).lower()
            return any(keyword in key_lower for keyword in sensitive_keywords)
        
        def _mask_value(value: str) -> str:
            """Mask a sensitive value, preserving first/last chars for debugging"""
            if not isinstance(value, str) or len(value) == 0:
                return "***masked***"
            if len(value) <= 8:
                return "***masked***"
            # Show first 4 and last 4 chars
            return f"{value[:4]}...{value[-4:]}"
        
        def _mask_recursive(obj: Any) -> Any:
            """Recursively traverse and mask sensitive data"""
            if isinstance(obj, dict):
                masked = {}
                for key, val in obj.items():
                    if _should_mask(key):
                        if isinstance(val, str):
                            masked[key] = _mask_value(val)
                        else:
                            masked[key] = "***masked***"
                    else:
                        masked[key] = _mask_recursive(val)
                return masked
            elif isinstance(obj, list):
                return [_mask_recursive(item) for item in obj]
            else:
                return obj
        
        return _mask_recursive(data)

    def request(self, method: str, path: str, token: Optional[str], params=None, json_body=None, extra_headers: Optional[Dict[str, str]] = None, 
                role: str = None, bac_type: str = None, test_context: Dict[str, Any] = None, data=None) -> Dict[str, Any]:
        """
        Execute HTTP request and save artifact with organized structure.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            token: Authorization token
            params: Query parameters
            json_body: JSON request body
            extra_headers: Additional headers
            role: User role for organizing artifacts (e.g., 'Admin_HC', 'Employee')
            bac_type: BAC test type - 'horizontal', 'vertical', 'baseline', 'auth'
            test_context: Additional context (self_access, mutation info) for metadata
        """
        # Normalize path defensively: strip BOM and enforce leading slash
        try:
            p = str(path)
        except Exception:
            p = "/"
        # Strip Unicode BOM if present
        p = p.lstrip('\ufeff')
        # Ensure leading slash
        if not p.startswith('/'):
            p = '/' + p
        url = f"{self.base_url}{p}"
        headers = {"Accept": "application/json"}
        # Do not send Authorization only on unauth auth flows (login/register/refresh).
        try:
            if str(bac_type).lower() == 'auth':
                pnorm = str(p).lower().rstrip('/')
                unauth_auth_paths = (
                    '/auth/login',
                    '/auth/signin',
                    '/auth/register',
                    '/auth/signup',
                    '/auth/refresh',
                    '/auth/token',
                )
                if any(pnorm.endswith(x) for x in unauth_auth_paths):
                    token = None
        except Exception:
            pass
        if token:
            # Use configured header and type
            if self.token_type:
                headers[self.token_header] = f"{self.token_type} {token}"
            else:
                headers[self.token_header] = token
        if extra_headers:
            headers.update({k: v for k, v in extra_headers.items() if v is not None})
        attempt = 0
        last_exc = None
        downgraded_localhost_https = False
        while attempt <= self.retries:
            try:
                if self.dry_run:
                    # HARDCODED DRY-RUN SENTINEL:
                    # HTTP 418 is intentionally non-successful so simulated traffic
                    # cannot be mistaken for a real authorization result.
                    resp = {"status_code": 418, "body": {"dry_run": True, "url": url}}
                else:
                    sess = self.session or requests
                    if data is not None:
                        # Form-encoded body (application/x-www-form-urlencoded); requests sets the header.
                        r = sess.request(method.upper(), url, headers=headers, params=params, data=data, timeout=self.timeout_s)
                    else:
                        r = sess.request(method.upper(), url, headers=headers, params=params, json=json_body, timeout=self.timeout_s)
                    resp = {"status_code": r.status_code, "body": self._safe_json(r)}
                # Save the artifact with an organized structure
                ts = int(time.time()*1000)
                # Human-friendly UTC time for logs (DD-MM-YYYY HH:MM)
                from datetime import datetime, timezone
                ts_str = datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M")
                safe = path.strip('/').replace('/','_').replace('?','_').replace('&','_').replace('=','-')
                safe = self._safe_segment(safe, fallback="root", max_len=96)
                status = int(resp.get("status_code") or 0)
                name = f"{ts}_{str(method).upper()}_{safe}_s{status}.json"
                
                # Derive optional target label for deeper categorization (best-practice):
                # - horizontal: same role, to_same_role or to_<role>
                # - vertical: privilege escalation attempts, try to infer target role
                target_label = None
                try:
                    ctx = test_context or {}
                    original_role = (ctx.get("original_role") or role or "").strip()
                    as_role = (ctx.get("as_role") or role or "").strip()
                    mut = ctx.get("mutation") or {}

                    def _norm(s: str) -> str:
                        return (s or "").strip().lower().replace(' ', '_').replace('-', '_')

                    if str(bac_type).lower() == 'horizontal':
                        # same privilege level; target is effectively the same role
                        # Use explicit same-role label for clarity
                        target_label = f"to_{_norm(original_role) or 'same_role'}"
                    elif str(bac_type).lower() == 'vertical':
                        # privilege escalation; if mutation specifies a different role, use it
                        mut_role = mut.get('as_role')
                        if isinstance(mut_role, str) and _norm(mut_role) and _norm(mut_role) != _norm(original_role):
                            target_label = f"to_{_norm(mut_role)}"
                        else:
                            target_label = 'to_unknown'
                except Exception:
                    target_label = None

                # Generate artifact path with role/type/target organization
                artifact_full_path = self._artifact_path(
                    name,
                    role=role,
                    bac_type=bac_type,
                    target_label=target_label,
                    method=str(method).upper(),
                    path=path,
                )
                
                # Mask sensitive data in request/response before saving
                masked_headers = self._mask_headers_for_artifact(headers)
                masked_params = self._mask_sensitive_data(params) if params else None
                masked_json_body = self._mask_sensitive_data(json_body) if json_body else None
                masked_response = self._mask_sensitive_data(resp)
                
                # Build artifact metadata
                artifact_data = {
                    "request": {
                        "method": method, 
                        "url": url, 
                        "headers": masked_headers, 
                        "params": masked_params, 
                        "json": masked_json_body
                    },
                    "response": masked_response,
                    "metadata": {
                        "role": role,
                        "bac_type": bac_type,
                        "timestamp": ts,
                        "timestamp_str": ts_str,
                        "folder_label": f"{(role or '').strip().lower().replace(' ','_').replace('-','_')} ({bac_type or 'baseline'})",
                        "artifact_path": artifact_full_path,
                        "test_context": test_context or {}
                    }
                }
                
                with open(artifact_full_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_data, f, indent=2)
                
                resp["artifact"] = artifact_full_path
                return resp
            except Exception as e:
                # FALLBACK KHUSUS LOCAL DEVELOPMENT:
                # Downgrade HTTPS -> HTTP only for localhost/127.0.0.1. Never
                # broaden this host allow-list because that would weaken TLS.
                try:
                    emsg = str(e)
                    parsed = urlsplit(url)
                    host = (parsed.hostname or "").lower()
                    is_local = host in ("127.0.0.1", "localhost")
                    is_https = (parsed.scheme or "").lower() == "https"
                    if (
                        not downgraded_localhost_https
                        and is_local
                        and is_https
                        and ("WRONG_VERSION_NUMBER" in emsg or "SSLError" in emsg)
                    ):
                        url = url.replace("https://", "http://", 1)
                        if isinstance(self.base_url, str):
                            self.base_url = self.base_url.replace("https://", "http://", 1)
                        downgraded_localhost_https = True
                        continue
                except Exception:
                    pass
                last_exc = e
                attempt += 1
                # Exponential backoff with jitter
                delay = min(0.25 * (2 ** attempt), 2.0) + random.random() * 0.2
                time.sleep(delay)
        raise last_exc

    def smart_request(self, method: str, path: str, token: Optional[str], openapi: Dict[str, Any], 
                      params=None, json_body=None, discovered_ids: Dict[str, Any] = None,
                      extra_headers: Optional[Dict[str, str]] = None, role: str = None, 
                      bac_type: str = None, test_context: Dict[str, Any] = None, max_retries: int = 2) -> Dict[str, Any]:
        """
        Smart HTTP request with automatic payload generation and adaptive retry on 400 errors.
        
        This method:
        1. If json_body is None and method requires body (POST/PUT/PATCH), auto-generate from OpenAPI schema
        2. Execute request normally
        3. If 400 error, analyze response and retry with adjusted payload
        4. Return final response with metadata about payload generation
        
        Args:
            method: HTTP method
            path: API path
            token: Auth token
            openapi: OpenAPI specification dict
            params: Query parameters (optional)
            json_body: Request body (if None, will auto-generate for POST/PUT/PATCH)
            discovered_ids: Dict of discovered resource IDs for payload generation
            extra_headers: Additional headers
            role: User role
            bac_type: BAC test type
            test_context: Test context metadata
            max_retries: Max retry attempts for 400 errors (default: 2)
        
        Returns:
            Response dict with additional 'payload_info' metadata
        """
        from ai_agent.core.utils import extract_request_schema, generate_payload_from_schema
        
        # Flag to track if we generated the payload
        payload_generated = False
        original_body = json_body
        schema_info = None
        if method.upper() in ('POST', 'PUT', 'PATCH'):
            schema_info = extract_request_schema(openapi, method, path)
        
        should_skip = False

        # DEBUG: Check conditions
        # Auto-generate payload for methods that typically require body (but skip auth endpoints)
        if json_body is None and method.upper() in ('POST', 'PUT', 'PATCH') and not should_skip:
            schema_info = extract_request_schema(openapi, method, path)
            
            # extract_request_schema returns wrapper with 'schema' key for the actual schema
            actual_schema = schema_info.get('schema', schema_info) if isinstance(schema_info, dict) else {}
            has_content = bool(actual_schema and (actual_schema.get('properties') or actual_schema.get('items') or actual_schema.get('type'))) or schema_info.get('example') is not None
            
            # Generate payload if we have usable schema
            if has_content:
                # Preserve media-level examples alongside the resolved schema.
                json_body = generate_payload_from_schema(schema_info, discovered_ids)
                payload_generated = True
                # Only print compact message
                schema_type = actual_schema.get('type', 'object')
                print(f"      [SMART] Auto-generated {schema_type} payload")
        
        # Try initial request
        attempt = 0
        last_response = None
        self_access = bool((test_context or {}).get("self_access", True))
        
        while attempt <= max_retries:
            try:
                # Final ID normalization before sending request:
                # prefer discovered IDs over synthetic defaults for *_id fields.
                if isinstance(json_body, dict) and isinstance(discovered_ids, dict) and discovered_ids:
                    for fk, fv in list(json_body.items()):
                        if not isinstance(fk, str):
                            continue
                        if "id" not in fk.lower():
                            continue
                        fkl = fk.lower()
                        # For user_id-like fields, always use discovered user candidate (self/other).
                        if "user" in fkl:
                            uid = self._pick_user_id(discovered_ids, self_access=self_access)
                            if isinstance(uid, int) and uid > 0 and fv != uid:
                                json_body[fk] = uid
                            continue
                        rid = self._resolve_discovered_id_for_field(fk, discovered_ids)
                        if isinstance(rid, int) and rid > 0:
                            # Replace missing/synthetic values with discovered resource IDs.
                            if fv is None or (isinstance(fv, int) and fv <= 1):
                                json_body[fk] = rid
                response = self.request(
                    method=method,
                    path=path,
                    token=token,
                    params=params,
                    json_body=json_body,
                    extra_headers=extra_headers,
                    role=role,
                    bac_type=bac_type,
                    test_context=test_context
                )
                
                last_response = response
                status_code = response.get('status_code', 0)

                # Success or non-400 error - return immediately
                if status_code != 400:
                    response['payload_info'] = {
                        'auto_generated': payload_generated,
                        'original_body': original_body,
                        'final_body': json_body,
                        'attempts': attempt + 1,
                        'schema_available': bool(schema_info) if payload_generated else None
                    }
                    return response
                
                # Got 400 - try to improve payload if we haven't exhausted retries
                if attempt < max_retries:
                    if self._repair_unique_conflict(method, json_body, response.get('body'), schema_info):
                        attempt += 1
                        continue
                    # Analyze error message
                    error_body = response.get('body', {})
                    error_msg = str(error_body.get('message') or error_body.get('detail') or '').lower()
                    
                    # Try to extract missing field from error message
                    missing_field = self._extract_missing_field(error_msg)
                    
                    # Also extract required fields when API returns {"field": ["...required..."]}
                    required_fields = self._extract_required_fields_from_body(error_body)
                    if required_fields and isinstance(json_body, dict):
                        changed = False
                        for rf in required_fields:
                            if rf in json_body:
                                continue
                            rfl = rf.lower()
                            schema_props = {}
                            if isinstance(schema_info, dict):
                                schema_props = schema_info.get('properties') or {}
                            json_body[rf] = self._default_for_required_field(
                                rf,
                                schema_props.get(rf) or {},
                                discovered_ids=discovered_ids,
                                self_access=self_access,
                            )
                            changed = True
                        if changed:
                            attempt += 1
                            continue

                    if missing_field and isinstance(json_body, dict):
                        # Add the missing field with a default value
                        if missing_field not in json_body:
                            # Try to infer type from field name
                            missing_l = missing_field.lower()
                            if 'id' in missing_l:
                                if "user" in missing_l:
                                    rid = self._pick_user_id(discovered_ids or {}, self_access=self_access)
                                else:
                                    rid = self._resolve_discovered_id_for_field(missing_field, discovered_ids or {})
                                json_body[missing_field] = rid if isinstance(rid, int) and rid > 0 else 1
                            elif 'email' in missing_field.lower():
                                json_body[missing_field] = "test@example.com"
                            elif 'name' in missing_field.lower():
                                json_body[missing_field] = "Test Name"
                            elif 'status' in missing_field.lower():
                                json_body[missing_field] = "active"
                            else:
                                # HARDCODED LAST RESORT: this placeholder only makes
                                # an inconclusive request possible; it is not domain-valid data.
                                json_body[missing_field] = "default_value"
                            
                            # Retry with updated payload
                            attempt += 1
                            continue
                
                # If we reach here, either exhausted retries or couldn't fix the issue
                break
                
            except Exception as e:
                # Network or other errors - don't retry, raise immediately
                raise e
        
        # Return last response with metadata
        if last_response:
            last_response['payload_info'] = {
                'auto_generated': payload_generated,
                'original_body': original_body,
                'final_body': json_body,
                'attempts': attempt + 1,
                'schema_available': bool(extract_request_schema(openapi, method, path)) if payload_generated else None,
                'failed_after_retries': True
            }
        
        return last_response or {'status_code': 0, 'body': {}, 'error': 'No response received'}
    
    @staticmethod
    def _default_for_required_field(
        field_name: str,
        field_schema: Dict[str, Any],
        discovered_ids: Dict[str, Any] = None,
        self_access: bool = True,
    ) -> Any:
        """
        Generate deterministic default from schema first, then field semantics.
        Avoid role- or project-specific hardcoded values.
        """
        import datetime as _dt
        discovered_ids = discovered_ids or {}

        ftype = str(field_schema.get("type") or "").lower()
        fmt = str(field_schema.get("format") or "").lower()
        name = str(field_name or "").lower()

        # For ID-like fields, prioritize discovered IDs over schema examples/defaults.
        if "id" in name:
            if "user" in name:
                uid = HttpClient._pick_user_id(discovered_ids, self_access=self_access)
                if isinstance(uid, int) and uid > 0:
                    return uid
            rid = HttpClient._resolve_discovered_id_for_field(field_name, discovered_ids)
            if isinstance(rid, int) and rid > 0:
                return rid

        enum_vals = field_schema.get("enum")
        if isinstance(enum_vals, list) and enum_vals:
            return enum_vals[0]
        if field_schema.get("example") is not None:
            return field_schema.get("example")
        if field_schema.get("default") is not None:
            return field_schema.get("default")

        if ftype in ("integer", "number") or "id" in name:
            minimum = field_schema.get("minimum")
            if isinstance(minimum, (int, float)):
                return int(minimum)
            if "year" in name:
                return _dt.date.today().year
            if "month" in name:
                return 1
            # HARDCODED LAST RESORT: prefer discovered IDs/schema examples. Value 1
            # may reference real data and must not be treated as a safe fixture.
            return 1
        if ftype == "boolean":
            return False
        if fmt == "email" or "email" in name:
            return "test@example.com"
        if fmt in ("date-time", "datetime"):
            return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        if fmt == "date" or "date" in name:
            return _dt.date.today().isoformat()
        if "password" in name:
            return "Test123!Secure"
        if "username" in name:
            return "tester_auto"
        if "name" in name:
            return "Test Name"
        if "salary" in name:
            return 1000
        if ftype == "array":
            return []
        if ftype == "object":
            return {}
        # HARDCODED LAST RESORT for an unknown string-like required field.
        return "default_value"

    @staticmethod
    def _pick_user_id(discovered_ids: Dict[str, Any], self_access: bool = True) -> Optional[int]:
        """
        Pick best user id from discovered IDs.
        - self_access=True  -> prefer user_id/user
        - self_access=False -> prefer other_user_id/other_user, fallback to user_id
        """
        try:
            if not isinstance(discovered_ids, dict):
                return None
            if not self_access:
                for k in ("other_user_id", "other_user"):
                    v = discovered_ids.get(k)
                    if isinstance(v, int) and v > 0:
                        return v
            for k in ("user_id", "id_user", "user"):
                v = discovered_ids.get(k)
                if isinstance(v, int) and v > 0:
                    return v
        except Exception:
            return None
        return None
    
    @staticmethod
    def _repair_unique_conflict(method, payload, body, schema_info):
        """Repair only fields explicitly rejected as duplicates by the API."""
        import re
        import uuid
        if not isinstance(payload, dict) or not isinstance(body, dict):
            return False
        schema = (schema_info or {}).get('schema', {})
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        details = body.get('details') or body.get('errors') or body
        message = ' '.join(str(body.get(k, '')) for k in ('error', 'detail', 'message'))
        changed = False
        for field in list(payload):
            field_error = details.get(field, '') if isinstance(details, dict) else ''
            if not field_error and re.search(r'\b' + re.escape(field) + r'\b', message, re.I):
                field_error = message
            if not re.search(r'already exists|duplicate|must be unique', str(field_error), re.I):
                continue
            prop = properties.get(field)
            if not isinstance(prop, dict):
                continue
            if method.upper() in ('PUT', 'PATCH') and field not in required:
                del payload[field]
                changed = True
                continue
            value = payload[field]
            if not isinstance(value, str) or prop.get('enum') or prop.get('pattern'):
                continue
            suffix = uuid.uuid4().hex[:10]
            limit = int(prop.get('maxLength', max(64, len(value) + 11)))
            tail = ''
            if prop.get('format') == 'email' and '@' in value:
                value, domain = value.rsplit('@', 1)
                tail = '@' + domain
            room = limit - len(tail) - len(suffix) - 1
            if room < 1:
                continue
            payload[field] = value[:room] + '_' + suffix + tail
            changed = True
        return changed

    @staticmethod
    def _error_text_from_response(response: Dict[str, Any]) -> str:
        try:
            body = (response or {}).get("body", {})
            if isinstance(body, dict):
                parts = []
                for k in ("message", "detail", "error", "text"):
                    v = body.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                if parts:
                    return " | ".join(parts).lower()
                return str(body).lower()
            return str(body).lower()
        except Exception:
            return ""

    @staticmethod
    def _resolve_discovered_id_for_field(field_name: str, discovered_ids: Dict[str, Any]) -> Optional[int]:
        """
        Resolve best discovered ID for an id-like field (e.g., user_id, id_user, payroll_id).
        """
        try:
            if not isinstance(discovered_ids, dict):
                return None
            fname = str(field_name or "").lower().strip()
            if not fname or "id" not in fname:
                return None

            # 1) Exact/alias matches first.
            aliases = [fname]
            if fname.startswith("id_"):
                aliases.append(fname[3:])
            if fname.endswith("_id"):
                aliases.append(fname[:-3])
            aliases.extend(["user_id", "id_user", "user"])
            for a in aliases:
                v = discovered_ids.get(a)
                if isinstance(v, int) and v > 0:
                    return v

            # 2) Token-based matching.
            token = fname.replace("id_", "").replace("_id", "")
            for k, v in discovered_ids.items():
                if not isinstance(k, str):
                    continue
                if not (isinstance(v, int) and v > 0):
                    continue
                kl = k.lower()
                if token and (token in kl or kl in token):
                    return v

            # 3) Fallback to first positive int ID-like key.
            for k, v in discovered_ids.items():
                if isinstance(k, str) and "id" in k.lower() and isinstance(v, int) and v > 0:
                    return v
        except Exception:
            return None
        return None
    
    @staticmethod
    def _extract_required_fields_from_body(error_body: Any) -> list[str]:
        """
        Extract required field names from structured 400 payloads, e.g.
        {"username": ["This field is required."], "password": ["This field is required."]}
        """
        try:
            out = []
            if isinstance(error_body, dict):
                for k, v in error_body.items():
                    if not isinstance(k, str):
                        continue
                    if isinstance(v, list):
                        joined = " ".join(str(x) for x in v).lower()
                        if "required" in joined:
                            out.append(k)
                    elif isinstance(v, str):
                        if "required" in v.lower():
                            out.append(k)
            return out
        except Exception:
            return []
    
    @staticmethod
    def _extract_missing_field(error_message: str) -> Optional[str]:
        """
        Extract missing field name from common error message patterns.
        
        Examples:
        - "field 'email' is required" -> "email"
        - "Missing required parameter: name" -> "name"
        - "The field status is required." -> "status"
        """
        import re
        
        # Pattern 1: field 'xxx' is required
        match = re.search(r"field\s+['\"]([^'\"]+)['\"].*required", error_message, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 2: Missing required parameter: xxx
        match = re.search(r"missing.*required.*parameter[:\s]+([a-z_]+)", error_message, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 3: field xxx is required
        match = re.search(r"field\s+([a-z_]+).*required", error_message, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 4: xxx is required
        match = re.search(r"([a-z_]+)\s+is\s+required", error_message, re.IGNORECASE)
        if match:
            field = match.group(1)
            # Avoid false positives like "authentication is required"
            if field not in ['authentication', 'authorization', 'permission', 'access']:
                return field
        
        return None

    @staticmethod
    def _safe_json(r):
        try:
            return r.json()
        except Exception:
            return {"text": r.text[:2000]}
