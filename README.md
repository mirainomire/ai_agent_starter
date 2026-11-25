# BYEBAC

BYEBAC is a policy-first, AI agent-based framework for testing Broken Access Control in REST APIs.

Current version: **1.1.0**

## Scope

BYEBAC evaluates whether observed API behavior conforms to an explicit authorization policy. It supports multi-account testing for:

- Broken Object Level Authorization (BOLA) and IDOR
- Broken Function Level Authorization (BFLA)
- Broken Object Property Level Authorization (BOPLA)
- Authentication and selected header or query controls

The framework does not test unrelated vulnerability classes such as injection, cross-site scripting, or request forgery.

## How it works

The runtime workflow is:

```text
Plan -> Discover -> Execute -> Observe -> Report
```

Planning, policy resolution, result classification, and metric calculation are deterministic. In the evaluated configuration, Gemini 3 Flash Preview is used only for:

1. Selecting suitable resource identifiers from candidates already observed by the framework.
2. Producing a natural-language report summary.

The LLM does not determine the expected authorization outcome or assign TP, FP, FN, or TN labels.

### Evaluated LLM configuration

- Model identifier: `models/gemini-3-flash-preview`
- SDK: `google-genai` 1.66.0
- Temperature: `0.0`
- Top-p: `1.0`
- Thinking level for structured selection: `MINIMAL`
- Maximum summary output: 2,000 tokens
- Identifier selection output: `max(2000, 120 * number_of_slots)` tokens

Structured identifier selection uses a JSON response schema. API requests are sent directly by `HttpClient` to the target API and are not routed through Gemini. The model is not given credentials, authorization headers, source code, raw request or response bodies, or complete OAS and policy files.

## Required inputs

Runtime inputs are intentionally excluded from Git when they may expose target details:

- `ai_agent/config/agent.yaml`: execution settings and target base URL fallback
- `ai_agent/config/auth.yaml`: account aliases, roles, and login behavior
- `ai_agent/config/policy.yaml`: authorization oracle
- `ai_agent/data/openapi.json`: API operation and schema inventory
- Environment variables: credentials, target URL, and Gemini API key

RBAC CSV files are optional supporting inputs:

- `roles.csv`
- `permissions.csv`
- `role_permission.csv`
- `rbac_matrix.csv`

Never commit populated `.env` files, passwords, bearer tokens, API keys, or production data.

## Installation

Requirements:

- Python 3.14 or compatible Python 3 release
- An isolated, authorized API target
- Synthetic test accounts and data

```bash
cd ai_agent_starter
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a local `.env` file and provide only the variables required by your account configuration. Typical variables include:

```dotenv
API_BASE_URL=http://127.0.0.1:8000/api
GEMINI_API_KEY=replace-me

ADMIN_USERNAME=replace-me
ADMIN_PASSWORD=replace-me
EMPLOYEE_USERNAME=replace-me
EMPLOYEE_PASSWORD=replace-me
EMPLOYEE_2_USERNAME=replace-me
EMPLOYEE_2_PASSWORD=replace-me
```

Variable names must match the placeholders referenced by `auth.yaml`. Use an isolated test environment. Do not run authorization mutations against production data.

## Usage

From the `ai_agent_starter` directory:

```bash
# Validate the local setup
python byebac.py /check

# Run the complete agent workflow
python byebac.py /runagent

# Show the latest run summary
python byebac.py /status

# Show report locations
python byebac.py /report

# Display active configuration without exposing credentials
python byebac.py /config
```

Generated runtime files are written under `ai_agent/runs/` and are ignored by Git.

## Reports and labels

Each run produces:

- A JSON report containing metadata, cases, observations, labels, metrics, and evidence references
- A Markdown report for human review
- Request and response artifacts where configured

Only the following labels enter the confusion matrix:

- `TP`: expected ALLOW and the API allowed the request
- `FN`: expected ALLOW and the API denied the request
- `TN`: expected DENY and the API denied the request or ignored the denied write
- `FP`: expected DENY and the API allowed the request or applied the restricted write

`SKIP`, `NOT_FOUND`, `INCONCLUSIVE`, `UNVERIFIED`, `OUTSIDE_POLICY`, and `ERROR` are reported separately and do not contribute to confusion-matrix metrics.

These metrics measure conformance to the configured policy. They are not a general estimate of vulnerability-detection accuracy across arbitrary applications.

## Evaluation snapshot

The reported evaluation used a self-curated HRIS back end with:

- 56 API operations on 32 paths
- 28 object-addressed operations
- Three configured test accounts
- 168 operation-account pairs
- Five runs on a secure target variant
- Five runs on a variant containing five documented seeded BAC flaws

Each automated run processed 321 planned cases and sent 317 HTTP requests. Secure runs produced 288 conclusive confusion-matrix outcomes; seeded runs produced 287. In every seeded run, eight false-allow labels mapped to all five documented seeded flaws. Secure runs did not label a policy violation.

Agreement across each five-run group supports repeatability under the reported configuration. It does not establish reproducibility across applications, environments, datasets, or model versions. The separate 228-scenario manual execution used a different case set, date, machine, and execution count, so its elapsed time is descriptive rather than a controlled benchmark.

## Tests

The test suite uses Python's standard `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
ai_agent_starter/
  byebac.py
  requirements.txt
  ai_agent/
    config/
    core/
      orchestrator.py
      tools_auth.py
      tools_http.py
      evaluators.py
      reporters.py
    data/
    prompts/
    scripts/
  tests/
```

## Security and data protection

- Test only systems for which you have explicit authorization.
- Use synthetic accounts and records in an isolated environment.
- Review the provider's retention, training, and data-processing terms before enabling an external LLM.
- BYEBAC applies payload minimization and redaction, but these controls cannot guarantee zero disclosure.
- Replace the external model with an approved local model when organizational policy prohibits external processing.

## Research artifacts

The evaluation dataset and run evidence are published separately from the source repository:

- Evaluation dataset: <https://doi.org/10.6084/m9.figshare.31742872>
- Source archive: <https://doi.org/10.5281/zenodo.18524561>

The ten evaluation reports reference source commit `4607a679`. Version 1.1.0 retains the evaluated behavior while removing unused legacy entry points and updating user-facing documentation. Published source and dataset archives must not contain populated environment files or runtime credentials.

## Limitations

- Policy quality determines the expected-access oracle.
- Workflow-dependent authorization may require additional state modeling.
- Some cases can remain inconclusive because of record state or unavailable identifiers.
- Results from one target application do not establish broader reproducibility or production readiness.
- CI/CD integration is a potential application and has not been evaluated as a production deployment.

## License

MIT License. See `ai_agent_starter/LICENSE`.
