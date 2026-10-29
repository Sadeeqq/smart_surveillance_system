# Smart Surveillance System

A local-first prototype for detecting and tracking people and creating an event when a tracked person crosses an operator-configured virtual line. Face processing is optional enrichment, and an unknown/no-match result is not evidence of wrongdoing. The system is being built incrementally according to the PRD; this Phase 0 foundation does not yet connect to a camera or run AI models.

## Requirements

- Python 3.12 (initial development baseline; model/runtime compatibility will be validated in later phases)
- Git

## Setup

From the repository root, create and activate a virtual environment, then install the phase 0 dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` for local overrides. `.env` is ignored by Git; never put secrets in source files or logs. Missing optional settings use safe defaults.

## Run and verify

```bash
python -m app.main
pytest
```

The Phase 0 application starts without camera or model packages. Camera, detection, tracking, boundary events, face processing, persistence, notifications, and the dashboard arrive in later approved phases. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/INTERFACES.md](docs/INTERFACES.md).

## Local data

`data/` and `models/` are excluded from Git. Event image retention, enrolment/removal, access controls and real-deployment policies must be defined before real use; see [docs/DECISIONS.md](docs/DECISIONS.md).
