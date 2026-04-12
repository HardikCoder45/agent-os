---
title: AGENT OS
emoji: "🧭"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: bsd-3-clause
---

# AGENT OS

AGENT OS is a deployment-ready OpenEnv benchmark for strategic business decision making. The repository is organized around a single shared benchmark engine, a typed OpenEnv server, a stronger deterministic-plus-LLM reward system, and a root-level `inference.py` that emits the required `[START]`, `[STEP]`, and `[END]` logs.

## Repository Layout

The repo is intentionally kept flat for import stability and hackathon deployment simplicity. The main areas are:

- `app.py`: AGENT OS Gradio UI.
- `server/`: Space and API entrypoints for deployment.
- `benchmark_engine.py`: shared state machine used by both the UI and OpenEnv environment.
- `hackathon_environment.py`: benchmark environment implementation.
- `reward_engine.py` and `judge_engine.py`: deterministic scoring and bounded LLM judging.
- `tool_schemas.py`: strict tool and argument validation.
- `benchmark_tasks.py`, `agents.py`, `situations.py`, `domains.py`: scenario catalog and task assembly.
- `inference.py`: required submission baseline script.
- `tests/`: regression coverage for leakage, scoring, progression, and proxy usage.
- `validate_local.py`: local end-to-end validation harness.
- `outputs/`: generated validation and scoring artifacts, recreated on demand.

## Required Environment Variables

Before submitting, define these variables in the Space or runtime configuration:

- `API_BASE_URL`: the OpenAI-compatible LLM endpoint.
- `MODEL_NAME`: the model name used for inference and optional judging.
- `API_KEY`: the API key used by the OpenAI client.

## Benchmark Overview

The environment currently ships with twelve graded canonical tasks:

- `ceo_fundraise_or_default`
- `cfr_burn_crisis`
- `cmo_category_creation`
- `cmo_social_crisis`
- `cmo_wrong_medication`
- `cso_safety_signal`
- `cto_scale_crisis`
- `ecom_ceo_brand_or_amazon`
- `hospital_ceo_insurance_battle`
- `pharma_ceo_bigpharma_offer`
- `pm_roadmap_vs_enterprise`
- `reg_nda_strategy`

Each task exposes:

- a typed action model
- a typed observation model
- a typed state model
- a role-specific tool registry
- a leak-free public task description
- deterministic checklist grading with risk gates
- optional bounded LLM semantic judging layered on top

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the AGENT OS UI locally:

```bash
python app.py
```

Run the API server locally:

```bash
python -m server.app --host 0.0.0.0 --port 7860
```

## Validation

Run the OpenEnv validator:

```bash
openenv validate .
```

Run the end-to-end local validation helper:

```bash
python validate_local.py
```

Start the server locally:

```bash
python -m server.app --host 0.0.0.0 --port 7860
```

Quick checks:

```bash
curl http://127.0.0.1:7860/
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/tasks
curl http://127.0.0.1:7860/metadata
curl http://127.0.0.1:7860/schema
```

## Baseline Inference

The required root-level baseline script is `inference.py`.

Run it locally:

```bash
python inference.py
```

The script:

- enumerates all canonical benchmark tasks
- emits strict `[START]`, `[STEP]`, and `[END]` structured logs
- writes reproducible scores to `outputs/baseline_scores.json`
- uses the OpenAI client when `API_BASE_URL`, `MODEL_NAME`, and `API_KEY` are available
- performs bounded retry-and-revise loops before advancing phases
- falls back to a deterministic benchmark reasoning policy if the LLM is unavailable

## Generated Artifacts

These files are generated locally and can be safely deleted at any time:

- `outputs/baseline_scores.json`
- `outputs/adversarial_regressions.json`
- `outputs/judge_consistency_report.json`
- `outputs/validation_report.json`
- `__pycache__/`
- `.pytest_cache/`
- `*.egg-info/`

## Docker

Build locally:

```bash
docker build -t hackathon-openenv .
```

Run locally:

```bash
docker run --rm -p 7860:7860 \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e API_KEY="$API_KEY" \
  hackathon-openenv
```

## Hugging Face Spaces

This repository is configured as a Docker Space and listens on port `7860`. The OpenEnv API remains available at:

- `/reset`
- `/step`
- `/state`
- `/schema`
- `/metadata`
- `/health`

The root page is the AGENT OS control surface so automated pings to the Space URL get a valid `200` response before interacting with the environment API.
