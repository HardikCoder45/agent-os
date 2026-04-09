---
title: Hackathon OpenEnv
emoji: "🧭"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: bsd-3-clause
---

# Hackathon OpenEnv

Hackathon OpenEnv is a deployment-ready OpenEnv benchmark for strategic business decision making. It now aligns the repo around a typed FastAPI/OpenEnv server, a deterministic benchmark task set, a stronger reward + grader pipeline, and a root-level `inference.py` that produces reproducible scores and strict `[START]`, `[STEP]`, `[END]` stdout lines.

## What Changed

- `reset()`, `step()`, and `state` now follow the installed OpenEnv interface directly.
- The root Space URL returns `200` with a working control surface instead of a blank / 404 landing page.
- Four benchmark tasks are exposed through `/tasks`, with deterministic graders and rewards in the `0.0` to `1.0` range.
- The judge path supports OpenAI-compatible endpoints via `API_BASE_URL`, `MODEL_NAME`, and `API_KEY`.
- The repository includes both a root `Dockerfile` for Hugging Face Spaces and the expected root `inference.py`.

## Required Environment Variables

Before submitting, define these variables in the Space or runtime configuration:

- `API_BASE_URL`: the OpenAI-compatible LLM endpoint.
- `MODEL_NAME`: the model name used for inference and optional judging.
- `API_KEY`: the API key used by the OpenAI client.

## Benchmark Tasks

The environment currently ships with four graded tasks:

- `startup_ceo_fundraise`
- `startup_cto_scale`
- `pharma_cso_signal`
- `healthcare_cmo_safety`

Each task exposes:

- a typed action model
- a typed observation model
- a typed state model
- a role-specific tool registry
- deterministic task grading
- optional LLM judging layered on top

## Local Validation

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

- enumerates all benchmark tasks
- emits strict `[START]`, `[STEP]`, and `[END]` structured logs
- writes reproducible scores to `outputs/baseline_scores.json`
- uses the OpenAI client when `API_BASE_URL`, `MODEL_NAME`, and `API_KEY` are available
- falls back to a deterministic benchmark reasoning policy if the LLM is unavailable

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

The root page is a lightweight control surface so automated pings to the Space URL get a valid `200` response before interacting with the environment API.
