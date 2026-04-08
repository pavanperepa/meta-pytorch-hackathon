---
title: Disaster Response Coordination Environment
emoji: siren
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Disaster Response Coordination Environment

This OpenEnv environment simulates an Emergency Operations Center allocating scarce resources across simultaneous disaster targets. The agent must reduce preventable deaths, critical injuries, exposure harm, and infrastructure failure across a ladder of progressively harder scenes.

## Motivation

This is a real-world coordination problem rather than a toy task. It evaluates whether an agent can:

- prioritize under time pressure
- reason about vulnerability and deadlines
- handle mixed rescue and infrastructure triage
- avoid harmful but superficially plausible actions

The environment is designed to provide rich per-step reward while keeping the true harm model hidden from the agent.

## Action Space

Each turn the agent submits a `DisasterAction` with zero or more resource assignments:

```json
{
  "assignments": [
    {"resource_id": "engineering_strike", "target_id": "hospital_power"},
    {"resource_id": "tunnel_rescue", "target_id": "tunnel_train"}
  ]
}
```

Constraints:

- a resource may be assigned at most once per turn
- unavailable resources must not be assigned
- resolved or failed targets should not be assigned
- assignments with no capability overlap are ineffective and penalized

## Observation Space

The agent sees:

- scene id, name, and level
- narrative briefing
- visible target state:
  - status
  - estimated people
  - observed risk
  - `critical_now`
  - `priority_band`
  - vulnerability label
  - progress
  - time remaining
  - recommended capabilities
- visible resource state:
  - capabilities
  - availability
  - remaining uses
  - available-until turn
- structured feedback from the previous step

The latent harm model remains hidden so the policy cannot self-score.

## Task Ladder

The environment contains a genuine easy-to-hard difficulty range:

1. `scene_1`: Flash Flood, Two Rescue Calls, One Boat
2. `scene_2`: Flood Rescue vs Medical Transport
3. `scene_3`: Building Collapse vs Highway Hazmat Crash
4. `scene_4`: Wildfire Suburb vs Nursing Home
5. `scene_5`: Hospital Backup Power vs Tunnel Train Entrapment
6. `scene_6`: Toxic Plume vs Downtown Office Tower Fire
7. `scene_7`: Bridge Collapse During VIP Event Weekend
8. `scene_8`: Regional Multi-Disaster with Scarce Air Assets

For submission purposes, this exceeds the minimum requirement of three tasks with easy, medium, and hard coverage.

## Reward And Grading

Per-step reward is dense and shaped:

- positive reward for reducing latent remaining harm
- penalties for invalid actions
- penalties for ineffective assignments
- penalties for leaving compatible resources idle during critical windows
- penalties for deadline misses, churn, and failed targets

Final evaluation uses a normalized score against a no-op baseline:

- `final_score` in `[0, 100]`
- `grader_score = final_score / 100.0` in `[0.0, 1.0]`

This keeps grading deterministic and reproducible while preserving a meaningful learning signal.

## Baselines

The repo-root [`inference.py`](/c:/Users/pavan/meta-pytorch-hackathon/inference.py) supports:

- `heuristic`
- `random`
- `llm`

Recent observed behavior:

- strong scenes: `scene_4`, `scene_6`, `scene_7`
- middling scenes: `scene_2`, `scene_5`
- weak scenes: `scene_1`, `scene_3`
- hard-fail scene: `scene_8`

## Validate Locally

From this directory:

```powershell
.\.venv\Scripts\openenv.exe validate
```

## Run Locally

Run the API locally:

```powershell
.\.venv\Scripts\python.exe -m server.app
```

Or:

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

Build from this directory:

```powershell
docker build -t hack_meta-env:latest -f server/Dockerfile .
```

Run:

```powershell
docker run --rm -p 8000:8000 hack_meta-env:latest
```

## Hugging Face Space

This package directory is the deployable environment root. Deploy from `hack_meta/`, not from the repo root.

Before pushing:

1. configure environment secrets in the Space settings
2. validate locally
3. confirm `reset()` responds successfully

## Package Layout

```text
hack_meta/
|-- client.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- README.md
`-- server/
    |-- app.py
    |-- Dockerfile
    `-- hack_meta_environment.py
```
