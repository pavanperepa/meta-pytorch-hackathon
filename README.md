# Disaster Response Coordination OpenEnv

This repository contains an OpenEnv environment for evaluating disaster-response coordination agents. The task is to allocate scarce emergency resources across simultaneous incidents while minimizing preventable harm.

## Environment Summary

The agent acts as an Emergency Operations Center coordinator. Each episode presents one scene from a difficulty ladder and asks the model to assign resources to operational targets.

The environment models:

- rescue triage under deadline pressure
- medical fragility
- infrastructure cascade risk
- unequal visibility and vulnerability
- scarce one-shot assets in hard scenarios

## Action Space

The agent submits a typed `DisasterAction`:

```json
{
  "assignments": [
    {"resource_id": "engineering_strike", "target_id": "hospital_power"},
    {"resource_id": "tunnel_rescue", "target_id": "tunnel_train"}
  ]
}
```

Rules:

- each resource can be assigned at most once per turn
- unavailable resources should not be assigned
- resolved or failed targets should not be assigned
- incompatible assignments are allowed by schema but penalized by the simulator

## Observation Space

The typed `DisasterObservation` includes:

- `scene_id`, `scene_name`, `level`
- `narrative`
- visible `targets`
- visible `resources`
- `resolved_count`
- `turn`, `max_turns`
- `feedback`
- `final_score` once done

Each target includes:

- status
- estimated people
- observed risk
- `critical_now`
- `priority_band`
- vulnerability
- visibility
- progress
- time remaining
- recommended capabilities

## Reward Model

The environment uses dense step reward plus a normalized end score.

Per-step reward is driven by latent harm reduction and penalties for:

- invalid actions
- ineffective assignments
- leaving compatible resources idle
- deadline misses
- failed targets
- reassignment churn

The typed reward contract is represented by `DisasterReward` in [models.py](/c:/Users/pavan/meta-pytorch-hackathon/models.py).

Final evaluation uses:

- `final_score` in `[0, 100]`
- `grader_score = final_score / 100.0` in `[0.0, 1.0]`

## Task Ladder

The environment contains eight scenes spanning easy to hard:

1. Flash Flood, Two Rescue Calls, One Boat
2. Flood Rescue vs Medical Transport
3. Building Collapse vs Highway Hazmat Crash
4. Wildfire Suburb vs Nursing Home
5. Hospital Backup Power vs Tunnel Train Entrapment
6. Toxic Plume vs Downtown Office Tower Fire
7. Bridge Collapse During VIP Event Weekend
8. Regional Multi-Disaster with Scarce Air Assets

This satisfies the minimum requirement of at least three tasks with a real difficulty range.

## Baseline Evaluation

[`inference.py`](/c:/Users/pavan/meta-pytorch-hackathon/inference.py) is the required baseline script. It supports:

- `heuristic`
- `random`
- `llm`

It uses the OpenAI client and reads configuration from environment variables.

Structured stdout uses exact `START`, `STEP`, and `END` event lines.

## Required Environment Variables

- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY`
- `HF_TOKEN`
- optional: `LOCAL_IMAGE_NAME`

Use [.env.example](/c:/Users/pavan/meta-pytorch-hackathon/.env.example) as the template.

## Validation

Validate from repo root:

```powershell
.\hack_meta\.venv\Scripts\openenv.exe validate .
```

## Docker

Build from repo root:

```powershell
docker build -t hack_meta-env:latest .
```

Run:

```powershell
docker run --rm -p 8000:8000 hack_meta-env:latest
```

Expected endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /docs`

## Local Usage

Run the server locally:

```powershell
.\hack_meta\.venv\Scripts\python.exe -m server.app
```

Run the baseline from repo root:

```powershell
hack_meta\.venv\Scripts\python.exe inference.py --local --policy heuristic --scene scene_1
```
