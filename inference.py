"""
Evaluation runner for the disaster response scene ladder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
sys.path.insert(0, os.path.dirname(__file__))

from models import DisasterAction, DisasterObservation, ResourceAssignment
from server.scene_catalog import DEFAULT_SCENE_ID, ordered_scene_ids

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5.4-mini")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
API_KEY: Optional[str] = OPENAI_API_KEY or HF_TOKEN
SERVER_URL: str = os.getenv("SERVER_URL", "http://localhost:8000")

TEMPERATURE: float = 0.1
MAX_TOKENS: int = 700
DEBUG: bool = os.getenv("DEBUG", "0") == "1"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the EOC decision policy for a disaster-response simulator.

    HARD CONSTRAINTS (violations cause large penalties):
    - NEVER assign a resource where available=False. Check the AVAILABLE RESOURCES list only.
    - NEVER assign a resource to a resolved or failed target.
    - Do not assign the same resource more than once in one turn.

    DEADLINE RULES:
    - t=time_remaining turns until critical window closes. t=0 means this is the LAST turn to act.
    - When t<=0: the window has already closed — focus available resources on the next most urgent target.
    - When t=1: this target MUST be worked on this turn or the window closes.
    - When multiple targets have t=1, split resources proportionally — do not abandon one entirely.
    - Once you start working a t=1 target, keep that resource on it next turn unless it resolves.

    ASSIGNMENT STRATEGY:
    - If any target has critical_now=true, prioritize all compatible resources on it.
    - Never leave a resource idle when compatible unresolved targets exist.
    - Assigning a resource to a target it cannot help is wasteful and penalized.
    - Priority order: t=1 deadline > critical_now=true > extreme vulnerability > highest risk > population size.

    Respond with ONLY a JSON array:
    [
      {"resource_id": "<resource_id>", "target_id": "<target_id>"},
      ...
    ]
    """
).strip()


def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    print(f"{event_type} {json.dumps(payload, ensure_ascii=True)}")


def _visibility_label(value: float) -> str:
    if value >= 0.8:
        return "very high"
    if value >= 0.55:
        return "high"
    if value >= 0.3:
        return "medium"
    return "low"


def _targets_table(obs: DisasterObservation) -> str:
    rows: List[str] = []
    for target_id, target in obs.targets.items():
        capabilities = ",".join(target.recommended_capabilities)
        rows.append(
            f"  {target_id:<20} risk={target.observed_risk:.2f}  progress={target.progress:>4.0%}  "
            f"t={target.time_remaining:>2}  status={target.status:<9}  "
            f"critical={str(target.critical_now):<5} prio={target.priority_band:<9}  "
            f"people={target.estimated_people:<26} vuln={target.vulnerability:<9} "
            f"vis={_visibility_label(target.visibility):<9} needs={capabilities}"
        )
    return "\n".join(rows)


def _resources_table(obs: DisasterObservation) -> str:
    available_rows: List[str] = []
    unavailable_rows: List[str] = []
    for resource_id, resource in obs.resources.items():
        use_note = (
            "inf"
            if resource.remaining_uses is None
            else str(resource.remaining_uses)
        )
        turn_note = (
            "-"
            if resource.available_until_turn is None
            else str(resource.available_until_turn)
        )
        line = (
            f"  {resource_id:<20} uses={use_note:<3} "
            f"until_turn={turn_note:<3} caps={','.join(resource.capabilities)}"
        )
        if resource.available:
            available_rows.append(line)
        else:
            unavailable_rows.append(f"  [UNAVAILABLE - DO NOT ASSIGN] {resource_id}")
    sections: List[str] = []
    if available_rows:
        sections.append("AVAILABLE (assign these):\n" + "\n".join(available_rows))
    if unavailable_rows:
        sections.append("UNAVAILABLE (never assign):\n" + "\n".join(unavailable_rows))
    return "\n".join(sections)


def build_user_prompt(obs: DisasterObservation, history: Sequence[str]) -> str:
    recent = "\n".join(f"  {item}" for item in history[-4:]) if history else "  (none yet)"
    return textwrap.dedent(
        f"""
        Scene {obs.scene_id} | Level {obs.level} | {obs.scene_name}
        Turn {obs.turn}/{obs.max_turns} | Resolved targets: {obs.resolved_count}/{len(obs.targets)}

        BRIEFING:
        {obs.narrative}

        TARGETS:
        {_targets_table(obs)}

        RESOURCES:
        {_resources_table(obs)}

        LAST FEEDBACK:
        {obs.feedback}

        RECENT HISTORY:
        {recent}

        Output the assignment JSON array now:
        """
    ).strip()


def parse_assignments(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    # Strip Qwen3 / any model's <think>...</think> reasoning blocks before parsing
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("assignments", "actions", "allocation", "allocations"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        pass
    return []


def assignments_to_action(raw: List[Dict[str, Any]]) -> DisasterAction:
    assignments: List[ResourceAssignment] = []
    for item in raw:
        try:
            assignments.append(
                ResourceAssignment(
                    resource_id=str(item["resource_id"]),
                    target_id=str(item["target_id"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            if DEBUG:
                print(f"[DEBUG] Dropping invalid assignment: {item}")
    return DisasterAction(assignments=assignments)


def extract_people_signal(text: str) -> float:
    numbers = [int(match) for match in re.findall(r"\d+", text)]
    if not numbers:
        return 1.0
    return float(sum(numbers))


def vulnerability_weight(label: str) -> float:
    weights = {
        "mixed": 1.0,
        "medium": 1.0,
        "high": 1.3,
        "very high": 1.55,
        "extreme": 1.8,
        "indirect": 0.8,
    }
    return weights.get(label.lower(), 1.0)


def target_priority(target: Any) -> float:
    deadline_pressure = 3.0 if target.time_remaining <= 1 else 1.7 if target.time_remaining == 2 else 1.0
    return (
        target.observed_risk * 12.0
        + vulnerability_weight(target.vulnerability) * 4.0
        + deadline_pressure * 3.0
        + min(6.0, extract_people_signal(target.estimated_people) / 50.0)
        + (1.0 - target.progress) * 4.0
    )


def capability_match(resource: Any, target: Any) -> float:
    overlap = set(resource.capabilities).intersection(target.recommended_capabilities)
    if not overlap:
        return 0.0
    return 1.0 + 0.2 * len(overlap)


def compatible_target_ids(obs: DisasterObservation, resource: Any) -> List[str]:
    return [
        target_id
        for target_id, target in obs.targets.items()
        if target.status not in {"resolved", "failed"}
        and capability_match(resource, target) > 0.0
    ]


def heuristic_policy(obs: DisasterObservation) -> DisasterAction:
    available_resources = sorted(
        (
            (resource_id, resource)
            for resource_id, resource in obs.resources.items()
            if resource.available
        ),
        key=lambda item: (
            item[1].remaining_uses is None,
            len(item[1].capabilities),
        ),
    )
    active_targets = [
        (target_id, target)
        for target_id, target in obs.targets.items()
        if target.status not in {"resolved", "failed"}
    ]
    assignments: List[ResourceAssignment] = []
    for resource_id, resource in available_resources:
        best_target_id: Optional[str] = None
        best_score = 0.0
        for target_id, target in active_targets:
            match = capability_match(resource, target)
            if match <= 0.0:
                continue
            score = match * target_priority(target)
            if score > best_score:
                best_score = score
                best_target_id = target_id
        if best_target_id is not None:
            assignments.append(
                ResourceAssignment(resource_id=resource_id, target_id=best_target_id)
            )
    return DisasterAction(assignments=assignments)


def random_policy(obs: DisasterObservation, rng: random.Random) -> DisasterAction:
    assignments: List[ResourceAssignment] = []
    active_targets = [
        (target_id, target)
        for target_id, target in obs.targets.items()
        if target.status not in {"resolved", "failed"}
    ]
    for resource_id, resource in obs.resources.items():
        if not resource.available:
            continue
        candidates = [
            target_id
            for target_id, target in active_targets
            if capability_match(resource, target) > 0.0
        ]
        if not candidates:
            continue
        assignments.append(
            ResourceAssignment(resource_id=resource_id, target_id=rng.choice(candidates))
        )
    return DisasterAction(assignments=assignments)


@dataclass
class StepResult:
    observation: DisasterObservation
    reward: float
    done: bool


@dataclass
class EpisodeStats:
    scene_id: str
    level: int
    scene_name: str
    total_reward: float
    final_score: float
    grader_score: float
    level_mastery: Optional[float]
    audit: Dict[str, Any]
    discipline: Dict[str, Any]
    turn_sequence: List[Dict[str, Any]]


def discipline_score(record: Dict[str, Any]) -> float:
    penalty = (
        10.0 * float(record.get("invalid_actions", 0.0))
        + 4.0 * float(record.get("ineffective_assignments", 0.0))
        + 8.0 * float(record.get("idle_feasible_turns", 0.0))
        + 3.0 * float(record.get("idle_feasible_resources", 0.0))
        + 6.0 * float(record.get("resolved_target_assignments", 0.0))
        + 8.0 * float(record.get("empty_plan_turns", 0.0))
        + 10.0 * float(record.get("empty_parse_turns", 0.0))
    )
    return max(0.0, 100.0 - penalty)


def compute_level_mastery(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    scores = [float(record["final_score"]) for record in records]
    score_mean = mean(scores) if scores else 0.0
    score_std = pstdev(scores) if len(scores) > 1 else 0.0
    consistency_bonus = max(0.0, 100.0 - 4.0 * score_std)
    discipline_bonus = mean(discipline_score(record) for record in records) if records else 0.0
    level_mastery = 0.70 * score_mean + 0.20 * consistency_bonus + 0.10 * discipline_bonus
    return {
        "mean_final_score": round(score_mean, 2),
        "score_stddev": round(score_std, 2),
        "consistency_bonus": round(consistency_bonus, 2),
        "discipline_bonus": round(discipline_bonus, 2),
        "level_mastery": round(level_mastery, 2),
    }


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class LocalEnvWrapper:
    def __init__(self) -> None:
        from server.hack_meta_environment import DisasterResponseEnvironment

        self._env = DisasterResponseEnvironment()

    def reset(self, **kwargs: Any) -> StepResult:
        obs = self._env.reset(**kwargs)
        return StepResult(observation=obs, reward=float(obs.reward or 0.0), done=obs.done)

    def step(self, action: DisasterAction) -> StepResult:
        obs = self._env.step(action)
        return StepResult(observation=obs, reward=float(obs.reward or 0.0), done=obs.done)

    def close(self) -> None:
        pass


def run_scene(
    env: Any,
    scene_id: str,
    policy: str,
    client: Optional[OpenAI],
    rng: random.Random,
    run_index: int,
    episode: int,
    mode: str,
) -> EpisodeStats:
    result = env.reset(scene_id=scene_id)
    obs = result.observation
    history: List[str] = []
    total_reward = 0.0
    turn_sequence: List[Dict[str, Any]] = []
    empty_plan_turns = 0
    idle_feasible_turns = 0
    idle_feasible_resources = 0
    resolved_target_assignments = 0
    empty_parse_turns = 0

    emit_event(
        "START",
        {
            "run_index": run_index,
            "episode": episode,
            "mode": mode,
            "policy": policy,
            "model": MODEL_NAME if policy == "llm" else policy,
            "scene_id": obs.scene_id,
            "scene_name": obs.scene_name,
            "level": obs.level,
            "max_turns": obs.max_turns,
        },
    )

    while not result.done:
        if policy == "heuristic":
            action = heuristic_policy(obs)
        elif policy == "random":
            action = random_policy(obs, rng)
        else:
            assert client is not None
            user_prompt = build_user_prompt(obs, history)
            if DEBUG:
                print(f"\n[DEBUG] PROMPT\n{user_prompt}\n")
            try:
                _messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    # Standard path: works for Groq, HF, and older OpenAI models
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=_messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                except Exception as _inner:
                    # Newer OpenAI models (gpt-4.1, gpt-5, o-series) reject max_tokens
                    # and/or temperature — retry with max_completion_tokens only
                    if "max_tokens" not in str(_inner) and "temperature" not in str(_inner):
                        raise
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=_messages,
                        max_completion_tokens=MAX_TOKENS,
                        stream=False,
                    )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                emit_event(
                    "STEP",
                    {
                        "run_index": run_index,
                        "episode": episode,
                        "scene_id": obs.scene_id,
                        "turn": obs.turn + 1,
                        "warning": f"LLM call failed: {exc}",
                    },
                )
                response_text = "[]"
            raw = parse_assignments(response_text)
            if response_text.strip() and not raw:
                empty_parse_turns += 1
            action = assignments_to_action(raw)

        # Defensive filter: strip unavailable resources and resolved/failed targets
        valid_assignments = [
            a for a in action.assignments
            if obs.resources.get(a.resource_id) is not None
            and obs.resources[a.resource_id].available
            and obs.targets.get(a.target_id) is not None
            and obs.targets[a.target_id].status not in {"resolved", "failed"}
        ]
        # Deduplicate: each resource used at most once
        seen_resources: set = set()
        deduped: List[ResourceAssignment] = []
        for a in valid_assignments:
            if a.resource_id not in seen_resources:
                seen_resources.add(a.resource_id)
                deduped.append(a)
        action = DisasterAction(assignments=deduped)

        used_resource_ids = {assignment.resource_id for assignment in action.assignments}
        compatible_idle_now = 0
        for resource_id, resource in obs.resources.items():
            if not resource.available or resource_id in used_resource_ids:
                continue
            compatible_targets = compatible_target_ids(obs, resource)
            if compatible_targets:
                compatible_idle_now += 1
        if compatible_idle_now > 0:
            idle_feasible_turns += 1
            idle_feasible_resources += compatible_idle_now

        for assignment in action.assignments:
            target = obs.targets.get(assignment.target_id)
            if target and target.status in {"resolved", "failed"}:
                resolved_target_assignments += 1

        summary = ", ".join(
            f"{assignment.resource_id}->{assignment.target_id}"
            for assignment in action.assignments
        ) or "(idle)"
        if not action.assignments:
            empty_plan_turns += 1

        result = env.step(action)
        next_obs = result.observation
        obs = result.observation
        total_reward += float(result.reward or 0.0)
        emit_event(
            "STEP",
            {
                "run_index": run_index,
                "episode": episode,
                "scene_id": obs.scene_id,
                "turn": int(next_obs.turn),
                "assignments": [
                    {
                        "resource_id": assignment.resource_id,
                        "target_id": assignment.target_id,
                    }
                    for assignment in action.assignments
                ],
                "summary": summary,
                "reward": round(float(result.reward or 0.0), 3),
                "resolved_count": int(obs.resolved_count),
                "target_count": len(obs.targets),
                "feedback": obs.feedback,
            },
        )

        turn_sequence.append(
            {
                "turn": int(next_obs.turn),
                "assignments": [
                    {
                        "resource_id": assignment.resource_id,
                        "target_id": assignment.target_id,
                    }
                    for assignment in action.assignments
                ],
                "summary": summary,
                "reward": float(result.reward or 0.0),
                "resolved_count": int(next_obs.resolved_count),
                "feedback": next_obs.feedback,
            }
        )
        history.append(f"t{obs.turn}: {summary} -> {float(result.reward or 0.0):+.2f}")

    final_score = float(obs.final_score or 0.0)
    audit = obs.metadata.get("audit_metrics", {})
    discipline = {
        "idle_feasible_turns": idle_feasible_turns,
        "idle_feasible_resources": idle_feasible_resources,
        "resolved_target_assignments": resolved_target_assignments,
        "empty_plan_turns": empty_plan_turns,
        "empty_parse_turns": empty_parse_turns,
    }
    emit_event(
        "END",
        {
            "run_index": run_index,
            "episode": episode,
            "scene_id": obs.scene_id,
            "scene_name": obs.scene_name,
            "level": int(obs.level),
            "final_score": round(final_score, 2),
            "grader_score": round(final_score / 100.0, 4),
            "total_reward": round(total_reward, 3),
            "fatalities": round(float(audit.get("fatalities", 0.0)), 2),
            "critical_injuries": round(float(audit.get("critical_injuries", 0.0)), 2),
            "deadline_misses": round(float(audit.get("deadline_misses", 0.0)), 2),
            "failed_targets": round(float(audit.get("failed_targets", 0.0)), 2),
            "discipline": discipline,
        },
    )
    return EpisodeStats(
        scene_id=obs.scene_id,
        level=int(obs.level),
        scene_name=obs.scene_name,
        total_reward=total_reward,
        final_score=final_score,
        grader_score=round(final_score / 100.0, 4),
        level_mastery=None,
        audit=audit,
        discipline=discipline,
        turn_sequence=turn_sequence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policies on the disaster response ladder.")
    parser.add_argument("--local", action="store_true", help="Run the environment in-process.")
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE_ID,
        help="Scene ID to run, or 'all' for the full ladder.",
    )
    parser.add_argument(
        "--policy",
        choices=["heuristic", "random", "llm"],
        default="heuristic",
        help="Policy to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many repeats to run for each selected scene.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used by the random baseline.",
    )
    parser.add_argument(
        "--log-path",
        default="logs/eval_runs.jsonl",
        help="JSONL file where per-episode records are appended.",
    )
    args = parser.parse_args()

    if args.policy == "llm" and not API_KEY:
        emit_event(
            "END",
            {
                "status": "error",
                "message": "No API key found. Set OPENAI_API_KEY or HF_TOKEN for --policy llm.",
            },
        )
        sys.exit(1)

    if args.scene == "all":
        selected_scenes = ordered_scene_ids()
    else:
        selected_scenes = [args.scene]

    rng = random.Random(args.seed)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if args.policy == "llm" else None
    mode = "local" if args.local else "server"

    if args.local:
        env: Any = LocalEnvWrapper()
    else:
        from client import DisasterResponseEnv

        env = DisasterResponseEnv(base_url=SERVER_URL).sync()

    log_path = Path(args.log_path)
    records: List[Dict[str, Any]] = []
    try:
        run_index = 1
        for episode in range(1, args.episodes + 1):
            for scene_id in selected_scenes:
                stats = run_scene(
                    env=env,
                    scene_id=scene_id,
                    policy=args.policy,
                    client=client,
                    rng=rng,
                    run_index=run_index,
                    episode=episode,
                    mode=mode,
                )
                record: Dict[str, Any] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "scene_id": stats.scene_id,
                    "level": stats.level,
                    "scene_name": stats.scene_name,
                    "model": MODEL_NAME if args.policy == "llm" else args.policy,
                    "policy": args.policy,
                    "episode": episode,
                    "run_index": run_index,
                    "final_score": round(stats.final_score, 2),
                    "grader_score": round(stats.grader_score, 4),
                    "total_reward": round(stats.total_reward, 3),
                    "fatalities": round(float(stats.audit.get("fatalities", 0.0)), 2),
                    "critical_injuries": round(float(stats.audit.get("critical_injuries", 0.0)), 2),
                    "exposure_harm": round(float(stats.audit.get("exposure_harm", 0.0)), 2),
                    "service_loss": round(float(stats.audit.get("service_loss", 0.0)), 2),
                    "deadline_misses": round(float(stats.audit.get("deadline_misses", 0.0)), 2),
                    "failed_targets": round(float(stats.audit.get("failed_targets", 0.0)), 2),
                    "invalid_actions": round(float(stats.audit.get("invalid_actions", 0.0)), 2),
                    "ineffective_assignments": round(float(stats.audit.get("ineffective_assignments", 0.0)), 2),
                    "reassignment_churn": round(float(stats.audit.get("reassignment_churn", 0.0)), 2),
                    "idle_feasible_turns": stats.discipline["idle_feasible_turns"],
                    "idle_feasible_resources": stats.discipline["idle_feasible_resources"],
                    "resolved_target_assignments": stats.discipline["resolved_target_assignments"],
                    "empty_plan_turns": stats.discipline["empty_plan_turns"],
                    "empty_parse_turns": stats.discipline["empty_parse_turns"],
                    "turn_sequence": stats.turn_sequence,
                }
                append_jsonl(log_path, record)
                records.append(record)
                run_index += 1
    finally:
        env.close()

    if len(records) > 1:
        rewards = [float(item["total_reward"]) for item in records]
        scores = [float(item["final_score"]) for item in records]

        per_scene: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            per_scene.setdefault(str(record["scene_id"]), []).append(record)
        for scene_id in selected_scenes:
            scene_records = per_scene.get(scene_id, [])
            if not scene_records:
                continue
            mastery = compute_level_mastery(scene_records)
            catastrophic_failures = sum(
                1
                for record in scene_records
                if float(record["final_score"]) < 25.0 or float(record["failed_targets"]) > 0.0
            )
            emit_event(
                "END",
                {
                    "aggregate": True,
                    "scene_id": scene_id,
                    "runs": len(scene_records),
                    "mean_final_score": mastery["mean_final_score"],
                    "score_stddev": mastery["score_stddev"],
                    "discipline_bonus": mastery["discipline_bonus"],
                    "level_mastery": mastery["level_mastery"],
                    "catastrophic_failures": catastrophic_failures,
                    "log_path": str(log_path),
                },
            )
    elif len(records) == 1:
        mastery = compute_level_mastery(records)
        emit_event(
            "END",
            {
                "aggregate": True,
                "runs": 1,
                "grader_score": records[0]["grader_score"],
                "level_mastery": mastery["level_mastery"],
                "log_path": str(log_path),
            },
        )


if __name__ == "__main__":
    main()
