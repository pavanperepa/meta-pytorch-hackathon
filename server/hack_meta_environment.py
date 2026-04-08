"""
Scene-based disaster response coordination environment.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DisasterAction,
        DisasterObservation,
        ResourceStatus,
        TargetStatus,
    )
    from .scene_catalog import DEFAULT_SCENE_ID, SCENE_CATALOG, SceneConfig, ordered_scene_ids
except ImportError:
    from models import DisasterAction, DisasterObservation, ResourceStatus, TargetStatus
    from server.scene_catalog import DEFAULT_SCENE_ID, SCENE_CATALOG, SceneConfig, ordered_scene_ids


class DisasterResponseEnvironment(Environment):
    """
    Multi-scene disaster response environment with hidden-state reward shaping.

    The agent sees targets, resources, and timing cues, but rewards come from a
    latent harm model so the policy cannot self-certify mediocre behavior.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scene: SceneConfig = SCENE_CATALOG[DEFAULT_SCENE_ID]
        self._targets: Dict[str, Dict[str, Any]] = {}
        self._resources: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, float] = {}
        self._turn: int = 0
        self._baseline_harm: float = 0.0
        self._final_score: Optional[float] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        level: Optional[int] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._scene = self._select_scene(scene_id=scene_id, level=level)
        self._targets = self._init_targets(self._scene)
        self._resources = self._init_resources(self._scene)
        self._metrics = {
            "fatalities": 0.0,
            "critical_injuries": 0.0,
            "exposure_harm": 0.0,
            "service_loss": 0.0,
            "invalid_actions": 0.0,
            "ineffective_assignments": 0.0,
            "deadline_misses": 0.0,
            "reassignment_churn": 0.0,
            "resolved_targets": 0.0,
            "failed_targets": 0.0,
        }
        self._turn = 0
        self._final_score = None
        self._baseline_harm = self._simulate_noop_baseline()

        feedback = (
            f"Level {self._scene.level}: {self._scene.name}\n"
            f"{self._scene.briefing}\n"
            f"Why this is hard: {self._scene.why_harder}\n"
            "Objective: minimize preventable deaths, critical injuries, exposure, and service collapse.\n"
            "Submit assignments as a JSON list of {resource_id, target_id} objects."
        )
        return self._build_observation(feedback=feedback, reward=0.0, done=False)

    def step(self, action: DisasterAction, **kwargs: Any) -> DisasterObservation:  # type: ignore[override]
        self._turn += 1
        self._state.step_count += 1

        feedback_parts: List[str] = []
        prev_potential = self._potential(self._targets)

        assignments_by_target: Dict[str, List[str]] = {tid: [] for tid in self._targets}
        used_resources: set[str] = set()
        penalty = 0.0

        for assignment in action.assignments:
            resource_id = assignment.resource_id
            target_id = assignment.target_id

            if resource_id not in self._resources:
                penalty += 6.0
                self._metrics["invalid_actions"] += 1
                feedback_parts.append(f"[ERR] Unknown resource '{resource_id}'")
                continue
            if target_id not in self._targets:
                penalty += 6.0
                self._metrics["invalid_actions"] += 1
                feedback_parts.append(f"[ERR] Unknown target '{target_id}'")
                continue
            if resource_id in used_resources:
                penalty += 5.0
                self._metrics["invalid_actions"] += 1
                feedback_parts.append(f"[ERR] Resource '{resource_id}' assigned more than once")
                continue
            if not self._resource_available(self._resources[resource_id], self._turn):
                penalty += 5.0
                self._metrics["invalid_actions"] += 1
                feedback_parts.append(f"[ERR] Resource '{resource_id}' is unavailable")
                continue
            if self._targets[target_id]["status"] == "resolved":
                penalty += 3.0
                self._metrics["ineffective_assignments"] += 1
                feedback_parts.append(f"[WARN] Target '{target_id}' already resolved")
                continue

            used_resources.add(resource_id)
            assignments_by_target[target_id].append(resource_id)

        penalty += self._apply_idle_penalty(used_resources)
        penalty += self._advance_system(assignments_by_target, feedback_parts)

        next_potential = self._potential(self._targets)
        reward = round((next_potential - prev_potential) / 10.0 - penalty, 3)

        done = self._all_targets_resolved() or self._turn >= self._scene.max_turns
        if done:
            self._final_score = self._compute_final_score()
            feedback_parts.append(
                f"Episode complete. Final score={self._final_score:.1f}/100."
            )

        feedback = " | ".join(feedback_parts) if feedback_parts else "Assignments executed."
        return self._build_observation(feedback=feedback, reward=reward, done=done)

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            scene_id=self._scene.scene_id,
            scene_name=self._scene.name,
            level=self._scene.level,
        )

    def _select_scene(
        self,
        scene_id: Optional[str],
        level: Optional[int],
    ) -> SceneConfig:
        if scene_id:
            if scene_id not in SCENE_CATALOG:
                raise ValueError(f"Unknown scene_id '{scene_id}'")
            return SCENE_CATALOG[scene_id]
        if level is not None:
            for candidate in SCENE_CATALOG.values():
                if candidate.level == level:
                    return candidate
            raise ValueError(f"Unknown level '{level}'")
        return SCENE_CATALOG[DEFAULT_SCENE_ID]

    def _init_targets(self, scene: SceneConfig) -> Dict[str, Dict[str, Any]]:
        targets: Dict[str, Dict[str, Any]] = {}
        for cfg in scene.targets:
            targets[cfg.target_id] = {
                "config": cfg,
                "status": "active",
                "progress": 0.0,
                "risk": cfg.initial_risk,
                "people_remaining": cfg.people_true,
                "time_remaining": cfg.deadline_turns,
                "last_assigned_resources": [],
                "deadline_missed": False,
                "failed": False,
            }
        return targets

    def _init_resources(self, scene: SceneConfig) -> Dict[str, Dict[str, Any]]:
        resources: Dict[str, Dict[str, Any]] = {}
        for cfg in scene.resources:
            resources[cfg.resource_id] = {
                "config": cfg,
                "remaining_uses": cfg.max_uses,
                "last_target_id": None,
            }
        return resources

    def _resource_available(self, resource: Dict[str, Any], turn: int) -> bool:
        cfg = resource["config"]
        if cfg.available_until_turn is not None and turn > cfg.available_until_turn:
            return False
        if resource["remaining_uses"] is not None and resource["remaining_uses"] <= 0:
            return False
        return True

    def _apply_idle_penalty(self, used_resources: set[str]) -> float:
        penalty = 0.0
        critical_targets = [
            target
            for target in self._targets.values()
            if target["status"] != "resolved" and target["time_remaining"] <= 2
        ]
        if not critical_targets:
            return penalty

        for resource_id, resource in self._resources.items():
            if resource_id in used_resources or not self._resource_available(resource, self._turn):
                continue
            if self._resource_can_help_any_target(resource["config"].capabilities, critical_targets):
                penalty += 3.0
        return penalty

    def _resource_can_help_any_target(
        self,
        capabilities: Dict[str, float],
        targets: List[Dict[str, Any]],
    ) -> bool:
        for target in targets:
            weights = target["config"].capability_weights
            if any(capability in weights for capability in capabilities):
                return True
        return False

    def _advance_system(
        self,
        assignments_by_target: Dict[str, List[str]],
        feedback_parts: List[str],
    ) -> float:
        penalty = 0.0
        newly_resolved: List[str] = []
        deadline_hits: List[str] = []

        for target_id, target in self._targets.items():
            cfg = target["config"]
            resource_ids = assignments_by_target.get(target_id, [])
            response_power = 0.0
            assigned_names: List[str] = []

            for resource_id in resource_ids:
                resource = self._resources[resource_id]
                resource_cfg = resource["config"]
                match = max(
                    (
                        resource_cfg.capabilities[capability] * weight
                        for capability, weight in cfg.capability_weights.items()
                        if capability in resource_cfg.capabilities
                    ),
                    default=0.0,
                )
                if match <= 0.0:
                    penalty += 3.0
                    self._metrics["ineffective_assignments"] += 1
                    feedback_parts.append(
                        f"[WARN] {resource_id} does not materially help {target_id}"
                    )
                    continue

                if resource["last_target_id"] not in (None, target_id):
                    penalty += 1.0
                    self._metrics["reassignment_churn"] += 1
                response_power += match
                assigned_names.append(resource_id)
                resource["last_target_id"] = target_id
                if resource["remaining_uses"] is not None:
                    resource["remaining_uses"] -= 1

            target["last_assigned_resources"] = assigned_names
            if target["status"] == "resolved" or target["failed"]:
                continue

            progress_gain = cfg.progress_per_power * response_power
            protection = min(0.92, target["progress"] * 0.55 + response_power * cfg.protection_per_power)
            target["progress"] = min(1.0, target["progress"] + progress_gain)

            target["risk"] = max(
                0.15,
                min(
                    2.5,
                    target["risk"] + cfg.escalation_rate - response_power * cfg.risk_reduction_per_power,
                ),
            )

            time_pressure = 1.0 + max(0, 1 - max(target["time_remaining"], 0) / max(1, cfg.deadline_turns)) * 0.6
            if target["time_remaining"] <= 0:
                time_pressure += 0.4

            protective_gap = max(0.05, 1.0 - protection)

            deaths_now = target["people_remaining"] * cfg.death_rate * target["risk"] * time_pressure * protective_gap
            critical_now = target["people_remaining"] * cfg.critical_rate * target["risk"] * time_pressure * protective_gap
            exposure_now = cfg.exposed_population * cfg.exposure_rate * target["risk"] * time_pressure * protective_gap
            service_now = cfg.service_scale * cfg.service_rate * target["risk"] * time_pressure * protective_gap

            self._metrics["fatalities"] += deaths_now
            self._metrics["critical_injuries"] += critical_now
            self._metrics["exposure_harm"] += exposure_now
            self._metrics["service_loss"] += service_now

            if target["people_remaining"] > 0.0:
                target["people_remaining"] = max(0.0, target["people_remaining"] - deaths_now)

            if target["progress"] >= 1.0 or (target["progress"] >= 0.86 and target["risk"] <= 0.25):
                if target["status"] != "resolved":
                    target["status"] = "resolved"
                    self._metrics["resolved_targets"] += 1
                    newly_resolved.append(cfg.name)
                continue

            if not target["deadline_missed"] and target["time_remaining"] <= 0 and target["progress"] < 0.60:
                target["deadline_missed"] = True
                weighted_miss = cfg.deadline_weight * cfg.vulnerability
                self._metrics["deadline_misses"] += weighted_miss
                penalty += 4.0 * weighted_miss
                deadline_hits.append(cfg.name)

            if target["time_remaining"] < -2 and target["progress"] < 0.35 and not target["failed"]:
                target["failed"] = True
                target["status"] = "failed"
                weighted_fail = cfg.deadline_weight * cfg.vulnerability
                self._metrics["failed_targets"] += weighted_fail
                penalty += 6.0 * weighted_fail
            elif target["progress"] >= 0.55:
                target["status"] = "contained"
            else:
                target["status"] = "active"

            target["time_remaining"] -= 1

        if newly_resolved:
            feedback_parts.append("Resolved: " + ", ".join(newly_resolved))
        if deadline_hits:
            feedback_parts.append("Critical window missed: " + ", ".join(deadline_hits))

        hot_targets = self._hot_target_summaries(limit=3)
        if hot_targets:
            feedback_parts.append("Hot targets: " + ", ".join(hot_targets))

        return penalty

    def _hot_target_summaries(self, limit: int) -> List[str]:
        active_targets = [
            target
            for target in self._targets.values()
            if target["status"] not in {"resolved", "failed"}
        ]
        active_targets.sort(
            key=lambda target: (
                -target["risk"],
                target["time_remaining"],
                -target["config"].vulnerability,
            )
        )
        summaries: List[str] = []
        for target in active_targets[:limit]:
            summaries.append(
                f"{target['config'].target_id}(risk={target['risk']:.2f}, t={target['time_remaining']})"
            )
        return summaries

    def _potential(self, targets: Dict[str, Dict[str, Any]]) -> float:
        total = 0.0
        for target in targets.values():
            if target["status"] == "resolved":
                continue
            cfg = target["config"]
            if target["failed"]:
                total += (
                    140.0 * max(0.0, target["people_remaining"])
                    + 24.0 * cfg.exposed_population
                    + 28.0 * cfg.service_scale
                    + 40.0 * cfg.deadline_weight * cfg.vulnerability
                )
                continue
            urgency = target["risk"] * (1.0 + max(0, 2 - target["time_remaining"]) * 0.35)
            protective_gap = max(0.05, 1.0 - target["progress"] * 0.75)
            expected_deaths = target["people_remaining"] * cfg.death_rate * urgency * protective_gap * cfg.vulnerability
            expected_critical = target["people_remaining"] * cfg.critical_rate * urgency * protective_gap * cfg.vulnerability
            expected_exposure = cfg.exposed_population * cfg.exposure_rate * urgency * protective_gap
            expected_service = cfg.service_scale * cfg.service_rate * urgency * protective_gap
            equity_gap = cfg.equity_weight * cfg.vulnerability * urgency * protective_gap * (1.0 - cfg.visibility)
            deadline_gap = max(0.0, 1.0 - max(target["time_remaining"], 0) / max(1, cfg.deadline_turns))
            total += (
                100.0 * expected_deaths
                + 35.0 * expected_critical
                + 12.0 * expected_exposure
                + 18.0 * expected_service
                + 10.0 * equity_gap
                + 8.0 * deadline_gap * cfg.deadline_weight
            )
        return -total

    def _simulate_noop_baseline(self) -> float:
        targets = deepcopy(self._targets)
        resources = deepcopy(self._resources)
        metrics = deepcopy(self._metrics)
        for turn in range(1, self._scene.max_turns + 1):
            empty_assignments = {target_id: [] for target_id in targets}
            self._advance_copy(targets, resources, metrics, empty_assignments, turn)
        return max(1.0, self._compute_total_harm(metrics))

    def _advance_copy(
        self,
        targets: Dict[str, Dict[str, Any]],
        resources: Dict[str, Dict[str, Any]],
        metrics: Dict[str, float],
        assignments_by_target: Dict[str, List[str]],
        turn: int,
    ) -> None:
        for target_id, target in targets.items():
            cfg = target["config"]
            response_power = 0.0
            for resource_id in assignments_by_target.get(target_id, []):
                resource = resources[resource_id]
                resource_cfg = resource["config"]
                match = max(
                    (
                        resource_cfg.capabilities[capability] * weight
                        for capability, weight in cfg.capability_weights.items()
                        if capability in resource_cfg.capabilities
                    ),
                    default=0.0,
                )
                if match <= 0.0:
                    metrics["ineffective_assignments"] += 1
                    continue
                response_power += match
                if resource["remaining_uses"] is not None:
                    resource["remaining_uses"] -= 1

            if target["status"] in {"resolved", "failed"}:
                continue

            progress_gain = cfg.progress_per_power * response_power
            protection = min(0.92, target["progress"] * 0.55 + response_power * cfg.protection_per_power)
            target["progress"] = min(1.0, target["progress"] + progress_gain)
            target["risk"] = max(
                0.15,
                min(
                    2.5,
                    target["risk"] + cfg.escalation_rate - response_power * cfg.risk_reduction_per_power,
                ),
            )

            time_pressure = 1.0 + max(0, 1 - max(target["time_remaining"], 0) / max(1, cfg.deadline_turns)) * 0.6
            if target["time_remaining"] <= 0:
                time_pressure += 0.4
            protective_gap = max(0.05, 1.0 - protection)

            deaths_now = target["people_remaining"] * cfg.death_rate * target["risk"] * time_pressure * protective_gap
            critical_now = target["people_remaining"] * cfg.critical_rate * target["risk"] * time_pressure * protective_gap
            exposure_now = cfg.exposed_population * cfg.exposure_rate * target["risk"] * time_pressure * protective_gap
            service_now = cfg.service_scale * cfg.service_rate * target["risk"] * time_pressure * protective_gap

            metrics["fatalities"] += deaths_now
            metrics["critical_injuries"] += critical_now
            metrics["exposure_harm"] += exposure_now
            metrics["service_loss"] += service_now

            if target["people_remaining"] > 0.0:
                target["people_remaining"] = max(0.0, target["people_remaining"] - deaths_now)

            if target["progress"] >= 1.0 or (target["progress"] >= 0.86 and target["risk"] <= 0.25):
                target["status"] = "resolved"
                metrics["resolved_targets"] += 1
                continue

            if not target["deadline_missed"] and target["time_remaining"] <= 0 and target["progress"] < 0.60:
                target["deadline_missed"] = True
                metrics["deadline_misses"] += cfg.deadline_weight * cfg.vulnerability

            if target["time_remaining"] < -2 and target["progress"] < 0.35 and not target["failed"]:
                target["failed"] = True
                target["status"] = "failed"
                metrics["failed_targets"] += cfg.deadline_weight * cfg.vulnerability
            elif target["progress"] >= 0.55:
                target["status"] = "contained"
            else:
                target["status"] = "active"

            target["time_remaining"] -= 1

    def _compute_total_harm(self, metrics: Dict[str, float]) -> float:
        return (
            100.0 * metrics["fatalities"]
            + 35.0 * metrics["critical_injuries"]
            + 12.0 * metrics["exposure_harm"]
            + 18.0 * metrics["service_loss"]
            + 18.0 * metrics["deadline_misses"]
            + 24.0 * metrics["failed_targets"]
            + 4.0 * metrics["invalid_actions"]
            + 2.0 * metrics["ineffective_assignments"]
            + 1.0 * metrics["reassignment_churn"]
        )

    def _compute_final_score(self) -> float:
        realized_harm = self._compute_total_harm(self._metrics)
        raw = 100.0 * (self._baseline_harm - realized_harm) / self._baseline_harm
        return max(0.0, min(100.0, round(raw, 2)))

    def _all_targets_resolved(self) -> bool:
        return all(target["status"] == "resolved" for target in self._targets.values())

    def _priority_band(self, target: Dict[str, Any]) -> str:
        cfg = target["config"]
        if target["failed"]:
            return "failed"
        urgency = target["risk"] * cfg.vulnerability
        if target["time_remaining"] <= 1 or urgency >= 1.6:
            return "immediate"
        if target["time_remaining"] <= 2 or urgency >= 1.15:
            return "high"
        if target["time_remaining"] <= 3 or urgency >= 0.8:
            return "medium"
        return "monitor"

    def _build_observation(
        self,
        feedback: str,
        reward: float,
        done: bool,
    ) -> DisasterObservation:
        targets = {
            target_id: TargetStatus(
                name=target["config"].name,
                category=target["config"].category,
                status=target["status"],
                estimated_people=target["config"].estimated_people,
                observed_risk=round(
                    max(
                        0.05,
                        min(
                            1.0,
                            target["config"].observed_risk
                            + (target["risk"] - target["config"].initial_risk) * 0.35,
                        ),
                    ),
                    3,
                ),
                critical_now=(target["time_remaining"] <= 1 and target["status"] not in {"resolved", "failed"}),
                priority_band=self._priority_band(target),
                vulnerability=target["config"].vulnerability_label,
                visibility=target["config"].visibility,
                progress=round(target["progress"], 3),
                time_remaining=target["time_remaining"],
                recommended_capabilities=list(target["config"].recommended_capabilities),
                last_assigned_resources=list(target["last_assigned_resources"]),
                description=(
                    f"{target['config'].description} Critical window: {target['config'].deadline_note}"
                ),
            )
            for target_id, target in self._targets.items()
        }
        resources = {
            resource_id: ResourceStatus(
                name=resource["config"].name,
                capabilities=sorted(resource["config"].capabilities.keys()),
                available=self._resource_available(resource, self._turn + 1 if not done else self._turn),
                remaining_uses=resource["remaining_uses"],
                available_until_turn=resource["config"].available_until_turn,
                description=resource["config"].description,
            )
            for resource_id, resource in self._resources.items()
        }
        resolved_count = sum(1 for target in self._targets.values() if target["status"] == "resolved")
        metadata: Dict[str, Any] = {
            "scene_ids": ordered_scene_ids(),
            "score_method": "normalized_against_noop_baseline",
        }
        if done and self._final_score is not None:
            metadata["audit_metrics"] = {
                key: round(value, 2) for key, value in self._metrics.items()
            }
            metadata["baseline_harm"] = round(self._baseline_harm, 2)

        return DisasterObservation(
            scene_id=self._scene.scene_id,
            scene_name=self._scene.name,
            level=self._scene.level,
            narrative=self._scene.briefing,
            targets=targets,
            resources=resources,
            resolved_count=resolved_count,
            turn=self._turn,
            max_turns=self._scene.max_turns,
            feedback=feedback,
            final_score=self._final_score if done else None,
            done=done,
            reward=reward,
            metadata=metadata,
        )
