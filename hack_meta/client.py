"""Disaster response scene ladder client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    DisasterAction,
    DisasterObservation,
    ResourceStatus,
    TargetStatus,
)


class DisasterResponseEnv(EnvClient[DisasterAction, DisasterObservation, State]):
    """Client for the scene-based disaster response environment."""

    def _step_payload(self, action: DisasterAction) -> Dict:
        return {
            "assignments": [
                {
                    "resource_id": assignment.resource_id,
                    "target_id": assignment.target_id,
                }
                for assignment in action.assignments
            ]
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        obs_data = payload.get("observation", {})

        targets = {
            target_id: TargetStatus(**target_data)
            for target_id, target_data in obs_data.get("targets", {}).items()
        }
        resources = {
            resource_id: ResourceStatus(**resource_data)
            for resource_id, resource_data in obs_data.get("resources", {}).items()
        }

        observation = DisasterObservation(
            scene_id=obs_data.get("scene_id", ""),
            scene_name=obs_data.get("scene_name", ""),
            level=obs_data.get("level", 0),
            narrative=obs_data.get("narrative", ""),
            targets=targets,
            resources=resources,
            resolved_count=obs_data.get("resolved_count", 0),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 0),
            feedback=obs_data.get("feedback", ""),
            final_score=obs_data.get("final_score"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scene_id=payload.get("scene_id"),
            scene_name=payload.get("scene_name"),
            level=payload.get("level"),
        )
