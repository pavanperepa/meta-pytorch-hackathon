"""
Data models for the scene-based disaster response environment.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class ResourceAssignment(BaseModel):
    """Assign one resource to one target for the current turn."""

    resource_id: str = Field(..., description="Resource to deploy this turn")
    target_id: str = Field(..., description="Target to support this turn")


class DisasterAction(Action):
    """
    Action for the disaster response ladder.

    Each turn the agent assigns scarce resources to targets. A resource may only
    appear once in the action list for the turn.
    """

    assignments: List[ResourceAssignment] = Field(
        default_factory=list,
        description=(
            "Per-turn resource assignments. Each item maps one resource_id to "
            "one target_id. Resources not listed remain idle."
        ),
    )


class ResourceStatus(BaseModel):
    """Visible status for a deployable resource."""

    name: str = Field(..., description="Human-readable resource name")
    capabilities: List[str] = Field(
        default_factory=list,
        description="Operational capabilities this resource can provide",
    )
    available: bool = Field(..., description="Whether the resource can be deployed")
    remaining_uses: Optional[int] = Field(
        default=None,
        description="How many episode-wide uses remain, if finite",
    )
    available_until_turn: Optional[int] = Field(
        default=None,
        description="Last turn on which the resource can still be used, if limited",
    )
    description: str = Field(..., description="Operational description")


class TargetStatus(BaseModel):
    """Visible status for a response target within the current scene."""

    name: str = Field(..., description="Human-readable target name")
    category: str = Field(..., description="Target category such as victims or infrastructure")
    status: str = Field(
        ...,
        description="One of: active, contained, resolved, or failed",
    )
    estimated_people: str = Field(
        ...,
        description="Visible people estimate or affected population note",
    )
    observed_risk: float = Field(
        ...,
        description="Observed urgency signal in the range 0.0 to 1.0",
    )
    critical_now: bool = Field(
        ...,
        description="Whether this target is in an immediate decision window",
    )
    priority_band: str = Field(
        ...,
        description="Model-facing priority label: immediate, high, medium, monitor, or failed",
    )
    vulnerability: str = Field(
        ...,
        description="Visible vulnerability band for the target population",
    )
    visibility: float = Field(
        ...,
        description="How visible the incident is publicly, 0.0 to 1.0",
    )
    progress: float = Field(
        ...,
        description="Mitigation progress from 0.0 to 1.0",
    )
    time_remaining: int = Field(
        ...,
        description="Approximate turns before the target becomes much harder to save",
    )
    recommended_capabilities: List[str] = Field(
        default_factory=list,
        description="Capabilities that can materially improve the target",
    )
    last_assigned_resources: List[str] = Field(
        default_factory=list,
        description="Resources deployed to this target on the previous turn",
    )
    description: str = Field(..., description="Operational context and constraints")


class DisasterObservation(Observation):
    """
    Observation returned after each turn of the scene ladder.

    The simulator exposes the operational picture but keeps the full latent harm
    model internal so rewards cannot be self-scored by the agent.
    """

    scene_id: str = Field(..., description="Stable scene identifier")
    scene_name: str = Field(..., description="Human-readable scene name")
    level: int = Field(..., description="Difficulty level for the scene")
    narrative: str = Field(..., description="Top-level scene briefing")
    targets: Dict[str, TargetStatus] = Field(
        default_factory=dict,
        description="Visible target statuses keyed by target ID",
    )
    resources: Dict[str, ResourceStatus] = Field(
        default_factory=dict,
        description="Deployable resource statuses keyed by resource ID",
    )
    resolved_count: int = Field(
        default=0,
        description="Number of targets resolved so far",
    )
    turn: int = Field(default=0, description="Current turn number")
    max_turns: int = Field(default=0, description="Maximum turns in the scene")
    feedback: str = Field(
        default="",
        description="Structured feedback on the last action and simulator update",
    )
    final_score: Optional[float] = Field(
        default=None,
        description="Normalized 0-100 score once the episode is complete",
    )


class DisasterReward(BaseModel):
    """
    Typed reward model for the disaster response ladder.

    OpenEnv responses still carry the scalar reward at step time, but this model
    makes the reward contract explicit for spec compliance and documentation.
    """

    value: float = Field(..., description="Scalar step reward returned by the environment")
    final_score: Optional[float] = Field(
        default=None,
        description="Normalized 0-100 end-of-episode score when available",
    )
    fatalities: Optional[float] = Field(
        default=None,
        description="Cumulative fatalities observed in audit metrics, if available",
    )
    critical_injuries: Optional[float] = Field(
        default=None,
        description="Cumulative critical injuries observed in audit metrics, if available",
    )
    deadline_misses: Optional[float] = Field(
        default=None,
        description="Weighted count of missed critical windows, if available",
    )
    failed_targets: Optional[float] = Field(
        default=None,
        description="Weighted count of targets that reached failed status, if available",
    )
