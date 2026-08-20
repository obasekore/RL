"""Dataclasses mirroring the `<rl_env>` XML document (see schema/rl_env.xsd).

The parser (parser.py) produces these from XML; generic_env.py consumes them
to build a gymnasium.Env. Keeping these as plain dataclasses (rather than
generating a dynamic gym.Env subclass) keeps the XML->space construction
logic in one testable place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ObservationSpec:
    kind: str  # joint_position | joint_velocity | object_pose | object_position | camera | custom
    key: str  # Dict observation-space key
    ref: str | None = None
    frame: str | None = None  # object_pose / object_position
    resolution: tuple[int, int] | None = None  # camera
    channels: str | None = None  # camera: rgb | depth
    callable: str | None = None  # custom


@dataclass
class ActionEntrySpec:
    kind: str  # joint_velocity | joint_position | event | custom
    key: str
    ref: str | None = None
    value_range: tuple[float, float] | None = None  # joint_*
    signal: str | None = None  # event
    callable: str | None = None  # custom


@dataclass
class ActionGroupSpec:
    action_type: str  # continuous | discrete
    entries: list[ActionEntrySpec] = field(default_factory=list)


@dataclass
class RewardTermSpec:
    kind: str  # distance | collision | signal_event | custom
    weight: float
    from_ref: str | None = None
    to_ref: str | None = None
    object1: str | None = None
    object2: str | None = None
    signal: str | None = None
    callable: str | None = None


@dataclass
class TerminationConditionSpec:
    kind: str  # signal | max_steps | custom
    name: str | None = None
    value: float | None = None
    callable: str | None = None


@dataclass
class TextureRandomizationSpec:
    ref: str
    pool: str


@dataclass
class CameraRandomizationSpec:
    ref: str
    jitter_pos: float | None = None
    jitter_rot_deg: float | None = None


@dataclass
class RangeRandomizationSpec:
    ref: str
    value_range: tuple[float, float]


@dataclass
class DomainRandomizationSpec:
    textures: list[TextureRandomizationSpec] = field(default_factory=list)
    cameras: list[CameraRandomizationSpec] = field(default_factory=list)
    masses: list[RangeRandomizationSpec] = field(default_factory=list)
    frictions: list[RangeRandomizationSpec] = field(default_factory=list)
    action_delay_steps: tuple[int, int] | None = None
    observation_noise_std: float | None = None
    resample_on: str = "episode_start"
    resample_every_n_steps: int | None = None  # required when resample_on == "n_steps"


@dataclass
class EnvSpec:
    name: str
    step_dt: float
    scene_path: Path
    observations: list[ObservationSpec]
    actions: ActionGroupSpec
    reward_terms: list[RewardTermSpec]
    termination_conditions: list[TerminationConditionSpec]
    domain_randomization: DomainRandomizationSpec | None = None
