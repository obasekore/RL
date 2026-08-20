"""GUI-framework-free state for the RL Env Editor add-on.

Owns an in-progress EnvSpec under construction: add/remove/update entries,
load from a parsed EnvSpec, validate refs against a SimClient (real or
FakeSim), and assemble the current state back into an EnvSpec. The add-on
script (addOns/RL Env Editor.py) is a thin simUI wrapper around this class,
kept that way specifically so this logic - the actual behavior - can be
pytest-covered without CoppeliaSim.
"""

from __future__ import annotations

from pathlib import Path

from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    EnvSpec,
    ObservationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
)
from coppelia_rl.randomization.randomizer_state import DomainRandomizerState
from coppelia_rl.sim_interface.client import SimClient


def _ref_path(ref: str) -> str:
    return ref if ref.startswith("/") else f"/{ref}"


class EnvBuilderState:
    def __init__(self) -> None:
        self.name: str = "new_env"
        self.step_dt: float = 0.05
        self.scene_path: Path | None = None
        self.observations: list[ObservationSpec] = []
        self.actions: ActionGroupSpec = ActionGroupSpec(action_type="continuous", entries=[])
        self.reward_terms: list[RewardTermSpec] = []
        self.termination_conditions: list[TerminationConditionSpec] = []
        self.dr = DomainRandomizerState()

    @property
    def domain_randomization(self):
        return self.dr.domain_randomization

    # -- loading / building -------------------------------------------------

    def load(self, spec: EnvSpec) -> None:
        self.name = spec.name
        self.step_dt = spec.step_dt
        self.scene_path = spec.scene_path
        self.observations = list(spec.observations)
        self.actions = ActionGroupSpec(action_type=spec.actions.action_type, entries=list(spec.actions.entries))
        self.reward_terms = list(spec.reward_terms)
        self.termination_conditions = list(spec.termination_conditions)
        self.dr.load(spec.domain_randomization)

    def build_spec(self) -> EnvSpec:
        if self.scene_path is None:
            raise ValueError("scene_path is not set")
        if not self.observations:
            raise ValueError("at least one observation is required")
        if not self.actions.entries:
            raise ValueError("at least one action entry is required")
        if not self.reward_terms:
            raise ValueError("at least one reward term is required")
        if not self.termination_conditions:
            raise ValueError("at least one termination condition is required")

        return EnvSpec(
            name=self.name,
            step_dt=self.step_dt,
            scene_path=self.scene_path,
            observations=list(self.observations),
            actions=ActionGroupSpec(action_type=self.actions.action_type, entries=list(self.actions.entries)),
            reward_terms=list(self.reward_terms),
            termination_conditions=list(self.termination_conditions),
            domain_randomization=self.dr.domain_randomization if self.dr.has_domain_randomization() else None,
        )

    # -- observations ---------------------------------------------------------

    def add_observation(
        self,
        kind: str,
        ref: str | None = None,
        name: str | None = None,
        frame: str | None = None,
        resolution: tuple[int, int] | None = None,
        channels: str | None = None,
        callable: str | None = None,
    ) -> ObservationSpec:
        if kind == "custom":
            if not name:
                raise ValueError("custom observations require a name")
            key = name
        else:
            if not ref:
                raise ValueError(f"{kind} observations require a ref")
            key = name or f"{kind}_{ref}"

        if key in {obs.key for obs in self.observations}:
            raise ValueError(f"Duplicate observation key {key!r}")

        obs = ObservationSpec(
            kind=kind, key=key, ref=ref, frame=frame, resolution=resolution, channels=channels, callable=callable
        )
        self.observations.append(obs)
        return obs

    def remove_observation(self, index: int) -> None:
        del self.observations[index]

    # -- actions ------------------------------------------------------------------

    def set_action_type(self, action_type: str) -> None:
        """Switches continuous<->discrete. Existing entries are type-specific
        (joint_velocity/joint_position vs. event), so they're cleared."""
        if action_type not in ("continuous", "discrete"):
            raise ValueError(f"Unknown action type {action_type!r}")
        self.actions = ActionGroupSpec(action_type=action_type, entries=[])

    def add_action_entry(
        self,
        kind: str,
        ref: str | None = None,
        name: str | None = None,
        value_range: tuple[float, float] | None = None,
        signal: str | None = None,
        callable: str | None = None,
    ) -> ActionEntrySpec:
        if kind in ("joint_velocity", "joint_position"):
            if self.actions.action_type != "continuous":
                raise ValueError(f'{kind!r} is only valid when action type is "continuous"')
            if not ref or value_range is None:
                raise ValueError(f"{kind} action entries require ref and value_range")
            entry = ActionEntrySpec(kind=kind, key=ref, ref=ref, value_range=value_range)
        elif kind == "event":
            if self.actions.action_type != "discrete":
                raise ValueError('"event" is only valid when action type is "discrete"')
            if not name or not signal:
                raise ValueError("event action entries require name and signal")
            entry = ActionEntrySpec(kind=kind, key=name, signal=signal)
        elif kind == "custom":
            if not name or not callable:
                raise ValueError("custom action entries require name and callable")
            entry = ActionEntrySpec(kind=kind, key=name, callable=callable)
        else:
            raise ValueError(f"Unknown action kind {kind!r}")

        self.actions.entries.append(entry)
        return entry

    def remove_action_entry(self, index: int) -> None:
        del self.actions.entries[index]

    # -- reward -----------------------------------------------------------------

    def add_reward_term(
        self,
        kind: str,
        weight: float,
        from_ref: str | None = None,
        to_ref: str | None = None,
        object1: str | None = None,
        object2: str | None = None,
        signal: str | None = None,
        callable: str | None = None,
    ) -> RewardTermSpec:
        if kind == "distance" and not (from_ref and to_ref):
            raise ValueError("distance reward terms require from_ref and to_ref")
        if kind == "collision" and not (object1 and object2):
            raise ValueError("collision reward terms require object1 and object2")
        if kind == "signal_event" and not signal:
            raise ValueError("signal_event reward terms require signal")
        if kind == "custom" and not callable:
            raise ValueError("custom reward terms require callable")

        term = RewardTermSpec(
            kind=kind,
            weight=weight,
            from_ref=from_ref,
            to_ref=to_ref,
            object1=object1,
            object2=object2,
            signal=signal,
            callable=callable,
        )
        self.reward_terms.append(term)
        return term

    def remove_reward_term(self, index: int) -> None:
        del self.reward_terms[index]

    # -- termination ------------------------------------------------------------

    def add_termination_condition(
        self,
        kind: str,
        name: str | None = None,
        value: float | None = None,
        callable: str | None = None,
    ) -> TerminationConditionSpec:
        if kind == "signal" and not name:
            raise ValueError("signal termination conditions require name")
        if kind == "max_steps" and value is None:
            raise ValueError("max_steps termination conditions require value")
        if kind == "custom" and not callable:
            raise ValueError("custom termination conditions require callable")

        cond = TerminationConditionSpec(kind=kind, name=name, value=value, callable=callable)
        self.termination_conditions.append(cond)
        return cond

    def remove_termination_condition(self, index: int) -> None:
        del self.termination_conditions[index]

    # -- domain randomization (delegates to self.dr - see randomizer_state.py) ----

    def add_texture_randomization(self, ref: str, pool: str) -> None:
        self.dr.add_texture_randomization(ref, pool)

    def remove_texture_randomization(self, index: int) -> None:
        self.dr.remove_texture_randomization(index)

    def add_camera_randomization(
        self, ref: str, jitter_pos: float | None = None, jitter_rot_deg: float | None = None
    ) -> None:
        self.dr.add_camera_randomization(ref, jitter_pos, jitter_rot_deg)

    def remove_camera_randomization(self, index: int) -> None:
        self.dr.remove_camera_randomization(index)

    def add_mass_randomization(self, ref: str, value_range: tuple[float, float]) -> None:
        self.dr.add_mass_randomization(ref, value_range)

    def remove_mass_randomization(self, index: int) -> None:
        self.dr.remove_mass_randomization(index)

    def add_friction_randomization(self, ref: str, value_range: tuple[float, float]) -> None:
        self.dr.add_friction_randomization(ref, value_range)

    def remove_friction_randomization(self, index: int) -> None:
        self.dr.remove_friction_randomization(index)

    def set_latency(
        self, action_delay_steps: tuple[int, int] | None = None, observation_noise_std: float | None = None
    ) -> None:
        self.dr.set_latency(action_delay_steps, observation_noise_std)

    def set_resample_on(self, resample_on: str, every_n_steps: int | None = None) -> None:
        self.dr.set_resample_on(resample_on, every_n_steps)

    # -- validation --------------------------------------------------------------

    def collect_scene_refs(self) -> list[tuple[str, str]]:
        """Returns (ref, resolver_kind) pairs for every scene-object ref in the
        current state (custom callables and signal names are excluded - those
        aren't scene refs)."""
        refs: list[tuple[str, str]] = []

        for obs in self.observations:
            if obs.kind in ("joint_position", "joint_velocity"):
                refs.append((obs.ref, "joint"))
            elif obs.kind in ("object_pose", "object_position"):
                refs.append((obs.ref, "object"))
                if obs.frame and obs.frame != "world":
                    refs.append((obs.frame, "object"))
            elif obs.kind == "camera":
                refs.append((obs.ref, "vision_sensor"))

        if self.actions.action_type == "continuous":
            for entry in self.actions.entries:
                if entry.kind in ("joint_velocity", "joint_position"):
                    refs.append((entry.ref, "joint"))

        for term in self.reward_terms:
            if term.kind == "distance":
                refs.append((term.from_ref, "object"))
                refs.append((term.to_ref, "object"))
            elif term.kind == "collision":
                refs.append((term.object1, "object"))
                refs.append((term.object2, "object"))

        return refs

    def validate_refs(self, client: SimClient) -> list[str]:
        """Attempts to resolve every scene ref against `client`. Returns the
        refs that failed to resolve (empty list means everything resolved)."""
        resolvers = {
            "joint": client.get_joint,
            "object": client.get_object,
            "vision_sensor": client.get_vision_sensor,
        }
        missing: list[str] = []
        for ref, kind in self.collect_scene_refs():
            try:
                resolvers[kind](_ref_path(ref))
            except Exception:
                missing.append(ref)
        return missing
