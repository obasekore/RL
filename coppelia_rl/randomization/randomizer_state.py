"""GUI-framework-free state for authoring a DomainRandomizationSpec.

Shared by both `EnvBuilderState` (the RL Env Editor's Domain Randomization
tab, where it's one part of a larger EnvSpec) and the standalone "RL Domain
Randomizer" add-on (where it's the whole thing - no EnvSpec involved). This
is the single place that owns add/remove/update logic for randomization
entries, so the two add-ons can't drift into separate implementations.
"""

from __future__ import annotations

from pathlib import Path

from coppelia_rl.env_schema.spec import (
    CameraRandomizationSpec,
    DomainRandomizationSpec,
    RangeRandomizationSpec,
    TextureRandomizationSpec,
)

_RESAMPLE_ON_VALUES = ("episode_start", "n_steps", "once")


class DomainRandomizerState:
    def __init__(self) -> None:
        self.domain_randomization: DomainRandomizationSpec = DomainRandomizationSpec()

    def load(self, dr: DomainRandomizationSpec | None) -> None:
        self.domain_randomization = dr or DomainRandomizationSpec()

    def load_from_env_xml(self, xml_path: str | Path) -> None:
        """Pulls just the <domain_randomization> block out of a full <rl_env>
        XML file - the "enable by parsing an rl-env definition" path for the
        standalone add-on."""
        from coppelia_rl.env_schema.parser import parse_env_xml

        spec = parse_env_xml(xml_path)
        self.load(spec.domain_randomization)

    def has_domain_randomization(self) -> bool:
        dr = self.domain_randomization
        return bool(
            dr.textures
            or dr.cameras
            or dr.masses
            or dr.frictions
            or dr.action_delay_steps is not None
            or dr.observation_noise_std is not None
        )

    # -- visual -----------------------------------------------------------------

    def add_texture_randomization(self, ref: str, pool: str) -> None:
        self.domain_randomization.textures.append(TextureRandomizationSpec(ref=ref, pool=pool))

    def remove_texture_randomization(self, index: int) -> None:
        del self.domain_randomization.textures[index]

    def add_camera_randomization(
        self, ref: str, jitter_pos: float | None = None, jitter_rot_deg: float | None = None
    ) -> None:
        self.domain_randomization.cameras.append(
            CameraRandomizationSpec(ref=ref, jitter_pos=jitter_pos, jitter_rot_deg=jitter_rot_deg)
        )

    def remove_camera_randomization(self, index: int) -> None:
        del self.domain_randomization.cameras[index]

    # -- dynamics ---------------------------------------------------------------

    def add_mass_randomization(self, ref: str, value_range: tuple[float, float]) -> None:
        self.domain_randomization.masses.append(RangeRandomizationSpec(ref=ref, value_range=value_range))

    def remove_mass_randomization(self, index: int) -> None:
        del self.domain_randomization.masses[index]

    def add_friction_randomization(self, ref: str, value_range: tuple[float, float]) -> None:
        self.domain_randomization.frictions.append(RangeRandomizationSpec(ref=ref, value_range=value_range))

    def remove_friction_randomization(self, index: int) -> None:
        del self.domain_randomization.frictions[index]

    # -- latency + resample policy ------------------------------------------------

    def set_latency(
        self, action_delay_steps: tuple[int, int] | None = None, observation_noise_std: float | None = None
    ) -> None:
        self.domain_randomization.action_delay_steps = action_delay_steps
        self.domain_randomization.observation_noise_std = observation_noise_std

    def set_resample_on(self, resample_on: str, every_n_steps: int | None = None) -> None:
        if resample_on not in _RESAMPLE_ON_VALUES:
            raise ValueError(f"Unknown resample_on {resample_on!r}")
        self.domain_randomization.resample_on = resample_on
        self.domain_randomization.resample_every_n_steps = every_n_steps
