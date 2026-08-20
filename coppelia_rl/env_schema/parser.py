"""Parses `<rl_env>` XML documents (validated against schema/rl_env.xsd) into EnvSpec."""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    CameraRandomizationSpec,
    DomainRandomizationSpec,
    EnvSpec,
    ObservationSpec,
    RangeRandomizationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
    TextureRandomizationSpec,
)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "rl_env.xsd"
_RANGE_RE = re.compile(r"\[\s*(-?[0-9]*\.?[0-9]+)\s*,\s*(-?[0-9]*\.?[0-9]+)\s*\]")


class EnvXmlValidationError(ValueError):
    pass


def _load_schema() -> etree.XMLSchema:
    return etree.XMLSchema(etree.parse(str(_SCHEMA_PATH)))


def _parse_range(text: str) -> tuple[float, float]:
    match = _RANGE_RE.fullmatch(text.strip())
    if not match:
        raise EnvXmlValidationError(f"Invalid range {text!r}, expected '[lo,hi]'")
    return float(match.group(1)), float(match.group(2))


def _parse_resolution(text: str) -> tuple[int, int]:
    width, _, height = text.partition("x")
    return int(width), int(height)


def parse_env_xml(path: str | Path) -> EnvSpec:
    path = Path(path)
    tree = etree.parse(str(path))

    schema = _load_schema()
    if not schema.validate(tree):
        errors = "\n".join(str(e) for e in schema.error_log)
        raise EnvXmlValidationError(f"{path} failed schema validation:\n{errors}")

    root = tree.getroot()
    scene_path = (path.parent / root.get("scene")).resolve()

    dr_elem = root.find("domain_randomization")

    return EnvSpec(
        name=root.get("name"),
        step_dt=float(root.get("step_dt")),
        scene_path=scene_path,
        observations=_parse_observations(root.find("observations")),
        actions=_parse_actions(root.find("actions")),
        reward_terms=_parse_reward(root.find("reward")),
        termination_conditions=_parse_termination(root.find("termination")),
        domain_randomization=_parse_domain_randomization(dr_elem) if dr_elem is not None else None,
    )


def _parse_observations(elem) -> list[ObservationSpec]:
    specs = []
    seen_keys = set()
    for child in elem:
        tag = etree.QName(child).localname
        ref = child.get("ref")
        # Default key includes the tag, not just ref: joint_position and
        # joint_velocity on the same joint must not collide in the Dict space.
        default_key = f"{tag}_{ref}" if ref else tag
        key = child.get("name") or default_key
        if key in seen_keys:
            raise EnvXmlValidationError(
                f"Duplicate observation key {key!r} - add a distinct name=\"...\" attribute"
            )
        seen_keys.add(key)
        if tag in ("joint_position", "joint_velocity"):
            specs.append(ObservationSpec(kind=tag, key=key, ref=ref))
        elif tag in ("object_pose", "object_position"):
            specs.append(ObservationSpec(kind=tag, key=key, ref=ref, frame=child.get("frame", "world")))
        elif tag == "camera":
            specs.append(
                ObservationSpec(
                    kind=tag,
                    key=key,
                    ref=ref,
                    resolution=_parse_resolution(child.get("resolution")),
                    channels=child.get("channels", "rgb"),
                )
            )
        elif tag == "custom":
            specs.append(ObservationSpec(kind=tag, key=child.get("name"), callable=child.get("callable")))
        else:
            raise EnvXmlValidationError(f"Unknown observation element <{tag}>")
    return specs


def _parse_actions(elem) -> ActionGroupSpec:
    action_type = elem.get("type")
    entries = []
    for child in elem:
        tag = etree.QName(child).localname
        if tag in ("joint_velocity", "joint_position"):
            if action_type != "continuous":
                raise EnvXmlValidationError(f'<{tag}> is only valid inside actions type="continuous"')
            ref = child.get("ref")
            entries.append(
                ActionEntrySpec(kind=tag, key=ref, ref=ref, value_range=_parse_range(child.get("range")))
            )
        elif tag == "event":
            if action_type != "discrete":
                raise EnvXmlValidationError('<event> is only valid inside actions type="discrete"')
            entries.append(ActionEntrySpec(kind=tag, key=child.get("name"), signal=child.get("signal")))
        elif tag == "custom":
            entries.append(ActionEntrySpec(kind=tag, key=child.get("name"), callable=child.get("callable")))
        else:
            raise EnvXmlValidationError(f"Unknown action element <{tag}>")
    return ActionGroupSpec(action_type=action_type, entries=entries)


def _parse_reward(elem) -> list[RewardTermSpec]:
    return [
        RewardTermSpec(
            kind=term_elem.get("type"),
            weight=float(term_elem.get("weight")),
            from_ref=term_elem.get("from"),
            to_ref=term_elem.get("to"),
            object1=term_elem.get("object1"),
            object2=term_elem.get("object2"),
            signal=term_elem.get("signal"),
            callable=term_elem.get("callable"),
        )
        for term_elem in elem.findall("term")
    ]


def _parse_termination(elem) -> list[TerminationConditionSpec]:
    conditions = []
    for cond_elem in elem.findall("condition"):
        value = cond_elem.get("value")
        conditions.append(
            TerminationConditionSpec(
                kind=cond_elem.get("type"),
                name=cond_elem.get("name"),
                value=float(value) if value is not None else None,
                callable=cond_elem.get("callable"),
            )
        )
    return conditions


def _parse_domain_randomization(elem) -> DomainRandomizationSpec:
    spec = DomainRandomizationSpec()

    visual = elem.find("visual")
    if visual is not None:
        for texture_elem in visual.findall("texture"):
            spec.textures.append(
                TextureRandomizationSpec(ref=texture_elem.get("ref"), pool=texture_elem.get("pool"))
            )
        for camera_elem in visual.findall("camera"):
            jitter_pos = camera_elem.get("jitter_pos")
            jitter_rot = camera_elem.get("jitter_rot_deg")
            spec.cameras.append(
                CameraRandomizationSpec(
                    ref=camera_elem.get("ref"),
                    jitter_pos=float(jitter_pos) if jitter_pos is not None else None,
                    jitter_rot_deg=float(jitter_rot) if jitter_rot is not None else None,
                )
            )

    dynamics = elem.find("dynamics")
    if dynamics is not None:
        for mass_elem in dynamics.findall("mass"):
            spec.masses.append(
                RangeRandomizationSpec(ref=mass_elem.get("ref"), value_range=_parse_range(mass_elem.get("range")))
            )
        for friction_elem in dynamics.findall("friction"):
            spec.frictions.append(
                RangeRandomizationSpec(
                    ref=friction_elem.get("ref"), value_range=_parse_range(friction_elem.get("range"))
                )
            )

    latency = elem.find("latency")
    if latency is not None:
        delay_elem = latency.find("action_delay_steps")
        if delay_elem is not None:
            lo, hi = _parse_range(delay_elem.get("range"))
            spec.action_delay_steps = (int(lo), int(hi))
        noise_elem = latency.find("observation_noise")
        if noise_elem is not None:
            spec.observation_noise_std = float(noise_elem.get("std"))

    resample_on_elem = elem.find("resample_on")
    if resample_on_elem is not None and resample_on_elem.text:
        spec.resample_on = resample_on_elem.text.strip()
        every_n_steps = resample_on_elem.get("every_n_steps")
        if every_n_steps is not None:
            spec.resample_every_n_steps = int(every_n_steps)

    return spec
