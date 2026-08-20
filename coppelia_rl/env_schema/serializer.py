"""Serializes an EnvSpec back to `<rl_env>` XML - the inverse of parser.py.

Used by the GUI editor's "View XML"/"Save XML" actions, and independently
round-trip tested (parse(serialize(spec)) == spec) without needing the GUI.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    DomainRandomizationSpec,
    EnvSpec,
    ObservationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
)


def _format_range(value_range: tuple[float, float]) -> str:
    lo, hi = value_range
    return f"[{lo:g},{hi:g}]"


def _set_if_not_none(elem, attr: str, value) -> None:
    if value is not None:
        elem.set(attr, str(value))


def _serialize_observation(parent, obs: ObservationSpec) -> None:
    elem = etree.SubElement(parent, obs.kind)
    if obs.kind == "custom":
        elem.set("name", obs.key)
        elem.set("callable", obs.callable)
        return

    elem.set("ref", obs.ref)
    default_key = f"{obs.kind}_{obs.ref}"
    if obs.key != default_key:
        elem.set("name", obs.key)

    if obs.kind in ("object_pose", "object_position"):
        elem.set("frame", obs.frame or "world")
    elif obs.kind == "camera":
        width, height = obs.resolution
        elem.set("resolution", f"{width}x{height}")
        elem.set("channels", obs.channels or "rgb")


def _serialize_action_entry(parent, entry: ActionEntrySpec) -> None:
    elem = etree.SubElement(parent, entry.kind)
    if entry.kind in ("joint_velocity", "joint_position"):
        elem.set("ref", entry.ref)
        elem.set("range", _format_range(entry.value_range))
    elif entry.kind == "event":
        elem.set("name", entry.key)
        elem.set("signal", entry.signal)
    elif entry.kind == "custom":
        elem.set("name", entry.key)
        elem.set("callable", entry.callable)


def _serialize_reward_term(parent, term: RewardTermSpec) -> None:
    elem = etree.SubElement(parent, "term")
    elem.set("type", term.kind)
    elem.set("weight", f"{term.weight:g}")
    if term.kind == "distance":
        elem.set("from", term.from_ref)
        elem.set("to", term.to_ref)
    elif term.kind == "collision":
        elem.set("object1", term.object1)
        elem.set("object2", term.object2)
    elif term.kind == "signal_event":
        elem.set("signal", term.signal)
    elif term.kind == "custom":
        elem.set("callable", term.callable)


def _serialize_termination_condition(parent, cond: TerminationConditionSpec) -> None:
    elem = etree.SubElement(parent, "condition")
    elem.set("type", cond.kind)
    if cond.kind == "signal":
        elem.set("name", cond.name)
    elif cond.kind == "max_steps":
        elem.set("value", f"{cond.value:g}")
    elif cond.kind == "custom":
        elem.set("callable", cond.callable)


def _serialize_domain_randomization(parent, dr: DomainRandomizationSpec) -> None:
    elem = etree.SubElement(parent, "domain_randomization")

    if dr.textures or dr.cameras:
        visual = etree.SubElement(elem, "visual")
        for texture in dr.textures:
            t = etree.SubElement(visual, "texture")
            t.set("ref", texture.ref)
            t.set("pool", texture.pool)
        for camera in dr.cameras:
            c = etree.SubElement(visual, "camera")
            c.set("ref", camera.ref)
            _set_if_not_none(c, "jitter_pos", camera.jitter_pos)
            _set_if_not_none(c, "jitter_rot_deg", camera.jitter_rot_deg)

    if dr.masses or dr.frictions:
        dynamics = etree.SubElement(elem, "dynamics")
        for mass in dr.masses:
            m = etree.SubElement(dynamics, "mass")
            m.set("ref", mass.ref)
            m.set("range", _format_range(mass.value_range))
        for friction in dr.frictions:
            f = etree.SubElement(dynamics, "friction")
            f.set("ref", friction.ref)
            f.set("range", _format_range(friction.value_range))

    if dr.action_delay_steps is not None or dr.observation_noise_std is not None:
        latency = etree.SubElement(elem, "latency")
        if dr.action_delay_steps is not None:
            delay = etree.SubElement(latency, "action_delay_steps")
            delay.set("range", _format_range(dr.action_delay_steps))
        if dr.observation_noise_std is not None:
            noise = etree.SubElement(latency, "observation_noise")
            noise.set("std", f"{dr.observation_noise_std:g}")

    resample_on = etree.SubElement(elem, "resample_on")
    resample_on.text = dr.resample_on
    if dr.resample_every_n_steps is not None:
        resample_on.set("every_n_steps", str(dr.resample_every_n_steps))


def spec_to_element(spec: EnvSpec, scene_path: str | Path | None = None) -> etree._Element:
    """Builds the `<rl_env>` XML element tree for `spec`.

    `scene_path` overrides `spec.scene_path` in the output (e.g. to write a
    path relative to where the caller is about to save the file); defaults to
    `spec.scene_path` (always absolute, since that's what parse_env_xml produces).
    """
    root = etree.Element("rl_env")
    root.set("name", spec.name)
    root.set("step_dt", f"{spec.step_dt:g}")
    root.set("scene", str(scene_path if scene_path is not None else spec.scene_path))

    observations = etree.SubElement(root, "observations")
    for obs in spec.observations:
        _serialize_observation(observations, obs)

    actions = etree.SubElement(root, "actions")
    actions.set("type", spec.actions.action_type)
    for entry in spec.actions.entries:
        _serialize_action_entry(actions, entry)

    reward = etree.SubElement(root, "reward")
    for term in spec.reward_terms:
        _serialize_reward_term(reward, term)

    termination = etree.SubElement(root, "termination")
    for cond in spec.termination_conditions:
        _serialize_termination_condition(termination, cond)

    if spec.domain_randomization is not None:
        _serialize_domain_randomization(root, spec.domain_randomization)

    return root


def spec_to_xml(spec: EnvSpec, scene_path: str | Path | None = None) -> str:
    """Renders `spec` as a pretty-printed, UTF-8 XML document string."""
    root = spec_to_element(spec, scene_path=scene_path)
    return etree.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")


def save_spec_xml(spec: EnvSpec, path: str | Path, scene_path: str | Path | None = None) -> None:
    Path(path).write_text(spec_to_xml(spec, scene_path=scene_path), encoding="utf-8")
