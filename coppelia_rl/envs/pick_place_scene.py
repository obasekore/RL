"""Procedurally builds (or reloads) the pick-and-place task scene.

Like the reach scene, this is a schema-generality proof, not a physically
realistic grasp task: the target cube is a plain static primitive shape
repositioned between episodes, not a dynamically simulated, graspable object.
Its purpose is validating that the XML schema/parser handle a second task
shape (plus a wrist camera, to exercise Dict-space vector+image mixing).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coppelia_rl.envs.ur5_arm import load_ur5_arm, resolve_ur5_arm
from coppelia_rl.sim_interface.client import SimClient
from coppelia_rl.sim_interface.objects import Joint, SceneObject
from coppelia_rl.sim_interface.vision import VisionSensor


@dataclass
class PickPlaceScene:
    base: SceneObject
    joints: list[Joint]
    tip: SceneObject
    target_cube: SceneObject
    wrist_cam: VisionSensor


def ensure_pick_and_place_scene(
    client: SimClient,
    scene_path: str | Path,
    ur5_model_path: str | Path | None = None,
) -> PickPlaceScene:
    scene_path = Path(scene_path)

    if scene_path.exists():
        client.load_scene(scene_path)
        arm = resolve_ur5_arm(client)
        target_cube = client.get_object("/target_cube")
        wrist_cam = client.get_vision_sensor("/wrist_cam")
        _position_wrist_cam(wrist_cam, arm.tip)
        return PickPlaceScene(
            base=arm.base, joints=arm.joints, tip=arm.tip, target_cube=target_cube, wrist_cam=wrist_cam
        )

    client.close_scene()
    arm = load_ur5_arm(client, ur5_model_path)

    target_cube = client.create_primitive_shape("cuboid", [0.05, 0.05, 0.05])
    target_cube.set_name("target_cube")
    target_cube.set_position(sample_cube_position(np.random.default_rng()))

    wrist_cam = client.create_vision_sensor(resolution=(128, 128))
    wrist_cam.set_name("wrist_cam")
    wrist_cam.set_parent(arm.tip, keep_in_place=False)
    _position_wrist_cam(wrist_cam, arm.tip)

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    client.save_scene(scene_path)

    return PickPlaceScene(
        base=arm.base, joints=arm.joints, tip=arm.tip, target_cube=target_cube, wrist_cam=wrist_cam
    )


def _position_wrist_cam(wrist_cam, tip) -> None:
    """Offset from the tip dummy the camera is parented to.

    The original [0, 0, 0.05] placement put the sensor close enough to the
    UR5's own wrist/flange geometry to self-occlude - the view was blank
    with a real object directly in front of it. Shifted -0.05 on X to clear
    that self-occlusion (found empirically against a live instance). Applied
    on every resolve, not just fresh builds, so it also self-heals
    already-saved scenes built before this fix.
    """
    wrist_cam.set_position([-0.05, 0.0, 0.05], relative_to=tip.handle)


def sample_cube_position(rng: np.random.Generator) -> np.ndarray:
    """Samples a target-cube position within a reachable envelope on a nominal tabletop."""
    x = rng.uniform(0.3, 0.6)
    y = rng.uniform(-0.3, 0.3)
    return np.array([x, y, 0.05], dtype=np.float64)
